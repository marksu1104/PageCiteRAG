"""Global questions over a whole document: map over sections, reduce over verified evidence.

Pattern: LangChain's map-reduce summarization and GraphRAG global search (a partial answer
per part of the corpus, then one answer built from those parts). Both steps reuse the product
writer, so every kept point rests on verbatim quotes that passed the citation checks:

- map: the document is cut into contiguous sections of about SECTION_TOKENS tokens; each
  section is asked what it says that helps answer the question;
- reduce: the blocks cited by the map answers, in document order, go through the writer once
  with the original question, plus up to eight verified map points as unverified hints.

A document that fits the writer's context is answered in one call, like the detail route.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from page_cite_rag.generation import ContextBudgetError, OutputValidationError
from page_cite_rag.models import Block

GLOBAL_VERSION = "global-map-reduce/v1"
SECTION_TOKENS = 5000
MAX_SECTIONS = 15
MAX_FINDINGS = 8
FINDING_CHARS = 200
_CJK = re.compile(r"[㐀-鿿豈-﫿]")

MAP_QUESTION = {
    "en": "Using only this part of the document, list what it says that helps answer this "
    "question: {question}",
    "zh-Hant": "只根據這一部分文件，列出其中有助於回答以下問題的內容：{question}",
}


@dataclass(frozen=True)
class GlobalAnswer:
    result: object
    trace: dict = field(default_factory=dict)


def estimated_tokens(text: str) -> int:
    """Rough token count for planning section sizes only (admission stays exact)."""
    cjk = len(_CJK.findall(text))
    return math.ceil(cjk / 1.5 + (len(text) - cjk) / 4)


def plan_sections(blocks: Sequence[Block], *, section_tokens: int = SECTION_TOKENS):
    """Contiguous sections of roughly equal size, between 2 and MAX_SECTIONS."""
    sizes = [estimated_tokens(block.text) for block in blocks]
    total = sum(sizes)
    count = max(2, min(MAX_SECTIONS, math.ceil(total / section_tokens)))
    count = min(count, len(blocks))
    target = total / count
    sections, current, running = [], [], 0
    for block, size in zip(blocks, sizes):
        current.append(block)
        running += size
        remaining_sections = count - len(sections) - 1
        if running >= target * (len(sections) + 1) and remaining_sections > 0:
            sections.append(current)
            current = []
    if current:
        sections.append(current)
    return sections


def _claims(result) -> tuple:
    return (*getattr(result, "answer_claims", ()), *getattr(result, "detail_claims", ()))


def _usage(result) -> tuple[int, int]:
    stats = getattr(result, "generation_stats", None)
    if stats is None:
        return 0, 0
    return stats.prompt_eval_count or 0, stats.eval_count or 0


def select_findings(section_results: Sequence, blocks_by_id: dict[str, Block]) -> list:
    """Round-robin over sections: each section's first verified point, then its second..."""
    queues = []
    for result in section_results:
        points = []
        for claim in _claims(result):
            cited = [c.block_id for c in claim.citations if c.block_id in blocks_by_id]
            if cited:
                text = " ".join(claim.text.split())[:FINDING_CHARS]
                points.append((text, f"P{blocks_by_id[cited[0]].ordinal}"))
        queues.append(points)
    findings, depth = [], 0
    while len(findings) < MAX_FINDINGS and any(depth < len(q) for q in queues):
        for queue in queues:
            if depth < len(queue) and len(findings) < MAX_FINDINGS:
                findings.append(queue[depth])
        depth += 1
    return findings


def priority_order(sections: Sequence[Sequence[Block]], cited: dict[str, int]) -> list[Block]:
    """Cited blocks, sections taking turns, most-cited first within a section."""
    queues = [
        sorted(
            (block for block in section if block.block_id in cited),
            key=lambda b: (-cited[b.block_id], b.ordinal),
        )
        for section in sections
    ]
    ordered, depth = [], 0
    while any(depth < len(q) for q in queues):
        ordered.extend(q[depth] for q in queues if depth < len(q))
        depth += 1
    return ordered


def reduce_context(workflow, question, language, sections, cited, order) -> list[Block]:
    """Keep what the writer admits in priority order, then restore document order."""
    candidates = priority_order(sections, cited)
    selected, _omitted, _messages, _budget = workflow._context(
        question, language, "ask", candidates, None
    )
    return sorted(selected, key=lambda b: order[b.block_id])


def answer_global(workflow, question: str, *, language: str = "zh-Hant") -> GlobalAnswer:
    """Answer a whole-document question; the trace records every step and its token use."""
    from page_cite_rag.workflows import resolve_answer_language

    language = resolve_answer_language(question, language)
    if workflow.whole_document_fits(question, language=language):
        blocks = workflow._store.get_blocks(workflow.document_id)
        result = workflow.answer_from_context(question, blocks, language=language)
        prompt, completion = _usage(result)
        return GlobalAnswer(
            result,
            {
                "version": GLOBAL_VERSION,
                "route": "whole_document",
                "calls": 1,
                "prompt_tokens": prompt,
                "completion_tokens": completion,
            },
        )
    blocks = workflow._store.get_blocks(workflow.document_id)
    sections = plan_sections(blocks)
    map_question = MAP_QUESTION[language].format(question=question)
    section_results, section_trace = [], []
    prompt_total = completion_total = 0
    last_error: Exception | None = None
    for index, section in enumerate(sections, 1):
        try:
            result = workflow.answer_from_context(map_question, section, language=language)
        except (OutputValidationError, ContextBudgetError) as exc:
            last_error = exc
            # One section's over-long or unparsable reply drops that section's points only;
            # GPU stops, timeouts and connection failures still end the whole answer.
            section_trace.append(
                {
                    "section": index,
                    "first_block": section[0].ordinal,
                    "last_block": section[-1].ordinal,
                    "status": "failed",
                    "error": type(exc).__name__,
                    "points": 0,
                }
            )
            continue
        prompt, completion = _usage(result)
        prompt_total += prompt
        completion_total += completion
        section_results.append(result)
        section_trace.append(
            {
                "section": index,
                "first_block": section[0].ordinal,
                "last_block": section[-1].ordinal,
                "status": result.status,
                "points": len(_claims(result)),
                "omitted_blocks": len(getattr(result, "omitted_block_ids", ())),
            }
        )
    order = {block.block_id: position for position, block in enumerate(blocks)}
    cited: dict[str, int] = {}
    for result in section_results:
        for claim in _claims(result):
            for citation in claim.citations:
                cited[citation.block_id] = cited.get(citation.block_id, 0) + 1
    trace = {
        "version": GLOBAL_VERSION,
        "route": "map_reduce",
        "sections": section_trace,
        "map_question": map_question,
    }
    if not cited:
        trace.update(
            reason="no_verified_points",
            calls=len(sections),
            prompt_tokens=prompt_total,
            completion_tokens=completion_total,
        )
        if not section_results:
            raise last_error
        return GlobalAnswer(section_results[0], trace)
    reduce_blocks = reduce_context(workflow, question, language, sections, cited, order)
    blocks_by_id = {block.block_id: block for block in reduce_blocks}
    findings = select_findings(section_results, blocks_by_id)
    final = workflow.answer_from_context(
        question, reduce_blocks, language=language, search_findings=findings
    )
    prompt, completion = _usage(final)
    trace.update(
        calls=len(sections) + 1,
        reduce_blocks=len(reduce_blocks),
        reduce_omitted_blocks=len(getattr(final, "omitted_block_ids", ())),
        findings=len(findings),
        prompt_tokens=prompt_total + prompt,
        completion_tokens=completion_total + completion,
    )
    return GlobalAnswer(final, trace)
