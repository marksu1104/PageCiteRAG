from dataclasses import dataclass, field

from page_cite_rag import global_answer as ga
from page_cite_rag.models import Block

DOC = "a" * 64


def block(ordinal: int, text: str = "word " * 400) -> Block:
    return Block(
        document_id=DOC,
        block_id=f"{ordinal:064x}",
        ordinal=ordinal,
        unit_index=1,
        line_start=ordinal,
        line_end=ordinal,
        char_start=0,
        char_end=len(text),
        text=text,
    )


@dataclass
class Citation:
    block_id: str


@dataclass
class Claim:
    text: str
    citations: tuple


@dataclass
class Stats:
    prompt_eval_count: int
    eval_count: int


@dataclass
class Result:
    status: str
    answer_claims: tuple = ()
    detail_claims: tuple = ()
    omitted_block_ids: tuple = ()
    generation_stats: Stats = field(default_factory=lambda: Stats(100, 10))


class Store:
    def __init__(self, blocks):
        self.blocks = blocks

    def get_blocks(self, document_id):
        return self.blocks


class Workflow:
    """Duck-typed writer: answers each section by citing its first block."""

    def __init__(self, blocks, fits=False, budget_blocks=None):
        self._store = Store(blocks)
        self.document_id = DOC
        self.fits = fits
        self.budget_blocks = budget_blocks
        self.calls = []

    def whole_document_fits(self, question, *, language):
        return self.fits

    def _context(self, question, language, mode, blocks, requested):
        limit = self.budget_blocks or len(blocks)
        return tuple(blocks[:limit]), (), (), None

    def answer_from_context(self, question, blocks, *, language, search_findings=()):
        self.calls.append((question, [b.ordinal for b in blocks], list(search_findings)))
        if self.fits or question.startswith("Q?"):
            return Result("answered", (Claim("final", (Citation(blocks[0].block_id),)),))
        first = blocks[0]
        return Result(
            "answered",
            (Claim(f"point from {first.ordinal}", (Citation(first.block_id),)),),
        )


def test_sections_are_contiguous_and_bounded():
    blocks = [block(i) for i in range(1, 101)]
    sections = ga.plan_sections(blocks)
    assert 2 <= len(sections) <= ga.MAX_SECTIONS
    assert [b.ordinal for s in sections for b in s] == list(range(1, 101))
    sizes = [len(s) for s in sections]
    assert max(sizes) - min(sizes) <= 2


def test_cjk_text_counts_more_tokens_per_character():
    assert ga.estimated_tokens("中文" * 300) > ga.estimated_tokens("ab" * 300)


def test_small_document_uses_one_whole_document_call():
    workflow = Workflow([block(1), block(2)], fits=True)
    answer = ga.answer_global(workflow, "Q? what is it about", language="en")
    assert answer.trace["route"] == "whole_document"
    assert answer.trace["calls"] == 1
    assert len(workflow.calls) == 1


def test_map_reduce_reads_every_section_then_reduces_cited_blocks_in_order():
    blocks = [block(i) for i in range(1, 41)]
    workflow = Workflow(blocks)
    answer = ga.answer_global(workflow, "Q? what are the contributions", language="en")
    sections = answer.trace["sections"]
    assert answer.trace["route"] == "map_reduce"
    assert answer.trace["calls"] == len(sections) + 1
    *maps, reduce = workflow.calls
    assert all(q.startswith("Using only this part of the document") for q, _, _ in maps)
    assert reduce[0] == "Q? what are the contributions"
    cited = [s["first_block"] for s in sections]
    assert reduce[1] == cited  # each section's cited block, in document order
    assert [pid for _, pid in reduce[2]] == [f"P{o}" for o in cited][: ga.MAX_FINDINGS]
    assert answer.trace["prompt_tokens"] == 100 * (len(sections) + 1)


def test_chinese_map_question_and_no_points_case():
    blocks = [block(i) for i in range(1, 41)]

    class Refusing(Workflow):
        def answer_from_context(self, question, blocks, *, language, search_findings=()):
            self.calls.append((question, [b.ordinal for b in blocks], []))
            return Result("insufficient_evidence")

    workflow = Refusing(blocks)
    answer = ga.answer_global(workflow, "這篇的貢獻是什麼？", language="zh-Hant")
    assert workflow.calls[0][0].startswith("只根據這一部分文件")
    assert answer.trace["reason"] == "no_verified_points"
    assert answer.trace["calls"] == len(answer.trace["sections"])


def test_budget_keeps_every_section_represented():
    sections = [[block(1), block(2)], [block(3), block(4)], [block(5), block(6)]]
    cited = {b.block_id: 1 for s in sections for b in s}
    cited[block(2).block_id] = 5
    ordered = ga.priority_order(sections, cited)
    assert [b.ordinal for b in ordered[:3]] == [2, 3, 5]  # one per section first
    blocks = [b for s in sections for b in s]
    workflow = Workflow(blocks, budget_blocks=3)
    order = {b.block_id: i for i, b in enumerate(blocks)}
    kept = ga.reduce_context(workflow, "Q?", "en", sections, cited, order)
    assert [b.ordinal for b in kept] == [2, 3, 5]


def test_findings_take_turns_and_are_capped():
    blocks = {f"{i:064x}": block(i) for i in range(1, 30)}
    results = [
        Result(
            "answered",
            tuple(Claim("x" * 300, (Citation(f"{i * 3 + j:064x}"),)) for j in range(1, 4)),
        )
        for i in range(6)
    ]
    findings = ga.select_findings(results, blocks)
    assert len(findings) == ga.MAX_FINDINGS
    assert all(len(text) == ga.FINDING_CHARS for text, _ in findings)
    assert [pid for _, pid in findings[:6]] == [f"P{i * 3 + 1}" for i in range(6)]
