"""Small shared application boundary for CLI, local API and evaluation adapters."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal, Sequence

from page_cite_rag.full_answer import FullAnswerWorkflow
from page_cite_rag.hybrid import Retriever
from page_cite_rag.models import Block
from page_cite_rag.retrieval import SearchResult
from page_cite_rag.retrieval_profile import FORMAL_RETRIEVAL
from page_cite_rag.store import SQLiteDocumentStore
from page_cite_rag.workflows import (
    AgentUsage,
    AnswerResult,
    AnswerWorkflow,
    Language,
    RetrievalTrace,
    WorkflowConfig,
    resolve_answer_language,
)

RetrievalMethod = Literal["bm25", "dense", "hybrid"]


def _is_hybrid_retriever(retriever) -> bool:
    from page_cite_rag.hybrid import HybridRetriever

    if isinstance(retriever, HybridRetriever) or getattr(retriever, "method", None) == "hybrid":
        return True
    if retriever is None:
        return False
    if getattr(retriever, "_dense", None) is not None:
        return True
    backend = getattr(retriever, "_search_backend", None)
    if getattr(backend, "method", None) == "hybrid":
        return True
    native = getattr(retriever, "_retriever", None)
    return bool(
        getattr(native, "method", None) == "hybrid"
        or (hasattr(native, "candidates") and hasattr(native, "encoder"))
        or (hasattr(retriever, "candidates") and hasattr(retriever, "encoder"))
    )


class FrozenSearchRetriever:
    """Carry one already performed scoped search into formal answer generation."""

    def __init__(self, result: SearchResult, *, top_k: int) -> None:
        self.result = result
        self.top_k = top_k

    def search(self, query: str, *, top_k: int = 5) -> SearchResult:
        if query != self.result.query or top_k != self.top_k:
            raise ValueError("staged retrieval request differs from the verified search")
        return self.result


def _with_reference_pinning(store: SQLiteDocumentStore, document_id: str, retriever):
    """Share explicit "paragraph N" address lookup across every retrieval entry point."""
    from page_cite_rag.references import ReferencePinningRetriever, numbered_paragraph_blocks

    markers = numbered_paragraph_blocks(
        store, document_id, native_source=getattr(retriever, "store", None)
    )
    if not markers:
        return retriever
    return ReferencePinningRetriever(retriever, store, document_id, markers)


def document_retriever(
    store: SQLiteDocumentStore,
    document_id: str,
    method: RetrievalMethod,
    *,
    encoder=None,
    versioned_dense: bool = False,
    candidates: int | None = None,
) -> Retriever:
    """Load existing snapshots only; never build or silently switch an index."""
    if candidates is not None and (type(candidates) is not int or candidates < 1):
        raise ValueError("hybrid candidate budget must be a positive integer")
    if candidates is not None and method != "hybrid":
        raise ValueError("candidate override is only valid for hybrid retrieval")
    candidate_count = candidates or FORMAL_RETRIEVAL.candidates_per_branch
    record = store.get_document(document_id)
    if (
        record is not None
        and record.parser_name == "docling"
        and store.get_retrieval_manifest(document_id, "docling-native-document-v1") is not None
    ):
        if method != "hybrid" or encoder is None:
            raise ValueError("Docling source requires its explicit hybrid encoder and index")
        from page_cite_rag.docling_retrieval import load_retriever

        return _with_reference_pinning(
            store,
            document_id,
            load_retriever(
                store,
                document_id,
                encoder,
                candidates=candidate_count,
                constant=FORMAL_RETRIEVAL.rrf_constant,
            ),
        )
    lexical, dense = document_retriever_parts(
        store, document_id, method, encoder=encoder, versioned_dense=versioned_dense
    )
    if method == "bm25":
        return _with_reference_pinning(store, document_id, lexical)
    if method == "dense":
        return _with_reference_pinning(store, document_id, dense)
    from page_cite_rag.hybrid import HybridRetriever

    return _with_reference_pinning(
        store,
        document_id,
        HybridRetriever(
            lexical,
            dense,
            candidates=candidate_count,
            constant=FORMAL_RETRIEVAL.rrf_constant,
        ),
    )


def document_retriever_parts(
    store: SQLiteDocumentStore,
    document_id: str,
    method: RetrievalMethod,
    *,
    encoder=None,
    versioned_dense: bool = False,
) -> tuple[Retriever | None, Retriever | None]:
    """Return the scoped lexical/dense branches for fixed or rewritten routing."""
    from page_cite_rag.retrieval import BM25Retriever

    if method not in ("bm25", "dense", "hybrid"):
        raise ValueError("unsupported retrieval method")
    lexical = BM25Retriever(store, document_id) if method != "dense" else None
    if method == "bm25":
        return lexical, None
    if encoder is None:
        raise ValueError("dense retrieval needs an explicit matching encoder")
    from page_cite_rag.dense import DenseRetriever

    dense = DenseRetriever(store, document_id, encoder, versioned=versioned_dense)
    return lexical, dense


def document_workflow(
    store: SQLiteDocumentStore,
    document_id: str,
    retriever: Retriever | None,
    model,
    budget_config,
    *,
    top_k: int = FORMAL_RETRIEVAL.final_k,
    neighbor_blocks: int = 0,
    answer_format: Literal["general", "short_answer", "multiple_choice"] = "general",
    prompt_revision: Literal["contract-consistent", "task-aligned-v5"] = ("contract-consistent"),
) -> AnswerWorkflow:
    """Use one answer contract regardless of retrieval method or parser."""
    return FullAnswerWorkflow(
        store,
        document_id,
        retriever,
        model,
        budget_config,
        WorkflowConfig(top_k=top_k, neighbor_blocks=neighbor_blocks),
        answer_format=answer_format,
        prompt_revision=prompt_revision,
    )


def ask_document(
    workflow: AnswerWorkflow, question: str, *, language: Language = "zh-Hant"
) -> AnswerResult:
    return workflow.ask(question, language=language)


def plan_document_question(
    workflow: FullAnswerWorkflow,
    question: str,
    *,
    language: Language,
    source_language: str,
    scoped: bool = False,
):
    """Create the same answer-free ReadingPlan used by the quality evaluator.

    ``scoped`` also asks whether the question needs the whole document (``ScopedReadingPlan``),
    which routes it to section reading instead of the fixed retrieval path.
    """
    from page_cite_rag.reading_plan import ReadingPlan, ScopedReadingPlan, reading_plan_messages

    outline = workflow.source_outline()
    titles = tuple(
        item.title for item in (outline.sections if outline is not None else ()) if item.title
    )[:80]
    messages = reading_plan_messages(
        question,
        source_language=(
            source_language if source_language in {"en", "zh-Hant", "zh-Hans"} else "unknown"
        ),
        output_language=language,
        outline_titles=titles,
        with_scope=scoped,
    )
    model = workflow._model
    if model is None:
        raise ValueError("a model is required to create a reading plan")
    generated = model.generate(messages, ScopedReadingPlan if scoped else ReadingPlan)
    return generated.value


def search_with_reading_plan(
    workflow: FullAnswerWorkflow,
    question: str,
    plan,
    *,
    candidates_per_query: int = 2 * FORMAL_RETRIEVAL.candidates_per_branch,
    fused_pool_size: int = 100,
):
    """Run the shared planned-query retrieval and materialize writer source blocks."""
    search, _results, trace = search_planned_hybrid(
        workflow._retriever,
        question,
        plan,
        candidates_per_query=candidates_per_query,
        fused_pool_size=fused_pool_size,
    )
    blocks = workflow._context_blocks(search, workflow._read_hits(search, question))
    return search, blocks, trace


def search_planned_hybrid(
    retriever,
    question: str,
    plan,
    *,
    candidates_per_query: int = 2 * FORMAL_RETRIEVAL.candidates_per_branch,
    fused_pool_size: int = 100,
    queries_override: tuple[str, ...] | None = None,
    query_to_needs_override: dict[str, tuple[str, ...]] | None = None,
):
    """Execute one validated ReadingPlan through the shared hybrid/RRF path."""
    from page_cite_rag.reading_plan import (
        ReadingPlan,
        fuse_planned_searches,
        planned_query_routes,
    )

    if not isinstance(plan, ReadingPlan):
        raise TypeError("a validated ReadingPlan is required")
    if (
        type(candidates_per_query) is not int
        or candidates_per_query < 1
        or type(fused_pool_size) is not int
        or fused_pool_size < candidates_per_query
    ):
        raise ValueError("invalid planned-retrieval candidate limits")
    if not _is_hybrid_retriever(retriever):
        raise ValueError("planned reading requires the hybrid BM25+dense retriever")
    if getattr(retriever, "reranker_model_path", None) is not None:
        raise ValueError("planned reading does not combine with the reranker profile")
    disable_rewrites = getattr(retriever, "disable_rewrites", None)
    if callable(disable_rewrites):
        disable_rewrites()
    if queries_override is None:
        queries, query_to_needs = planned_query_routes(question, plan)
    else:
        if (
            not queries_override
            or queries_override[0].strip() != question.strip()
            or any(not isinstance(query, str) or not query.strip() for query in queries_override)
            or len({query.casefold() for query in queries_override}) != len(queries_override)
        ):
            raise ValueError("planned retrieval override must begin with the original query")
        queries = tuple(query.strip() for query in queries_override)
        query_to_needs = query_to_needs_override or {}
    results = tuple(retriever.search(query, top_k=candidates_per_query) for query in queries)
    if any(
        result.query != query or result.method != "hybrid"
        for query, result in zip(queries, results, strict=True)
    ):
        raise ValueError("planned retrieval returned an unexpected query or method")
    search = fuse_planned_searches(question, results, top_k=fused_pool_size)
    search = search.model_copy(
        update={
            "branches": tuple(branch for result in results for branch in result.branches),
        }
    )
    trace = {
        "profile": "planned-hybrid-rrf/v1",
        "queries": list(queries),
        "query_to_need_ids": {query: list(ids) for query, ids in query_to_needs.items()},
        "candidates_per_query": candidates_per_query,
        "fused_pool_size": fused_pool_size,
        "retrieved_block_ids": [hit.block.block_id for hit in search.hits],
    }
    return search, results, trace


def prepare_agent_question(
    workflow: FullAnswerWorkflow,
    question: str,
    *,
    language: Language,
    source_language: str,
):
    """Use the shared planner, hybrid query routes and RRF before Agent navigation."""
    if not _is_hybrid_retriever(workflow._retriever):
        raise ValueError("the reading Agent requires an existing hybrid BM25+dense index")
    plan = plan_document_question(
        workflow,
        question,
        language=language,
        source_language=source_language,
    )
    planned_workflow = workflow.with_reading_plan(plan)
    search, blocks, retrieval_trace = search_with_reading_plan(planned_workflow, question, plan)
    return (
        planned_workflow,
        search,
        blocks,
        {
            "plan": plan.model_dump(mode="json"),
            "plan_deviations": list(getattr(plan, "_deviations", ())),
            "retrieval": retrieval_trace,
        },
    )


@dataclass(frozen=True)
class AgentAnswer:
    result: AnswerResult
    trace: dict


Route = Literal["auto", "fixed", "whole-document"]
ROUTES: tuple[str, ...] = ("auto", "fixed", "whole-document")


def answer_question(
    workflow: FullAnswerWorkflow,
    question: str,
    *,
    language: Language = "zh-Hant",
    source_language: str = "unknown",
    route: Route = "auto",
) -> AgentAnswer:
    """The single product entry, routed as in the plan 5.13 evaluation.

    ``auto`` asks the scoped planner whether the question needs the whole document: a
    lookup goes to fixed retrieval with the original question, a whole-document task to
    section-note reading with the planned retrieval. A failed planner falls back to the
    fixed path and says so in the trace, as in the evaluation. ``fixed`` skips the planner;
    ``whole-document`` always reads by sections.
    """
    from page_cite_rag.generation import ModelError

    if route not in ROUTES:
        raise ValueError("route must be auto, fixed or whole-document")
    language = resolve_answer_language(question, language)
    started = perf_counter()
    from page_cite_rag.reading_plan import READING_PLAN_VERSION

    trace: dict = {
        "version": "routed-answer/v2",
        "reading_plan_version": READING_PLAN_VERSION,
        "requested_route": route,
    }
    if route == "fixed":
        result = ask_document(workflow, question, language=language)
        trace.update(route="fixed", reason="requested")
        return AgentAnswer(result, trace)
    if not _is_hybrid_retriever(workflow._retriever):
        from page_cite_rag.retrieval import RetrievalError

        raise RetrievalError(
            "routing and whole-document reading need the hybrid BM25+dense index; "
            "build it or ask with route fixed"
        )
    outline = workflow.source_outline()
    if outline is None or len(outline.sections) < 2:
        # Section reading was validated on documents with an outline (Docling PDFs, Markdown
        # headings); without one, never substitute another reader under its name.
        if route == "whole-document":
            from page_cite_rag.retrieval import RetrievalError

            raise RetrievalError(
                "whole-document reading needs the document's section outline; ingest PDFs "
                "with Docling or ask with route fixed"
            )
        result = ask_document(workflow, question, language=language)
        trace.update(route="fixed", reason="no_outline")
        return AgentAnswer(result, trace)
    try:
        plan = plan_document_question(
            workflow,
            question,
            language=language,
            source_language=source_language
            if source_language in {"en", "zh-Hant", "zh-Hans"}
            else "unknown",
            scoped=True,
        )
    except ModelError as error:
        if route == "whole-document":
            raise
        result = ask_document(workflow, question, language=language)
        trace.update(route="fixed", reason="planner_failed", planner_error=type(error).__name__)
        return AgentAnswer(result, trace)
    trace["plan"] = plan.model_dump(mode="json")
    trace["plan_deviations"] = list(getattr(plan, "_deviations", ()))
    from page_cite_rag.reading_plan import is_foreign_language_query

    trace["dropped_foreign_queries"] = [
        query
        for need in plan.requirements
        for query in need.queries
        if is_foreign_language_query(query, question)
    ]
    if route == "auto" and plan.scope == "lookup":
        result = ask_document(workflow, question, language=language)
        trace.update(route="fixed", reason="planner_scope_lookup")
        return AgentAnswer(result, trace)
    planned = workflow.with_reading_plan(plan)
    search, blocks, retrieval_trace = search_with_reading_plan(planned, question, plan)
    reading = ask_with_agent(
        planned,
        question,
        language=language,
        initial_search=search,
        initial_blocks=blocks,
        strategy="outline",
    )
    agent = reading.trace["agent"]
    titles = {s.section_id: s.title or "(untitled)" for s in outline.sections}

    def named(key: str) -> list[str]:
        return [titles.get(section_id, section_id) for section_id in agent.get(key, ())]

    coverage = agent.get("point_coverage") or {}
    trace.update(
        route="whole_document",
        reason="planner_scope_whole_document" if route == "auto" else "requested",
        planned_retrieval=retrieval_trace,
        reading_summary={
            "sections_chosen": named("selected_section_ids"),
            "sections_from_search": named("seed_section_ids"),
            "sections_followed_up": named("follow_up_section_ids"),
            "sections_read": named("read_section_ids"),
            "notes": agent.get("reading_point_count", 0),
            "notes_handed_to_writer": agent.get(
                "writer_reading_point_count", agent.get("reading_point_count", 0)
            ),
            "notes_rejected_for_scope": sum(
                p["reason"] == "requirement_scope" for p in agent.get("reader_rejected_points", ())
            ),
            "notes_not_in_answer": coverage.get("uncovered", 0),
            "sources_skipped_for_read_limit": agent.get("sources_skipped_for_read_limit", 0),
            "sources_skipped_for_budget": agent.get("sources_skipped_for_budget", 0),
            "notes_omitted_for_writer_budget": agent.get("writer_omitted_point_count", 0),
            # The scoped planner call plus every reading and writer call.
            "model_calls": 1 + (agent.get("model_calls_attempted") or 0),
            "navigation_degraded": agent.get("navigation_degraded", False),
            "elapsed_seconds": round(perf_counter() - started, 1),
        },
        reading=reading.trace,
    )
    return AgentAnswer(reading.result, trace)


def ask_with_agent(
    workflow: FullAnswerWorkflow,
    question: str,
    *,
    language: Language = "zh-Hant",
    max_model_calls: int | None = None,
    initial_search: SearchResult | None = None,
    initial_blocks: Sequence[Block] | None = None,
    on_model_call: Callable[[dict], bool | None] | None = None,
    strategy: Literal["gap", "focus", "outline"] = "outline",
) -> AgentAnswer:
    """Read with one program-driven reader, then use the shared cited-answer writer.

    ``outline`` (the product path) reads the sections chosen from the document outline and
    notes, per window, what the task asks for; ``gap`` asks only which question parts are
    missing and lets the program search and read; ``focus`` narrows the writer to the few
    sources that state each need. One budget ledger covers reading and the writer.
    """
    from page_cite_rag.agent import (
        AGENT_WRITER_RESERVE_SECONDS,
        MAX_INTEGRATED_AGENT_CALL_INPUT_TOKENS,
        MAX_INTEGRATED_AGENT_CALL_OUTPUT_TOKENS,
        AgentRunFailure,
        _AgentBudgetLedger,
        _AgentBudgetStop,
        _spans,
    )
    from page_cite_rag.gap_agent import (
        READING_VERSION,
        run_focus_reading,
        run_gap_reading,
        run_outline_reading,
    )

    if (initial_search is None) != (initial_blocks is None):
        raise ValueError("initial search and source blocks must be supplied together")
    language = resolve_answer_language(question, language)
    search, blocks = (
        workflow.search_sources(question)
        if initial_search is None
        else (initial_search, initial_blocks)
    )
    if search.document_id != workflow.document_id or search.query != question:
        raise ValueError("initial search does not match the bound document and question")
    blocks = workflow._context_blocks(search, blocks)
    disable_rewrites = getattr(workflow._retriever, "disable_rewrites", None)
    if callable(disable_rewrites):
        disable_rewrites()
    model = workflow._model
    started = perf_counter()

    def navigation_cooldown() -> float:
        remaining_timeout = getattr(model, "remaining_request_timeout_seconds", None)
        wait_for_headroom = getattr(model, "wait_for_gpu_headroom", None)
        # A navigation call must afford its bounded GPU wait without spending the
        # writer's reservation. Recheck after waiting: cooling is wall time too.
        wait_limit = 120.0
        gpu_wait = (
            wait_limit
            if callable(wait_for_headroom)
            and getattr(getattr(model, "config", None), "device", None) != "cpu"
            else 0.0
        )
        if (
            callable(remaining_timeout)
            and remaining_timeout() <= AGENT_WRITER_RESERVE_SECONDS + gpu_wait
        ):
            raise _AgentBudgetStop("writer_reserve_deadline")
        waited = (
            wait_for_headroom(timeout_seconds=wait_limit) if callable(wait_for_headroom) else 0.0
        )
        if callable(remaining_timeout) and remaining_timeout() <= AGENT_WRITER_RESERVE_SECONDS:
            raise _AgentBudgetStop("writer_reserve_deadline")
        return waited

    output_limit = min(MAX_INTEGRATED_AGENT_CALL_OUTPUT_TOKENS, workflow._budget_config.num_predict)
    if max_model_calls is not None and (type(max_model_calls) is not int or max_model_calls < 1):
        raise ValueError("an optional reading model-call limit must be positive")
    ledger = _AgentBudgetLedger(
        model_call_limit=max_model_calls + 1 if max_model_calls is not None else None,
        call_input_limit=MAX_INTEGRATED_AGENT_CALL_INPUT_TOKENS,
        total_input_limit=None,
        call_output_limit=output_limit,
        total_output_limit=None,
        tool_call_limit=None,
    )
    readers = {"gap": run_gap_reading, "focus": run_focus_reading, "outline": run_outline_reading}
    if strategy not in readers:
        raise ValueError("unknown reading strategy")
    reading = readers[strategy](
        workflow, question, language, search, blocks, cooldown=navigation_cooldown
    )
    for call in reading.trace["gap_model_calls"]:
        ledger.admit(call["input_tokens"], workflow._budget_config.num_ctx)
        ledger.complete(call["output_tokens"])
    retrieval_trace = RetrievalTrace(
        method=search.method,
        index_id=search.index_id,
        query=search.query,
        hit_count=len(search.hits),
        elapsed_ms=search.elapsed_ms,
        query_plan=workflow._query_plan(),
        branches=search.branches,
    )
    acquired = reading.acquired_context or reading.context
    report = reading.structured_response
    report_block_ids = {id for finding in report.requirements for id in finding.source_ids}
    initial_writer_ids = tuple(reading.trace["initial_answer_block_ids"])
    eligible_ids = set(initial_writer_ids)
    read_ids: set[str] = set()
    lookups: list[dict] = []
    for event in reading.trace.get("tool_trace", ()):
        if not isinstance(event, dict):
            continue
        need = event.get("information_need")
        need_id = event.get("need_id")
        source_ids = []
        if event.get("tool") == "read_source" and event.get("status") in {
            "ok",
            "partial",
            "invalid",
            "skipped_prompt_budget",
        }:
            source_ids = [
                value for value in event.get("returned_block_ids", ()) if isinstance(value, str)
            ]
            eligible_ids.update(source_ids)
            read_ids.update(source_ids)
        if isinstance(need, str) and need.strip():
            lookups.append(
                {
                    "need_id": need_id if isinstance(need_id, str) else None,
                    "need": need.strip()[:240],
                    "acquired_block_ids": source_ids,
                }
            )

    block_by_id = {block.block_id: block for block in acquired.blocks}
    if reading.trace.get("navigation_degraded"):
        eligible_ids.update(block.block_id for block in acquired.blocks)
        read_ids.update(
            block.block_id for block in acquired.blocks if block.block_id not in initial_writer_ids
        )
    # Successful reads are authoritative originals; fail closed if the agent trace
    # claims a read that is absent from the returned, source-bound context.
    try:
        for block_id in eligible_ids:
            if block_id not in block_by_id:
                block_by_id[block_id] = workflow._read_block(block_id)
        writer_blocks = tuple(
            block_by_id[block_id]
            for block_id in dict.fromkeys(
                [
                    *initial_writer_ids,
                    *(
                        block_id
                        for event in reading.trace.get("tool_trace", ())
                        if isinstance(event, dict)
                        and event.get("tool") == "read_source"
                        and event.get("status")
                        in {"ok", "partial", "invalid", "skipped_prompt_budget"}
                        for block_id in event.get("returned_block_ids", ())
                        if isinstance(block_id, str)
                    ),
                    *(
                        block.block_id
                        for block in acquired.blocks
                        if reading.trace.get("navigation_degraded")
                    ),
                ]
            )
            if block_id in eligible_ids
        )
    except Exception as exc:
        failure_trace = dict(reading.trace)
        failure_trace.update(failure_stage="writer_source_scope", failure_type=type(exc).__name__)
        raise AgentRunFailure(type(exc).__name__, failure_trace) from exc
    groups_by_id = {
        group.chunk_index: group
        for group in (*search.context_groups, *acquired.context_groups)
        if set(group.source_block_ids).issubset(eligible_ids)
    }
    required_ids = set(report_block_ids)
    if report_block_ids:
        # Report anchors are required, but initial context remains available to
        # fill the budget. Preserve complete native groups around those anchors.
        required_ids.update(
            id
            for group in groups_by_id.values()
            if report_block_ids.intersection(group.source_block_ids)
            for id in group.source_block_ids
        )
        new_required_ids = report_block_ids - set(initial_writer_ids)
        groups_by_id = dict(
            sorted(
                groups_by_id.items(),
                key=lambda item: (
                    not bool(new_required_ids.intersection(item[1].source_block_ids)),
                    not bool(report_block_ids.intersection(item[1].source_block_ids)),
                    not bool(read_ids.intersection(item[1].source_block_ids)),
                ),
            )
        )
        writer_blocks = tuple(
            sorted(
                writer_blocks,
                key=lambda block: (
                    block.block_id not in new_required_ids,
                    block.block_id not in required_ids,
                    block.block_id not in read_ids,
                ),
            )
        )
    writer_context_groups = tuple(groups_by_id.values())
    if writer_context_groups:
        grouped_ids = {
            block_id for group in writer_context_groups for block_id in group.source_block_ids
        }
        if grouped_ids != {block.block_id for block in writer_blocks}:
            # Do not mix contextual chunk text with partial source groups. Plain
            # blocks remain complete originals and retain their exact citations.
            writer_context_groups = ()
    spans = _spans(writer_blocks)
    source_ids_by_block: dict[str, list[str]] = {}
    for source_id, span in spans.items():
        source_ids_by_block.setdefault(span.block_id, []).append(source_id)
    initial_ids = set(initial_writer_ids)
    for finding in report.requirements:
        available = {
            source_id
            for block_id in finding.source_ids
            for source_id in source_ids_by_block[block_id]
        }
        if finding.source_span_ids and not set(finding.source_span_ids).issubset(available):
            # T4 H14 hard: reader spans and inference citations must belong to document originals.
            raise ValueError("reader span handles do not belong to its handed-off originals")
    handoff = {
        "version": "navigation-handoff/v2",
        "status": "unverified",
        "navigation_degraded": reading.trace.get("navigation_degraded", False),
        "navigation_failure_type": reading.trace.get("navigation_failure_type"),
        "reading_report": {
            "requirements": [
                {
                    "need_id": finding.need_id,
                    "need": finding.need,
                    "observation": finding.observation,
                    "source_ids": finding.source_span_ids
                    or [
                        source_id
                        for block_id in finding.source_ids
                        for source_id in source_ids_by_block[block_id]
                    ],
                }
                for finding in report.requirements
            ],
            "unresolved": report.unresolved,
        },
        "reader_note_status": reading.trace.get("navigation_handoff_note_status"),
        "reader_note_sha256": reading.trace.get("navigation_handoff_note_sha256"),
        "initial_source_ids": [
            source_id
            for block in writer_blocks
            if block.block_id in initial_ids
            for source_id in source_ids_by_block.get(block.block_id, ())
        ],
        "lookups": [
            {
                "need_id": item["need_id"],
                "need": item["need"],
                "acquired_source_ids": [
                    source_id
                    for block_id in item["acquired_block_ids"]
                    for source_id in source_ids_by_block.get(block_id, ())
                ],
                "state": (
                    "acquired_unverified"
                    if any(
                        block_id in source_ids_by_block for block_id in item["acquired_block_ids"]
                    )
                    else "not_read_or_unresolved"
                ),
            }
            for item in lookups[:8]
        ],
    }
    writer_call: dict = {"role": "writer", "status": "not_started"}

    def before_writer(budget) -> None:
        if type(budget.exact_input_tokens) is not int:
            raise ValueError("writer input token count is not exact")
        if budget.output_reserved != ledger.call_output_limit:
            raise ValueError("writer output reservation differs from the shared agent ledger")
        writer_call.update(
            call_index=ledger.model_calls + 1,
            input_tokens_exact=budget.exact_input_tokens,
            output_token_limit=budget.output_reserved,
            status="ready_to_dispatch",
        )
        if on_model_call is not None and on_model_call(dict(writer_call)) is False:
            writer_call["status"] = "blocked_before_dispatch"
            on_model_call(dict(writer_call))
            raise _AgentBudgetStop("global_generation_budget")
        ledger.admit(budget.exact_input_tokens, workflow._budget_config.num_ctx)
        writer_call.update(
            call_index=ledger.model_calls,
            input_tokens_exact=budget.exact_input_tokens,
            output_token_limit=budget.output_reserved,
            status="dispatched",
        )
        if on_model_call is not None:
            on_model_call(dict(writer_call))

    def after_writer(stats) -> None:
        ledger.input_tokens += stats.prompt_eval_count - writer_call["input_tokens_exact"]
        ledger.complete(stats.eval_count)
        writer_call.update(
            input_tokens_local=writer_call["input_tokens_exact"],
            input_tokens_exact=stats.prompt_eval_count,
            output_tokens_observed=stats.eval_count,
            elapsed_ms=stats.elapsed_ms,
            status="completed",
        )
        if on_model_call is not None:
            on_model_call(dict(writer_call))

    # Preserve need coverage while the writer reasons from the original evidence;
    # free-text reader notes remain available in the trace only.
    writer_workflow = (
        workflow.with_requirement_coverage(True) if strategy == "outline" else workflow
    )
    try:
        if writer_blocks:
            result = writer_workflow.answer_from_agent_context(
                question,
                writer_blocks,
                navigation_handoff=handoff,
                language=language,
                context_groups=writer_context_groups,
                required_block_ids=tuple(required_ids),
                before_generation=before_writer,
                after_generation=after_writer,
            )
        else:
            result = workflow.answer_from_context(question, (), language=language)
        if reading.trace.get("navigation_report_partial"):
            if reading.trace.get("navigation_failure_type") == "writer_reserve_deadline":
                limitation = (
                    "為保留作答時間，閱讀已提前結束；本回答依已讀原文生成，可能未涵蓋全部內容。"
                    if language == "zh-Hant"
                    else (
                        "Reading ended early to reserve time for the answer; this answer "
                        "uses the sources already read and may be incomplete."
                    )
                )
            else:
                limitation = (
                    "Agent 未能完成結構化閱讀報告；本回答僅依可驗證原文生成，內容可能不完整。"
                    if language == "zh-Hant"
                    else (
                        "The Agent could not complete its structured reading report; this answer "
                        "uses verified originals and may be incomplete."
                    )
                )
            # Keep the writer's own status; degradation is a separate trace field.
            result = result.model_copy(update={"limitations": (*result.limitations, limitation)})
        result = result.model_copy(
            update={
                "retrieval": retrieval_trace,
                "elapsed_ms": (perf_counter() - started) * 1000,
            }
        )
    except Exception as exc:
        failure_trace = dict(reading.trace)
        failure_trace.update(
            failure_stage="common_writer",
            failure_type=type(exc).__name__,
            writer_call=writer_call,
            writer_source_block_ids=[block.block_id for block in writer_blocks],
            navigation_handoff=handoff,
            agent_input_tokens_exact=ledger.input_tokens,
            agent_output_tokens_observed=(
                ledger.output_tokens if ledger.output_usage_complete else None
            ),
            agent_output_tokens_reserved=ledger.output_tokens + ledger.output_reserved_tokens,
            agent_output_tokens_unobserved_reserved=ledger.output_reserved_tokens,
            model_calls_attempted=ledger.model_calls,
        )
        if accounting := getattr(exc, "accounting", None):
            failure_trace["writer_failure_accounting"] = accounting.model_dump(mode="json")
        raise AgentRunFailure(type(exc).__name__, failure_trace) from exc

    # Plan 5.13: a code check of which reading notes reached the answer. Recorded only; a
    # one-shot rewrite (5.13) and an add-only supplement call (5.15) by the same 9B writer
    # both failed their development checks.
    points = reading.trace.get("writer_reading_points", reading.trace.get("reading_points")) or ()
    coverage = (
        _point_coverage(result, points)
        if strategy == "outline" and writer_call["status"] == "completed" and points
        else None
    )

    if writer_call["status"] == "completed":
        usage = AgentUsage(
            model_calls=ledger.model_calls,
            tool_calls=reading.trace["tool_calls_recorded"],
            input_tokens=ledger.input_tokens,
            output_tokens=ledger.output_tokens if ledger.output_usage_complete else None,
            output_tokens_reserved=ledger.output_reserved_tokens,
            output_usage_complete=ledger.output_usage_complete,
            final_input_tokens=writer_call["input_tokens_exact"],
            final_output_tokens=writer_call["output_tokens_observed"],
            final_output_limit=writer_call["output_token_limit"],
        )
        result = result.model_copy(
            update={
                "agent_usage": usage,
                "prompt_version": (
                    READING_VERSION if strategy == "outline" else "agentic-reading-v2"
                ),
            }
        )

    agent_trace = dict(reading.trace)
    writer_completed = writer_call["status"] == "completed"
    agent_trace.update(
        {
            "architecture": "single_agent_navigation_with_shared_writer",
            "response_strategy": "navigation_handoff_then_common_writer",
            "agent_budget_scope": "single_agent_navigation_and_shared_writer",
            "model_calls_attempted": ledger.model_calls,
            "model_calls_completed": reading.trace["model_calls_completed"] + int(writer_completed),
            "model_call_limit": ledger.model_call_limit,
            "agent_input_tokens_exact": ledger.input_tokens,
            "agent_output_tokens_observed": (
                ledger.output_tokens if ledger.output_usage_complete else None
            ),
            "agent_output_tokens_reserved": ledger.output_tokens + ledger.output_reserved_tokens,
            "agent_output_tokens_unobserved_reserved": ledger.output_reserved_tokens,
            "writer_call": writer_call,
            "point_coverage": coverage,
            "claim_ids": [claim.claim_id for claim in result.claims],
            "requirement_coverage": [
                item.model_dump(mode="json") for item in result.requirement_coverage
            ],
            "claim_duplicates_removed": result.claim_duplicates_removed,
            "output_deviations": list(result.output_deviations),
            "writer_source_block_ids": [block.block_id for block in writer_blocks],
            "writer_context_group_ids": [group.chunk_index for group in writer_context_groups],
            "navigation_handoff": handoff,
            "final_model_call": writer_call if writer_completed else None,
            "final_answer_input_tokens": writer_call.get("input_tokens_exact"),
            "selected_block_ids": list(result.included_block_ids),
            "omitted_block_ids": list(result.omitted_block_ids),
            "total_elapsed_ms": result.elapsed_ms,
        }
    )
    return AgentAnswer(
        result,
        {
            "version": "agentic-reading-v2",
            "strategy": "single_agent_navigation_then_shared_writer",
            "agent": agent_trace,
        },
    )


def _cited_block_ids(result: AnswerResult) -> set[str]:
    return {citation.block_id for claim in result.claims for citation in claim.citations}


def _point_coverage(result: AnswerResult, points) -> dict:
    """Which reading notes the answer cites at least one source of (code check, no model)."""
    cited = _cited_block_ids(result)
    uncovered = [
        index
        for index, point in enumerate(points)
        if point["block_ids"] and not cited.intersection(point["block_ids"])
    ]
    return {
        "version": "reading-point-coverage/v2",
        "points": len(points),
        "uncovered": len(uncovered),
        "uncovered_point_indices": uncovered,
    }


def explain_source(
    workflow: AnswerWorkflow,
    block_id: str,
    question: str,
    *,
    language: Language = "zh-Hant",
) -> AnswerResult:
    return workflow.explain(block_id, question=question, language=language)
