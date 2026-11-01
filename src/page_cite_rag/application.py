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

        return load_retriever(
            store,
            document_id,
            encoder,
            candidates=candidate_count,
            constant=FORMAL_RETRIEVAL.rrf_constant,
        )
    lexical, dense = document_retriever_parts(
        store, document_id, method, encoder=encoder, versioned_dense=versioned_dense
    )
    if method == "bm25":
        return lexical
    if method == "dense":
        return dense
    from page_cite_rag.hybrid import HybridRetriever

    return HybridRetriever(
        lexical,
        dense,
        candidates=candidate_count,
        constant=FORMAL_RETRIEVAL.rrf_constant,
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
):
    """Create the same answer-free ReadingPlan used by the quality evaluator."""
    from page_cite_rag.reading_plan import ReadingPlan, reading_plan_messages

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
    )
    model = workflow._model
    if model is None:
        raise ValueError("a model is required to create a reading plan")
    generated = model.generate(messages, ReadingPlan)
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
            "retrieval": retrieval_trace,
        },
    )


@dataclass(frozen=True)
class AgentAnswer:
    result: AnswerResult
    trace: dict


_FATAL_AGENT_TOOL_STATUSES = {
    "scope_unavailable",
    "invalid_source_or_cursor",
    "invalid_source_ids",
    "source_unavailable",
    "outside_source_scope",
    "outside_reader_scope",
    "invalid_terminal_state",
}


def _recoverable_navigation_failure(error: Exception) -> bool:
    """Recover only a malformed report or context exhaustion from verified originals."""
    from page_cite_rag.agent import AgentRunFailure

    if not isinstance(error, AgentRunFailure) or error.error_type not in {
        "InvalidAgentTerminalState",
        "InvalidStructuredAnswerSchema",
        "AgentBudgetExceeded",
    }:
        return False
    trace = error.trace
    terminal = trace.get("navigator_terminal_state")
    stop_reason = trace.get("budget_stop_reason")
    recoverable_context_stop = stop_reason in {
        "bootstrap_input_limit",
        "context_limit",
        "per_call_input_limit",
        "writer_reserve_deadline",
    }
    # ToolStrategy rejects malformed report arguments before executing tools.
    # Discard that one report; pending data tools or invalid calls remain fatal.
    invalid_report_only = (
        isinstance(terminal, dict)
        and error.error_type == "InvalidStructuredAnswerSchema"
        and terminal.get("pending_tool_call_count") == 1
        and terminal.get("pending_structured_tool_call_count") == 1
        and terminal.get("pending_data_tool_call_count") == 0
        and terminal.get("terminal_tool_call_count") == 1
        and terminal.get("terminal_invalid_tool_call_count") == 0
        and terminal.get("finish_reason") in {"tool_calls", "function_call"}
    )
    if (
        not isinstance(terminal, dict)
        or terminal.get("cancelled_or_timed_out") is not False
        or (terminal.get("pending_tool_call_count") != 0 and not invalid_report_only)
        or terminal.get("terminal_invalid_tool_call_count", 0) != 0
        or trace.get("cooldown_error_type") is not None
    ):
        return False
    if error.error_type == "AgentBudgetExceeded":
        if not recoverable_context_stop or terminal.get("status") != "budget_stopped":
            return False
    elif stop_reason is not None:
        return False
    if error.error_type == "InvalidAgentTerminalState":
        if terminal.get("reason") != "missing_structured_response":
            return False
        if terminal.get("finish_reason") not in {"length", "stop"}:
            return False
        if terminal.get("terminal_tool_call_count") != 0:
            return False
    elif (
        error.error_type == "InvalidStructuredAnswerSchema"
        and terminal.get("status") != "invalid_structured_answer"
    ):
        return False
    events = trace.get("tool_trace")
    if not isinstance(events, list) or any(
        not isinstance(event, dict) or event.get("status") in _FATAL_AGENT_TOOL_STATUSES
        for event in events
    ):
        return False
    calls = trace.get("model_call_records")
    if not isinstance(calls, list) or not calls:
        return False
    if stop_reason == "writer_reserve_deadline":
        if not isinstance(calls[-1], dict) or calls[-1].get("status") != "blocked_before_dispatch":
            return False
        completed_calls = calls[:-1]
    else:
        completed_calls = calls
    return all(
        isinstance(call, dict)
        and call.get("status") == "completed"
        and type(call.get("input_tokens_exact")) is int
        and type(call.get("output_tokens_observed")) is int
        for call in completed_calls
    )


def _recover_original_context(
    workflow: FullAnswerWorkflow,
    question: str,
    search: SearchResult,
    initial_blocks: Sequence[Block],
    error: Exception,
):
    """Rebuild only program-verified initial/read originals after a recoverable report failure."""
    from page_cite_rag.agent import AgentContext, AgentRun, ReadingReport

    if not _recoverable_navigation_failure(error):
        raise error
    trace = dict(error.trace)
    initial_ids = trace.get("initial_answer_block_ids")
    if (
        not isinstance(initial_ids, list)
        or not initial_ids
        or any(not isinstance(block_id, str) for block_id in initial_ids)
    ):
        raise error
    initial_by_id = {block.block_id: block for block in initial_blocks}
    if len(initial_by_id) != len(initial_blocks) or not set(initial_ids).issubset(initial_by_id):
        raise error

    read_ids = []
    returned_group_ids: list[int] = []
    for event in trace["tool_trace"]:
        if event.get("tool") != "read_source" or event.get("status") not in {"ok", "partial"}:
            continue
        returned = event.get("returned_block_ids")
        if not isinstance(returned, list) or any(not isinstance(item, str) for item in returned):
            raise error
        read_ids.extend(returned)
        group_ids = event.get("returned_context_group_ids", [])
        if not isinstance(group_ids, list) or any(type(item) is not int for item in group_ids):
            raise error
        returned_group_ids.extend(group_ids)
    read_ids = list(dict.fromkeys(read_ids))
    selected_group_ids = trace.get("selected_context_group_ids", [])
    if not isinstance(selected_group_ids, list) or any(
        type(item) is not int for item in selected_group_ids
    ):
        raise error
    initial_id_set = set(initial_ids)
    groups = {
        group.chunk_index: group
        for group in search.context_groups
        if group.chunk_index in selected_group_ids
        and set(group.source_block_ids).issubset(initial_id_set)
    }
    native_source = getattr(workflow, "_native_source", None)
    group_block_ids: set[str] = set()
    if returned_group_ids and native_source is None:
        raise error
    if native_source is not None:
        for group_id in set(returned_group_ids):
            if not 0 <= group_id < len(native_source.chunk_block_ids):
                raise error
            from page_cite_rag.retrieval import SearchContextGroup

            group = SearchContextGroup(
                chunk_index=group_id,
                text=native_source.contextualized_chunk_text(group_id),
                source_refs=native_source.context_refs_for_chunk(group_id),
                source_block_ids=native_source.context_block_ids_for_chunk(group_id),
            )
            groups[group_id] = group
            group_block_ids.update(group.source_block_ids)
    eligible_read_ids = list(dict.fromkeys([*read_ids, *sorted(group_block_ids)]))
    try:
        read_blocks = tuple(
            workflow._read_block(block_id)
            for block_id in eligible_read_ids
            if block_id not in initial_by_id
        )
    except Exception:
        raise error from None
    expected_read_ids = [value for value in eligible_read_ids if value not in initial_by_id]
    if any(
        block.block_id != block_id
        for block, block_id in zip(read_blocks, expected_read_ids, strict=True)
    ):
        raise error
    verified_groups = tuple(groups[index] for index in sorted(groups))
    for group in verified_groups:
        try:
            workflow._validate_context_group(group)
        except Exception:
            raise error from None

    original_blocks_list = []
    seen_block_ids = set()
    for block in [initial_by_id[block_id] for block_id in initial_ids] + list(read_blocks):
        if block.block_id not in seen_block_ids:
            original_blocks_list.append(block)
            seen_block_ids.add(block.block_id)
    original_blocks = tuple(original_blocks_list)
    available_ids = {block.block_id for block in original_blocks}
    if any(not set(group.source_block_ids).issubset(available_ids) for group in verified_groups):
        raise error
    trace.update(
        navigation_degraded=True,
        navigation_recovery="verified_originals_after_incomplete_structured_report",
        navigation_failure_type=error.error_type,
        navigation_report_partial=True,
        navigation_handoff_note=None,
        navigation_handoff_note_status="recovered_without_structured_report",
    )
    report = ReadingReport(requirements=[], unresolved=[question[:240] or "requested question"])
    return AgentRun(
        context=AgentContext(
            tuple(initial_by_id[block_id] for block_id in initial_ids),
            tuple(
                group
                for group in verified_groups
                if set(group.source_block_ids).issubset(initial_id_set)
            ),
            trace.get("final_answer_input_tokens"),
        ),
        trace=trace,
        structured_response=report,
        acquired_context=AgentContext(original_blocks, verified_groups, None),
    )


def ask_with_agent(
    workflow: FullAnswerWorkflow,
    question: str,
    *,
    language: Language = "zh-Hant",
    delegated: bool = False,
    data_tools_enabled: bool = True,
    max_model_calls: int | None = None,
    initial_search: SearchResult | None = None,
    initial_blocks: Sequence[Block] | None = None,
    on_model_call: Callable[[dict], bool | None] | None = None,
    on_wire_request: Callable[[dict], None] | None = None,
) -> AgentAnswer:
    """Navigate with one request-scoped agent, then use the shared cited-answer writer."""
    from page_cite_rag.agent import (
        AGENT_WRITER_RESERVE_SECONDS,
        MAX_INTEGRATED_AGENT_CALL_INPUT_TOKENS,
        MAX_INTEGRATED_AGENT_CALL_OUTPUT_TOKENS,
        AgentRunFailure,
        _AgentBudgetLedger,
        _AgentBudgetStop,
        _spans,
        local_chat_model,
        run_bounded_reading_agent,
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
        if callable(remaining_timeout) and remaining_timeout() <= AGENT_WRITER_RESERVE_SECONDS:
            raise _AgentBudgetStop("writer_reserve_deadline")
        wait_for_headroom = getattr(model, "wait_for_gpu_headroom", None)
        return wait_for_headroom() if callable(wait_for_headroom) else 0.0

    # Keep old isolated diagnostics available; the normal single-agent path uses
    # one ledger for navigation and the common writer.
    legacy_integrated = not delegated and not data_tools_enabled
    output_limit = min(MAX_INTEGRATED_AGENT_CALL_OUTPUT_TOKENS, workflow._budget_config.num_predict)
    ledger = None
    navigation_calls = max_model_calls
    if not delegated and not legacy_integrated:
        if navigation_calls is not None and (
            type(navigation_calls) is not int or navigation_calls < 1
        ):
            raise ValueError("an optional single-agent model-call limit must be positive")
        ledger = _AgentBudgetLedger(
            model_call_limit=navigation_calls + 1 if navigation_calls is not None else None,
            call_input_limit=MAX_INTEGRATED_AGENT_CALL_INPUT_TOKENS,
            total_input_limit=None,
            call_output_limit=output_limit,
            total_output_limit=None,
            tool_call_limit=None,
        )
    try:
        chat_model_context = (
            local_chat_model(model, on_request=on_wire_request)
            if on_wire_request is not None
            else local_chat_model(model)
        )
        with chat_model_context as chat_model:
            reading = run_bounded_reading_agent(
                workflow,
                question,
                language,
                search,
                blocks,
                chat_model,
                count_chat_input_tokens=model.count_chat_input_tokens,
                delegated=delegated,
                cooldown=navigation_cooldown,
                max_model_calls=(
                    max_model_calls
                    if max_model_calls is not None
                    else 7
                    if delegated
                    else 1
                    if legacy_integrated
                    else navigation_calls
                ),
                final_answer=legacy_integrated,
                shared_ledger=ledger,
                data_tools_enabled=data_tools_enabled,
                on_model_call=on_model_call,
            )
    except AgentRunFailure as exc:
        if delegated or legacy_integrated:
            raise
        reading = _recover_original_context(workflow, question, search, blocks, exc)
    retrieval_trace = RetrievalTrace(
        method=search.method,
        index_id=search.index_id,
        query=search.query,
        hit_count=len(search.hits),
        elapsed_ms=search.elapsed_ms,
        query_plan=workflow._query_plan(),
        branches=search.branches,
    )
    if delegated:
        result = workflow.answer_from_context(
            question,
            reading.context.blocks,
            language=language,
            context_groups=reading.context.context_groups,
        )
        result = result.model_copy(update={"retrieval": retrieval_trace})
        return AgentAnswer(
            result,
            {
                "version": "source-navigation/v1",
                "strategy": "delegated",
                "agent": reading.trace,
            },
        )

    if legacy_integrated:
        final_call = reading.trace.get("final_model_call")
        if not isinstance(final_call, dict) or final_call.get("status") != "completed":
            failure_trace = dict(reading.trace)
            failure_trace["failure_stage"] = "structured_final_call"
            failure_trace["failure_type"] = "missing_completed_final_call"
            raise AgentRunFailure("InvalidAgentTerminalState", failure_trace)
        output_tokens = reading.trace.get("agent_output_tokens_observed")
        usage = AgentUsage(
            model_calls=reading.trace["model_calls_completed"],
            tool_calls=reading.trace["tool_calls_recorded"],
            input_tokens=reading.trace["agent_input_tokens_exact"],
            output_tokens=output_tokens,
            output_tokens_reserved=reading.trace["agent_output_tokens_unobserved_reserved"],
            output_usage_complete=output_tokens is not None,
            final_input_tokens=final_call["input_tokens_exact"],
            final_output_tokens=final_call["output_tokens_observed"],
            final_output_limit=final_call["output_token_limit"],
        )
        try:
            result = workflow.answer_from_agent_draft(
                question,
                reading.context.blocks,
                reading.structured_response,
                language=language,
                context_groups=reading.context.context_groups,
                omitted_block_ids=reading.context.omitted_block_ids,
                retrieval=retrieval_trace,
                agent_usage=usage,
                elapsed_ms=(perf_counter() - started) * 1000,
            )
        except Exception as exc:
            failure_trace = dict(reading.trace)
            failure_trace["failure_stage"] = "answer_validation"
            failure_trace["failure_type"] = type(exc).__name__
            raise AgentRunFailure(type(exc).__name__, failure_trace) from exc
        return AgentAnswer(
            result,
            {
                "version": "integrated-reading-agent/v1",
                "strategy": "integrated_single_agent_diagnostic",
                "agent": reading.trace,
            },
        )

    if ledger is None:
        raise RuntimeError("single-agent shared writer ledger was not initialized")
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
        if event.get("tool") == "read_source" and event.get("status") in {"ok", "partial"}:
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
                        and event.get("status") in {"ok", "partial"}
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
                    "source_ids": [
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
        ledger.complete(stats.eval_count)
        writer_call.update(
            output_tokens_observed=stats.eval_count,
            elapsed_ms=stats.elapsed_ms,
            status="completed",
        )
        if on_model_call is not None:
            on_model_call(dict(writer_call))

    try:
        if writer_blocks:
            result = workflow.answer_from_agent_context(
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
            limitation = (
                "Agent 未能完成結構化閱讀報告；本回答僅依可驗證原文生成，內容可能不完整。"
                if language == "zh-Hant"
                else (
                    "The Agent could not complete its structured reading report; this answer "
                    "uses verified originals and may be incomplete."
                )
            )
            result = result.model_copy(
                update={
                    "status": "partial"
                    if result.status != "insufficient_evidence"
                    else result.status,
                    "stop_reason": "partial"
                    if result.status != "insufficient_evidence"
                    else result.stop_reason,
                    "limitations": (*result.limitations, limitation),
                }
            )
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
            update={"agent_usage": usage, "prompt_version": "agentic-reading-v2"}
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


def explain_source(
    workflow: AnswerWorkflow,
    block_id: str,
    question: str,
    *,
    language: Language = "zh-Hant",
) -> AnswerResult:
    return workflow.explain(block_id, question=question, language=language)
