"""Small shared application boundary for CLI, local API and evaluation adapters."""

from __future__ import annotations

import re
from copy import copy
from dataclasses import dataclass
from typing import Literal

from page_cite_rag.full_answer import FullAnswerWorkflow
from page_cite_rag.hybrid import Retriever
from page_cite_rag.retrieval import SearchResult
from page_cite_rag.retrieval_profile import FORMAL_RETRIEVAL
from page_cite_rag.store import SQLiteDocumentStore
from page_cite_rag.workflows import (
    AnswerResult,
    AnswerWorkflow,
    Language,
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
    writer: Literal["structured", "quote_first"] = "structured",
    quote_first_revision: Literal["v1", "v3"] = "v1",
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
        writer=writer,
        prompt_revision=prompt_revision,
        quote_first_revision=quote_first_revision,
    )


def ask_document(
    workflow: AnswerWorkflow, question: str, *, language: Language = "auto"
) -> AnswerResult:
    return workflow.ask(question, language=language)


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


@dataclass(frozen=True)
class AgentAnswer:
    result: AnswerResult
    trace: dict


Mode = Literal["auto", "detail", "global", "deep"]
MODES: tuple[str, ...] = ("auto", "detail", "global", "deep")

# LlamaIndex 0.14.25, llama_index/core/selectors/prompts.py:
# DEFAULT_SINGLE_SELECT_PROMPT_TMPL, copied verbatim.
ROUTER_PROMPT = (
    "Some choices are given below. It is provided in a numbered list "
    "(1 to {num_choices}), where each item in the list corresponds to a summary.\n"
    "---------------------\n{context_list}\n---------------------\n"
    "Using only the choices above and not prior knowledge, return the choice that "
    "is most relevant to the question: '{query_str}'\n"
    # Project addition: require a number-only output.
    "Return only the number of the choice."
)
ROUTER_CHOICES = (
    "(1) Useful for questions about specific details in the document: facts, numbers, "
    "definitions, methods, or explaining a particular concept or passage.\n\n"
    "(2) Useful for questions about the document as a whole: its main idea or summary, "
    "its contributions, strengths and limitations, or comparisons and lists that span "
    "many sections."
)


def choose_mode(client, question: str) -> tuple[str, dict]:
    from page_cite_rag.generation import ChatMessage, ModelError

    record = {"mode": "detail", "reply": "", "error": None, "stats": None}
    prompt = ROUTER_PROMPT.format(num_choices=2, context_list=ROUTER_CHOICES, query_str=question)
    try:
        response = client.generate([ChatMessage(role="user", content=prompt)], None)
        reply = response.value
        record["reply"] = reply[:40]
        record["stats"] = response.stats.model_dump(mode="json")
        choice = re.search(r"[12]", reply)
        if choice:
            record["mode"] = "global" if choice.group() == "2" else "detail"
        else:
            record["error"] = "no_valid_choice"
    except ModelError as error:
        record["error"] = str(error)
    return record["mode"], record


Route = Literal["auto", "fixed", "agent"]
ROUTES: tuple[str, ...] = ("auto", "fixed", "agent")


def answer_question(
    workflow: FullAnswerWorkflow,
    question: str,
    *,
    language: Language = "auto",
    source_language: str = "unknown",
    mode: Mode = "detail",
    route: Route | None = None,
) -> AgentAnswer:
    """Select evidence using the shared writer admission and retrieval contracts."""
    from page_cite_rag.gpu_retrieval import StagedGpuSearch
    from page_cite_rag.retrieval_profile import RERANK_KEEP

    if mode not in MODES:
        raise ValueError("mode must be auto, detail, global or deep")
    if route is not None and route not in ROUTES:
        raise ValueError("route must be auto, fixed or agent")
    language = resolve_answer_language(question, language)
    requested_mode = mode
    router = None
    if route is None:
        if mode == "auto":
            mode, router = choose_mode(workflow._model, question)
        route = "agent" if mode == "deep" else "auto"
    else:
        mode = "deep" if route == "agent" else "detail"
        requested_mode = mode
    trace = {
        "version": "routed-answer/v5",
        "requested_mode": requested_mode,
        "mode": mode,
        "requested_route": route,
        "route": "fixed",
        "reason": "requested" if route == "fixed" else "rerank_unavailable",
    }
    if router is not None:
        trace["router"] = router
    if mode == "global":
        from page_cite_rag.global_answer import answer_global

        run = answer_global(workflow, question, language=language)
        trace.update(route="global", reason="requested", global_trace=run.trace)
        return AgentAnswer(run.result, trace)
    if route == "agent":
        from page_cite_rag.evidence_agent import run_evidence_agent
        from page_cite_rag.ollama_backend import OllamaGgufClient

        retriever = workflow._retriever
        if not isinstance(retriever, StagedGpuSearch) or retriever.method != "hybrid":
            raise ValueError("agent requires --native-gpu and --method hybrid")
        if retriever.reranker_model_path is not None:
            raise ValueError("agent requires retrieval without reranking")
        if not isinstance(workflow._model, OllamaGgufClient):
            raise ValueError("agent requires --backend pinned-ollama (OllamaGgufClient)")
        with retriever.session():
            run = run_evidence_agent(workflow, workflow._model, question, language=language)
        trace.update(route="agent", reason="requested")
        trace.update(
            {
                "agent_version": run.trace["version"],
                "model_calls": len(run.trace["model_calls"]),
                **{
                    key: run.trace[key]
                    for key in (
                        "tool_call_repairs",
                        "handoff_reason",
                        "context_check",
                        "search_steps",
                        "token_usage",
                    )
                },
            }
        )
        return AgentAnswer(run.answer, trace)
    if route == "auto":
        if workflow.whole_document_fits(question, language=language):
            result = workflow.answer_from_context(
                question, workflow._store.get_blocks(workflow.document_id), language=language
            )
            trace.update(route="whole_document_context", reason="fits_context")
            return AgentAnswer(result, trace)
        if (
            isinstance(workflow._retriever, StagedGpuSearch)
            and workflow._retriever.reranker_model_path is not None
        ):
            workflow = copy(workflow)
            workflow.config = workflow.config.model_copy(update={"top_k": RERANK_KEEP})
            trace.update(route="reranked", reason="whole_document_exceeds_context")
    return AgentAnswer(ask_document(workflow, question, language=language), trace)


def explain_source(
    workflow: AnswerWorkflow,
    block_id: str,
    question: str,
    *,
    language: Language = "auto",
) -> AnswerResult:
    return workflow.explain(block_id, question=question, language=language)
