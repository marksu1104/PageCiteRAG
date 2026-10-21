"""Small shared application boundary for CLI, local API and evaluation adapters."""

from __future__ import annotations

from typing import Literal

from page_cite_rag.full_answer import FullAnswerWorkflow
from page_cite_rag.hybrid import Retriever
from page_cite_rag.retrieval import SearchResult
from page_cite_rag.retrieval_profile import FORMAL_RETRIEVAL
from page_cite_rag.store import SQLiteDocumentStore
from page_cite_rag.workflows import AnswerResult, AnswerWorkflow, Language, WorkflowConfig

RetrievalMethod = Literal["bm25", "dense", "hybrid"]


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
) -> Retriever:
    """Load existing snapshots only; never build or silently switch an index."""
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
            candidates=FORMAL_RETRIEVAL.candidates_per_branch,
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
        candidates=FORMAL_RETRIEVAL.candidates_per_branch,
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
    )


def ask_document(
    workflow: AnswerWorkflow, question: str, *, language: Language = "en"
) -> AnswerResult:
    return workflow.ask(question, language=language)


def explain_source(
    workflow: AnswerWorkflow,
    block_id: str,
    question: str,
    *,
    language: Language = "en",
) -> AnswerResult:
    return workflow.explain(block_id, question=question, language=language)
