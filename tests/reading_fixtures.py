"""Small, self-authored reading fixtures shared by the reading tests."""

from __future__ import annotations

from page_cite_rag.full_answer import FullAnswerWorkflow
from page_cite_rag.generation import GenerationConfig
from page_cite_rag.ingest import prepare_source
from page_cite_rag.retrieval import SearchHit, SearchResult
from page_cite_rag.store import SQLiteDocumentStore


class _Retriever:
    def __init__(self, result):
        self.result = result
        self.queries = []
        self.top_k_values = []

    def search(self, query, *, top_k=5):
        self.queries.append(query)
        self.top_k_values.append(top_k)
        return self.result.model_copy(update={"query": query})


class _Model:
    identity = {"model": "scripted-test"}


def _setup(tmp_path):
    source = tmp_path / "paper.md"
    source.write_text(
        "# Intro\n\nIntro paragraph.\n\n## Method\n\n"
        "Method paragraph one.\n\nMethod paragraph two.\n\n"
        "## Results\n\nResult paragraph.\n",
        encoding="utf-8",
    )
    prepared = prepare_source(source, usage_scope="self-authored-test")
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    blocks = tuple(store.get_blocks(prepared.document.document_id))
    result = SearchResult(
        document_id=prepared.document.document_id,
        index_id="synthetic-index",
        method="bm25",
        query="method details",
        hits=(SearchHit(rank=1, score=1.0, block=blocks[1]),),
        elapsed_ms=0.1,
    )
    workflow = FullAnswerWorkflow(
        SQLiteDocumentStore(store.database_path, read_only=True),
        prepared.document.document_id,
        _Retriever(result),
        _Model(),
        GenerationConfig(num_predict=3072),
    )
    return workflow, result, blocks
