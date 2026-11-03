from __future__ import annotations

import pytest

from page_cite_rag.ingest import prepare_source
from page_cite_rag.references import (
    ReferencePinningRetriever,
    numbered_paragraph_blocks,
    referenced_numbers,
)
from page_cite_rag.retrieval import BM25Retriever, build_bm25_index
from page_cite_rag.store import SQLiteDocumentStore


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("請整合報告第124、129及132段", ("124", "129", "132")),
        ("Explain paragraphs 12 and 14, and item 3.", ("12", "14", "3")),
        ("Summarize the 3 main points in 2 sentences.", ()),
        ("說明第 7 條與第9項", ("7", "9")),
    ],
)
def test_only_explicit_reference_phrases_are_resolved(question, expected):
    assert referenced_numbers(question) == expected


def _store(tmp_path):
    text = "\n\n".join(
        [f"{n}. Filler paragraph number {n} about unrelated gardening topics." for n in range(1, 8)]
        + ["8. The pinned obligation requires an annual audit of every deployed system."]
    )
    path = tmp_path / "numbered.txt"
    path.write_text(text, encoding="utf-8")
    prepared = prepare_source(path, usage_scope="synthetic-test")
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    build_bm25_index(store, prepared.document.document_id)
    return store, prepared.document.document_id


def test_numbered_paragraph_lookup_pins_only_the_named_paragraph(tmp_path):
    store, document_id = _store(tmp_path)
    markers = numbered_paragraph_blocks(store, document_id)
    block_8 = markers["8"][0]
    assert "annual audit" in store.get_block(document_id, block_8).text

    retriever = ReferencePinningRetriever(
        BM25Retriever(store, document_id), store, document_id, markers
    )
    plain = retriever._inner.search("gardening topics", top_k=5)
    assert block_8 not in [hit.block.block_id for hit in plain.hits]

    pinned = retriever.search("gardening topics in paragraph 8", top_k=5)
    assert pinned.hits[0].block.block_id == block_8
    assert [hit.rank for hit in pinned.hits] == list(range(1, len(pinned.hits) + 1))
    assert retriever.last_pin["numbers"] == ["8"]

    untouched = retriever.search("gardening topics", top_k=5)
    assert [hit.block.block_id for hit in untouched.hits] == [
        hit.block.block_id for hit in plain.hits
    ]
    assert retriever.last_pin["block_ids"] == []
