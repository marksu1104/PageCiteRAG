"""Synthetic source-contract tests; without parser runtime or external documents."""

from __future__ import annotations

import numpy as np
import pytest

from page_cite_rag.docling_adapter import DoclingChunkRetriever, DoclingSourceStore
from page_cite_rag.docling_source import DoclingSourceError
from page_cite_rag.span_citations import (
    SpanAnswerDraft,
    SpanCitationDraft,
    SpanClaimDraft,
    source_spans,
    validate_span_claims,
)
from page_cite_rag.workflows import read_verified_source


def _source():
    document = {
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "text",
                "text": "Only when the condition holds.",
                "prov": [
                    {"page_no": 1, "bbox": {"l": 1, "t": 5, "r": 4, "b": 2}},
                    {"page_no": 2, "bbox": {"l": 1, "t": 5, "r": 4, "b": 2}},
                ],
            },
            {
                "self_ref": "#/texts/1",
                "label": "formula",
                "text": "",
                "prov": [{"page_no": 2, "bbox": {"l": 1, "t": 5, "r": 4, "b": 2}}],
            },
        ],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "label": "table",
                "prov": [{"page_no": 2, "bbox": {"l": 1, "t": 5, "r": 4, "b": 2}}],
                "data": {
                    "num_rows": 1,
                    "num_cols": 2,
                    "table_cells": [
                        {
                            "text": "Metric",
                            "start_row_offset_idx": 0,
                            "end_row_offset_idx": 1,
                            "start_col_offset_idx": 0,
                            "end_col_offset_idx": 1,
                        },
                        {
                            "text": "42",
                            "start_row_offset_idx": 0,
                            "end_row_offset_idx": 1,
                            "start_col_offset_idx": 1,
                            "end_col_offset_idx": 2,
                        },
                    ],
                },
            }
        ],
    }
    chunks = [
        {
            "chunk": {
                "meta": {
                    "doc_items": [
                        {"self_ref": "#/texts/0"},
                        {"self_ref": "#/texts/1"},
                        {"self_ref": "#/tables/0"},
                    ]
                }
            },
            "search_text": "Only when the condition holds. Metric 42",
        }
    ]
    return document, chunks


def _store():
    document, chunks = _source()
    return DoclingSourceStore(
        document, chunks, source_sha256="a" * 64, source_name="sample.pdf", page_count=2
    )


def test_virtual_units_are_exact_native_items_or_cells_not_pdf_offsets():
    store = _store()
    blocks = store.get_blocks(store.document_id)
    assert [block.text for block in blocks] == ["Only when the condition holds.", "Metric", "42"]
    assert all(block.page_number is None for block in blocks)
    assert store.chunk_block_ids == [tuple(block.block_id for block in blocks)]
    assert read_verified_source(store, store.document_id, blocks[0].block_id) == blocks[0]
    assert store.location_for(store.document_id, blocks[0].block_id)["pages"] == [1, 2]
    cell = store.location_for(store.document_id, blocks[2].block_id)
    assert cell["native_ref"] == "#/tables/0"
    assert cell["cell"]["column_start"] == 1
    assert cell["cell"]["cell_page_mapping"] == "unverified"
    assert cell["pdf_character_offsets"] == "unavailable"


def test_existing_span_validator_accepts_exact_cell_but_no_fabricated_page():
    store = _store()
    block = store.get_blocks(store.document_id)[2]
    span = source_spans((block,))[0]
    draft = SpanAnswerDraft(
        status="answered",
        claims=(
            SpanClaimDraft(
                text="The value is 42.", citations=(SpanCitationDraft(span_id=span.span_id),)
            ),
        ),
    )
    citation = validate_span_claims(store, store.document_id, draft, (block,))[0].citations[0]
    assert citation.quote == "42"
    assert citation.page_number is None
    assert citation.char_start == 0 and citation.char_end == 2


def test_unreadable_only_chunk_and_scope_violation_fail_closed():
    document, chunks = _source()
    chunks[0]["chunk"]["meta"]["doc_items"] = [{"self_ref": "#/texts/1"}]
    with pytest.raises(DoclingSourceError, match="no readable"):
        DoclingSourceStore(
            document, chunks, source_sha256="a" * 64, source_name="sample.pdf", page_count=2
        )
    store = _store()
    block = store.get_blocks(store.document_id)[0]
    assert store.get_block("b" * 64, block.block_id) is None
    assert store.location_for("b" * 64, block.block_id) is None


class _FakeEncoder:
    dimension = 2
    identity = {"model": "synthetic"}

    def encode_documents(self, texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_queries(self, texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


def test_chunk_retrieval_returns_native_blocks_not_search_representation():
    document, chunks = _source()
    chunks[0]["search_text"] = "Metadata heading: condition; Metric 42"
    store = DoclingSourceStore(
        document, chunks, source_sha256="a" * 64, source_name="sample.pdf", page_count=2
    )
    retriever = DoclingChunkRetriever(store, chunks, _FakeEncoder())
    result = retriever.search("condition", top_k=3)
    assert result.method == "hybrid"
    assert result.document_id == store.document_id
    assert len(result.hits) == 3
    assert result.hits[0].block.text == "Only when the condition holds."
    assert all("Metadata heading" not in hit.block.text for hit in result.hits)
    assert retriever.last_query_plan["lexical_query"] == "condition"


def test_chunk_retrieval_can_restrict_candidate_chunks_before_ranking():
    document, chunks = _source()
    store = DoclingSourceStore(
        document, chunks, source_sha256="a" * 64, source_name="sample.pdf", page_count=2
    )
    retriever = DoclingChunkRetriever(store, chunks, _FakeEncoder())

    scoped = retriever.search("condition", top_k=1, allowed_chunk_indices=(0,))

    assert [group.chunk_index for group in scoped.context_groups] == [0]
    assert retriever.last_query_plan["scoped_chunk_count"] == 1
    with pytest.raises(ValueError, match="allowed chunk scope"):
        retriever.search("condition", top_k=1, allowed_chunk_indices=(1,))


def test_table_hit_keeps_all_native_cells_available_with_query_relevance_order():
    document, chunks = _source()
    document["tables"][0]["data"]["num_cols"] = 12
    document["tables"][0]["data"]["table_cells"] = [
        {
            "text": f"value-{index}",
            "start_row_offset_idx": 0,
            "end_row_offset_idx": 1,
            "start_col_offset_idx": index,
            "end_col_offset_idx": index + 1,
        }
        for index in range(12)
    ]
    chunks[0]["search_text"] = "Only when the condition holds. Metric " + " ".join(
        f"value-{index}" for index in range(12)
    )
    store = DoclingSourceStore(
        document, chunks, source_sha256="a" * 64, source_name="sample.pdf", page_count=2
    )
    assert len(store.get_blocks(store.document_id)) == 13
    retriever = DoclingChunkRetriever(store, chunks, _FakeEncoder())
    result = retriever.search("condition", top_k=1)
    assert len(result.context_groups) == 1
    assert len(result.hits) == 13
    assert result.hits[0].block.text == "Only when the condition holds."
    assert (
        sum(
            store.location_for(store.document_id, hit.block.block_id)["kind"] == "table_cell"
            for hit in result.hits
        )
        == 12
    )
    assert result.context_groups[0].text == chunks[0]["search_text"]
    assert result.context_groups[0].source_refs == ("#/texts/0", "#/texts/1", "#/tables/0")
    assert set(result.context_groups[0].source_block_ids) == {
        hit.block.block_id for hit in result.hits
    }
    assert retriever.last_query_plan["selected_context_chunk_count"] == 1
    assert retriever.last_query_plan["selected_source_block_count"] == 13
    assert "context_groups" not in result.model_dump()
