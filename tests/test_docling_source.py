"""Candidate Docling read-back contract; fixtures contain synthetic text only."""

from __future__ import annotations

import pytest

from page_cite_rag.docling_source import (
    DoclingSourceError,
    check_search_chunk_admission,
    read_native_item,
)


def _document():
    return {
        "texts": [
            {
                "self_ref": "#/texts/0",
                "label": "text",
                "text": "A condition applies across pages.",
                "prov": [
                    {"page_no": 1, "bbox": {"l": 1, "t": 4, "r": 5, "b": 2}, "charspan": [0, 20]},
                    {"page_no": 2, "bbox": {"l": 1, "t": 4, "r": 5, "b": 2}, "charspan": [21, 35]},
                ],
            },
            {
                "self_ref": "#/texts/1",
                "label": "caption",
                "text": "Table 1: example",
                "prov": [
                    {"page_no": 2, "bbox": {"l": 1, "t": 4, "r": 5, "b": 2}, "charspan": [0, 16]}
                ],
            },
            {
                "self_ref": "#/texts/2",
                "label": "formula",
                "text": "",
                "prov": [
                    {"page_no": 2, "bbox": {"l": 1, "t": 4, "r": 5, "b": 2}, "charspan": [0, 9]}
                ],
            },
        ],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "label": "table",
                "prov": [{"page_no": 2, "bbox": {"l": 1, "t": 4, "r": 5, "b": 2}}],
                "captions": [{"$ref": "#/texts/1"}],
                "data": {
                    "num_rows": 1,
                    "num_cols": 2,
                    "table_cells": [
                        {
                            "text": "Measure",
                            "start_row_offset_idx": 0,
                            "end_row_offset_idx": 1,
                            "start_col_offset_idx": 0,
                            "end_col_offset_idx": 1,
                            "column_header": True,
                        },
                        {
                            "text": "12",
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


def test_text_uses_saved_item_not_search_representation_or_fake_single_page():
    source = read_native_item(_document(), "#/texts/0", page_count=2)
    assert source["saved_item_text"] == "A condition applies across pages."
    assert source["pages"] == [1, 2]
    assert source["location_scope"] == "multi_page_item_regions"
    assert source["span_alignment"] == "mismatch"


def test_table_cells_and_caption_remain_separate_native_sources():
    document = _document()
    table = read_native_item(document, "#/tables/0", page_count=2)
    caption = read_native_item(document, "#/texts/1", page_count=2)
    assert table["saved_item_text"] is None
    assert table["caption_refs"] == ["#/texts/1"]
    assert table["table"]["cells"][1]["text"] == "12"
    assert table["table"]["cells"][1]["column_start"] == 1
    assert table["table"]["cell_page_mapping"] == "unverified"
    assert caption["saved_item_text"] == "Table 1: example"
    assert caption["span_alignment"] == "within_saved_text_bounds_unverified"


def test_empty_formula_is_not_presented_as_quoteable_text():
    source = read_native_item(_document(), "#/texts/2", page_count=2)
    assert source["saved_item_text"] is None
    assert source["text_scope"] == "unavailable"
    assert source["span_alignment"] == "mismatch"


def test_unknown_ref_and_unlocated_source_are_rejected():
    document = _document()
    with pytest.raises(DoclingSourceError, match="missing"):
        read_native_item(document, "#/texts/99", page_count=2)
    document["texts"][0]["prov"][0]["page_no"] = 3
    with pytest.raises(DoclingSourceError, match="page"):
        read_native_item(document, "#/texts/0", page_count=2)


def test_invalid_table_cell_coordinates_are_rejected():
    document = _document()
    document["tables"][0]["data"]["table_cells"][1]["end_col_offset_idx"] = 3
    with pytest.raises(DoclingSourceError, match="coordinates"):
        read_native_item(document, "#/tables/0", page_count=2)


class _CountTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(range(len(text.split())))


def test_admission_recounts_actual_search_text_and_rejects_over_budget():
    entry = {
        "search_text": "long heading plus source text",
        "search_tokens": 1,
        "chunk": {"meta": {"doc_items": [{"self_ref": "#/texts/0"}]}},
    }
    result = check_search_chunk_admission(
        _document(), entry, page_count=2, tokenizer=_CountTokenizer(), max_tokens=3
    )
    assert result == {
        "accepted": False,
        "reason": "token_budget_exceeded",
        "search_tokens": 5,
        "max_tokens": 3,
        "native_refs": ["#/texts/0"],
        "unsupported_refs": [],
    }
    assert (
        check_search_chunk_admission(
            _document(), entry, page_count=2, tokenizer=_CountTokenizer(), max_tokens=5
        )["accepted"]
        is True
    )


def test_admission_rejects_missing_source_before_any_index_decision():
    entry = {
        "search_text": "a searchable chunk",
        "chunk": {"meta": {"doc_items": [{"self_ref": "#/texts/99"}]}},
    }
    with pytest.raises(DoclingSourceError, match="missing"):
        check_search_chunk_admission(_document(), entry, page_count=2, tokenizer=_CountTokenizer())


def test_admission_excludes_unreadable_formula_but_keeps_readable_text():
    entry = {
        "search_text": "a formula followed by readable text",
        "chunk": {
            "meta": {
                "doc_items": [
                    {"self_ref": "#/texts/0"},
                    {"self_ref": "#/texts/2"},
                ]
            }
        },
    }
    result = check_search_chunk_admission(
        _document(), entry, page_count=2, tokenizer=_CountTokenizer()
    )
    assert result["native_refs"] == ["#/texts/0"]
    assert result["unsupported_refs"] == ["#/texts/2"]


def test_admission_rejects_formula_only_chunk():
    entry = {
        "search_text": "formula without recovered source text",
        "chunk": {"meta": {"doc_items": [{"self_ref": "#/texts/2"}]}},
    }
    with pytest.raises(DoclingSourceError, match="no readable text"):
        check_search_chunk_admission(_document(), entry, page_count=2, tokenizer=_CountTokenizer())


def test_admission_keeps_readable_table_cells_without_flattening_them():
    entry = {
        "search_text": "Measure 12",
        "chunk": {"meta": {"doc_items": [{"self_ref": "#/tables/0"}]}},
    }
    result = check_search_chunk_admission(
        _document(), entry, page_count=2, tokenizer=_CountTokenizer()
    )
    assert result["accepted"] is True
    assert result["native_refs"] == ["#/tables/0"]
