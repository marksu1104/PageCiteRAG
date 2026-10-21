"""Read saved Docling items without confusing search context with source text.

This is an experimental, read-only source adapter. It accepts a replay-validated
DoclingDocument dictionary; it does not parse a PDF, index text, or grant an item
more precise citation coordinates than Docling saved.
"""

from __future__ import annotations

import math
import re

_ITEM_REF = re.compile(r"^#/(texts|tables|pictures|key_value_items|form_items)/([0-9]+)$")


class DoclingSourceError(ValueError):
    """The requested native item cannot be read back safely."""


def _item_by_ref(document: dict, ref: str) -> tuple[str, dict]:
    match = _ITEM_REF.fullmatch(ref)
    if match is None:
        raise DoclingSourceError("unsupported native item ref")
    collection, index_text = match.groups()
    entries = document.get(collection)
    index = int(index_text)
    if not isinstance(entries, list) or index >= len(entries):
        raise DoclingSourceError("native item ref is missing")
    item = entries[index]
    if not isinstance(item, dict) or item.get("self_ref") != ref:
        raise DoclingSourceError("native item identity changed")
    return collection, item


def _regions(item: dict, page_count: int) -> list[dict]:
    provenance = item.get("prov")
    if not isinstance(provenance, list) or not provenance:
        raise DoclingSourceError("native item has no page provenance")
    regions = []
    for prov in provenance:
        if not isinstance(prov, dict):
            raise DoclingSourceError("native item provenance is malformed")
        page = prov.get("page_no")
        bbox = prov.get("bbox")
        if not isinstance(page, int) or not 1 <= page <= page_count:
            raise DoclingSourceError("native item page is invalid")
        if not isinstance(bbox, dict) or any(
            not isinstance(bbox.get(key), (int, float)) or not math.isfinite(bbox[key])
            for key in ("l", "t", "r", "b")
        ):
            raise DoclingSourceError("native item bbox is invalid")
        if bbox["l"] >= bbox["r"] or bbox["t"] == bbox["b"]:
            raise DoclingSourceError("native item bbox is empty")
        regions.append({"page_number": page, "bbox": dict(bbox)})
    return regions


def _span_alignment(item: dict) -> str:
    text = item.get("text")
    if not isinstance(text, str):
        return "not_applicable"
    spans = [prov.get("charspan") for prov in item["prov"]]
    if any(span is None for span in spans):
        return "missing"
    if any(
        not isinstance(span, (list, tuple))
        or len(span) != 2
        or not all(isinstance(value, int) for value in span)
        or not 0 <= span[0] < span[1] <= len(text)
        for span in spans
    ):
        return "mismatch"
    return "within_saved_text_bounds_unverified"


def _table_cells(item: dict) -> dict:
    data = item.get("data")
    if not isinstance(data, dict):
        raise DoclingSourceError("native table data is missing")
    rows, cols, cells = data.get("num_rows"), data.get("num_cols"), data.get("table_cells")
    if not isinstance(rows, int) or not isinstance(cols, int) or rows < 1 or cols < 1:
        raise DoclingSourceError("native table dimensions are invalid")
    if not isinstance(cells, list):
        raise DoclingSourceError("native table cells are missing")
    result = []
    for cell in cells:
        if not isinstance(cell, dict) or not isinstance(cell.get("text"), str):
            raise DoclingSourceError("native table cell text is invalid")
        r0, r1 = cell.get("start_row_offset_idx"), cell.get("end_row_offset_idx")
        c0, c1 = cell.get("start_col_offset_idx"), cell.get("end_col_offset_idx")
        if not all(isinstance(value, int) for value in (r0, r1, c0, c1)) or not (
            0 <= r0 < r1 <= rows and 0 <= c0 < c1 <= cols
        ):
            raise DoclingSourceError("native table cell coordinates are invalid")
        result.append(
            {
                "text": cell["text"],
                "row_start": r0,
                "row_end": r1,
                "column_start": c0,
                "column_end": c1,
                "row_header": bool(cell.get("row_header", False)),
                "column_header": bool(cell.get("column_header", False)),
                "bbox": dict(cell["bbox"]) if isinstance(cell.get("bbox"), dict) else None,
            }
        )
    return {
        "rows": rows,
        "columns": cols,
        "cells": result,
        "cell_page_mapping": "unverified",
    }


def read_native_item(document: dict, ref: str, *, page_count: int) -> dict:
    """Return saved item text/structure and honest, item-level location limits.

    The caller must first validate the capture manifest and document hash. A
    bounded charspan is not proof of an exact quote in the original PDF.
    """
    if not isinstance(document, dict) or not isinstance(page_count, int) or page_count < 1:
        raise DoclingSourceError("native document or page count is invalid")
    collection, item = _item_by_ref(document, ref)
    regions = _regions(item, page_count)
    pages = sorted({region["page_number"] for region in regions})
    caption_refs = []
    for caption in item.get("captions", []):
        caption_ref = caption.get("$ref") if isinstance(caption, dict) else None
        if not isinstance(caption_ref, str):
            raise DoclingSourceError("native caption ref is malformed")
        caption_collection, _ = _item_by_ref(document, caption_ref)
        if caption_collection != "texts":
            raise DoclingSourceError("native caption is not an independent text item")
        caption_refs.append(caption_ref)
    text = item.get("text")
    saved_text = text if isinstance(text, str) and text.strip() else None
    return {
        "ref": ref,
        "kind": collection,
        "label": item.get("label"),
        "saved_item_text": saved_text,
        "text_scope": "whole_saved_item" if saved_text is not None else "unavailable",
        "pages": pages,
        "location_scope": "item_regions" if len(pages) == 1 else "multi_page_item_regions",
        "regions": regions,
        "span_alignment": _span_alignment(item),
        "caption_refs": caption_refs,
        "table": _table_cells(item) if collection == "tables" else None,
    }


def check_search_chunk_admission(
    document: dict,
    entry: dict,
    *,
    page_count: int,
    tokenizer: object,
    max_tokens: int = 512,
) -> dict:
    """Fail closed on an experimental chunk's actual search text and source refs.

    The caller still owns the index decision. Saved ``search_tokens`` is not
    trusted: count the exact representation that would be embedded instead.
    """
    if not isinstance(max_tokens, int) or max_tokens < 1:
        raise DoclingSourceError("search token budget is invalid")
    if not isinstance(entry, dict) or not isinstance(entry.get("search_text"), str):
        raise DoclingSourceError("search representation is missing")
    search_text = entry["search_text"]
    if not search_text.strip():
        raise DoclingSourceError("search representation is empty")
    chunk = entry.get("chunk")
    meta = chunk.get("meta") if isinstance(chunk, dict) else None
    items = meta.get("doc_items") if isinstance(meta, dict) else None
    if not isinstance(items, list) or not items:
        raise DoclingSourceError("search chunk has no native source refs")
    refs = []
    unsupported_refs = []
    seen_refs = set()
    for item in items:
        ref = item.get("self_ref") if isinstance(item, dict) else None
        if not isinstance(ref, str):
            raise DoclingSourceError("search chunk contains an invalid source ref")
        if ref in seen_refs:
            raise DoclingSourceError("search chunk repeats a native source ref")
        seen_refs.add(ref)
        source = read_native_item(document, ref, page_count=page_count)
        if source["saved_item_text"] is None and source["table"] is None:
            unsupported_refs.append(ref)
        else:
            refs.append(ref)
    if not refs:
        raise DoclingSourceError("search chunk has no readable text or table source")
    encoded = tokenizer.encode(search_text, add_special_tokens=False)
    if not isinstance(encoded, list):
        raise DoclingSourceError("tokenizer did not return token ids")
    token_count = len(encoded)
    return {
        "accepted": token_count <= max_tokens,
        "reason": "accepted" if token_count <= max_tokens else "token_budget_exceeded",
        "search_tokens": token_count,
        "max_tokens": max_tokens,
        "native_refs": refs,
        "unsupported_refs": unsupported_refs,
    }
