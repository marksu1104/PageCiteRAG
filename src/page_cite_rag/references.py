"""Resolve explicit paragraph references (e.g. "paragraph 124") from document structure.

This is address lookup, not retrieval: it never guesses by meaning, never reads an
answer, and uses only numbering saved in the source. Every flow shares it through
``document_retriever`` so a fixed pipeline and an Agent see the same pinned sources.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from page_cite_rag.retrieval import SearchContextGroup, SearchHit, SearchResult

_NUMBER_LIST = r"([0-9]+(?:\s*(?:[,，、&]|and\b|與|及|和)\s*(?:第\s*)?[0-9]+)*)"
_PATTERNS = (
    re.compile(
        r"\b(?:paragraphs?|paras?\.?|clauses?|items?)\s*(?:nos?\.?\s*)?" + _NUMBER_LIST, re.I
    ),
    re.compile(r"(?:第\s*)?" + _NUMBER_LIST + r"\s*(?:段|條|項)", re.I),
)
_LINE_MARKER = re.compile(r"(?m)^\s*([0-9]{1,4})[.．)]\s")
_MARKER_NUMBER = re.compile(r"\s*([0-9]+)[.．)]\s*")
MAX_PINNED_NUMBERS = 8


def referenced_numbers(question: str) -> tuple[str, ...]:
    """Numbers bound to an explicit reference phrase, not counts such as "in 3 points"."""
    numbers: dict[str, None] = {}
    for pattern in _PATTERNS:
        for match in pattern.finditer(question):
            for number in re.findall(r"[0-9]+", match[1]):
                numbers[number] = None
    return tuple(numbers)[:MAX_PINNED_NUMBERS]


def numbered_paragraph_blocks(store, document_id: str, native_source=None) -> dict[str, list[str]]:
    """Map a paragraph number to the source blocks that carry that exact numbering."""
    result: dict[str, list[str]] = {}
    if native_source is not None:
        for item in getattr(native_source, "document_data", {}).get("texts", ()):
            marker, original, text = item.get("marker"), item.get("orig"), item.get("text")
            if not (
                item.get("label") == "list_item"
                and item.get("enumerated") is True
                and isinstance(marker, str)
                and isinstance(original, str)
                and isinstance(text, str)
                and original.startswith(marker)
                and original[len(marker) :].lstrip() == text
                and isinstance(item.get("self_ref"), str)
            ):
                continue
            number = _MARKER_NUMBER.fullmatch(marker)
            if number is not None:
                result.setdefault(number[1], []).extend(
                    native_source.block_ids_for_ref(document_id, item["self_ref"])
                )
        return {key: list(dict.fromkeys(ids)) for key, ids in result.items()}
    for block in store.get_blocks(document_id):
        for match in _LINE_MARKER.finditer(block.text):
            result.setdefault(match[1], []).append(block.block_id)
    return {key: list(dict.fromkeys(ids)) for key, ids in result.items()}


class ReferencePinningRetriever:
    """Place explicitly numbered paragraphs first, leaving other ranking untouched."""

    def __init__(self, inner, store, document_id: str, markers: dict[str, list[str]]) -> None:
        self._inner = inner
        self._store = store
        self._document_id = document_id
        self._markers = markers
        self._native = getattr(inner, "store", None)
        self._chunks_by_block: dict[str, list[int]] = {}
        if self._native is not None and hasattr(self._native, "chunk_block_ids"):
            for index in range(len(self._native.chunk_block_ids)):
                for block_id in self._native.context_block_ids_for_chunk(index):
                    self._chunks_by_block.setdefault(block_id, []).append(index)
        self.last_pin: dict | None = None

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    def pinned_block_ids(self, query: str) -> list[str]:
        numbers = referenced_numbers(query)
        return list(
            dict.fromkeys(
                block_id for number in numbers for block_id in self._markers.get(number, ())
            )
        )

    def search(self, query: str, *, top_k: int = 5, **kwargs) -> SearchResult:
        result = self._inner.search(query, top_k=top_k, **kwargs)
        pinned = self.pinned_block_ids(query) if not kwargs else []
        self.last_pin = {"numbers": list(referenced_numbers(query)), "block_ids": pinned}
        if not pinned or not isinstance(result, SearchResult):
            return result
        return self._pin(result, pinned, top_k)

    def _hit(self, block_id: str, rank: int, score: float, existing: dict) -> SearchHit:
        previous = existing.get(block_id)
        block = (
            previous.block
            if previous is not None
            else self._store.get_block(self._document_id, block_id)
        )
        if block is None:
            raise ValueError("a referenced source block is missing from the store")
        return SearchHit(rank=rank, score=previous.score if previous else score, block=block)

    def _pin(self, result: SearchResult, pinned: Sequence[str], top_k: int) -> SearchResult:
        existing = {hit.block.block_id: hit for hit in result.hits}
        floor = max((hit.score for hit in result.hits), default=0.0)
        if result.context_groups:
            groups_by_index = {group.chunk_index: group for group in result.context_groups}
            pinned_groups: list[SearchContextGroup] = []
            for block_id in pinned:
                for index in self._chunks_by_block.get(block_id, ()):
                    group = groups_by_index.get(index) or SearchContextGroup(
                        chunk_index=index,
                        text=self._native.contextualized_chunk_text(index),
                        source_refs=self._native.context_refs_for_chunk(index),
                        source_block_ids=self._native.context_block_ids_for_chunk(index),
                    )
                    if group not in pinned_groups:
                        pinned_groups.append(group)
            if not pinned_groups:
                return result
            others = [g for g in result.context_groups if g not in pinned_groups]
            groups = tuple((pinned_groups + others)[: max(top_k, len(pinned_groups))])
            order = list(
                dict.fromkeys(block_id for group in groups for block_id in group.source_block_ids)
            )
            hits = tuple(
                self._hit(block_id, rank, floor, existing) for rank, block_id in enumerate(order, 1)
            )
            return result.model_copy(update={"hits": hits, "context_groups": groups})
        order = list(dict.fromkeys([*pinned, *(hit.block.block_id for hit in result.hits)]))[
            : max(top_k, len(pinned))
        ]
        hits = tuple(
            self._hit(block_id, rank, floor, existing) for rank, block_id in enumerate(order, 1)
        )
        return result.model_copy(update={"hits": hits})
