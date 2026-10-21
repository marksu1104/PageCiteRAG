"""Unweighted reciprocal-rank fusion; lexical scores never mix with cosine scores."""

from __future__ import annotations

import hashlib
import time
from typing import Protocol

from page_cite_rag.retrieval import RetrievalError, SearchBranchTrace, SearchHit, SearchResult


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int = 5) -> SearchResult: ...


def reciprocal_rank_fusion(
    results: list[SearchResult], *, top_k: int = 5, constant: int = 60
) -> tuple[SearchHit, ...]:
    if type(top_k) is not int or top_k < 1 or type(constant) is not int or constant < 1:
        raise ValueError("RRF budgets must be positive integers")
    if len({r.document_id for r in results}) > 1 or len({r.query for r in results}) > 1:
        raise RetrievalError("cannot fuse different document scopes or queries")
    scores = {}
    blocks = {}
    for result in results:
        seen = set()
        for rank, hit in enumerate(result.hits, 1):
            block = hit.block
            if (
                block.document_id != result.document_id
                or block.block_id in seen
                or hit.rank != rank
            ):
                raise RetrievalError("invalid ranking scope, duplicate or rank in RRF input")
            seen.add(block.block_id)
            if block.block_id in blocks and blocks[block.block_id] != block:
                raise RetrievalError("RRF branches disagree about source blocks")
            blocks[block.block_id] = block
            scores[block.block_id] = scores.get(block.block_id, 0) + 1 / (constant + rank)
    ordered = sorted(scores, key=lambda key: (-scores[key], blocks[key].ordinal, key))[:top_k]
    return tuple(
        SearchHit(rank=rank, score=scores[key], block=blocks[key])
        for rank, key in enumerate(ordered, 1)
    )


class HybridRetriever:
    def __init__(
        self, lexical: Retriever, dense: Retriever, *, candidates: int = 20, constant: int = 60
    ) -> None:
        if (
            type(candidates) is not int
            or candidates < 1
            or type(constant) is not int
            or constant < 1
        ):
            raise ValueError("hybrid budgets must be positive integers")
        self._lexical = lexical
        self._dense = dense
        self._candidates = candidates
        self._constant = constant

    def search(self, query: str, *, top_k: int = 5) -> SearchResult:
        if (
            not isinstance(query, str)
            or type(top_k) is not int
            or not 1 <= top_k <= self._candidates
        ):
            raise ValueError("top_k must be positive and no larger than the candidate budget")
        start = time.perf_counter()
        results = [
            retriever.search(query, top_k=self._candidates)
            for retriever in (self._lexical, self._dense)
        ]
        hits = reciprocal_rank_fusion(results, top_k=top_k, constant=self._constant)
        identity = hashlib.sha256(
            repr(([r.index_id for r in results], self._candidates, self._constant)).encode()
        ).hexdigest()
        return SearchResult(
            document_id=results[0].document_id,
            index_id=identity,
            method="hybrid",
            query=query,
            hits=hits,
            elapsed_ms=(time.perf_counter() - start) * 1000,
            branches=tuple(
                SearchBranchTrace(
                    name=name,
                    index_id=result.index_id,
                    elapsed_ms=result.elapsed_ms,
                    block_ids=tuple(hit.block.block_id for hit in result.hits),
                )
                for name, result in zip(("lexical", "dense"), results, strict=True)
            ),
        )

    def search_scoped(
        self,
        query: str,
        *,
        allowed_block_ids: set[str],
        candidate_k: int,
        top_k: int = 5,
    ) -> SearchResult:
        """Rank the authorized block subset using the same lexical/dense RRF pipeline."""
        if (
            not isinstance(query, str)
            or not query.strip()
            or type(candidate_k) is not int
            or candidate_k < 1
            or type(top_k) is not int
            or not 1 <= top_k <= self._candidates
            or not allowed_block_ids
            or any(not isinstance(block_id, str) for block_id in allowed_block_ids)
        ):
            raise ValueError("invalid scoped hybrid retrieval request")
        started = time.perf_counter()
        restricted = []
        for branch in (self._lexical, self._dense):
            result = branch.search(query, top_k=candidate_k)
            hits = tuple(hit for hit in result.hits if hit.block.block_id in allowed_block_ids)
            restricted.append(
                result.model_copy(
                    update={
                        "hits": tuple(
                            hit.model_copy(update={"rank": rank})
                            for rank, hit in enumerate(hits, 1)
                        ),
                        "branches": (),
                    }
                )
            )
        hits = reciprocal_rank_fusion(restricted, top_k=top_k, constant=self._constant)
        identity = {
            "branches": [result.index_id for result in restricted],
            "candidates": candidate_k,
            "constant": self._constant,
            "scope": sorted(allowed_block_ids),
            "version": "section-scoped-hybrid-rerank/v1",
        }
        index_id = hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()
        return SearchResult(
            document_id=restricted[0].document_id,
            index_id=index_id,
            method="hybrid",
            query=query,
            hits=hits,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            branches=tuple(
                SearchBranchTrace(
                    name=name,
                    index_id=result.index_id,
                    elapsed_ms=result.elapsed_ms,
                    block_ids=tuple(hit.block.block_id for hit in result.hits),
                )
                for name, result in zip(("lexical", "dense"), restricted, strict=True)
            ),
        )
