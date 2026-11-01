"""Unweighted reciprocal-rank fusion; lexical scores never mix with cosine scores."""

from __future__ import annotations

import hashlib
import math
import time
from typing import Protocol

from page_cite_rag.retrieval import RetrievalError, SearchBranchTrace, SearchHit, SearchResult
from page_cite_rag.retrieval_profile import FORMAL_RETRIEVAL


class Retriever(Protocol):
    def search(self, query: str, *, top_k: int = 5) -> SearchResult: ...


def rerank_search_result(result: SearchResult, ranker, *, top_k: int, store=None) -> SearchResult:
    """Rank whole reading chunks while keeping exact authoritative source identities."""
    if top_k < 1:
        raise ValueError("rerank top_k must be positive")
    groups = result.context_groups
    texts = [group.text for group in groups] if groups else [hit.block.text for hit in result.hits]
    if not texts:
        return result
    started = time.perf_counter()
    ranked = ranker.rerank(result.query, texts)
    if (
        len(ranked) != len(texts)
        or {index for index, _ in ranked} != set(range(len(texts)))
        or any(
            type(index) is not int or type(score) not in (int, float) or not math.isfinite(score)
            for index, score in ranked
        )
    ):
        raise RetrievalError("reranker returned an incomplete or invalid candidate ranking")
    ranked = sorted(ranked, key=lambda item: (-item[1], item[0]))[:top_k]
    selected_groups = tuple(groups[index] for index, _ in ranked) if groups else ()
    blocks = {hit.block.block_id: hit.block for hit in result.hits}
    selected = {}
    if groups:
        for index, score in ranked:
            for block_id in groups[index].source_block_ids:
                block = blocks.get(block_id)
                if block is None and store is not None:
                    block = store.get_block(result.document_id, block_id)
                if block is None:
                    raise RetrievalError("reranked chunk source is missing")
                selected.setdefault(block_id, (block, score))
    else:
        for index, score in ranked:
            block = result.hits[index].block
            selected[block.block_id] = (block, score)
    selected_blocks = list(selected.values())
    return result.model_copy(
        update={
            "index_id": hashlib.sha256(
                f"{result.index_id}/qwen3-reranker-0.6b/v1".encode()
            ).hexdigest(),
            "hits": tuple(
                SearchHit(rank=rank, score=score, block=block)
                for rank, (block, score) in enumerate(
                    selected_blocks if groups else selected_blocks[:top_k], 1
                )
            ),
            "context_groups": selected_groups,
            "elapsed_ms": result.elapsed_ms + (time.perf_counter() - started) * 1000,
        }
    )


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


def fuse_routed_hybrid_results(
    query: str,
    lexical_results: list[SearchResult],
    dense_result: SearchResult,
    *,
    lexical_top_k: int = 20,
    top_k: int = 40,
    constant: int = 60,
) -> SearchResult:
    """Fuse one or more BM25 query routes with one selected dense query route.

    Multiple BM25 lists are fused into one lexical list before the outer hybrid
    fusion. Scores are rank-only and never compared across retriever families.
    """
    if (
        not isinstance(query, str)
        or not query.strip()
        or not lexical_results
        or type(lexical_top_k) is not int
        or lexical_top_k < 1
        or type(top_k) is not int
        or not lexical_top_k <= top_k <= 2 * lexical_top_k
        or type(constant) is not int
        or constant < 1
    ):
        raise ValueError("invalid routed hybrid fusion budget")
    document_ids = {result.document_id for result in [*lexical_results, dense_result]}
    if len(document_ids) != 1:
        raise RetrievalError("routed retrieval branches must share one document scope")
    if any(result.method != "bm25" for result in lexical_results) or dense_result.method != "dense":
        raise RetrievalError("routed hybrid fusion requires BM25 and dense branch results")
    for result in [*lexical_results, dense_result]:
        if len(result.hits) > lexical_top_k:
            raise RetrievalError("routed retrieval branch exceeded its candidate limit")
        if any(hit.rank != rank for rank, hit in enumerate(result.hits, 1)):
            raise RetrievalError("routed retrieval branch has invalid ranks")

    normalized_lexical = [result.model_copy(update={"query": query}) for result in lexical_results]
    lexical_hits = (
        reciprocal_rank_fusion(normalized_lexical, top_k=lexical_top_k, constant=constant)
        if len(normalized_lexical) > 1
        else normalized_lexical[0].hits
    )
    lexical_identity = hashlib.sha256(
        repr(
            (tuple(result.index_id for result in normalized_lexical), lexical_top_k, constant)
        ).encode()
    ).hexdigest()
    lexical = SearchResult(
        document_id=normalized_lexical[0].document_id,
        index_id=lexical_identity,
        method="bm25",
        query=query,
        hits=tuple(
            hit.model_copy(update={"rank": rank}) for rank, hit in enumerate(lexical_hits, 1)
        ),
        elapsed_ms=sum(result.elapsed_ms for result in normalized_lexical),
    )
    normalized_dense = dense_result.model_copy(update={"query": query})
    hits = reciprocal_rank_fusion([lexical, normalized_dense], top_k=top_k, constant=constant)
    identity = hashlib.sha256(
        repr(
            (
                tuple(result.index_id for result in normalized_lexical),
                dense_result.index_id,
                lexical_top_k,
                top_k,
                constant,
                "routed-hybrid-rrf/v1",
            )
        ).encode()
    ).hexdigest()
    branches = (
        SearchBranchTrace(
            name="lexical",
            index_id=lexical.index_id,
            elapsed_ms=lexical.elapsed_ms,
            block_ids=tuple(hit.block.block_id for hit in lexical.hits),
        ),
        SearchBranchTrace(
            name="dense",
            index_id=normalized_dense.index_id,
            elapsed_ms=normalized_dense.elapsed_ms,
            block_ids=tuple(hit.block.block_id for hit in normalized_dense.hits),
        ),
    )
    return SearchResult(
        document_id=lexical.document_id,
        index_id=identity,
        method="hybrid",
        query=query,
        hits=hits,
        elapsed_ms=sum(result.elapsed_ms for result in [*normalized_lexical, dense_result]),
        branches=branches,
    )


class HybridRetriever:
    def __init__(
        self,
        lexical: Retriever,
        dense: Retriever,
        *,
        candidates: int = FORMAL_RETRIEVAL.candidates_per_branch,
        constant: int = FORMAL_RETRIEVAL.rrf_constant,
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
        return self.search_with_lexical_query(query, query, top_k=top_k)

    def search_with_lexical_query(
        self, query: str, lexical_query: str, *, top_k: int = 5
    ) -> SearchResult:
        """Expand only BM25 terms while dense retrieval stays on the original query."""
        if (
            not isinstance(query, str)
            or not query.strip()
            or not isinstance(lexical_query, str)
            or not lexical_query.strip()
            or type(top_k) is not int
            or not 1 <= top_k <= 2 * self._candidates
        ):
            raise ValueError("top_k must be positive and no larger than the candidate budget")
        start = time.perf_counter()
        lexical = self._lexical.search(lexical_query, top_k=self._candidates)
        dense = self._dense.search(query, top_k=self._candidates)
        results = [lexical.model_copy(update={"query": query}), dense]
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
            or not 1 <= top_k <= 2 * self._candidates
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
