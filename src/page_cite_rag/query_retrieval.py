"""Opt-in lexical query rewriting without changing the frozen retrieval baseline."""

from __future__ import annotations

import hashlib
import json
import math
import time

from page_cite_rag.hybrid import Retriever, reciprocal_rank_fusion
from page_cite_rag.query import QueryRewriter
from page_cite_rag.retrieval import RetrievalError, SearchBranchTrace, SearchResult
from page_cite_rag.retrieval_profile import FORMAL_RETRIEVAL, MAX_CONTEXT_GROUPS


class RewritingRetriever:
    """Rewrite only lexical queries; dense and answering retain the original question.

    Existing HybridRetriever and benchmark rankings are unchanged. The last query
    plan makes translation cost and degradation visible; this instance is intended
    for sequential use, just like the resident model client.
    """

    def __init__(
        self,
        lexical: Retriever | None,
        rewriter: QueryRewriter,
        dense: Retriever | None = None,
        *,
        search_backend: Retriever | None = None,
        candidates: int = FORMAL_RETRIEVAL.candidates_per_branch,
        constant: int = FORMAL_RETRIEVAL.rrf_constant,
    ) -> None:
        if (
            type(candidates) is not int
            or not 1 <= candidates <= 100
            or type(constant) is not int
            or not 1 <= constant <= 1000
            or (search_backend is not None and (lexical is not None or dense is not None))
            or (search_backend is None and lexical is None)
            or (search_backend is None and dense is not None and lexical is None)
        ):
            raise ValueError("invalid rewriting retrieval candidate or RRF budget")
        self._lexical = lexical
        self._dense = dense
        self._search_backend = search_backend
        self._rewriter = rewriter
        self._candidates = candidates
        self._constant = constant
        self.last_query_plan: dict | None = None
        self._rewrite_enabled = True

    def disable_rewrites(self) -> None:
        """Keep later Agent searches focused without spending another rewrite call."""
        self._rewrite_enabled = False

    @staticmethod
    def _check_result(
        result: SearchResult, expected_query: str, limit: int, expected_method: str
    ) -> None:
        if not isinstance(result, SearchResult) or result.query != expected_query:
            raise RetrievalError("query-rewrite backend returned an invalid query scope")
        if (
            result.method != expected_method
            or not result.index_id
            or type(result.elapsed_ms) not in (float, int)
            or not math.isfinite(result.elapsed_ms)
            or result.elapsed_ms < 0
            or (len(result.context_groups) if result.context_groups else len(result.hits)) > limit
        ):
            raise RetrievalError("query-rewrite branch returned invalid query or index metadata")
        seen = set()
        for rank, hit in enumerate(result.hits, 1):
            if (
                type(hit.rank) is not int
                or hit.rank != rank
                or hit.block.document_id != result.document_id
                or hit.block.block_id in seen
                or type(hit.score) not in (float, int)
                or not math.isfinite(hit.score)
            ):
                raise RetrievalError("query-rewrite branch returned invalid scoped ranking")
            seen.add(hit.block.block_id)

    def search(self, query: str, *, top_k: int = 5) -> SearchResult:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("rewriting retrieval requires a nonempty query")
        if type(top_k) is not int or not 1 <= top_k <= MAX_CONTEXT_GROUPS:
            raise ValueError("top_k must be within the shared context-group limit")
        started = time.perf_counter()
        self.last_query_plan = None
        plan = self._rewriter.rewrite(query) if self._rewrite_enabled else None
        lexical_query = plan.lexical_query if plan is not None else query
        if self._search_backend is not None:
            if plan is None:
                result = self._search_backend.search(query, top_k=top_k)
            else:
                search_with_lexical_query = getattr(
                    self._search_backend, "search_with_lexical_query", None
                )
                if not callable(search_with_lexical_query):
                    raise RetrievalError("staged search does not support lexical query expansion")
                result = search_with_lexical_query(query, lexical_query, top_k=top_k)
            self._check_result(result, query, top_k, "hybrid")
            self.last_query_plan = {
                **(
                    plan.model_dump(mode="json")
                    if plan is not None
                    else {
                        "original_query": query,
                        "lexical_query": query,
                        "status": "not_needed",
                        "reason": "focused_agent_query_no_extra_rewrite",
                        "elapsed_ms": 0.0,
                        "cache_hit": False,
                    }
                ),
                "search_backend_elapsed_ms": result.elapsed_ms,
                "identity": {
                    "backend": type(self._search_backend).__name__,
                    "rewriter": self._rewriter.identity if plan is not None else None,
                },
            }
            return result
        limit = self._candidates if self._dense is not None else min(top_k, self._candidates)
        lexical = self._lexical.search(lexical_query, top_k=limit)
        self._check_result(lexical, lexical_query, limit, "bm25")
        branch_ids = [lexical.index_id]
        if self._dense is None:
            # Keep the raw lexical query in the trace, not the public query used
            # for document scope validation and answering.
            # BM25-only rewriting must retain original BM25 scores, not pretend
            # a one-branch RRF score was the lexical score.
            hits = tuple(lexical.hits[:top_k])
            method = "bm25"
        else:
            dense = self._dense.search(query, top_k=self._candidates)
            self._check_result(dense, query, self._candidates, "dense")
            branch_ids.append(dense.index_id)
            hits = reciprocal_rank_fusion(
                [lexical.model_copy(update={"query": query}), dense],
                top_k=top_k,
                constant=self._constant,
            )
            method = "hybrid"
        identity = {
            "branches": branch_ids,
            "rewriter": self._rewriter.identity if plan is not None else None,
            "candidates": self._candidates,
            "constant": self._constant,
            "version": "original-dense/optional-expanded-lexical/v2",
        }
        index_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest()
        self.last_query_plan = {
            **(
                plan.model_dump(mode="json")
                if plan is not None
                else {
                    "original_query": query,
                    "lexical_query": query,
                    "status": "not_needed",
                    "reason": "focused_agent_query_no_extra_rewrite",
                    "elapsed_ms": 0.0,
                    "cache_hit": False,
                }
            ),
            "dense_query": query if self._dense is not None else None,
            "lexical_elapsed_ms": lexical.elapsed_ms,
            "dense_elapsed_ms": dense.elapsed_ms if self._dense is not None else None,
            "identity": identity,
        }
        return SearchResult(
            document_id=lexical.document_id,
            index_id=index_id,
            method=method,
            query=query,
            hits=hits,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            branches=(
                SearchBranchTrace(
                    name="lexical",
                    index_id=lexical.index_id,
                    elapsed_ms=lexical.elapsed_ms,
                    block_ids=tuple(hit.block.block_id for hit in lexical.hits),
                ),
                *(
                    (
                        SearchBranchTrace(
                            name="dense",
                            index_id=dense.index_id,
                            elapsed_ms=dense.elapsed_ms,
                            block_ids=tuple(hit.block.block_id for hit in dense.hits),
                        ),
                    )
                    if self._dense is not None
                    else ()
                ),
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
        """Rewrite only the lexical branch, then rank within the authorized source IDs."""
        if (
            not isinstance(query, str)
            or not query.strip()
            or type(candidate_k) is not int
            or candidate_k < 1
            or type(top_k) is not int
            or not 1 <= top_k <= self._candidates
            or not allowed_block_ids
        ):
            raise ValueError("invalid scoped query-rewrite request")
        if self._search_backend is not None:
            if self._rewrite_enabled:
                raise RetrievalError("scoped staged search is reserved for Agent follow-up queries")
            scoped = getattr(self._search_backend, "search_scoped", None)
            if not callable(scoped):
                raise RetrievalError("staged search does not support scoped retrieval")
            return scoped(
                query,
                allowed_block_ids=allowed_block_ids,
                candidate_k=candidate_k,
                top_k=top_k,
            )
        started = time.perf_counter()
        plan = self._rewriter.rewrite(query) if self._rewrite_enabled else None
        lexical_query = plan.lexical_query if plan is not None else query
        lexical = self._lexical.search(lexical_query, top_k=candidate_k).model_copy(
            update={"query": query}
        )
        lexical_hits = tuple(hit for hit in lexical.hits if hit.block.block_id in allowed_block_ids)
        lexical = lexical.model_copy(
            update={
                "hits": tuple(
                    hit.model_copy(update={"rank": rank})
                    for rank, hit in enumerate(lexical_hits, 1)
                ),
                "branches": (),
            }
        )
        branch_results = [lexical]
        if self._dense is not None:
            dense_result = self._dense.search(query, top_k=candidate_k)
            dense_hits = tuple(
                hit for hit in dense_result.hits if hit.block.block_id in allowed_block_ids
            )
            branch_results.append(
                dense_result.model_copy(
                    update={
                        "hits": tuple(
                            hit.model_copy(update={"rank": rank})
                            for rank, hit in enumerate(dense_hits, 1)
                        ),
                        "branches": (),
                    }
                )
            )
            hits = reciprocal_rank_fusion(branch_results, top_k=top_k, constant=self._constant)
            method = "hybrid"
        else:
            hits = lexical.hits[:top_k]
            method = "bm25"
        identity = {
            "branches": [branch.index_id for branch in branch_results],
            "candidate_k": candidate_k,
            "constant": self._constant,
            "rewrite": (
                plan.model_dump(mode="json")
                if plan is not None
                else {
                    "status": "not_needed",
                    "reason": "focused_agent_query_no_extra_rewrite",
                    "original_query": query,
                    "lexical_query": query,
                    "elapsed_ms": 0.0,
                }
            ),
            "scope": sorted(allowed_block_ids),
            "version": "section-scoped-rewrite/v1",
        }
        index_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True, allow_nan=False).encode("utf-8")
        ).hexdigest()
        self.last_query_plan = {
            **(
                plan.model_dump(mode="json")
                if plan is not None
                else {
                    "original_query": query,
                    "lexical_query": query,
                    "status": "not_needed",
                    "reason": "focused_agent_query_no_extra_rewrite",
                    "elapsed_ms": 0.0,
                    "cache_hit": False,
                }
            ),
            "dense_query": query if self._dense is not None else None,
            "scoped_candidate_count": candidate_k,
            "scoped_block_count": len(allowed_block_ids),
            "identity": identity,
        }
        return SearchResult(
            document_id=lexical.document_id,
            index_id=index_id,
            method=method,
            query=query,
            hits=hits,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            branches=tuple(
                SearchBranchTrace(
                    name=name,
                    index_id=branch.index_id,
                    elapsed_ms=branch.elapsed_ms,
                    block_ids=tuple(hit.block.block_id for hit in branch.hits),
                )
                for name, branch in zip(
                    ("lexical", "dense")[: len(branch_results)], branch_results, strict=True
                )
            ),
        )
