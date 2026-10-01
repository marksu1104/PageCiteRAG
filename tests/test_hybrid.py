from __future__ import annotations

import hashlib

import pytest

from doc_rag.hybrid import HybridRetriever, reciprocal_rank_fusion
from doc_rag.models import Block
from doc_rag.retrieval import RetrievalError, SearchHit, SearchResult


def result(ordinals, scores=None, document_id="1" * 64, query="q"):
    return SearchResult(
        document_id=document_id,
        index_id="synthetic",
        query=query,
        elapsed_ms=0,
        hits=tuple(
            SearchHit(
                rank=rank,
                score=(scores[rank - 1] if scores else 1),
                block=Block(
                    block_id=hashlib.sha256(f"{document_id}-{ordinal}".encode()).hexdigest(),
                    document_id=document_id,
                    unit_index=1,
                    ordinal=ordinal,
                    text="abc",
                    char_start=0,
                    char_end=3,
                    line_start=1,
                    line_end=1,
                ),
            )
            for rank, ordinal in enumerate(ordinals, 1)
        ),
    )


def test_rrf_uses_ranks_not_incompatible_scores_and_is_stable():
    a = result([1, 2], [10000, 3])
    b = result([2, 1], [0.99, -0.8])
    hits = reciprocal_rank_fusion([a, b])
    assert [h.block.ordinal for h in hits] == [1, 2]
    assert hits[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert hits == reciprocal_rank_fusion([b, a])
    assert reciprocal_rank_fusion([result([]), b]) == reciprocal_rank_fusion([b])
    assert reciprocal_rank_fusion([result([]), result([])]) == ()


def test_rrf_rejects_mixed_scope_queries_and_duplicate_hits():
    with pytest.raises(RetrievalError, match="scopes"):
        reciprocal_rank_fusion([result([1]), result([1], document_id="2" * 64)])
    with pytest.raises(RetrievalError, match="queries"):
        reciprocal_rank_fusion([result([1]), result([2], query="other")])
    with pytest.raises(RetrievalError, match="duplicate"):
        reciprocal_rank_fusion([result([1, 1])])
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([], constant=0)


def test_hybrid_limits_candidates_and_reuses_branches():
    class Branch:
        def __init__(self, ordinals):
            self.ordinals = ordinals
            self.calls = []

        def search(self, query, *, top_k):
            self.calls.append((query, top_k))
            return result(self.ordinals, query=query)

    lexical, dense = Branch([]), Branch([1, 2])
    retriever = HybridRetriever(lexical, dense, candidates=20)
    assert retriever.search("q", top_k=1).method == "hybrid"
    assert lexical.calls == dense.calls == [("q", 20)]
    with pytest.raises(ValueError):
        retriever.search("q", top_k=21)
