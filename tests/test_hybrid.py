from __future__ import annotations

import hashlib

import pytest

from page_cite_rag.hybrid import (
    HybridRetriever,
    fuse_routed_hybrid_results,
    reciprocal_rank_fusion,
)
from page_cite_rag.models import Block
from page_cite_rag.retrieval import (
    RetrievalError,
    SearchContextGroup,
    SearchHit,
    SearchResult,
)


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


def test_routed_hybrid_fuses_two_lexical_queries_before_the_dense_route():
    original_lexical = result([1, 2], query="original")
    rewritten_lexical = result([3, 2], query="rewritten")
    dense = result([2, 4], query="rewritten").model_copy(update={"method": "dense"})

    combined = fuse_routed_hybrid_results(
        "original", [original_lexical, rewritten_lexical], dense, lexical_top_k=3, top_k=6
    )

    assert combined.method == "hybrid"
    assert combined.query == "original"
    assert {hit.block.ordinal for hit in combined.hits} == {1, 2, 3, 4}
    assert [branch.name for branch in combined.branches] == ["lexical", "dense"]
    assert [len(branch.block_ids) for branch in combined.branches] == [3, 2]
    assert len(combined.hits) <= 6


def test_routed_hybrid_rejects_wrong_branch_type_or_scope():
    lexical = result([1], query="q")
    dense = result([2], document_id="2" * 64, query="q").model_copy(update={"method": "dense"})
    with pytest.raises(RetrievalError, match="document scope"):
        fuse_routed_hybrid_results("q", [lexical], dense, lexical_top_k=2, top_k=4)

    with pytest.raises(RetrievalError, match="BM25 and dense"):
        fuse_routed_hybrid_results("q", [lexical], result([2], query="q"), lexical_top_k=2, top_k=4)


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
    combined = retriever.search("q", top_k=1)
    assert combined.method == "hybrid"
    assert [branch.name for branch in combined.branches] == ["lexical", "dense"]
    assert [len(branch.block_ids) for branch in combined.branches] == [0, 2]
    assert lexical.calls == dense.calls == [("q", 20)]
    with pytest.raises(ValueError):
        retriever.search("q", top_k=41)


def test_reranker_uses_full_union_and_preserves_source_blocks():
    from page_cite_rag.hybrid import rerank_search_result

    source = result([1, 2, 3])

    class Ranker:
        def rerank(self, query, documents):
            assert query == source.query
            assert documents == [hit.block.text for hit in source.hits]
            return [(2, 0.9), (0, 0.1), (1, 0.1)]

    reranked = rerank_search_result(source, Ranker(), top_k=2)
    assert [hit.block for hit in reranked.hits] == [source.hits[2].block, source.hits[0].block]
    assert [hit.rank for hit in reranked.hits] == [1, 2]
    assert reranked.document_id == source.document_id
    assert source.hits[0].block.ordinal == 1


def test_reranker_missing_candidate_is_rejected():
    from page_cite_rag.hybrid import rerank_search_result

    class Ranker:
        def rerank(self, query, documents):
            return [(0, 1.0), (0, 0.5)]

    with pytest.raises(RetrievalError, match="invalid candidate ranking"):
        rerank_search_result(result([1, 2]), Ranker(), top_k=1)


def test_reranker_keeps_all_source_blocks_for_selected_context_groups():
    from page_cite_rag.hybrid import rerank_search_result

    original = result([1, 2, 3, 4])
    groups = (
        SearchContextGroup(
            chunk_index=10,
            text="first complete chunk",
            source_refs=("#/texts/0", "#/texts/1"),
            source_block_ids=(original.hits[0].block.block_id, original.hits[1].block.block_id),
        ),
        SearchContextGroup(
            chunk_index=11,
            text="second complete chunk",
            source_refs=("#/texts/2", "#/texts/3"),
            source_block_ids=(original.hits[2].block.block_id, original.hits[3].block.block_id),
        ),
    )
    original = original.model_copy(update={"context_groups": groups})

    class Ranker:
        def rerank(self, _query, documents):
            assert documents == [group.text for group in groups]
            return [(1, 0.9), (0, 0.2)]

    reranked = rerank_search_result(original, Ranker(), top_k=1)

    assert reranked.context_groups == (groups[1],)
    assert {hit.block.block_id for hit in reranked.hits} == set(groups[1].source_block_ids)
    assert [hit.rank for hit in reranked.hits] == [1, 2]
