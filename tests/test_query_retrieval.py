from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import pytest

from page_cite_rag.generation import GenerationResult, GenerationStats, ModelError, check_budget
from page_cite_rag.hybrid import HybridRetriever
from page_cite_rag.models import Block
from page_cite_rag.query import QueryRewriter
from page_cite_rag.query_retrieval import RewritingRetriever
from page_cite_rag.retrieval import RetrievalError, SearchHit, SearchResult

DOCUMENT = "1" * 64


@dataclass
class Budget:
    num_ctx: int = 8192
    num_predict: int = 512
    template_reserve: int = 256


def stats():
    return GenerationStats(
        prompt_eval_count=50,
        eval_count=10,
        total_duration=100,
        load_duration=None,
        prompt_eval_duration=30,
        eval_duration=70,
        done_reason="stop",
        elapsed_ms=1.0,
    )


class Model:
    identity = {"backend": "self-authored-test", "revision": "1"}

    def __init__(self, value="Explain BM25"):
        self.value = value
        self.calls = []

    def generate(self, messages, response_model):
        self.calls.append((messages, response_model))
        if isinstance(self.value, Exception):
            raise self.value
        return GenerationResult(
            response_model(query=self.value),
            stats(),
            check_budget(messages, response_model.model_json_schema(), Budget()),
        )

    def close(self):
        raise AssertionError("the wrapper must not close caller-owned models")


class Branch:
    def __init__(self, method, ordinals=(1, 2), *, scores=None, document=DOCUMENT, mutation=None):
        self.method = method
        self.ordinals = ordinals
        self.scores = scores
        self.document = document
        self.mutation = mutation
        self.calls = []
        self.results = []

    def search(self, query, *, top_k):
        self.calls.append((query, top_k))
        result = SearchResult(
            document_id=self.document,
            index_id=f"self-authored-{self.method}",
            method=self.method,
            query=query,
            elapsed_ms=2.0 if self.method == "bm25" else 3.0,
            hits=tuple(
                SearchHit(
                    rank=rank,
                    score=(self.scores[rank - 1] if self.scores else float(10 - rank)),
                    block=Block(
                        document_id=self.document,
                        block_id=sha256(f"{self.document}-{ordinal}".encode()).hexdigest(),
                        ordinal=ordinal,
                        unit_index=1,
                        line_start=ordinal,
                        line_end=ordinal,
                        char_start=0,
                        char_end=3,
                        text="abc",
                    ),
                )
                for rank, ordinal in enumerate(self.ordinals[:top_k], 1)
            ),
        )
        if self.mutation:
            result = self.mutation(result)
        self.results.append(result)
        return result


def setup(*, dense=True, model_value="Explain BM25", candidates=4):
    model = Model(model_value)
    lexical = Branch("bm25", (1, 2), scores=(100.0, 50.0))
    semantic = Branch("dense", (2, 1), scores=(0.9, -0.8)) if dense else None
    wrapper = RewritingRetriever(
        lexical, QueryRewriter(model, Budget()), semantic, candidates=candidates
    )
    return wrapper, lexical, semantic, model


def test_dense_uses_original_lexical_uses_rewrite_and_public_result_retains_original():
    wrapper, lexical, dense, model = setup()
    question = "說明 BM25"
    result = wrapper.search(question, top_k=2)
    assert lexical.calls == [("Explain BM25", 4)]
    assert dense.calls == [(question, 4)]
    assert result.query == question and result.document_id == DOCUMENT
    assert result.method == "hybrid"
    assert [hit.block.ordinal for hit in result.hits] == [1, 2]
    assert result.hits[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert len(model.calls) == 1
    # Returned branch records remain unmodified; query normalization is only a copy.
    assert lexical.results[0].query == "Explain BM25"
    assert dense.results[0].query == question


def test_trace_records_both_branch_costs_rewrite_cost_and_actual_cache_behavior():
    wrapper, _lexical, _dense, model = setup()
    wrapper.search("說明 BM25", top_k=2)
    first = wrapper.last_query_plan
    assert first["original_query"] == "說明 BM25"
    assert first["lexical_query"] == "Explain BM25"
    assert first["dense_query"] == "說明 BM25"
    assert first["lexical_elapsed_ms"] == 2.0
    assert first["dense_elapsed_ms"] == 3.0
    assert first["elapsed_ms"] >= 0
    assert first["stats"]["eval_count"] == 10
    assert first["cache_hit"] is False
    assert first["meaning_requires_human_review"] is True
    wrapper.search("說明 BM25", top_k=2)
    second = wrapper.last_query_plan
    assert second["cache_hit"] is True and second["stats"] is None
    assert second["elapsed_ms"] >= 0 and len(model.calls) == 1
    assert first["stats"]["eval_count"] == 10


def test_index_identity_does_not_depend_on_query_or_cache_statistics():
    wrapper, _lexical, _dense, model = setup()
    first = wrapper.search("說明 BM25", top_k=2)
    model.value = "Compare BM25"
    different = wrapper.search("比較 BM25", top_k=1)
    cached = wrapper.search("說明 BM25", top_k=2)
    assert first.index_id == different.index_id == cached.index_id
    assert len(first.index_id) == 64


def test_lexical_only_keeps_bm25_scores_and_reports_no_dense_cost():
    wrapper, lexical, _dense, _model = setup(dense=False)
    result = wrapper.search("說明 BM25", top_k=1)
    assert lexical.calls == [("Explain BM25", 1)]
    assert result.method == "bm25" and result.query == "說明 BM25"
    assert result.hits[0].score == 100.0
    assert wrapper.last_query_plan["dense_query"] is None
    assert wrapper.last_query_plan["dense_elapsed_ms"] is None


def test_english_query_is_passed_through_without_model_call():
    wrapper, lexical, dense, model = setup()
    wrapper.search("Explain BM25", top_k=2)
    assert lexical.calls == dense.calls == [("Explain BM25", 4)]
    assert not model.calls
    assert wrapper.last_query_plan["status"] == "not_needed"


def test_scoped_rewrite_filters_branches_before_reciprocal_rank_fusion():
    wrapper, lexical, dense, model = setup()
    allowed = sha256(f"{DOCUMENT}-2".encode()).hexdigest()

    result = wrapper.search_scoped(
        "說明 BM25",
        allowed_block_ids={allowed},
        candidate_k=2,
        top_k=1,
    )

    assert lexical.calls == [("Explain BM25", 2)]
    assert dense.calls == [("說明 BM25", 2)]
    assert [hit.block.block_id for hit in result.hits] == [allowed]
    assert result.hits[0].score == pytest.approx(2 / 61)
    assert result.method == "hybrid"
    assert wrapper.last_query_plan["scoped_block_count"] == 1
    assert len(model.calls) == 1


def test_degraded_translation_explicitly_falls_back_to_original_lexical_query():
    wrapper, lexical, dense, model = setup(model_value=ModelError("unavailable"))
    question = "說明 BM25"
    result = wrapper.search(question, top_k=2)
    assert lexical.calls == dense.calls == [(question, 4)]
    assert result.query == question
    assert wrapper.last_query_plan["status"] == "degraded"
    assert wrapper.last_query_plan["reason"] == "generation_or_schema_failure"
    assert wrapper.last_query_plan["stats"] is None
    assert len(model.calls) == 1


def test_existing_hybrid_baseline_queries_and_scores_are_unchanged():
    lexical, dense = Branch("bm25", (1, 2)), Branch("dense", (2, 1))
    baseline = HybridRetriever(lexical, dense, candidates=4)
    first = baseline.search("說明 BM25", top_k=2)
    wrapper = RewritingRetriever(lexical, QueryRewriter(Model(), Budget()), dense, candidates=4)
    wrapper.search("說明 BM25", top_k=2)
    second = baseline.search("說明 BM25", top_k=2)
    assert first.hits == second.hits and first.index_id == second.index_id
    assert lexical.calls == [("說明 BM25", 4), ("Explain BM25", 4), ("說明 BM25", 4)]
    assert dense.calls == [("說明 BM25", 4)] * 3


@pytest.mark.parametrize("question", ["", "  ", None, 42])
def test_invalid_query_fails_before_model_or_retrieval_calls(question):
    wrapper, lexical, dense, model = setup()
    with pytest.raises(ValueError):
        wrapper.search(question, top_k=1)
    assert not lexical.calls and not dense.calls and not model.calls
    assert wrapper.last_query_plan is None


@pytest.mark.parametrize("top_k", [0, 5, True, 1.5])
def test_invalid_top_k_fails_before_model_or_retrieval_calls(top_k):
    wrapper, lexical, dense, model = setup()
    with pytest.raises(ValueError):
        wrapper.search("說明 BM25", top_k=top_k)
    assert not lexical.calls and not dense.calls and not model.calls
    assert wrapper.last_query_plan is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"candidates": 0},
        {"candidates": 101},
        {"candidates": True},
        {"constant": 0},
        {"constant": 1001},
        {"constant": True},
    ],
)
def test_constructor_budgets_are_strict_and_bounded(kwargs):
    with pytest.raises(ValueError):
        RewritingRetriever(Branch("bm25"), QueryRewriter(Model(), Budget()), **kwargs)


def duplicated(result):
    first = result.hits[0]
    return result.model_copy(update={"hits": (first, first.model_copy(update={"rank": 2}))})


def wrong_scope(result):
    first = result.hits[0]
    return result.model_copy(
        update={
            "hits": (
                first.model_copy(
                    update={"block": first.block.model_copy(update={"document_id": "2" * 64})}
                ),
            )
        }
    )


def wrong_rank(result):
    return result.model_copy(update={"hits": (result.hits[0].model_copy(update={"rank": 3}),)})


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize(
    "mutation",
    [
        duplicated,
        wrong_scope,
        wrong_rank,
        lambda result: result.model_copy(update={"query": "wrong query"}),
        lambda result: result.model_copy(update={"index_id": ""}),
        lambda result: result.model_copy(update={"method": "dense"}),
        lambda result: result.model_copy(update={"elapsed_ms": float("nan")}),
        lambda result: result.model_copy(update={"elapsed_ms": True}),
        lambda result: result.model_copy(
            update={"hits": (result.hits[0].model_copy(update={"score": float("inf")}),)}
        ),
        lambda result: result.model_copy(
            update={"hits": (result.hits[0].model_copy(update={"score": True}),)}
        ),
        lambda result: result.model_copy(
            update={"hits": (result.hits[0].model_copy(update={"rank": True}),)}
        ),
    ],
)
def test_invalid_lexical_result_is_not_accepted_or_fused(mutation, dense):
    wrapper, lexical, semantic, _model = setup(dense=dense)
    lexical.mutation = mutation
    with pytest.raises(RetrievalError):
        wrapper.search("說明 BM25", top_k=2)
    assert wrapper.last_query_plan is None
    if semantic:
        assert not semantic.calls


def test_different_document_branches_cannot_be_fused():
    wrapper, _lexical, dense, _model = setup()
    dense.document = "2" * 64
    with pytest.raises(RetrievalError):
        wrapper.search("說明 BM25", top_k=2)
    assert wrapper.last_query_plan is None


def test_dense_query_must_be_original_not_lexical_rewrite():
    wrapper, _lexical, dense, _model = setup()
    dense.mutation = lambda result: result.model_copy(update={"query": "Explain BM25"})
    with pytest.raises(RetrievalError):
        wrapper.search("說明 BM25", top_k=2)
    assert wrapper.last_query_plan is None


def test_dense_branch_cannot_report_a_different_retrieval_method():
    wrapper, _lexical, dense, _model = setup()
    dense.mutation = lambda result: result.model_copy(update={"method": "bm25"})
    with pytest.raises(RetrievalError):
        wrapper.search("說明 BM25", top_k=2)
    assert wrapper.last_query_plan is None


def test_overbudget_branch_results_are_rejected():
    wrapper, lexical, _dense, _model = setup(dense=False)
    lexical.mutation = lambda result: result.model_copy(
        update={"hits": (result.hits[0], result.hits[0].model_copy(update={"rank": 2}))}
    )
    with pytest.raises(RetrievalError):
        wrapper.search("說明 BM25", top_k=1)


def test_partial_failure_clears_previous_successful_trace():
    wrapper, _lexical, dense, _model = setup()
    wrapper.search("說明 BM25", top_k=2)
    assert wrapper.last_query_plan is not None
    dense.mutation = lambda result: result.model_copy(update={"query": "wrong query"})
    with pytest.raises(RetrievalError):
        wrapper.search("比較 BM25", top_k=2)
    assert wrapper.last_query_plan is None


def test_disagreeing_source_text_for_same_block_id_is_rejected():
    wrapper, _lexical, dense, _model = setup()
    dense.mutation = lambda result: result.model_copy(
        update={
            "hits": (
                result.hits[0].model_copy(
                    update={"block": result.hits[0].block.model_copy(update={"text": "xyz"})}
                ),
            )
        }
    )
    with pytest.raises(RetrievalError):
        wrapper.search("說明 BM25", top_k=2)
    assert wrapper.last_query_plan is None
