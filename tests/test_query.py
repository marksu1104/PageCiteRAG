from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
from pydantic import ValidationError

from page_cite_rag.generation import GenerationResult, GenerationStats, ModelError, check_budget
from page_cite_rag.query import EnglishQuery, QueryRewriteConfig, QueryRewriter


@dataclass
class Budget:
    num_ctx: int = 8192
    num_predict: int = 512
    template_reserve: int = 256


STATS = GenerationStats(
    prompt_eval_count=50,
    eval_count=10,
    total_duration=100,
    load_duration=None,
    prompt_eval_duration=30,
    eval_duration=70,
    done_reason="stop",
    elapsed_ms=1.0,
)


class FakeModel:
    def __init__(self, query="Why does BM25 fail when recall < 0.5%?"):
        self.identity = {"backend": "fake", "revision": "test-1"}
        self.query = query
        self.calls = []
        self.closed = False

    def generate(self, messages, response_model):
        self.calls.append((messages, response_model))
        if isinstance(self.query, Exception):
            raise self.query
        value = response_model(query=self.query)
        return GenerationResult(
            value, STATS, check_budget(messages, response_model.model_json_schema(), Budget())
        )

    def close(self):
        self.closed = True


def test_chinese_question_rewrites_only_lexical_and_records_real_cost():
    model = FakeModel()
    rewriter = QueryRewriter(model, Budget())
    original = "為什麼 BM25 在 recall < 0.5% 時失效？"
    result = rewriter.rewrite(original)
    assert result.original_query == original
    assert result.lexical_query == model.query
    assert result.status == "rewritten"
    assert result.stats == STATS
    assert result.meaning_requires_human_review
    assert result.elapsed_ms >= 0
    assert not result.cache_hit
    assert len(model.calls) == 1
    assert model.calls[0][1] is EnglishQuery
    assert "negation" in model.calls[0][0][0].content
    assert not model.closed


def test_english_passthrough_never_calls_or_inspects_model():
    class Untouchable:
        @property
        def identity(self):
            raise AssertionError("English passthrough must not inspect model")

        def generate(self, *_args):
            raise AssertionError("English passthrough must not generate")

    result = QueryRewriter(Untouchable(), Budget()).rewrite("Why does BM25 fail?")
    assert result.status == "not_needed"
    assert result.lexical_query == result.original_query
    assert result.stats is None and result.model_identity_sha256 is None
    assert not result.meaning_requires_human_review


@pytest.mark.parametrize(
    ("original", "candidate"),
    [
        ("BM25 是否在 n > 64 時有效？", "Is BM25 effective when n > 65?"),
        ("BM25 是否在 n > 64 時有效？", "Is BM25 effective when n >= 64?"),
        ("模型 ID abc-12 與 F1 的結果？", "Results of model ID abc-13 and F1?"),
        ("差異是否為 -3.25%？", "Is the difference 3.25%?"),
        ("差異是否為 +3.25%？", "Is the difference 3.25%?"),
        ("BM25 與 BM25 的不同？", "How is BM25 different?"),
        ("BM25 的效果？", "How effective is BM250?"),
        ("Qwen 的效果？", "How effective is qwen?"),
        ("門檻是 .5 與 1e-3？", "Are thresholds 0.5 and 0.001?"),
    ],
)
def test_dropped_or_changed_literals_degrade_without_silent_repair(original, candidate):
    model = FakeModel(candidate)
    result = QueryRewriter(model, Budget()).rewrite(original)
    assert result.status == "degraded"
    assert result.reason == "required_literal_missing"
    assert result.lexical_query == original
    assert result.stats == STATS
    assert len(model.calls) == 1


@pytest.mark.parametrize("candidate", ["", "只有中文", "123", "English\nquery", "  "])
def test_nonenglish_empty_and_control_outputs_degrade(candidate):
    result = QueryRewriter(FakeModel(candidate), Budget()).rewrite("說明方法")
    assert result.status == "degraded"
    assert result.lexical_query == "說明方法"


def test_english_query_output_limit_is_bounded():
    result = QueryRewriter(
        FakeModel("An excessively long query"), Budget(), QueryRewriteConfig(max_query_chars=10)
    ).rewrite("說明方法")
    assert result.reason == "invalid_english_query"


def test_exact_quoted_english_phrase_and_numbers_are_preserved():
    original = '「結果」中的 "not significant" 是否僅在 n > 64 才成立？'
    model = FakeModel('Does "not significant" only hold when n > 64?')
    result = QueryRewriter(model, Budget()).rewrite(original)
    assert result.status == "rewritten"
    assert result.original_query == original
    assert result.lexical_query == model.query
    assert result.meaning_requires_human_review


@pytest.mark.parametrize(
    "candidate",
    [
        'Does "significant not" only hold when n > 64?',
        "Does not significant only hold when n > 64?",
    ],
)
def test_same_words_are_not_proof_of_preserved_quoted_phrase(candidate):
    original = '"not significant" 是否僅在 n > 64 才成立？'
    result = QueryRewriter(FakeModel(candidate), Budget()).rewrite(original)
    assert result.reason == "required_literal_missing"
    assert result.lexical_query == original


def test_numeric_facts_cannot_be_added_to_query():
    result = QueryRewriter(FakeModel("Explain the method with 99 samples"), Budget()).rewrite(
        "說明方法"
    )
    assert result.reason == "numeric_or_comparison_literal_added"
    assert result.status == "degraded"


def test_model_failure_is_safe_explicit_and_not_retried():
    model = FakeModel(ModelError("private server error with question contents"))
    result = QueryRewriter(model, Budget()).rewrite("說明方法")
    assert result.status == "degraded"
    assert result.reason == "generation_or_schema_failure"
    assert "private" not in result.reason
    assert result.stats is None
    assert len(model.calls) == 1


def test_budget_rejects_before_any_model_generation():
    model = FakeModel()
    result = QueryRewriter(model, Budget(num_ctx=1000)).rewrite("請說明方法")
    assert result.reason == "context_budget_exceeded"
    assert result.stats is None and not model.calls


def test_character_limit_rejects_before_model_generation():
    model = FakeModel()
    result = QueryRewriter(model, Budget(), QueryRewriteConfig(max_question_chars=3)).rewrite(
        "請說明方法"
    )
    assert result.reason == "question_character_limit"
    assert not model.calls


def test_cache_hit_has_no_new_stats_and_changed_identity_invalidates():
    model = FakeModel("Explain the method")
    rewriter = QueryRewriter(model, Budget())
    first = rewriter.rewrite("說明方法")
    cached = rewriter.rewrite("說明方法")
    assert first.stats == STATS and cached.stats is None
    assert cached.cache_hit and cached.lexical_query == first.lexical_query
    assert cached.meaning_requires_human_review
    assert len(model.calls) == 1
    model.identity["revision"] = "test-2"
    changed = rewriter.rewrite("說明方法")
    assert changed.model_identity_sha256 != first.model_identity_sha256
    assert not changed.cache_hit and len(model.calls) == 2
    rewriter.clear_cache()
    assert not rewriter.rewrite("說明方法").cache_hit
    assert len(model.calls) == 3


def test_cache_budget_change_cannot_bypass_preflight():
    model = FakeModel("Explain the method")
    budget = Budget()
    rewriter = QueryRewriter(model, budget)
    assert rewriter.rewrite("說明方法").status == "rewritten"
    budget.num_ctx = 1000
    result = rewriter.rewrite("說明方法")
    assert result.reason == "context_budget_exceeded"
    assert not result.cache_hit and len(model.calls) == 1


def test_lru_eviction_and_disabled_cache_are_bounded():
    model = FakeModel("Explain the method")
    rewriter = QueryRewriter(model, Budget(), QueryRewriteConfig(cache_size=1))
    rewriter.rewrite("方法一")
    rewriter.rewrite("方法二")
    assert not rewriter.rewrite("方法一").cache_hit
    assert len(model.calls) == 3
    uncached = QueryRewriter(model, Budget(), QueryRewriteConfig(cache_size=0))
    uncached.rewrite("方法一")
    assert not uncached.rewrite("方法一").cache_hit
    assert len(model.calls) == 5


def test_failed_rewrite_is_cached_without_new_generation_or_fake_cost():
    model = FakeModel(ModelError("unavailable"))
    rewriter = QueryRewriter(model, Budget())
    assert rewriter.rewrite("說明方法").status == "degraded"
    cached = rewriter.rewrite("說明方法")
    assert cached.status == "degraded" and cached.cache_hit and cached.stats is None
    assert len(model.calls) == 1


def test_invalid_model_identity_degrades_without_generation():
    model = FakeModel()
    model.identity = {"backend": object()}
    result = QueryRewriter(model, Budget()).rewrite("說明方法")
    assert result.reason == "model_identity_or_budget_invalid"
    assert not model.calls


@pytest.mark.parametrize("question", ["", "  ", None, 123])
def test_empty_and_nonstring_questions_are_rejected(question):
    with pytest.raises(ValueError, match="nonempty"):
        QueryRewriter(FakeModel(), Budget()).rewrite(question)


@pytest.mark.parametrize(
    "config",
    [
        {"target_language": "zh"},
        {"cache_size": 129},
        {"max_question_chars": 2001},
        {"max_query_chars": 2001},
        {"cache_size": True},
    ],
)
def test_configuration_bounds_and_english_only_target(config):
    with pytest.raises(ValidationError):
        QueryRewriteConfig(**config)


def test_identity_records_version_configuration_and_no_semantic_claim():
    identity = QueryRewriter(FakeModel(), Budget()).identity
    assert identity["config"]["target_language"] == "en"
    assert identity["version"].endswith("/v2")
    assert len(identity["model_identity_sha256"]) == 64
    assert identity["meaning_verified"] is False


def test_translation_prompt_and_schema_require_english_without_answer_evidence():
    model = FakeModel("What limits the sample size?")
    QueryRewriter(model, Budget()).rewrite("樣本數有哪些限制？")
    messages, schema = model.calls[0]
    assert messages[-1].role == "user"
    assert '"target_language": "English"' in messages[-1].content
    assert "English-language" in schema.model_json_schema()["properties"]["query"]["description"]
    assert len(messages) == 4
    assert messages[2].role == "assistant"
    assert "evidence" not in messages[-1].content


class MalformedModel(FakeModel):
    def __init__(self, *, value=None, stats=None, returned_budget=None):
        super().__init__()
        self.value = EnglishQuery(query="Explain the method") if value is None else value
        self.stats = STATS if stats is None else stats
        self.returned_budget = returned_budget

    def generate(self, messages, response_model):
        self.calls.append((messages, response_model))
        budget = check_budget(messages, response_model.model_json_schema(), Budget())
        if self.returned_budget:
            budget = self.returned_budget(budget)
        return GenerationResult(self.value, self.stats, budget)


@pytest.mark.parametrize(
    "value",
    [
        EnglishQuery.model_construct(query=123),
        EnglishQuery.model_construct(query="x" * 2001),
    ],
)
def test_constructed_invalid_values_are_revalidated_without_warning_or_exception(value):
    import warnings

    with warnings.catch_warnings(record=True) as recorded:
        result = QueryRewriter(MalformedModel(value=value), Budget()).rewrite("說明方法")
    assert result.status == "degraded"
    assert result.reason == "generation_or_schema_failure"
    assert not recorded


def test_output_dictionary_cannot_add_unexpected_schema_fields():
    model = MalformedModel(value={"query": "Explain the method", "unexpected": "field"})
    result = QueryRewriter(model, Budget()).rewrite("說明方法")
    assert result.reason == "generation_or_schema_failure"


@pytest.mark.parametrize(
    "stats",
    [
        STATS.model_copy(update={"done_reason": "length"}),
        STATS.model_copy(update={"eval_count": -1}),
        STATS.model_copy(update={"elapsed_ms": float("nan")}),
        STATS.model_copy(update={"prompt_eval_count": True}),
    ],
)
def test_constructed_invalid_stats_are_rejected_not_reported_as_real_cost(stats):
    result = QueryRewriter(MalformedModel(stats=stats), Budget()).rewrite("說明方法")
    assert result.status == "degraded"
    assert result.reason == "generation_or_schema_failure"
    assert result.stats is None


@pytest.mark.parametrize(
    "stats",
    [
        STATS.model_copy(update={"eval_count": 513}),
        STATS.model_copy(update={"prompt_eval_count": 8192}),
    ],
)
def test_actual_valid_stats_outside_budget_are_reported_but_answer_is_not_accepted(stats):
    result = QueryRewriter(MalformedModel(stats=stats), Budget()).rewrite("說明方法")
    assert result.status == "degraded"
    assert result.reason == "generation_budget_mismatch"
    assert result.stats == stats


def test_returned_budget_must_exactly_match_preflight():
    model = MalformedModel(
        returned_budget=lambda budget: replace(
            budget, input_upper_estimate=budget.input_upper_estimate + 1
        )
    )
    result = QueryRewriter(model, Budget()).rewrite("說明方法")
    assert result.status == "degraded"
    assert result.reason == "generation_budget_mismatch"
    assert result.stats == STATS


def test_malformed_generated_object_degrades_safely_without_retry():
    class MissingAttributes(FakeModel):
        def generate(self, messages, response_model):
            self.calls.append((messages, response_model))
            return object()

    model = MissingAttributes()
    result = QueryRewriter(model, Budget()).rewrite("說明方法")
    assert result.status == "degraded"
    assert result.reason == "generation_or_schema_failure"
    assert result.stats is None and len(model.calls) == 1


@pytest.mark.parametrize(
    "budget", [Budget(num_ctx=True), Budget(num_predict=-1), Budget(template_reserve=-1)]
)
def test_invalid_budget_types_and_ranges_degrade_before_generation(budget):
    model = FakeModel()
    result = QueryRewriter(model, budget).rewrite("說明方法")
    assert result.reason == "model_identity_or_budget_invalid"
    assert not model.calls
