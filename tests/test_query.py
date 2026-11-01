from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from page_cite_rag.generation import GenerationResult, GenerationStats, ModelError, check_budget
from page_cite_rag.query import QueryRewriteConfig, QueryRewriter


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
    def __init__(self, value="retrieval, ranking"):
        self.identity = {"backend": "fake", "revision": "test-1"}
        self.value = value
        self.calls = []

    def generate(self, messages, response_model):
        self.calls.append((messages, response_model))
        if isinstance(self.value, Exception):
            raise self.value
        schema = None if response_model is None else response_model.model_json_schema()
        return GenerationResult(self.value, STATS, check_budget(messages, schema, Budget()))


def test_keyword_expansion_appends_terms_but_retains_original_verbatim():
    model = FakeModel("hybrid retrieval, ranking methods")
    question = "BM25 為什麼在 recall < 0.5% 時失效？"
    result = QueryRewriter(model, Budget(), QueryRewriteConfig.for_document_language("en")).rewrite(
        question
    )
    assert result.status == "rewritten"
    assert result.lexical_query.startswith(question + " ")
    assert result.expansion_terms == ("hybrid retrieval", "ranking methods")
    assert result.meaning_requires_human_review
    assert len(model.calls) == 1 and model.calls[0][1] is None


@pytest.mark.parametrize(
    ("language", "question", "rewritten"),
    [
        (
            "en",
            "How does model A compare with model B when accuracy is >= 90%?",
            "Compare model A and model B when accuracy is >= 90%.",
        ),
        (
            "zh-Hant",
            "方法在準確率 >= 90% 時，與基準相比有何限制？",
            "說明該方法在準確率 >= 90% 時相較基準的限制。",
        ),
    ],
)
def test_full_sentence_mode_returns_one_bounded_query_in_document_language(
    language, question, rewritten
):
    model = FakeModel(rewritten)
    result = QueryRewriter(
        model,
        Budget(),
        QueryRewriteConfig.for_document_language(language, mode="full_sentence"),
    ).rewrite(question)

    assert result.status == "rewritten"
    assert result.mode == "full_sentence"
    assert result.lexical_query == rewritten
    assert result.expansion_terms == ()
    assert result.version == "full-sentence-document-language/v1"
    assert result.prompt_id == "retrieval.full-sentence.v1"
    assert result.meaning_requires_human_review
    assert len(model.calls) == 1
    system, user = model.calls[0][0]
    assert "Do not answer the question" in system.content
    assert "Traditional Chinese" in system.content if language == "zh-Hant" else True
    assert question in user.content


@pytest.mark.parametrize(
    "rewritten",
    [
        "The accuracy is >= 91%.",
        "The accuracy is <= 90%.",
        "The accuracy is >= 90%.\nAlso include other models.",
        "   ",
    ],
)
def test_full_sentence_mode_falls_back_if_output_changes_constraints_or_is_unbounded(
    rewritten,
):
    question = "What happens at accuracy >= 90%?"
    result = QueryRewriter(
        FakeModel(rewritten),
        Budget(),
        QueryRewriteConfig(source_language="en", mode="full_sentence"),
    ).rewrite(question)

    assert result.status == "degraded"
    assert result.lexical_query == question
    assert result.expansion_terms == ()


def test_full_sentence_mode_reports_unchanged_question_without_changing_it():
    question = "Explain this method."
    result = QueryRewriter(
        FakeModel(question),
        Budget(),
        QueryRewriteConfig(source_language="en", mode="full_sentence"),
    ).rewrite(question)

    assert result.status == "not_needed"
    assert result.lexical_query == question
    assert result.reason == "full_sentence_unchanged"


def test_budget_aware_rewrite_uses_the_same_schema_as_plain_text_generation():
    class BudgetAwareFakeModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.prepared_schemas = []

        def prepare_budget(self, messages, schema):
            self.prepared_schemas.append(schema)
            return check_budget(messages, schema, Budget())

        def generate(self, messages, response_model):
            self.calls.append((messages, response_model))
            return GenerationResult(
                self.value,
                STATS,
                self.prepare_budget(messages, None),
            )

    model = BudgetAwareFakeModel()

    result = QueryRewriter(model, Budget()).rewrite("What retrieval method is used?")

    assert result.status == "rewritten"
    assert result.reason != "generation_budget_mismatch"
    assert model.prepared_schemas == [None, None]


def test_chinese_keyword_separators_keep_terms_and_limit_individual_terms():
    result = QueryRewriter(
        FakeModel("效能評估，準確率、推論速度"),
        Budget(),
        QueryRewriteConfig.for_document_language("zh-Hant"),
    ).rewrite("方法的準確率與速度如何？")
    assert result.status == "rewritten"
    assert result.expansion_terms == ("效能評估", "準確率", "推論速度")

    too_many = QueryRewriter(FakeModel("一，二，三，四，五，六，七，八，九"), Budget()).rewrite(
        "方法比較"
    )
    assert too_many.status == "degraded"
    assert too_many.reason == "keyword_output_out_of_bounds"


@pytest.mark.parametrize(
    ("document_language", "expected"),
    [("en", "English"), ("zh-Hant", "Traditional Chinese"), (None, "English")],
)
def test_querygym_prompt_is_one_call_and_targets_known_document_language(
    document_language, expected
):
    model = FakeModel()
    result = QueryRewriter(
        model, Budget(), QueryRewriteConfig.for_document_language(document_language)
    ).rewrite("What limits the method?")
    assert result.status == "rewritten"
    messages, response_model = model.calls[0]
    assert response_model is None
    assert len(messages) == 2
    assert "comma-separated keywords" in messages[0].content
    assert expected in messages[0].content
    assert "Suggest keywords to improve retrieval" in messages[1].content


@pytest.mark.parametrize(
    "value",
    ["", ", ,", "one, two, three, four, five, six, seven, eight, nine", "term\nextra"],
)
def test_invalid_or_unbounded_expansion_degrades_to_original(value):
    question = "Explain the method"
    result = QueryRewriter(FakeModel(value), Budget()).rewrite(question)
    assert result.status == "degraded"
    assert result.lexical_query == question
    assert result.expansion_terms == ()


def test_expansion_cannot_inject_new_numeric_or_comparison_claims():
    result = QueryRewriter(FakeModel("sample size 99, method"), Budget()).rewrite(
        "sample size in method"
    )
    assert result.status == "degraded"
    assert result.reason == "keyword_added_literal"
    assert result.lexical_query == "sample size in method"


def test_query_is_not_rewritten_if_it_exceeds_the_bound():
    model = FakeModel()
    result = QueryRewriter(model, Budget(), QueryRewriteConfig(max_question_chars=3)).rewrite(
        "question"
    )
    assert result.status == "degraded"
    assert result.reason == "question_character_limit"
    assert not model.calls


def test_failure_is_explicit_and_never_retried():
    model = FakeModel(ModelError("private content must not leak"))
    result = QueryRewriter(model, Budget()).rewrite("why this method?")
    assert result.status == "degraded"
    assert result.reason == "generation_failure"
    assert "private" not in result.reason
    assert len(model.calls) == 1


@pytest.mark.parametrize(
    "message",
    [
        "GPU temperature limit exceeded",
        "GPU thermal slowdown observed; own server stopped",
        "another GPU compute process is present; own server stopped",
        "GPU status unavailable; stop rather than assume it is safe",
    ],
)
def test_gpu_safety_stops_are_not_hidden_by_rewrite_fallback(message):
    model = FakeModel(ModelError(message))
    with pytest.raises(ModelError, match="GPU"):
        QueryRewriter(model, Budget()).rewrite("find the evidence")
    assert len(model.calls) == 1


def test_cache_records_cost_once_and_changes_with_model_identity():
    model = FakeModel("retrieval")
    rewriter = QueryRewriter(model, Budget())
    first = rewriter.rewrite("Explain the method")
    second = rewriter.rewrite("Explain the method")
    assert first.stats == STATS
    assert second.stats is None and second.cache_hit
    assert len(model.calls) == 1
    model.identity["revision"] = "test-2"
    third = rewriter.rewrite("Explain the method")
    assert not third.cache_hit
    assert third.model_identity_sha256 != first.model_identity_sha256
    assert len(model.calls) == 2


def test_source_language_mapping_is_conservative():
    assert QueryRewriteConfig.for_document_language("en-US").source_language == "en"
    assert QueryRewriteConfig.for_document_language("zh-Hant").source_language == "zh-Hant"
    assert QueryRewriteConfig.for_document_language("mixed").source_language == "mixed"
    assert QueryRewriteConfig.for_document_language(None).source_language == "unknown"


@pytest.mark.parametrize(
    "config",
    [
        {"source_language": "zh"},
        {"max_terms": 9},
        {"cache_size": 513},
        {"max_question_chars": True},
    ],
)
def test_configuration_rejects_invalid_values(config):
    with pytest.raises(ValidationError):
        QueryRewriteConfig(**config)
