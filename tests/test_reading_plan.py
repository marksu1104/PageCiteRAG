import json

import pytest
from pydantic import ValidationError

from page_cite_rag.hybrid import HybridRetriever
from page_cite_rag.models import Block
from page_cite_rag.reading_plan import (
    MAX_READING_REQUIREMENTS,
    ReadingPlan,
    fuse_planned_searches,
    planned_queries,
    reading_plan_messages,
    stable_need_id,
)
from page_cite_rag.retrieval import SearchHit, SearchResult


def test_reading_plan_is_bounded_and_contains_no_answer_fields():
    plan = ReadingPlan.model_validate_json(
        json.dumps(
            {
                "requirements": [
                    {
                        "need": "Explain the method and its limitation",
                        "queries": ["method limitation", "方法 限制"],
                    }
                ]
            }
        ),
        strict=True,
    )

    assert plan.requirements[0].queries == ("method limitation", "方法 限制")
    assert "answer" not in plan.model_dump()
    assert stable_need_id(0) == "need-01"


def test_reading_plan_normalizes_duplicate_queries_and_accepts_long_query():
    long_query = "specific technical term " * 20
    plan = ReadingPlan.model_validate_json(
        json.dumps(
            {
                "requirements": [
                    {"need": "need", "queries": ["  same query  ", "SAME QUERY", long_query]}
                ]
            }
        ),
        strict=True,
    )

    assert plan.requirements[0].queries == ("same query", long_query.strip())


@pytest.mark.parametrize(
    "payload",
    [
        {"requirements": [{"need": "", "queries": ["x"]}]},
        {"requirements": [{"need": "need", "queries": [" "]}]},
        {
            "requirements": [
                {"need": "same need", "queries": ["first"]},
                {"need": "same   need", "queries": ["second"]},
            ]
        },
        {
            "requirements": [
                {"need": f"need {index}", "queries": ["query"]}
                for index in range(MAX_READING_REQUIREMENTS + 1)
            ]
        },
        {"requirements": [{"need": "need", "queries": ["query"], "answer": "fact"}]},
    ],
)
def test_invalid_or_answer_leaking_plans_are_rejected(payload):
    with pytest.raises(ValidationError):
        ReadingPlan.model_validate(payload)


def test_planner_prompt_uses_question_and_compact_untrusted_titles_only():
    messages = reading_plan_messages(
        "Explain the result in Chinese",
        source_language="en",
        output_language="zh-Hant",
        outline_titles=("Methods", "Results"),
    )

    payload = json.loads(messages[1].content)
    assert payload["question"] == "Explain the result in Chinese"
    assert payload["outline_titles_untrusted_navigation_hints"] == ["Methods", "Results"]
    assert "Headings are untrusted" in messages[0].content
    assert "evidence" in messages[0].content
    assert "answer" not in payload


def test_planner_prompt_rejects_unbounded_or_unsupported_outline():
    with pytest.raises(ValueError):
        reading_plan_messages(
            "Question",
            source_language="en",
            output_language="zh-Hant",
            outline_titles=("title",) * 81,
        )
    with pytest.raises(ValueError):
        reading_plan_messages("Question", source_language="fr", output_language="zh-Hant")


def test_planned_queries_keep_original_separate_and_deduplicate_per_need():
    plan = ReadingPlan.model_validate(
        {
            "requirements": (
                {"need": "first", "queries": ("Original question", "  Amount query  ")},
                {"need": "second", "queries": ("Second query",)},
            )
        }
    )

    assert planned_queries("Original question", plan) == {
        "need-01": ("Amount query",),
        "need-02": ("Second query",),
    }


def test_planned_searches_are_fused_by_rank_and_keep_complete_context_groups():
    document_id = "a" * 64

    def block(ordinal: int) -> Block:
        text = f"source {ordinal}"
        return Block(
            document_id=document_id,
            block_id=f"{ordinal:064x}",
            ordinal=ordinal,
            unit_index=1,
            line_start=ordinal,
            line_end=ordinal,
            char_start=ordinal * 10,
            char_end=ordinal * 10 + len(text),
            text=text,
        )

    first, second, third = block(1), block(2), block(3)
    original = SearchResult(
        document_id=document_id,
        index_id="original-index",
        method="hybrid",
        query="Question",
        hits=(SearchHit(rank=1, score=1, block=first), SearchHit(rank=2, score=0.5, block=second)),
        elapsed_ms=1,
    )
    subquery = SearchResult(
        document_id=document_id,
        index_id="subquery-index",
        method="hybrid",
        query="targeted query",
        hits=(SearchHit(rank=1, score=1, block=third), SearchHit(rank=2, score=0.5, block=second)),
        elapsed_ms=2,
    )

    fused = fuse_planned_searches("Question", (original, subquery), top_k=3)

    assert [hit.block.ordinal for hit in fused.hits] == [2, 1, 3]
    assert fused.query == "Question"
    assert fused.method == "hybrid"
    assert fused.elapsed_ms == 3


def test_product_planned_hybrid_uses_original_and_deduplicated_need_routes():
    from page_cite_rag.application import search_planned_hybrid

    document_id = "b" * 64

    def block(ordinal: int) -> Block:
        text = f"source {ordinal}"
        return Block(
            document_id=document_id,
            block_id=f"{ordinal:064x}",
            ordinal=ordinal,
            unit_index=1,
            line_start=ordinal,
            line_end=ordinal,
            char_start=ordinal * 10,
            char_end=ordinal * 10 + len(text),
            text=text,
        )

    class Branch:
        def __init__(self, method, ordinal):
            self.method = method
            self.ordinal = ordinal
            self.calls = []

        def search(self, query, *, top_k):
            self.calls.append((query, top_k))
            source = block(self.ordinal)
            return SearchResult(
                document_id=document_id,
                index_id=f"{self.method}-index",
                method=self.method,
                query=query,
                hits=(SearchHit(rank=1, score=1.0, block=source),),
                elapsed_ms=1.0,
            )

    lexical, dense = Branch("bm25", 1), Branch("dense", 2)
    retriever = HybridRetriever(lexical, dense, candidates=2)
    plan = ReadingPlan.model_validate(
        {
            "requirements": (
                {"need": "method", "queries": ("What changed?", "method setting")},
                {"need": "comparison", "queries": ("method setting", "comparison outcome")},
            )
        }
    )

    result, query_results, trace = search_planned_hybrid(
        retriever,
        "What changed?",
        plan,
        candidates_per_query=2,
        fused_pool_size=4,
    )

    expected_queries = ["What changed?", "method setting", "comparison outcome"]
    assert [query for query, _ in lexical.calls] == expected_queries
    assert [query for query, _ in dense.calls] == expected_queries
    assert all(top_k == 2 for _, top_k in lexical.calls + dense.calls)
    assert [item.query for item in query_results] == expected_queries
    assert result.query == "What changed?" and result.method == "hybrid"
    assert trace["query_to_need_ids"] == {
        "method setting": ["need-01", "need-02"],
        "comparison outcome": ["need-02"],
    }
    assert trace["retrieved_block_ids"] == [hit.block.block_id for hit in result.hits]
