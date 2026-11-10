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
from page_cite_rag.retrieval import SearchContextGroup, SearchHit, SearchResult


def test_reading_plan_is_bounded_and_contains_no_answer_fields():
    plan = ReadingPlan.model_validate_json(
        json.dumps(
            {
                "requirements": [
                    {
                        "need": "Explain the method and its limitation",
                        "task": "lookup",
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
                    {
                        "need": "need",
                        "task": "lookup",
                        "queries": ["  same query  ", "SAME QUERY", long_query],
                    }
                ]
            }
        ),
        strict=True,
    )

    assert plan.requirements[0].queries == ("same query", long_query.strip())


@pytest.mark.parametrize(
    "payload",
    [
        {"requirements": [{"need": "", "task": "lookup", "queries": ["x"]}]},
        {"requirements": [{"need": "need", "task": "lookup", "queries": [" "]}]},
        {
            "requirements": [
                {"need": "same need", "task": "lookup", "queries": ["first"]},
                {"need": "same   need", "task": "lookup", "queries": ["second"]},
            ]
        },
        {
            "requirements": [
                {"need": f"need {index}", "task": "lookup", "queries": ["query"]}
                for index in range(MAX_READING_REQUIREMENTS + 1)
            ]
        },
        {
            "requirements": [
                {"need": "need", "task": "lookup", "queries": ["query"], "answer": "fact"}
            ]
        },
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
                {
                    "need": "first",
                    "task": "lookup",
                    "queries": ("Original question", "  Amount query  "),
                },
                {"need": "second", "task": "lookup", "queries": ("Second query",)},
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
                {
                    "need": "method",
                    "task": "lookup",
                    "queries": ("What changed?", "method setting"),
                },
                {
                    "need": "comparison",
                    "task": "lookup",
                    "queries": ("method setting", "comparison outcome"),
                },
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


def test_fusion_cut_never_leaves_a_context_group_with_missing_source_blocks():
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

    blocks = [block(n) for n in range(1, 5)]
    group = SearchContextGroup(
        chunk_index=0,
        text="chunk text",
        source_refs=("#/texts/0",),
        source_block_ids=tuple(item.block_id for item in blocks),
    )
    result = SearchResult(
        document_id=document_id,
        index_id="chunked-index",
        method="hybrid",
        query="Question",
        hits=tuple(
            SearchHit(rank=rank, score=1 / rank, block=item) for rank, item in enumerate(blocks, 1)
        ),
        elapsed_ms=1,
        context_groups=(group,),
    )

    fused = fuse_planned_searches("Question", (result,), top_k=2)

    assert {hit.block.block_id for hit in fused.hits} == set(group.source_block_ids)
    assert [hit.rank for hit in fused.hits] == [1, 2, 3, 4]
    assert fused.context_groups == (group,)


def test_fusion_accepts_an_original_question_with_surrounding_whitespace():
    document_id = "a" * 64
    block = Block(
        document_id=document_id,
        block_id=f"{1:064x}",
        ordinal=1,
        unit_index=1,
        line_start=1,
        line_end=1,
        char_start=0,
        char_end=8,
        text="source 1",
    )
    result = SearchResult(
        document_id=document_id,
        index_id="index",
        method="hybrid",
        query="Question?",
        hits=(SearchHit(rank=1, score=1.0, block=block),),
        elapsed_ms=1,
    )

    fused = fuse_planned_searches("Question?  ", (result,), top_k=1)

    assert fused.query == "Question?  "


def test_third_language_planner_queries_are_dropped_but_the_question_stays():
    from page_cite_rag.reading_plan import is_foreign_language_query, planned_query_routes

    question = "How was Country C built into the illustrative dataset?"
    assert is_foreign_language_query("Comment sont définis l'opérateur minimum ?", question)
    assert is_foreign_language_query("Wie wird der Wert für die Debatte bestimmt", question)
    assert not is_foreign_language_query("Country C dataset construction", question)
    assert not is_foreign_language_query("C 國資料集如何建立", question)
    assert is_foreign_language_query("Quel classificateur est utilisé ?", "分類器是什麼？")
    # One accented name in an English query is not a language switch.
    assert not is_foreign_language_query("Visit to the longhouse at Brattahlíð", question)
    # A question already written in French keeps its French queries.
    assert not is_foreign_language_query("Comment sont définis les seuils", "Où est défini C ?")
    plan = ReadingPlan.model_validate_json(
        json.dumps(
            {
                "requirements": [
                    {
                        "need": "How Country C was built",
                        "task": "lookup",
                        "queries": [
                            "Country C construction",
                            "Commentaire sur la construction de Country C",
                        ],
                    }
                ]
            }
        ),
        strict=True,
    )
    queries, _routes = planned_query_routes(question, plan)
    assert queries == (question, "Country C construction")
