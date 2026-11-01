"""Behavioral checks for one source-grounded repair, not model-quality scores."""

import json

import pytest

from page_cite_rag.generation import (
    GenerationConfig,
    GenerationResult,
    GenerationStats,
    ModelError,
    check_budget,
)
from page_cite_rag.review import ReviewRunFailure, SourceReview, answer_with_review
from test_full_answer import _Model, _Retriever


def _workflow(tmp_path, reviews):
    from page_cite_rag.full_answer import FullAnswerWorkflow
    from page_cite_rag.ingest import prepare_source
    from page_cite_rag.store import SQLiteDocumentStore

    path = tmp_path / "source.txt"
    path.write_text("The result rose by 42%.\n\nOnly under condition A.\n", encoding="utf-8")
    prepared = prepare_source(path)
    store = SQLiteDocumentStore(tmp_path / "source.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    blocks = store.get_blocks(prepared.document.document_id)
    config = GenerationConfig(num_ctx=32768, num_predict=1536)

    class Model(_Model):
        def generate(self, messages, response_model):
            if issubclass(response_model, SourceReview):
                payload = json.loads(messages[1].content)
                self.seen_requests.append((response_model, payload))
                value = reviews.pop(0)
                if value == "foreign":
                    value = {
                        "verdict": "needs_revision",
                        "needs_reading": False,
                        "issues": [
                            {
                                "kind": "contradiction",
                                "description": "Wrong value.",
                                "source_block_ids": ["f" * 64],
                            }
                        ],
                    }
                return GenerationResult(
                    SourceReview.model_validate_json(json.dumps(value)),
                    GenerationStats(
                        prompt_eval_count=100,
                        eval_count=20,
                        total_duration=100,
                        load_duration=None,
                        prompt_eval_duration=40,
                        eval_duration=60,
                        done_reason="stop",
                        elapsed_ms=1.0,
                    ),
                    check_budget(messages, response_model.model_json_schema(), config),
                )
            return super().generate(messages, response_model)

    model = Model(config)
    workflow = FullAnswerWorkflow(
        store,
        prepared.document.document_id,
        _Retriever(prepared.document.document_id, blocks),
        model,
        config,
    )
    return workflow, model


PASS = {"verdict": "supported", "needs_reading": False, "issues": []}


def _issue(kind="omission", needs_reading=False):
    return {
        "verdict": "needs_revision",
        "needs_reading": needs_reading,
        "issues": [
            {"kind": kind, "description": "Check the requested condition.", "source_block_ids": []}
        ],
    }


def test_supported_draft_stops_without_tools_or_rewrite(tmp_path):
    workflow, model = _workflow(tmp_path, [PASS])
    first = workflow._store.get_blocks(workflow.document_id)[0]
    workflow._section_titles[first.ordinal] = "Results"
    run = answer_with_review(workflow, "How much and under what condition?", strategy="once")
    assert run.result.status == "answered"
    assert [stage["stage"] for stage in run.trace["stages"]] == ["draft", "check"]
    assert run.trace["agent"] is None
    assert run.trace["checker_is_quality_evaluator"] is False
    checked = next(
        payload for schema, payload in model.seen_requests if issubclass(schema, SourceReview)
    )
    assert all(
        "quote" not in anchor for claim in checked["draft"]["claims"] for anchor in claim["sources"]
    )
    assert all("text" in source for source in checked["evidence"])
    indexed = {source["source_id"]: source for source in checked["evidence"]}
    assert len(indexed) == len(checked["evidence"])
    for item, claim in zip(checked["draft"]["claims"], run.result.claims, strict=True):
        for anchor, citation in zip(item["sources"], claim.citations, strict=True):
            selected = indexed[anchor["source_id"]]
            assert selected["text"] == citation.quote
            assert selected["block_id"] == citation.block_id
    assert (
        next(source for source in indexed.values() if source["block_id"] == first.block_id)[
            "section"
        ]
        == "Results"
    )
    assert checked["draft"]["limitations"] == []
    assert checked["other_context"] == []

    from page_cite_rag.review import _check_source_answer
    from page_cite_rag.workflows import WorkflowError

    original = run.result.answer_claims[0]
    wrong = original.citations[0].model_copy(update={"quote": "Invented."})
    corrupt = run.result.model_copy(
        update={"answer_claims": (original.model_copy(update={"citations": (wrong,)}),)}
    )
    with pytest.raises(WorkflowError, match="review citation differs"):
        _check_source_answer(
            workflow,
            run.result.question,
            run.result.language,
            corrupt,
            workflow._store.get_blocks(workflow.document_id),
            (),
        )


def test_repair_sees_draft_and_feedback_without_changing_next_request(tmp_path):
    workflow, model = _workflow(tmp_path, [_issue(), PASS])
    run = answer_with_review(workflow, "How much and under what condition?", strategy="once")
    assert [stage["stage"] for stage in run.trace["stages"]] == [
        "draft",
        "check",
        "repair",
        "recheck",
    ]
    repair = next(
        payload for schema, payload in model.seen_requests if "revision_feedback" in payload
    )
    assert (
        repair["revision_feedback"]["draft"]["claims"][0]["text"]
        == (run.trace["initial_answer"]["answer_claims"][0]["text"])
    )
    assert repair["revision_feedback"]["draft"]["claims"][0]["source_block_ids"]
    assert repair["revision_feedback"]["source_review"]["issues"][0]["kind"] == "omission"
    workflow.ask("Another question")
    assert "revision_feedback" not in model.seen


def test_revision_rejects_original_evidence_loss_before_generation(tmp_path):
    from page_cite_rag.generation import ContextBudgetError

    workflow, model = _workflow(tmp_path, [])
    question = "How much and under what condition?"
    answer = workflow.ask(question, language="en")
    call_count = len(model.seen_requests)

    def limited_budget(messages, schema):
        payload = json.loads(messages[1].content)
        if "revision_feedback" in payload and len(payload["evidence"]) > 1:
            raise ContextBudgetError("controlled backend input limit")
        return check_budget(messages, schema, model.config)

    model.prepare_budget = limited_budget
    with pytest.raises(ContextBudgetError, match="revision would omit original evidence"):
        workflow.revise_from_context(
            question,
            workflow._store.get_blocks(workflow.document_id),
            draft=answer,
            feedback={"issues": ["Preserve the requested condition."]},
            language="en",
        )
    assert len(model.seen_requests) == call_count


@pytest.mark.parametrize(
    "kind,status", [("contradiction", "insufficient_evidence"), ("omission", "partial")]
)
def test_unresolved_contradiction_is_withheld_not_relabelled_partial(tmp_path, kind, status):
    workflow, _ = _workflow(tmp_path, [_issue(kind), _issue(kind)])
    run = answer_with_review(workflow, "How much?", strategy="once")
    assert run.result.status == status
    if kind == "contradiction":
        assert not run.result.claims
        assert run.trace["proposed_answer"]["answer"]


def test_review_cannot_invent_source_or_silently_pass_on_failure(tmp_path):
    workflow, _ = _workflow(tmp_path, ["foreign"])
    with pytest.raises(ReviewRunFailure) as error:
        answer_with_review(workflow, "How much?", strategy="once")
    assert isinstance(error.value, ModelError)
    assert error.value.error_type == "WorkflowError"


def test_deterministic_supplement_is_only_used_for_missing_evidence(tmp_path):
    review = _issue(needs_reading=True)
    review["issues"].append(
        {"kind": "omission", "description": "A second gap.", "source_block_ids": []}
    )
    workflow, model = _workflow(tmp_path, [review, PASS])
    run = answer_with_review(workflow, "How much?", strategy="once")
    assert run.trace["supplement"]["query"] == "How much?\nCheck the requested condition."
    repair = next(payload for _, payload in model.seen_requests if "revision_feedback" in payload)
    assert len(repair["revision_feedback"]["source_review"]["issues"]) == 2
    assert run.result.status == "answered"


def test_review_schema_limits_sources_and_review_output_before_decoding():
    from pydantic import ValidationError

    from page_cite_rag.review import _provided_review_model

    allowed = ("a" * 64, "b" * 64)
    model = _provided_review_model(allowed)
    schema = model.model_json_schema()
    issue_schema = schema["$defs"]["ProvidedSourceIssue"]["properties"]
    assert issue_schema["source_block_ids"]["items"]["enum"] == list(allowed)
    assert issue_schema["description"]["maxLength"] == 400
    assert schema["properties"]["issues"]["maxItems"] == 3
    payload = _issue()
    payload["issues"][0]["source_block_ids"] = ["f" * 64]
    with pytest.raises(ValidationError, match="literal_error"):
        model.model_validate_json(json.dumps(payload))
    payload["issues"][0]["source_block_ids"] = [allowed[0]]
    assert isinstance(model.model_validate_json(json.dumps(payload)), SourceReview)
