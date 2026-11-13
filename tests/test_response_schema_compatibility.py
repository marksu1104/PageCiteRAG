"""Offline regression for the HEAD structured-output schema vocabulary."""

import json

import pytest
from pydantic import ValidationError

from page_cite_rag import full_answer
from page_cite_rag.agent import ReadingReport
from page_cite_rag.citations import AnswerDraft
from page_cite_rag.reading_plan import ReadingPlan, RequirementCoverage, ScopedReadingPlan
from page_cite_rag.span_citations import SpanAnswerDraft

# Union of keywords from HEAD response schemas, generated with git show HEAD:<path>.
# Property/definition names and discriminator mapping entries are not schema keywords.
_HEAD_KEYWORDS = {
    "$defs",
    "$ref",
    "additionalProperties",
    "anyOf",
    "const",
    "default",
    "description",
    "discriminator",
    "enum",
    "items",
    "maxItems",
    "maxLength",
    "minItems",
    "minLength",
    "oneOf",
    "pattern",
    "properties",
    "required",
    "title",
    "type",
}


def _check_schema(schema, *, patterns_allowed=True):
    if isinstance(schema, list):
        for item in schema:
            _check_schema(item, patterns_allowed=patterns_allowed)
    elif isinstance(schema, dict):
        assert set(schema) <= _HEAD_KEYWORDS
        if "pattern" in schema:
            assert patterns_allowed
        for key, value in schema.items():
            if key in {"properties", "$defs"}:
                for name, child in value.items():
                    # HEAD only used patterns for citation identifiers in these schemas.
                    allowed = (
                        patterns_allowed
                        if key == "$defs"
                        else name
                        in {
                            "source_id",
                            "span_id",
                            "block_id",
                        }
                    )
                    _check_schema(child, patterns_allowed=allowed)
            elif key not in {"discriminator", "default", "const", "enum"}:
                _check_schema(value, patterns_allowed=patterns_allowed)


@pytest.mark.parametrize("count", [0, 1, 2, 12])
def test_all_response_schema_families_use_head_keywords(monkeypatch, count):
    ids = tuple(f"S{i + 1}" for i in range(count))
    needs = tuple(f"need-{i + 1:02d}" for i in range(count)) or ("need-01",)
    models = [ReadingPlan, ScopedReadingPlan, ReadingReport, AnswerDraft, SpanAnswerDraft]
    monkeypatch.setattr(full_answer, "_spans", lambda blocks: dict.fromkeys(ids if blocks else ()))
    for observations in ((), ({"point": "reader note"},)):
        for kind in ("general", "short_answer", "multiple_choice"):
            workflow = object.__new__(full_answer.FullAnswerWorkflow)
            workflow.answer_format = kind
            workflow._reader_observations = observations
            workflow._reading_plan = [{"need_id": nid} for nid in needs]
            models.extend(
                [
                    workflow._response_model_for((object(),)),
                    workflow._response_model_for(()),
                    workflow.agent_response_model(),
                    workflow._selection_model_for(ids),
                ]
            )
    for model in models:
        _check_schema(model.model_json_schema())


@pytest.mark.parametrize("claim_id", ["C0", "1", ""])
def test_irregular_coverage_ids_are_left_for_the_answer_to_resolve(claim_id):
    # The answer relinks or discards links it cannot resolve (coverage_absent_claim).
    item = RequirementCoverage.model_validate_json(
        json.dumps({"need_id": "need-01", "status": "answered", "claim_ids": [claim_id]})
    )
    assert item.claim_ids == (claim_id,)


def test_non_string_coverage_ids_are_rejected_after_receiving_json():
    with pytest.raises(ValidationError):
        RequirementCoverage.model_validate_json(
            json.dumps({"need_id": "need-01", "status": "answered", "claim_ids": [True]})
        )


def test_coverage_need_id_remains_validated_after_receiving_json():
    with pytest.raises(ValidationError, match="stable need ID"):
        RequirementCoverage.model_validate_json(
            '{"need_id":"invalid","status":"answered","claim_ids":["C1"]}'
        )
