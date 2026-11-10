"""Offline regression for the HEAD structured-output schema vocabulary."""

import json

import pytest
from pydantic import ValidationError

from page_cite_rag import full_answer, gap_agent
from page_cite_rag.agent import ReadingReport
from page_cite_rag.citations import AnswerDraft
from page_cite_rag.reading_plan import ReadingPlan, RequirementCoverage, ScopedReadingPlan
from page_cite_rag.span_citations import SpanAnswerDraft
from tests.test_gap_agent import _markdown_reading, _ScriptedModel, _section_ids

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
    models.append(gap_agent.gap_check_model(needs, ids))
    if ids:
        models.append(gap_agent.focus_selection_model(needs, ids))
        for name in ("OutlineSelection", "FollowUpSelection"):
            models.append(gap_agent._selection_model(name, needs, ids, 12))
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


def test_reader_schema_and_received_control_character_validation(tmp_path):
    workflow, result, _, first = _markdown_reading(
        tmp_path, "## A\n\nAlpha.\n\n## B\n\nBeta.\n", "Find Alpha", "Alpha"
    )
    captured = []

    class CaptureModel(_ScriptedModel):
        def generate(self, messages, response_model):
            _check_schema(response_model.model_json_schema(), patterns_allowed=False)
            if response_model.__name__ == "SectionPoints":
                captured.append(response_model)
            return super().generate(messages, response_model)

    workflow._model = CaptureModel(
        workflow, [lambda p: _section_ids(p, "A"), {"points": []}, {}], "S1"
    )
    gap_agent.run_outline_reading(workflow, result.query, "en", result, (first,))
    assert captured
    model = captured[0]
    point_schema = model.model_json_schema()["$defs"]["SectionPoint"]["properties"]
    source_id = point_schema["source_ids"]["items"].get("const")
    if source_id is None:
        source_id = point_schema["source_ids"]["items"]["enum"][0]
    payload = {"points": [{"need_id": "need-01", "point": "valid", "source_ids": [source_id]}]}
    for code in range(32):
        payload["points"][0]["point"] = f"before{chr(code)}after"
        raw = json.dumps(payload)
        if code in (9, 10, 13):
            assert model.model_validate_json(raw).points[0].point == payload["points"][0]["point"]
        else:
            with pytest.raises(ValidationError, match="forbidden control characters"):
                model.model_validate_json(raw)


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
