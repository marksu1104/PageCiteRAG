"""Synthetic scope, capacity and writer-contract behavior; no model or network calls."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from page_cite_rag.agent import _spans
from page_cite_rag.full_answer import FullAnswerDraft, FullAnswerWorkflow
from page_cite_rag.generation import (
    ContextBudgetError,
    GenerationResult,
    GenerationStats,
    PromptBudget,
)
from page_cite_rag.reading_plan import (
    ReadingNeed,
    ReadingPlan,
    ScopedReadingPlan,
    reading_plan_messages,
)
from page_cite_rag.retrieval import SearchHit, SearchResult
from tests.reading_fixtures import _Retriever


def _plan(*tasks):
    return ReadingPlan(
        requirements=tuple(
            ReadingNeed(need=f"Requested part {i}", task=task, queries=(f"part {i}",))
            for i, task in enumerate(tasks, 1)
        )
    )


def test_scope_must_follow_requested_operations_and_keep_conditions():
    need = ReadingNeed(
        need="Every completed workshop in Annex B, with its power source",
        task="inventory",
        queries=("Annex B completed workshop power",),
    )
    repaired = ScopedReadingPlan(requirements=(need,), scope="lookup")
    assert repaired.scope == "whole_document"
    assert repaired._deviations == [{"type": "scope_mismatch", "count": 1}]
    plan = ScopedReadingPlan(requirements=(need,), scope="whole_document")
    assert plan.as_prompt_context()[0] == {
        "need_id": "need-01",
        "need": need.need,
        "task": "inventory",
        "queries": ["Annex B completed workshop power"],
    }
    localized_needs = _plan("lookup").requirements
    preserved = ScopedReadingPlan(requirements=localized_needs, scope="whole_document")
    assert preserved.scope == "whole_document"
    assert preserved._deviations == []
    with pytest.raises(ValidationError):
        ReadingNeed(need="Missing operation", queries=("query",))


def test_scope_prompt_decides_from_original_question_before_decomposition():
    messages = reading_plan_messages(
        "比較文件不同部分提到的多項指標與限制。",
        source_language="en",
        output_language="zh-Hant",
        with_scope=True,
    )
    prompt = messages[0].content
    assert "complete original question before splitting" in prompt
    assert "Do not downgrade whole_document" in prompt
    assert "need.task describes each decomposed retrieval operation" in prompt


def test_fixed_writer_schema_forbids_unrequested_coverage(tmp_path):
    workflow, result, _, first = _markdown_reading(
        tmp_path, "## Results\n\nAlpha improved by 4%.\n", "How much?", "Alpha improved"
    )
    response_model = workflow._response_model_for((first,))
    schema = response_model.model_json_schema()
    coverage_fields = []

    def collect_coverage_fields(value):
        if isinstance(value, dict):
            properties = value.get("properties", {})
            if "requirement_coverage" in properties:
                coverage_fields.append(properties["requirement_coverage"])
            for child in value.values():
                collect_coverage_fields(child)
        elif isinstance(value, list):
            for child in value:
                collect_coverage_fields(child)

    collect_coverage_fields(schema)
    assert coverage_fields
    assert all(field.get("maxItems") == 0 for field in coverage_fields)

    source_id = next(iter(_spans((first,))))
    claim = {
        "claim_id": "C1",
        "text": "Alpha improved by 4%.",
        "citations": [{"source_id": source_id}],
        "support": "direct",
    }
    payload = {
        "status": "answered",
        "answer_claims": [claim],
        "detail_claims": [],
        "limitations": [],
    }
    parsed = response_model.model_validate_json(json.dumps(payload))
    assert parsed.answer_claims[0].text == claim["text"]
    assert parsed.requirement_coverage == ()
    with pytest.raises(ValidationError):
        response_model.model_validate_json(
            json.dumps(
                {
                    **payload,
                    "requirement_coverage": [
                        {"need_id": "C1", "status": "answered", "claim_ids": ["C1"]}
                    ],
                }
            )
        )

    system, user = workflow._messages("How much?", "en", "ask", (first,), None)
    user_payload = json.loads(user.content)
    assert "reading_plan" not in user_payload
    assert "coverage" not in user_payload["output_budget"]["includes"]
    assert "Do not add requirement_coverage" in system.content


def _draft(status="answered", coverage_status="answered", claim_ids=("C1",), gap=""):
    return {
        "status": status,
        "answer_claims": [
            {
                "claim_id": "C1",
                "support": "direct",
                "text": "A uses a hand pump",
                "citations": [{"source_id": "S1"}],
            }
        ],
        "detail_claims": [],
        "limitations": [gap] if gap else [],
        "requirement_coverage": [
            {"need_id": "need-01", "status": coverage_status, "claim_ids": claim_ids, "gap": gap}
        ],
    }


@pytest.mark.parametrize(
    "status,coverage,claim_ids,gap",
    [
        ("answered", "answered", ("C2",), ""),
        ("partial", "partial", ("C1",), ""),
    ],
)
def test_writer_discards_inconsistent_completion_links(status, coverage, claim_ids, gap):
    payload = _draft(status, coverage, claim_ids, gap)
    if status == "partial":
        payload["limitations"] = ["Unknown duration"]
    draft = FullAnswerDraft.model_validate_json(json.dumps(payload))
    assert draft.answer_claims
    assert draft.requirement_coverage == ()
    assert draft._deviations


def test_general_schema_requires_all_need_ids_and_normalizes_exact_duplicate_claims(tmp_path):
    workflow, result, _, first = _markdown_reading(
        tmp_path, "# R\n\nA uses a hand pump.\n\n## Other\n\nOther.\n", "List trials", "A uses"
    )
    workflow = workflow.with_reading_plan(_plan("inventory")).with_reader_observations(True)
    model = workflow._response_model_for((first,))
    sid = next(iter(_spans((first,))))
    draft = _draft()
    draft["answer_claims"][0]["citations"][0]["source_id"] = sid
    draft["detail_claims"] = [{**draft["answer_claims"][0], "claim_id": "C2"}]
    draft["requirement_coverage"][0]["claim_ids"] = ("C1", "C2")
    value = model.model_validate_json(json.dumps(draft))
    verified, fields = workflow._verify_output(value, (first,), language="en")
    assert len(verified) == 1
    assert fields["claim_duplicates_removed"] == 1
    assert fields["requirement_coverage"][0].claim_ids == ("C1",)
    assert fields["detail_claims"] == ()
    assert fields["claim_support"] == ("direct",)
    draft.pop("requirement_coverage")
    with pytest.raises(ValidationError):
        model.model_validate_json(json.dumps(draft))


def test_partial_answer_and_refusal_keep_supported_conclusions_out_of_gaps(tmp_path):
    workflow, _, _, first = _markdown_reading(
        tmp_path,
        "# R\n\nTrial A used a motor.\n\n## Other\n\nOther.\n",
        "Was A manual, and how long did it run?",
        "Trial A",
    )
    workflow = workflow.with_reading_plan(_plan("lookup", "lookup")).with_reader_observations(True)
    model = workflow._response_model_for((first,))
    sid = next(iter(_spans((first,))))
    payload = {
        "status": "partial",
        "answer_claims": [
            {
                "claim_id": "C1",
                "text": "No",
                "support": "limited_inference",
                "inference_basis": "A used a motor, so it was not solely manual.",
                "citations": [{"source_id": sid}],
            }
        ],
        "detail_claims": [
            {
                "claim_id": "C2",
                "text": "A used a motor, so it was not solely manual.",
                "support": "limited_inference",
                "inference_basis": "A used a motor, so it was not solely manual.",
                "citations": [{"source_id": sid}],
            }
        ],
        "limitations": ["Duration is not supplied."],
        "requirement_coverage": [
            {"need_id": "need-01", "status": "answered", "claim_ids": ["C1", "C2"], "gap": ""},
            {
                "need_id": "need-02",
                "status": "missing",
                "claim_ids": [],
                "gap": "Duration is not supplied.",
            },
        ],
    }
    draft = model.model_validate_json(json.dumps(payload))
    claims, metadata = workflow._verify_output(draft, (first,), language="en")
    assert len(claims) == 2
    assert metadata["claim_support"] == ("limited_inference", "limited_inference")
    payload["status"] = "insufficient_evidence"
    with pytest.raises(ValidationError):
        model.model_validate_json(json.dumps(payload))
    payload["answer_claims"] = []
    payload["detail_claims"] = []
    payload["requirement_coverage"][0] = {
        "need_id": "need-01",
        "status": "missing",
        "claim_ids": [],
        "gap": "No premise.",
    }
    assert model.model_validate_json(json.dumps(payload)).status == "insufficient_evidence"
    # The validator verifies the contract; it cannot detect that this model judgment is wrong.


def test_duplicate_check_preserves_case_sensitive_units_and_different_evidence(tmp_path):
    workflow, _, _, first = _markdown_reading(
        tmp_path,
        "# R\n\nDevice A: 1 mW; device B: 1 MW.\n\n## Other\n\nOther.\n",
        "List powers",
        "Device A",
    )
    sid = next(iter(_spans((first,))))
    payload = {
        "status": "answered",
        "answer_claims": [
            {
                "claim_id": "C1",
                "support": "direct",
                "text": "1 mW",
                "citations": [{"source_id": sid}],
            },
            {
                "claim_id": "C2",
                "support": "direct",
                "text": "1 MW",
                "citations": [{"source_id": sid}],
            },
        ],
        "detail_claims": [],
        "limitations": [],
    }
    draft = workflow._response_model_for((first,)).model_validate_json(json.dumps(payload))
    claims, metadata = workflow._verify_output(draft, (first,), language="en")
    assert len(claims) == 2
    assert [claim.claim_id for claim in claims] == ["C1", "C2"]
    assert metadata["claim_duplicates_removed"] == 0


@pytest.mark.parametrize("basis", [None, "", "   "])
def test_limited_inference_requires_structured_basis(basis):
    from page_cite_rag.full_answer import FullClaimDraft

    claim = {
        "claim_id": "C1",
        "text": "Not all trials were manual",
        "citations": [{"source_id": "S1"}],
    }
    with pytest.raises(ValidationError, match="support"):
        FullClaimDraft.model_validate_json(json.dumps(claim))
    claim["support"] = "limited_inference"
    if basis is not None:
        claim["inference_basis"] = basis
    with pytest.raises(ValidationError):
        FullClaimDraft.model_validate_json(json.dumps(claim))


@pytest.mark.parametrize("answer_format", ["general", "short_answer", "multiple_choice"])
@pytest.mark.parametrize("support", ["direct", "limited_inference"])
def test_every_writer_format_preserves_and_verifies_support(
    tmp_path, monkeypatch, answer_format, support
):
    from page_cite_rag import full_answer

    workflow, _, _, first = _markdown_reading(
        tmp_path,
        "# R\n\nTrial A used a motor.\n\n## Other\n\nOther.\n",
        "Was every trial manual?",
        "Trial A",
    )
    workflow.answer_format = answer_format
    sid = next(iter(_spans((first,))))
    basis = "A used a motor; A is a counterexample to every trial being manual."
    claim = {"claim_id": "C1", "text": "No", "support": support, "citations": [{"source_id": sid}]}
    if support == "limited_inference":
        claim["inference_basis"] = basis
    payload = {"status": "answered", "detail_claims": [], "limitations": []}
    if answer_format == "general":
        payload["answer_claims"] = [claim]
    elif answer_format == "short_answer":
        payload["short_answer"] = claim
    else:
        payload.update(
            choice_claim_id="C1",
            choice="B",
            choice_citations=claim["citations"],
            choice_support=support,
        )
        if support == "limited_inference":
            payload["choice_inference_basis"] = basis
    schema = workflow._response_model_for((first,))
    if support == "limited_inference":
        missing = json.loads(json.dumps(payload))
        if answer_format == "multiple_choice":
            missing.pop("choice_inference_basis")
        else:
            (
                missing["answer_claims"][0]
                if answer_format == "general"
                else missing["short_answer"]
            ).pop("inference_basis")
        with pytest.raises(ValidationError):
            schema.model_validate_json(json.dumps(missing))
    checked = []
    original = full_answer.validate_span_claims

    def record(store, document_id, draft, blocks):
        checked.extend(c.text for c in draft.claims)
        return original(store, document_id, draft, blocks)

    monkeypatch.setattr(full_answer, "validate_span_claims", record)
    draft = schema.model_validate_json(json.dumps(payload))
    claims, metadata = workflow._verify_output(draft, (first,), language="en")
    assert len(claims) == 1
    assert claims[0].claim_id == metadata["answer_claims"][0].claim_id == "C1"
    assert metadata["claim_support"] == (support,)
    if support == "limited_inference":
        verified_basis = metadata["claim_inference_basis"][0]
        assert verified_basis.text == basis and basis in checked
        assert verified_basis.citations == claims[0].citations
    else:
        assert metadata["claim_inference_basis"] == (None,)


def test_dedup_revalidates_primary_slots_coverage_ids_and_status(tmp_path, monkeypatch):
    workflow, _, _, first = _markdown_reading(
        tmp_path, "# R\n\nA uses a hand pump.\n\n## Other\n\nOther.\n", "List trials", "A uses"
    )
    payload = _draft()
    payload["answer_claims"][0]["citations"] = [{"source_id": next(iter(_spans((first,))))}]
    claim = payload["answer_claims"][0]
    payload["answer_claims"] = [claim, {**claim, "claim_id": "C2"}]
    payload["detail_claims"] = [
        {**claim, "claim_id": "C3"},
        {**claim, "claim_id": "C4", "text": "A is a completed trial"},
    ]
    payload["requirement_coverage"][0]["claim_ids"] = ["C2", "C3", "C4"]
    draft = FullAnswerDraft.model_validate_json(json.dumps(payload))
    validated = []
    original = FullAnswerDraft.model_validate

    def record(value, **kwargs):
        result = original(value, **kwargs)
        validated.append(result)
        return result

    monkeypatch.setattr(FullAnswerDraft, "model_validate", record)
    claims, metadata = workflow._verify_output(draft, (first,), language="en")
    assert len(validated) == 2
    normalized = validated[-1]
    assert normalized.status == "answered"
    assert len(normalized.answer_claims) == len(normalized.detail_claims) == 1
    assert len(claims) == 2
    assert [claim.claim_id for claim in claims] == ["C1", "C2"]
    assert metadata["claim_duplicates_removed"] == 2
    assert normalized.requirement_coverage[0].claim_ids == ("C1", "C2")
    assert metadata["requirement_coverage"] == normalized.requirement_coverage
    assert metadata["claim_support"] == ("direct", "direct")


def test_writer_actual_budget_includes_support_schema_and_full_output_contract(tmp_path):
    workflow, _, _, first = _markdown_reading(
        tmp_path, "# R\n\nA uses a hand pump.\n\n## Other\n\nOther.\n", "List trials", "A uses"
    )
    seen = []

    class BudgetModel(_ScriptedModel):
        def prepare_budget(self, messages, schema):
            seen.append((json.loads(messages[1].content), schema))
            return super().prepare_budget(messages, schema)

    workflow._model = BudgetModel(workflow, [], "S1")
    _, _, _, budget = workflow._context("List trials", "en", "local", (first,), None)
    payload, schema = seen[-1]
    assert payload["output_budget"]["version"] == "writer-support-fields/v2"
    assert (
        payload["output_budget"]["tokens"]
        == budget.output_reserved
        == workflow._budget_config.num_predict
    )
    claim = schema["$defs"]["ProvidedFullAnswerClaimDraft"]
    assert "support" in claim["required"]
    assert "inference_basis" in claim["properties"]


@pytest.mark.parametrize("answer_format", ["general", "short_answer", "multiple_choice"])
@pytest.mark.parametrize("ids", [("C1", "C2"), ("C1", "C1"), ("C2", "C1"), ("C1", "C3")])
def test_writer_claim_ids_require_unique_identity_but_allow_advisory_order(answer_format, ids):
    workflow = object.__new__(FullAnswerWorkflow)
    workflow.answer_format = answer_format
    claim = {
        "claim_id": ids[0],
        "text": "A",
        "support": "direct",
        "citations": [{"source_id": "S1"}],
    }
    payload = {"status": "answered", "detail_claims": [{**claim, "claim_id": ids[1], "text": "B"}]}
    if answer_format == "general":
        payload["answer_claims"] = [claim]
    elif answer_format == "short_answer":
        payload["short_answer"] = claim
    else:
        payload.update(
            choice="B",
            choice_claim_id=ids[0],
            choice_citations=claim["citations"],
            choice_support="direct",
        )
    schema = workflow.agent_response_model()
    if ids[0] == ids[1] and answer_format == "general":
        # General answers keep both claims under fresh IDs; benchmark formats stay strict.
        draft = schema.model_validate_json(json.dumps(payload))
        assert draft.detail_claims[0].claim_id == "C2"
        assert {"type": "duplicate_claim_id", "count": 1} in draft._deviations
    elif ids[0] == ids[1]:
        with pytest.raises(ValidationError, match="unique and nonblank"):
            schema.model_validate_json(json.dumps(payload))
    else:
        assert schema.model_validate_json(json.dumps(payload)).detail_claims[0].claim_id == ids[1]


def test_coverage_accepts_existing_ids_and_discards_absent_ids():
    payload = _draft()
    assert FullAnswerDraft.model_validate_json(json.dumps(payload)).requirement_coverage[
        0
    ].claim_ids == ("C1",)
    payload["requirement_coverage"][0]["claim_ids"] = ["C99"]
    draft = FullAnswerDraft.model_validate_json(json.dumps(payload))
    assert not draft.requirement_coverage
    assert {"type": "coverage_absent_claim", "count": 1} in draft._deviations


def test_writer_normalizes_claim_order_and_duplicate_links_with_trace_metadata(tmp_path):
    workflow, _, _, first = _markdown_reading(
        tmp_path, "## A\n\nAlpha.\n\n## B\n\nBeta.\n", "Find Alpha", "Alpha"
    )
    payload = _draft(claim_ids=("C8", "C8"))
    payload["answer_claims"][0]["claim_id"] = "C8"
    payload["answer_claims"][0]["citations"][0]["source_id"] = next(iter(_spans((first,))))
    payload["answer_claims"][0]["inference_basis"] = "Alpha is documented."
    draft = FullAnswerDraft.model_validate_json(json.dumps(payload))
    verified, fields = workflow._verify_output(draft, (first,), language="en")
    assert verified[0].claim_id == "C1"
    assert fields["requirement_coverage"][0].claim_ids == ("C1",)
    assert fields["claim_support"] == ("limited_inference",)
    assert fields["output_deviations"] == (
        {"type": "support_with_basis", "count": 1},
        {"type": "duplicate_coverage_link", "count": 1},
        {"type": "claim_id_order", "count": 1},
    )


@pytest.mark.parametrize("coverage_status,claim_ids", [("missing", ()), ("answered", ("C1",))])
def test_writer_preserves_content_when_completion_metadata_disagrees(coverage_status, claim_ids):
    payload = _draft(coverage_status=coverage_status, claim_ids=claim_ids, gap="Unknown duration")
    draft = FullAnswerDraft.model_validate_json(json.dumps(payload))
    assert draft.status == "partial"
    assert draft.answer_claims[0].text == payload["answer_claims"][0]["text"]
    assert draft.requirement_coverage[0].gap == "Unknown duration"
    assert {"type": "coverage_status_mismatch", "count": 1} in draft._deviations


def test_writer_merges_identical_need_coverage_and_discards_conflicting_judgments():
    payload = _draft()
    payload["requirement_coverage"] *= 2
    draft = FullAnswerDraft.model_validate_json(json.dumps(payload))
    assert len(draft.requirement_coverage) == 1
    assert draft._deviations == [{"type": "duplicate_need_coverage", "count": 1}]
    payload["requirement_coverage"][1] = {
        "need_id": "need-01",
        "status": "missing",
        "claim_ids": [],
        "gap": "Unknown duration",
    }
    draft = FullAnswerDraft.model_validate_json(json.dumps(payload))
    assert not draft.requirement_coverage
    assert {"type": "conflicting_need_coverage", "count": 1} in draft._deviations


@pytest.mark.parametrize(
    "coverage",
    [
        [{"need_id": "need-01", "status": "answered", "claim_ids": [], "gap": ""}],
        [{"need_id": "need-01", "status": "missing", "claim_ids": ["C1"], "gap": "unknown"}],
        [{"need_id": "need-01", "status": "answered", "claim_ids": ["C99"], "gap": ""}],
        [{"need_id": "need-02", "status": "answered", "claim_ids": ["C1"], "gap": ""}],
    ],
)
def test_inconsistent_coverage_delivers_verified_claims_with_deviations(tmp_path, coverage):
    workflow, _, _, first = _markdown_reading(
        tmp_path, "# R\n\nA uses a hand pump.\n\n## Other\n\nOther.\n", "List trials", "A uses"
    )
    workflow = workflow.with_reading_plan(_plan("inventory")).with_requirement_coverage(True)
    payload = _draft()
    payload["answer_claims"][0]["citations"] = [{"source_id": next(iter(_spans((first,))))}]
    payload["requirement_coverage"] = coverage
    schema = workflow._response_model_for((first,))
    draft = schema.model_validate_json(json.dumps(payload))
    claims, fields = workflow._verify_output(draft, (first,), language="en")
    assert len(claims) == 1 and claims[0].citations
    assert fields["output_deviations"]
    assert not fields["requirement_coverage"]
    # Coverage tolerance cannot admit a source outside the provided evidence.
    payload["answer_claims"][0]["citations"] = [{"source_id": "S999"}]
    with pytest.raises(ValidationError):
        schema.model_validate_json(json.dumps(payload))


def test_repeated_claim_ids_keep_claims_and_drop_ambiguous_coverage(tmp_path):
    workflow, _, _, first = _markdown_reading(
        tmp_path, "# R\n\nA uses a hand pump.\n\n## Other\n\nOther.\n", "List trials", "A uses"
    )
    workflow = workflow.with_reading_plan(_plan("inventory")).with_requirement_coverage(True)
    sid = next(iter(_spans((first,))))
    payload = _draft()
    payload["answer_claims"][0]["citations"] = [{"source_id": sid}]
    payload["detail_claims"] = [
        {**payload["answer_claims"][0], "text": "A uses a pump"},
    ]
    schema = workflow._response_model_for((first,))
    draft = schema.model_validate_json(json.dumps(payload))
    assert [c.claim_id for c in (*draft.answer_claims, *draft.detail_claims)] == ["C1", "C2"]
    assert not draft.requirement_coverage
    claims, fields = workflow._verify_output(draft, (first,), language="en")
    assert len(claims) == 2 and all(c.citations for c in claims)
    deviations = fields["output_deviations"]
    assert {"type": "duplicate_claim_id", "count": 1} in deviations
    assert {"type": "coverage_ambiguous_claim", "count": 1} in deviations
    # Citations stay strict: an unknown source still rejects the whole output.
    payload["detail_claims"][0]["citations"] = [{"source_id": "S999"}]
    with pytest.raises(ValidationError):
        schema.model_validate_json(json.dumps(payload))


def test_irregular_claim_ids_are_renumbered_and_coverage_relinked(tmp_path):
    workflow, _, _, first = _markdown_reading(
        tmp_path, "# R\n\nA uses a hand pump.\n\n## Other\n\nOther.\n", "List trials", "A uses"
    )
    workflow = workflow.with_reading_plan(_plan("inventory")).with_requirement_coverage(True)
    sid = next(iter(_spans((first,))))
    payload = _draft(claim_ids=("C1", "D1"))
    payload["answer_claims"][0]["citations"] = [{"source_id": sid}]
    payload["detail_claims"] = [
        {**payload["answer_claims"][0], "claim_id": "D1", "text": "A uses a pump"},
    ]
    draft = workflow._response_model_for((first,)).model_validate_json(json.dumps(payload))
    assert [c.claim_id for c in draft.detail_claims] == ["C2"]
    assert draft.requirement_coverage[0].claim_ids == ("C1", "C2")
    claims, fields = workflow._verify_output(draft, (first,), language="en")
    assert len(claims) == 2 and all(c.citations for c in claims)
    assert {"type": "irregular_claim_id", "count": 1} in fields["output_deviations"]


def _stats(prompt=120, output=30):
    return GenerationStats(
        prompt_eval_count=prompt,
        eval_count=output,
        total_duration=1,
        load_duration=None,
        prompt_eval_duration=1,
        eval_duration=1,
        done_reason="stop",
        elapsed_ms=1.0,
    )


class _ScriptedModel:
    """Return a fixed cited answer for retained writer contract tests."""

    identity = {"model": "scripted-gap-test"}

    def __init__(self, workflow, gap_replies, cite_source_id):
        self.config = workflow._budget_config
        self.gap_replies = list(gap_replies)
        self.cite_source_id = cite_source_id
        self.gap_payloads: list[dict] = []
        self.writer_payloads: list[dict] = []
        self.writer_systems: list[str] = []
        self.writer_schemas: list[dict] = []
        self.writer_citations: list = []  # per writer call: source IDs or an exception
        self.evidence_limit: int | None = None

    def count_chat_input_tokens(self, payload):
        return 100

    def _budget(self):
        return PromptBudget(
            input_upper_estimate=100 + self.config.template_reserve,
            output_reserved=self.config.num_predict,
            context_limit=self.config.num_ctx,
            method="scripted-exact-token-test/v1",
            exact_input_tokens=100,
        )

    def prepare_budget(self, messages, _schema):
        evidence = json.loads(messages[1].content).get("evidence", ())
        if self.evidence_limit is not None and len(evidence) > self.evidence_limit:
            raise ContextBudgetError("scripted writer context is full")
        return self._budget()

    def generate(self, messages, response_model):
        payload = json.loads(messages[1].content)
        self.writer_payloads.append(payload)
        self.writer_systems.append(messages[0].content)
        self.writer_schemas.append(response_model.model_json_schema())
        cited = self.writer_citations.pop(0) if self.writer_citations else [self.cite_source_id]
        if isinstance(cited, Exception):
            raise cited
        claims = tuple(
            {
                "claim_id": f"C{n + 1}",
                "support": "direct",
                "text": f"ok {n}",
                "citations": ({"source_id": source_id},),
            }
            for n, source_id in enumerate(cited)
        )
        value = response_model.model_validate(
            {
                "status": "answered",
                "answer_claims": claims[:3],
                "detail_claims": claims[3:],
                "limitations": (),
                **(
                    {
                        "requirement_coverage": tuple(
                            {
                                "need_id": need["need_id"],
                                "status": "answered",
                                "claim_ids": ("C1",),
                                "gap": "",
                            }
                            for need in payload["reading_plan"]
                        )
                    }
                    if response_model.__name__ == "CoveredFullAnswerDraft"
                    else {}
                ),
            }
        )
        return GenerationResult(value, _stats(100, 20), self._budget(), value.model_dump_json())


def _markdown_reading(tmp_path, text: str, question: str, hit_prefix: str):
    """Workflow over a self-authored Markdown file whose outline comes from its headings."""
    from page_cite_rag.full_answer import FullAnswerWorkflow
    from page_cite_rag.generation import GenerationConfig
    from page_cite_rag.ingest import prepare_source
    from page_cite_rag.store import SQLiteDocumentStore

    source = tmp_path / "report.md"
    source.write_text(text, encoding="utf-8")
    prepared = prepare_source(source, usage_scope="self-authored-test")
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    blocks = store.get_blocks(prepared.document.document_id)
    hit = next(block for block in blocks if block.text.startswith(hit_prefix))
    result = SearchResult(
        document_id=prepared.document.document_id,
        index_id="synthetic-index",
        method="bm25",
        query=question,
        hits=(SearchHit(rank=1, score=1.0, block=hit),),
        elapsed_ms=0.1,
    )
    workflow = FullAnswerWorkflow(
        SQLiteDocumentStore(store.database_path, read_only=True),
        prepared.document.document_id,
        _Retriever(result),
        None,
        GenerationConfig(num_predict=3072),
    )
    return workflow, result, blocks, hit


def test_scoped_plan_adds_the_routing_field_only_when_asked():
    from page_cite_rag.reading_plan import (
        SCOPE_INSTRUCTION,
        ReadingPlan,
        ScopedReadingPlan,
        reading_plan_messages,
    )

    plain = reading_plan_messages("Q?", source_language="en", output_language="en")
    scoped = reading_plan_messages(
        "Q?", source_language="en", output_language="en", with_scope=True
    )
    assert SCOPE_INSTRUCTION not in plain[0].content
    assert scoped[0].content.endswith(SCOPE_INSTRUCTION)
    assert "scope" not in ReadingPlan.model_json_schema()["properties"]
    plan = ScopedReadingPlan.model_validate_json(
        json.dumps(
            {
                "requirements": [{"need": "n", "task": "inventory", "queries": ["q"]}],
                "scope": "whole_document",
            }
        )
    )
    assert plan.scope == "whole_document"
    assert ScopedReadingPlan.model_json_schema()["required"] == ["requirements", "scope"]


def test_general_answers_allow_more_detail_claims_than_benchmark_formats():
    from page_cite_rag.full_answer import (
        MAX_GENERAL_DETAIL_CLAIMS,
        FullAnswerDraft,
        ShortBenchmarkAnswerDraft,
    )

    assert MAX_GENERAL_DETAIL_CLAIMS == 15
    assert ShortBenchmarkAnswerDraft.model_fields["detail_claims"].metadata[-1].max_length == 9

    def draft(details: int) -> dict:
        claim = {"support": "direct", "text": "item", "citations": [{"source_id": "S1"}]}
        return {
            "status": "answered",
            "answer_claims": [{**claim, "claim_id": f"C{i}"} for i in range(1, 4)],
            "detail_claims": [{**claim, "claim_id": f"C{i}"} for i in range(4, 4 + details)],
            "limitations": [],
        }

    # The whole contract, not only the field limit, must accept 3 + 15 claims.
    assert len(FullAnswerDraft.model_validate_json(json.dumps(draft(15))).detail_claims) == 15
    with pytest.raises(ValidationError):
        FullAnswerDraft.model_validate_json(json.dumps(draft(16)))


def test_answers_say_when_evidence_had_chat_markers_and_quote_the_original(tmp_path):
    question = "How does the prompt template end a turn?"
    workflow, _result, _blocks, hit = _markdown_reading(
        tmp_path,
        "# Paper\n\nThe template ends each turn with an <|im_end|> token.\n",
        question,
        "The template",
    )
    workflow._model = _ScriptedModel(workflow, [], next(iter(_spans((hit,)))))
    answer = workflow.ask(question, language="en")
    assert any("chat-template markers" in note for note in answer.limitations)
    assert "<|im_end|>" in answer.answer_claims[0].citations[0].quote


def test_a_full_list_of_eighteen_claims_passes_writer_and_citation_checks(tmp_path):
    workflow, _, blocks, _ = _markdown_reading(
        tmp_path,
        "## Results\n\nAlpha.\n\n## Discussion\n\nBeta.\n",
        "List all findings",
        "Alpha",
    )
    evidence = tuple(block for block in blocks if block.text.strip() in {"Alpha.", "Beta."})
    s1, s2 = _spans(evidence)
    model = _ScriptedModel(workflow, [], s1)
    model.writer_citations = [[s1, s2] * 9]
    workflow._model = model
    answer = workflow.answer_from_context("List all findings", evidence, language="en")

    assert answer.status == "answered"
    assert (len(answer.answer_claims), len(answer.detail_claims)) == (3, 15)
    assert [claim.claim_id for claim in answer.claims] == [f"C{i}" for i in range(1, 19)]
    result_type = type(answer)
    valid = {**answer.model_dump(), "budget": answer.budget}
    assert result_type.model_validate(valid) == answer
    invalid = {**answer.model_dump(), "budget": answer.budget}
    invalid["answer_claims"][1]["claim_id"] = "C1"
    with pytest.raises(ValidationError, match="unique and sequential"):
        result_type.model_validate(invalid)
    invalid = {**answer.model_dump(), "budget": answer.budget}
    invalid["requirement_coverage"] = (
        {"need_id": "need-01", "status": "answered", "claim_ids": ("C99",), "gap": ""},
    )
    with pytest.raises(ValidationError, match="refer to delivered claims"):
        result_type.model_validate(invalid)
