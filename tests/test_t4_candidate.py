"""Synthetic scope, capacity and writer-contract behavior; no model or network calls."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from pydantic import ValidationError

from page_cite_rag.agent import _spans
from page_cite_rag.full_answer import FullAnswerDraft, FullAnswerWorkflow
from page_cite_rag.reading_plan import (
    ReadingNeed,
    ReadingPlan,
    ScopedReadingPlan,
    reading_plan_messages,
)
from tests.test_gap_agent import _ask, _mark, _markdown_reading, _ScriptedModel, _section_ids


def _plan(*tasks):
    return ReadingPlan(
        requirements=tuple(
            ReadingNeed(need=f"Requested part {i}", task=task, queries=(f"part {i}",))
            for i, task in enumerate(tasks, 1)
        )
    )


def test_outline_pages_keep_each_needs_selection_in_the_writer_handoff(tmp_path, monkeypatch):
    from page_cite_rag import gap_agent

    workflow, result, _, first = _markdown_reading(
        tmp_path,
        "## A\n\nAlpha.\n\n## B\n\nBeta.\n\n## C\n\nGamma.\n\n## D\n\nDelta.\n",
        "Compare parts",
        "Alpha",
    )
    workflow = workflow.with_reading_plan(_plan("comparison", "comparison"))
    monkeypatch.setattr(gap_agent, "OUTLINE_PAGE_SECTIONS", 2)
    selected = {}

    def choose(p):
        assert [s["title"] for s in p["outline"]] == (
            ["A", "B"] if p["outline_part"] == "1 of 2" else ["C", "D"]
        )
        entries = []
        for need, section in zip(p["needs"], p["outline"], strict=True):
            selected[section["section_id"]] = need["need_id"]
            entries.append({"need_id": need["need_id"], "section_ids": [section["section_id"]]})
        return {"selections": entries}

    def notes(p):
        return {
            "points": [
                {
                    "need_id": selected[section["section_id"]],
                    "point": section["title"],
                    "source_ids": [section["sources"][-1]["source_id"]],
                }
                for section in p["sections"]
            ]
        }

    model = _ScriptedModel(
        workflow, [choose, choose, notes, notes, {}, {}], next(iter(_spans((first,))))
    )
    workflow._model = model
    answer = _ask(workflow, result.query, result, first)
    trace = answer.trace["agent"]
    assert not model.gap_replies
    assert trace["selected_section_ids"] == list(selected)
    assert trace["seed_section_ids"] == []
    assert trace["section_need_ids"] == {sid: [nid] for sid, nid in selected.items()}
    assert len(trace["reading_points"]) == len(trace["writer_reading_points"]) == 4
    report = model.writer_payloads[0]["navigation_handoff"]["reading_report"]
    assert report["unresolved"] == []
    assert all("observation" not in requirement for requirement in report["requirements"])
    assert {sid for r in report["requirements"] for sid in r["source_ids"]} == {
        sid for point in trace["writer_reading_points"] for sid in point["source_ids"]
    }
    assert {(r["need_id"], tuple(r["source_ids"])) for r in report["requirements"]} == {
        (point["need_id"], tuple(point["source_ids"])) for point in trace["writer_reading_points"]
    }
    assert {r["need"] for r in report["requirements"]} == {
        "Requested part 1",
        "Requested part 2",
    }
    assert "need-01" in json.dumps(model.writer_schemas[0])


@pytest.mark.parametrize("stage", ["select_sections", "follow_up_sections"])
@pytest.mark.parametrize("invalid_ids", [("need-01", "need-01"), ("need-01",), (None, "need-02")])
def test_selection_rejects_duplicate_or_missing_need_ids(tmp_path, stage, invalid_ids):
    from page_cite_rag.gap_agent import run_outline_reading

    workflow, result, _, first = _markdown_reading(
        tmp_path, "## A\n\nAlpha.\n\n## B\n\nBeta.\n", "Compare parts", "Alpha"
    )
    workflow = workflow.with_reading_plan(_plan("comparison", "comparison"))
    workflow._budget_config = replace(workflow._budget_config, num_predict=8192)

    def invalid(p):
        key = "outline" if stage == "select_sections" else "unread_outline"
        sid = p[key][0]["section_id"]
        return {
            "selections": [
                {**({"need_id": nid} if nid else {}), "section_ids": [sid]} for nid in invalid_ids
            ]
        }

    replies = (
        [invalid, {"points": []}, {}]
        if stage == "select_sections"
        else [lambda p: _section_ids(p, "A"), {"points": []}, invalid]
    )
    model = _ScriptedModel(workflow, replies, "S1")
    workflow._model = model
    run = run_outline_reading(workflow, result.query, "en", result, (first,))
    events = [e for e in run.trace["tool_trace"] if e["tool"] == stage]
    assert len(events) == 1 and events[0]["status"] == "invalid"
    assert run.trace["follow_up_section_ids"] == []
    assert len([p for p in model.gap_payloads if "sections" in p]) == 1
    assert not run.trace["reading_points"]
    assert not model.gap_replies
    if stage == "select_sections":
        assert run.trace["selected_section_ids"] == []
        assert run.trace["seed_section_ids"]


@pytest.mark.parametrize("cross_scope", [False, True])
def test_follow_up_checks_each_sources_requirement_scope(tmp_path, cross_scope):
    from page_cite_rag.gap_agent import run_outline_reading

    workflow, result, _, first = _markdown_reading(
        tmp_path,
        "## A\n\nInitial.\n\n## B\n\nBeta.\n\n## C\n\nGamma.\n",
        "Compare parts",
        "Initial",
    )
    workflow = workflow.with_reading_plan(_plan("comparison", "comparison"))
    workflow._budget_config = replace(workflow._budget_config, num_predict=8192)

    def follow_up(p):
        assert {s["title"] for s in p["unread_outline"]} == {"B", "C"}
        return {
            "selections": [
                {
                    "need_id": "need-01",
                    "section_ids": [
                        _section_ids(p, title)["section_ids"][0] for title in ("B", "C")
                    ],
                },
                {"need_id": "need-02", **_section_ids(p, "B")},
            ]
        }

    def note(p):
        assert {n["need_id"] for n in p["needs"]} == {"need-01", "need-02"}
        gamma = next(s for s in p["sections"] if s["title"] == "C")
        return {
            "points": [
                {
                    "need_id": "need-02" if cross_scope else "need-01",
                    "point": "Gamma fact",
                    "source_ids": [gamma["sources"][-1]["source_id"]],
                }
            ]
        }

    model = _ScriptedModel(
        workflow, [lambda p: _section_ids(p, "A"), {"points": []}, follow_up, note], "S1"
    )
    workflow._model = model
    run = run_outline_reading(workflow, result.query, "en", result, (first,))
    assert not model.gap_replies
    assert len(run.trace["follow_up_section_ids"]) == 2
    assert bool(run.trace["reader_rejected_points"]) is cross_scope
    if cross_scope:
        assert run.trace["reader_rejected_points"][0]["reason"] == "requirement_scope"
        assert run.trace["reading_points"] == run.trace["writer_reading_points"] == []
        assert run.structured_response.requirements == []
    else:
        assert run.trace["writer_reading_points"][0]["need_id"] == "need-01"
        assert run.structured_response.requirements[0].observation == "Gamma fact"


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


def test_reader_rejects_a_real_source_assigned_to_another_requirement(tmp_path):
    workflow, result, _, first = _markdown_reading(
        tmp_path, "# Report\n\n## A\n\nAlpha.\n\n## B\n\nBeta.\n", "Compare A and B", "Alpha"
    )
    workflow = workflow.with_reading_plan(_plan("comparison", "comparison"))

    def select(p):
        return {
            "selections": [
                {"need_id": "need-01", **_section_ids(p, "A")},
                {"need_id": "need-02", **_section_ids(p, "B")},
            ]
        }

    def wrong(p):
        sid = next(s["source_id"] for part in p["sections"] for s in part["sources"])
        return {"points": [{"need_id": "need-02", "point": "Misassigned", "source_ids": [sid]}]}

    model = _ScriptedModel(
        workflow, [select, wrong, _mark("Beta"), {}], next(iter(_spans((first,))))
    )
    # B is read with need-02; no writer call is needed for checking the reader handoff.
    workflow._model = model
    from page_cite_rag.gap_agent import run_outline_reading

    # The ordinary fixture fills absent need IDs, preserving explicitly invalid ones.
    run = run_outline_reading(workflow, result.query, "en", result, (first,))
    assert any(e["status"] == "invalid" for e in run.trace["tool_trace"])
    assert all(p["need_id"] == "need-02" for p in run.trace["reading_points"])
    assert run.structured_response.unresolved == ["Requested part 1"]


@pytest.mark.parametrize(
    "bad_kind,include_valid", [("scope", True), ("empty", True), ("scope", False), (None, True)]
)
def test_scope_validation_keeps_valid_notes_and_records_only_rejections(
    tmp_path, bad_kind, include_valid
):
    from page_cite_rag.gap_agent import run_outline_reading

    workflow, result, _, first = _markdown_reading(
        tmp_path, "# R\n\n## A\n\nAlpha.\n\n## B\n\nBeta.\n", "Compare parts", "Alpha"
    )
    workflow = workflow.with_reading_plan(_plan("comparison", "comparison"))

    workflow._budget_config = replace(workflow._budget_config, num_predict=8192)

    def select(p):
        a, b = (_section_ids(p, name)["section_ids"][0] for name in ("A", "B"))
        return {
            "selections": [
                {"need_id": "need-01", "section_ids": [a, b]},
                {"need_id": "need-02", "section_ids": [a]},
            ]
        }

    def notes(p):
        sid = next(
            source["source_id"]
            for part in p["sections"]
            if part["title"] == "B"
            for source in part["sources"]
            if "Beta" in source["text"]
        )
        points = []
        if include_valid:
            points.append({"need_id": "need-01", "point": "Beta fact", "source_ids": [sid]})
        if bad_kind:
            points.append(
                {
                    "need_id": "need-02" if bad_kind == "scope" else "need-01",
                    "point": "Misassigned" if bad_kind == "scope" else "   ",
                    "source_ids": [sid],
                }
            )
        return {"points": points}

    workflow._model = _ScriptedModel(workflow, [select, notes, {}], "S1")
    run = run_outline_reading(workflow, result.query, "en", result, (first,))
    rejected = run.trace["reader_rejected_points"]
    assert len(rejected) == int(bad_kind is not None)
    if bad_kind:
        assert rejected[0]["reason"] == (
            "requirement_scope" if bad_kind == "scope" else "empty_point"
        )
        assert rejected[0]["section_ids"]
    assert [p["point"] for p in run.trace["reading_points"]] == (
        ["Beta fact"] if include_valid else []
    )
    assert [p["point"] for p in run.trace["writer_reading_points"]] == (
        ["Beta fact"] if include_valid else []
    )
    event = next(e for e in run.trace["tool_trace"] if e["tool"] == "read_source")
    assert event["rejected_point_count"] == len(rejected)
    assert event["rejected_point_reasons"] == [p["reason"] for p in rejected]


def test_read_quota_is_shared_fairly_between_needs_without_saturation_continuation(
    tmp_path, monkeypatch
):
    from page_cite_rag import gap_agent

    workflow, result, _, first = _markdown_reading(
        tmp_path,
        "# Report\n\n## A\n\nAlpha one.\n\nAlpha two.\n\n## B\n\nBeta one.\n\nBeta two.\n",
        "Compare parts",
        "Alpha",
    )
    workflow = workflow.with_reading_plan(_plan("comparison", "comparison"))
    monkeypatch.setattr(gap_agent, "MAX_WINDOW_SOURCES", 1)
    monkeypatch.setattr(gap_agent, "MAX_SECTION_READ_CALLS", 2)

    def choose(p):
        return {
            "selections": [
                {"need_id": "need-01", **_section_ids(p, "A")},
                {"need_id": "need-02", **_section_ids(p, "B")},
            ]
        }

    model = _ScriptedModel(workflow, [choose, {"points": []}, {"points": []}], "S1")
    workflow._model = model
    run = gap_agent.run_outline_reading(workflow, result.query, "en", result, (first,))
    reads = [p for p in model.gap_payloads if "sections" in p]
    assert [p["needs"][0]["need_id"] for p in reads] == ["need-01", "need-02"]
    assert run.trace["sources_skipped_for_read_limit"] == 4
    assert run.trace["model_calls_completed"] == 3
    assert run.trace["navigation_failure_type"] == "no_reading_points"
    assert run.trace["tool_trace"][-1]["status"] == "degraded_initial_evidence"
    pending = [
        source
        for sources in run.trace["follow_up_candidate_sources"].values()
        for source in sources
    ]
    assert len([s for s in pending if s["reason"] == "read_call_limit"]) == 4
    assert all(
        len(s["need_ids"]) == 1 for s in pending if s["source_id"] in run.trace["read_source_ids"]
    )


@pytest.mark.parametrize("need_count", [1, 4, 10, 12])
def test_shared_window_serves_all_needs_once_with_transferable_slots(tmp_path, need_count):
    from page_cite_rag import gap_agent

    workflow, result, _, first = _markdown_reading(
        tmp_path, "## A\n\nShared evidence.\n\n## B\n\nOther.\n", "Compare", "Shared"
    )
    workflow = workflow.with_reading_plan(_plan(*(["comparison"] * need_count)))

    class SharedModel(_ScriptedModel):
        def generate(self, messages, response_model):
            if response_model.__name__ == "SectionPoints":
                assert response_model.model_json_schema()["properties"]["points"]["maxItems"] == 10
                assert "Unused slots may serve other" in messages[0].content
                assert "at least one note for each need with evidence" in messages[0].content
            return super().generate(messages, response_model)

    def notes(payload):
        ids = [need["need_id"] for need in payload["needs"]]
        assert len(ids) == need_count
        assert "max_points_per_need" not in payload
        source_id = next(iter(_spans((first,))))
        # All available evidence belongs to the first need: all ten slots transfer to it.
        return {
            "points": [
                {"need_id": ids[0], "point": f"Fact {i}", "source_ids": [source_id]}
                for i in range(10)
            ]
        }

    model = SharedModel(workflow, [lambda p: _section_ids(p, "A"), notes, {}], "S1")
    workflow._model = model
    run = gap_agent.run_outline_reading(workflow, result.query, "en", result, (first,))
    assert len([p for p in model.gap_payloads if "sections" in p]) == 1
    assert len(run.trace["reading_points"]) == 10
    assert not any("need_groups" in e for e in run.trace["tool_trace"])
    assert gap_agent.MAX_SECTION_READ_CALLS == run.trace["reading_budget"]["read_calls"] == 16
    assert run.trace["reading_version"] == "agentic-reading-v11-incremental-follow-up"
    assert run.trace["reading_budget_version"].startswith("need-round-robin-token-windows/v7-")
    assert not model.gap_replies


def test_all_need_reader_calls_share_the_sixteen_call_limit(tmp_path, monkeypatch):
    from page_cite_rag import gap_agent

    paragraphs = "\n\n".join(f"Fact {i}." for i in range(20))
    workflow, result, _, first = _markdown_reading(
        tmp_path, f"# R\n\n## A\n\n{paragraphs}\n", "List facts", "Fact"
    )
    workflow = workflow.with_reading_plan(_plan(*(["comparison"] * 12)))
    monkeypatch.setattr(gap_agent, "MAX_WINDOW_SOURCES", 1)
    model = _ScriptedModel(
        workflow, [lambda p: _section_ids(p, "A"), *([{"points": []}] * 16)], "S1"
    )
    workflow._model = model
    run = gap_agent.run_outline_reading(workflow, result.query, "en", result, (first,))
    reads = [p for p in model.gap_payloads if "sections" in p]
    assert len(reads) == 16
    assert not model.gap_replies
    skipped = [e for e in run.trace["tool_trace"] if e["status"] == "skipped_read_limit"]
    assert skipped
    assert all(e["reason"] == "read_call_limit" and e["need_ids"] for e in skipped)


@pytest.mark.parametrize("excess", ["total", "need"])
def test_reader_accepts_unequal_allocation_but_rejects_total_overflow(tmp_path, excess):
    from page_cite_rag.gap_agent import run_outline_reading

    workflow, result, _, first = _markdown_reading(
        tmp_path, "# R\n\n## A\n\nFact.\n\n## B\n\nOther.\n", "List facts", "Fact"
    )
    workflow = workflow.with_reading_plan(_plan("comparison", "comparison"))

    def over_quota(payload):
        ids = [n["need_id"] for n in payload["needs"]]
        source_id = next(iter(_spans((first,))))
        counts = [6, 5] if excess == "total" else [6, 0]
        return {
            "points": [
                {"need_id": nid, "point": "Fact", "source_ids": [source_id]}
                for nid, count in zip(ids, counts, strict=True)
                for _ in range(count)
            ]
        }

    model = _ScriptedModel(workflow, [lambda p: _section_ids(p, "A"), over_quota, {}], "S1")
    workflow._model = model
    run = run_outline_reading(workflow, result.query, "en", result, (first,))
    event = next(e for e in run.trace["tool_trace"] if e["tool"] == "read_source")
    if excess == "total":
        assert not run.trace["reading_points"]
        assert event["status"] == "invalid"
    else:
        assert len(run.trace["reading_points"]) == 6
        assert event["status"] == "ok"
        assert event["deviations"] == []


@pytest.mark.parametrize("text_size", [6000, 28000])
def test_reader_uses_full_context_and_splits_sources_without_loss(tmp_path, monkeypatch, text_size):
    from page_cite_rag import gap_agent, span_citations
    from page_cite_rag.generation import ContextBudgetError

    # Exercise a genuinely oversized single span, beyond the ordinary span slicer.
    monkeypatch.setattr(span_citations, "MAX_SPAN_CHARS", 40000)
    text = "證" * text_size
    workflow, result, _, first = _markdown_reading(
        tmp_path, f"## A\n\n{text}\n\n## B\n\nOther.\n", "List findings", "證"
    )
    workflow._budget_config = replace(workflow._budget_config, num_ctx=16384, num_predict=3072)

    class CountedModel(_ScriptedModel):
        def prepare_budget(self, messages, schema):
            payload = json.loads(messages[1].content)
            if "sections" not in payload:
                return super().prepare_budget(messages, schema)
            # Synthetic exact tokenizer: all framing/schema/need costs plus one
            # token per source character. It rejects before returning metadata.
            tokens = 1500 + sum(
                len(source["text"])
                for section in payload["sections"]
                for source in section["sources"]
            )
            if (
                tokens + self.config.template_reserve + self.config.num_predict
                > self.config.num_ctx
            ):
                raise ContextBudgetError("full context exhausted")
            return replace(
                self._budget(),
                exact_input_tokens=tokens,
                input_upper_estimate=tokens + self.config.template_reserve,
            )

    model = CountedModel(
        workflow,
        [lambda p: _section_ids(p, "A"), *([{"points": []}] * (2 if text_size == 6000 else 4)), {}],
        "S1",
    )
    workflow._model = model
    run = gap_agent.run_outline_reading(workflow, result.query, "en", result, (first,))
    offered = [
        source
        for payload in model.gap_payloads
        if "sections" in payload
        for section in payload["sections"]
        for source in section["sources"]
        if "證" in source["text"]
    ]
    assert "".join(source["text"] for source in offered) == first.text
    assert len({source["source_id"] for source in offered}) == 1
    assert len(offered) == (1 if text_size == 6000 else 3)
    assert len(offered[0]["text"]) >= 6000
    assert offered[0]["source_id"] in run.trace["read_source_ids"]
    assert run.trace["sources_skipped_for_budget"] == 0
    assert all(
        event["input_upper_estimate"] + workflow._budget_config.num_predict <= 16384
        for event in run.trace["tool_trace"]
        if event.get("mode") == "section"
    )


def test_tokenizer_failures_propagate_instead_of_switching_to_character_estimates(tmp_path):
    from page_cite_rag.gap_agent import run_outline_reading

    workflow, result, _, first = _markdown_reading(
        tmp_path, "# R\n\n## A\n\nFact.\n\n## B\n\nOther.\n", "List findings", "Fact"
    )
    model = _ScriptedModel(workflow, [lambda p: _section_ids(p, "A")], "S1")

    def broken(*_):
        raise RuntimeError("tokenizer identity mismatch")

    model.prepare_budget = broken
    workflow._model = model
    with pytest.raises(RuntimeError, match="identity mismatch"):
        run_outline_reading(workflow, result.query, "en", result, (first,))
    assert len(model.gap_payloads) == 1


def test_partial_source_stays_unread_when_split_windows_hit_call_limit(tmp_path, monkeypatch):
    from page_cite_rag import gap_agent
    from page_cite_rag.generation import ContextBudgetError

    workflow, result, _, first = _markdown_reading(
        tmp_path, "## A\n\n" + "證" * 28000 + "\n\n## B\n\nOther.\n", "Find facts", "證"
    )
    from page_cite_rag import span_citations

    monkeypatch.setattr(span_citations, "MAX_SPAN_CHARS", 40000)
    monkeypatch.setattr(gap_agent, "MAX_WINDOW_SOURCES", 1)
    monkeypatch.setattr(gap_agent, "MAX_SECTION_READ_CALLS", 2)

    class LimitedModel(_ScriptedModel):
        def prepare_budget(self, messages, schema):
            payload = json.loads(messages[1].content)
            if any(
                len(source["text"]) > 6000
                for section in payload.get("sections", ())
                for source in section["sources"]
            ):
                raise ContextBudgetError("synthetic source capacity")
            return super().prepare_budget(messages, schema)

    model = LimitedModel(
        workflow, [lambda p: _section_ids(p, "A"), {"points": []}, {"points": []}], "S1"
    )
    workflow._model = model
    run = gap_agent.run_outline_reading(workflow, result.query, "en", result, (first,))
    skipped = [e for e in run.trace["tool_trace"] if e["status"] == "skipped_read_limit"]
    assert skipped and not set(skipped[0]["source_ids"]) & set(run.trace["read_source_ids"])
    pending = [
        source
        for sources in run.trace["follow_up_candidate_sources"].values()
        for source in sources
    ]
    assert set(skipped[0]["source_ids"]).issubset({source["source_id"] for source in pending})
    assert all(
        source["reason"] == "read_call_limit"
        for source in pending
        if source["source_id"] in skipped[0]["source_ids"]
    )


def test_follow_up_omission_records_the_complete_note_and_budget_reason(tmp_path, monkeypatch):
    from page_cite_rag import gap_agent

    workflow, result, _, first = _markdown_reading(
        tmp_path, "# R\n\n## A\n\nFact.\n\n## B\n\nOther.\n", "List facts", "Fact"
    )
    monkeypatch.setattr(gap_agent, "FOLLOW_UP_TOTAL_CHARS", 1)
    model = _ScriptedModel(workflow, [lambda p: _section_ids(p, "A"), _mark("Fact"), {}], "S1")
    workflow._model = model
    run = gap_agent.run_outline_reading(workflow, result.query, "en", result, (first,))
    event = next(e for e in run.trace["tool_trace"] if e["tool"] == "follow_up_context")
    assert event["points_offered"] == 0
    assert event["points_omitted"] == len(event["omitted_notes"]) == 1
    omitted = event["omitted_notes"][0]
    assert omitted["reason"] == "follow_up_character_budget"
    assert omitted["point"] == run.trace["reading_points"][0]["point"]
    assert omitted["source_ids"] == run.trace["reading_points"][0]["source_ids"]


@pytest.mark.parametrize("budget_error", [False, True])
def test_all_writer_notes_rejected_use_degraded_application_handoff(
    tmp_path, monkeypatch, budget_error
):
    from page_cite_rag import gap_agent
    from page_cite_rag.generation import ContextBudgetError

    workflow, result, _, first = _markdown_reading(
        tmp_path,
        "# R\n\n## A\n\nLeft result.\n\nRight result.\n\n## B\n\nOther.\n",
        "Compare left and right",
        "Left",
    )

    def note(p):
        ids = [
            s["source_id"]
            for part in p["sections"]
            for s in part["sources"]
            if "result" in s["text"]
        ]
        return {"points": [{"point": "A relation needs both originals", "source_ids": ids}]}

    model = _ScriptedModel(
        workflow,
        [lambda p: _section_ids(p, "A"), note, {}],
        next(iter(_spans((first,)))),
    )
    model.evidence_limit = 1
    if budget_error:

        def reject(*args):
            raise ContextBudgetError("complete note cannot fit")

        monkeypatch.setattr(gap_agent, "_writer_admits", reject)
    workflow._model = model
    answer = _ask(workflow, result.query, result, first)
    trace = answer.trace["agent"]
    assert trace["writer_omitted_point_count"] == 1
    assert trace["navigation_degraded"] is True
    assert trace["navigation_failure_type"] == "complete_note_writer_budget"
    fallback = next(e for e in trace["tool_trace"] if e["tool"] == "writer_context")
    assert fallback["status"] == "degraded_initial_evidence"
    assert fallback["reason"] == "complete_note_writer_budget"
    assert fallback["block_ids"] == [first.block_id]
    assert len(model.writer_payloads) == 1
    handoff = trace["navigation_handoff"]
    assert handoff["navigation_degraded"] is True
    assert handoff["navigation_failure_type"] == "complete_note_writer_budget"
    assert handoff["reading_report"]["requirements"] == []
    assert handoff["initial_source_ids"] == list(_spans((first,)))
    assert (
        model.writer_payloads[0]["navigation_handoff"]["reading_report"]
        == handoff["reading_report"]
    )
    assert answer.result.answer_claims[0].citations[0].block_id == first.block_id
    assert trace["reading_point_count"] == len(trace["reading_points"]) == 1
    assert trace["writer_reading_point_count"] == len(trace["writer_reading_points"]) == 0
    point = trace["reading_points"][0]
    omitted = trace["writer_omitted_points"][0]
    assert omitted["reason"] == "complete_note_writer_budget"
    assert point.items() <= omitted.items()


def test_cli_distinguishes_all_notes_from_writer_notes_and_discloses_omissions(capsys):
    from page_cite_rag.cli import _print_route

    _print_route(
        {
            "route": "whole_document",
            "reason": "requested",
            "reading_summary": {
                "sections_chosen": ["A", "B"],
                "sections_from_search": [],
                "sections_followed_up": [],
                "sections_read": ["A"],
                "notes": 3,
                "notes_handed_to_writer": 1,
                "notes_not_in_answer": 0,
                "notes_omitted_for_writer_budget": 2,
                "notes_rejected_for_scope": 1,
                "sources_skipped_for_budget": 4,
                "model_calls": 3,
                "elapsed_seconds": 1.0,
                "navigation_degraded": False,
            },
        }
    )
    text = capsys.readouterr().out
    assert "Sections read (1): A" in text
    assert "Reading notes: 3; handed to writer: 1, of which 0 are not cited" in text
    assert "2 reading notes were omitted because of the writer budget" in text
    assert "1 reading notes were rejected for crossing requirement scope" in text
    assert "4 source passages exceeded the reading budget" in text


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


def test_follow_up_keeps_actual_span_handle_for_long_original(tmp_path):
    workflow, result, _, first = _markdown_reading(
        tmp_path,
        "# R\n\n## A\n\n" + "x" * 4100 + " tail fact.\n\n## B\n\nOther.\n",
        "Find the tail",
        "xxxx",
    )
    ids = list(_spans((first,)))
    assert len(ids) == 2

    def read(p):
        texts = {s["source_id"]: s["text"] for part in p["sections"] for s in part["sources"]}
        assert "tail fact" in texts[ids[1]]
        return {"points": [{"point": "tail fact", "source_ids": [ids[1]]}]}

    model = _ScriptedModel(workflow, [lambda p: _section_ids(p, "A"), read, {}], ids[1])
    workflow._model = model
    answer = _ask(workflow, result.query, result, first)
    handoff = model.writer_payloads[0]["navigation_handoff"]["reading_report"]["requirements"]
    assert handoff[0]["source_ids"] == [ids[1]]
    assert answer.result.answer_claims[0].citations[0].quote.rstrip().endswith("tail fact.")


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


def test_reader_over_capacity_output_is_rejected_without_truncation(tmp_path):
    from page_cite_rag.gap_agent import run_outline_reading

    workflow, result, _, first = _markdown_reading(
        tmp_path, "# R\n\n## A\n\nFact.\n\n## Other\n\nOther.\n", "List facts", "Fact"
    )
    offered = []

    def too_long(payload):
        text = "X" * (payload["max_point_chars"] + 1)
        offered.append(text)
        return {"points": [{"point": text, "source_ids": [next(iter(_spans((first,))))]}]}

    workflow._model = _ScriptedModel(workflow, [lambda p: _section_ids(p, "A"), too_long, {}], "S1")
    run = run_outline_reading(workflow, result.query, "en", result, (first,))
    assert offered
    assert run.trace["reading_points"] == []
    assert run.trace["navigation_failure_type"] == "no_reading_points"
    event = next(e for e in run.trace["tool_trace"] if e["tool"] == "read_source")
    assert event["status"] == "invalid"
    assert event["max_point_chars"] + 1 == len(offered[0])
    assert event["max_point_chars"] == 200


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


@pytest.mark.parametrize("name", ["OutlineSelection", "FollowUpSelection"])
def test_selection_merges_repeated_entries_and_retains_all_sections(name):
    from page_cite_rag.gap_agent import _selected_by_need, _selection_model

    schema = _selection_model(name, ["need-01"], ["A", "B", "C"], 2)
    section_schema = schema.model_json_schema()["$defs"][name + "Need"]["properties"]["section_ids"]
    if name == "FollowUpSelection":
        assert section_schema["maxItems"] == 2
        with pytest.raises(ValidationError):
            schema.model_validate(
                {"selections": [{"need_id": "need-01", "section_ids": ["A"] * 3}]}
            )
    else:
        assert "maxItems" not in section_schema
    value = schema.model_validate(
        {
            "selections": [
                {"need_id": "need-01", "section_ids": ["A", "B"]},
                {"need_id": "need-01", "section_ids": ["B", "C"]},
            ]
        }
    )
    assert _selected_by_need(value, ["need-01"]) == {"need-01": ["A", "B", "C"]}
    with pytest.raises(ValidationError):
        schema.model_validate({"selections": [{"need_id": "need-01", "section_ids": ["UNKNOWN"]}]})


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


@pytest.mark.parametrize("stage", ["select_sections"])
def test_selection_advice_deviations_are_recorded_without_losing_sections(
    tmp_path, monkeypatch, stage
):
    from page_cite_rag import gap_agent

    workflow, result, _, first = _markdown_reading(
        tmp_path, "## A\n\nAlpha.\n\n## B\n\nBeta.\n\n## C\n\nGamma.\n", "Find facts", "Alpha"
    )
    monkeypatch.setattr(gap_agent, "MAX_OUTLINE_SECTIONS", 1)
    monkeypatch.setattr(gap_agent, "MAX_FOLLOW_UP_SECTIONS", 1)
    selected = []

    def extra(payload):
        key = "outline" if stage == "select_sections" else "unread_outline"
        selected.extend(s["section_id"] for s in payload[key][:2])
        assert len(selected) == 2
        nid = payload["needs"][0]["need_id"]
        return {
            "selections": [
                {"need_id": nid, "section_ids": selected},
                {"need_id": nid, "section_ids": selected[:1]},
            ]
        }

    replies = (
        [extra, {"points": []}, {}]
        if stage == "select_sections"
        else [lambda p: _section_ids(p, "A"), {"points": []}, extra, {"points": []}]
    )
    workflow._model = _ScriptedModel(workflow, replies, "S1")
    run = gap_agent.run_outline_reading(workflow, result.query, "en", result, (first,))
    event = next(e for e in run.trace["tool_trace"] if e["tool"] == stage)
    assert event["status"] == "ok"
    assert event["selections"]["need-01"] == selected
    assert event["deviations"] == [
        {"type": "duplicate_selection", "count": 2},
        {"type": "selection_quota_exceeded", "count": 1},
    ]


@pytest.mark.parametrize("stage", ["follow_up_sections"])
def test_selection_over_limit_is_skipped_without_losing_read_sources(tmp_path, monkeypatch, stage):
    from page_cite_rag import gap_agent

    workflow, result, _, first = _markdown_reading(
        tmp_path, "## A\n\nAlpha.\n\n## B\n\nBeta.\n\n## C\n\nGamma.\n", "Find facts", "Alpha"
    )
    monkeypatch.setattr(gap_agent, "MAX_OUTLINE_SECTIONS", 1)
    monkeypatch.setattr(gap_agent, "MAX_FOLLOW_UP_SECTIONS", 1)
    selected = []

    def extra(payload):
        key = "outline" if stage == "select_sections" else "unread_outline"
        selected.extend(s["section_id"] for s in payload[key][:2])
        assert len(selected) == 2
        nid = payload["needs"][0]["need_id"]
        return {
            "selections": [
                {"need_id": nid, "section_ids": selected},
                {"need_id": nid, "section_ids": selected[:1]},
            ]
        }

    replies = (
        [extra, {"points": []}, {}]
        if stage == "select_sections"
        else [lambda p: _section_ids(p, "A"), {"points": []}, extra, {"points": []}]
    )
    workflow._model = _ScriptedModel(workflow, replies, "S1")
    run = gap_agent.run_outline_reading(workflow, result.query, "en", result, (first,))
    event = next(e for e in run.trace["tool_trace"] if e["tool"] == stage)
    assert event["status"] == "invalid"
    assert not run.trace["follow_up_section_ids"]
    assert run.context.blocks


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


@pytest.mark.parametrize("framing_fits", [True, False])
def test_follow_up_counts_only_note_bodies_and_checks_complete_prompt(
    tmp_path, monkeypatch, framing_fits
):
    from page_cite_rag import gap_agent
    from page_cite_rag.generation import ContextBudgetError

    paragraphs = "\n\n".join(f"Fact {index}." for index in range(4))
    workflow, result, _, first = _markdown_reading(
        tmp_path, f"## A\n\n{paragraphs}\n\n## B\n\nOther.\n", "List facts", "Fact"
    )
    monkeypatch.setattr(gap_agent, "MAX_WINDOW_SOURCES", 1)
    checked = []

    class FramingModel(_ScriptedModel):
        def prepare_budget(self, messages, schema):
            payload = json.loads(messages[1].content)
            if "unread_outline" in payload:
                checked.append(payload)
                assert len(payload["points"]) == 40
                assert sum(len(note["point"]) for note in payload["points"]) == 8000
                complete_size = sum(len(message.content) for message in messages) + len(
                    json.dumps(schema)
                )
                assert complete_size > 8000
                if not framing_fits:
                    raise ContextBudgetError("framing exhausts complete prompt budget")
            return super().prepare_budget(messages, schema)

    def notes(payload):
        sid = payload["sections"][0]["sources"][0]["source_id"]
        return {"points": [{"point": "證" * 200, "source_ids": [sid]} for _ in range(10)]}

    model = FramingModel(
        workflow,
        [lambda p: _section_ids(p, "A"), *([notes] * 5), *([{}] if framing_fits else [])],
        "S1",
    )
    workflow._model = model
    run = gap_agent.run_outline_reading(workflow, result.query, "en", result, (first,))
    assert checked and not model.gap_replies
    event = next(e for e in run.trace["tool_trace"] if e["tool"] == "follow_up_context")
    assert event["points_offered"] == 40
    assert event["points_omitted"] == 10
    follow = next(e for e in run.trace["tool_trace"] if e["tool"] == "follow_up_sections")
    assert follow["status"] == ("ok" if framing_fits else "skipped_prompt_budget")
    assert bool([p for p in model.gap_payloads if "unread_outline" in p]) is framing_fits


@pytest.mark.parametrize("covered_ids", [(), ("need-01",), ("need-01", "need-02")])
def test_writer_probe_uses_candidate_findings_actual_unresolved(tmp_path, monkeypatch, covered_ids):
    from page_cite_rag.gap_agent import _writer_admits
    from page_cite_rag.generation import ContextBudgetError

    workflow, _, _, first = _markdown_reading(
        tmp_path, "## A\n\nAlpha.\n\n## B\n\nBeta.\n", "Compare parts", "Alpha"
    )
    workflow = workflow.with_reading_plan(_plan("comparison", "comparison"))
    findings = [
        {
            "need_id": nid,
            "need": "Requested part " + nid[-1],
            "observation": "Fact",
            "source_ids": [next(iter(_spans((first,))))],
        }
        for nid in covered_ids
    ]
    expected = [f"Requested part {i}" for i in (1, 2) if f"need-0{i}" not in covered_ids]
    captured = []
    original = FullAnswerWorkflow._context

    def context(probe, *args):
        report = probe._navigation_handoff["reading_report"]
        captured.append(report)
        assert report["unresolved"] == expected
        # A candidate covering all needs fits exactly; any unresolved label would
        # overflow. Genuinely unresolved needs must still consume prompt space.
        if report["unresolved"]:
            raise ContextBudgetError("unresolved labels fill remaining writer space")
        return original(probe, *args)

    monkeypatch.setattr(FullAnswerWorkflow, "_context", context)
    workflow._model = _ScriptedModel(workflow, [], "S1")
    if expected:
        with pytest.raises(ContextBudgetError):
            _writer_admits(workflow, "Compare parts", "en", (first,), findings)
    else:
        assert _writer_admits(workflow, "Compare parts", "en", (first,), findings) == {
            first.block_id
        }
    assert len(captured) == 1
    assert [entry["need_id"] for entry in captured[0]["requirements"]] == list(covered_ids)


def test_reader_packs_original_text_toward_head_character_target(tmp_path):
    from page_cite_rag import gap_agent

    paragraphs = "\n\n".join(str(i) + "證" * 1700 for i in range(8))
    workflow, result, _, first = _markdown_reading(
        tmp_path, f"## A\n\n{paragraphs}\n\n## B\n\nOther.\n", "List", "0證"
    )
    model = _ScriptedModel(
        workflow, [lambda p: _section_ids(p, "A"), *([{"points": []}] * 3), {}], "S1"
    )
    workflow._model = model
    run = gap_agent.run_outline_reading(workflow, result.query, "en", result, (first,))
    reads = [p for p in model.gap_payloads if "sections" in p]
    assert len(reads) == 3
    assert gap_agent.MAX_SECTION_WINDOW_CHARS == 6000
    assert all(
        sum(len(s["text"]) for part in p["sections"] for s in part["sources"]) <= 6000
        for p in reads
    )
    assert any(e.get("reason") == "target_window_chars" for e in run.trace["tool_trace"])
    assert not model.gap_replies


def test_follow_up_can_reassign_a_read_source_to_an_unprocessed_need(tmp_path):
    from page_cite_rag import gap_agent

    workflow, result, _, first = _markdown_reading(
        tmp_path, "## A\n\nShared fact.\n\n## B\n\nOther.\n", "Compare", "Shared"
    )
    workflow = workflow.with_reading_plan(_plan("comparison", "comparison"))
    offered = []

    def choose(payload):
        return {
            "selections": [
                {"need_id": "need-01", **_section_ids(payload, "A")},
                {"need_id": "need-02", "section_ids": []},
            ]
        }

    def first_notes(payload):
        offered.extend(s["source_id"] for p in payload["sections"] for s in p["sources"])
        return {"points": [{"point": "Shared fact", "source_ids": [offered[-1]]}]}

    def follow(payload):
        sid = _section_ids(payload, "A")["section_ids"][0]
        assert all(s["need_ids"] == ["need-02"] for s in payload["unread_sources"][sid])
        return {
            "selections": [
                {"need_id": "need-01", "section_ids": []},
                {"need_id": "need-02", "section_ids": [sid]},
            ]
        }

    def reread(payload):
        assert offered == [s["source_id"] for p in payload["sections"] for s in p["sources"]]
        return {
            "points": [
                {
                    "need_id": "need-02",
                    "point": "Shared fact for second need",
                    "source_ids": [offered[-1]],
                }
            ]
        }

    model = _ScriptedModel(workflow, [choose, first_notes, follow, reread], "S1")
    workflow._model = model
    run = gap_agent.run_outline_reading(workflow, result.query, "en", result, (first,))
    assert set(offered) <= set(run.trace["read_source_ids"])
    assert {p["need_id"] for p in run.trace["reading_points"]} == {"need-01", "need-02"}
    assert not model.gap_replies


@pytest.mark.parametrize("single_fits", [True, False])
def test_follow_up_paginates_with_full_prompt_schema_budget_or_degrades(tmp_path, single_fits):
    from page_cite_rag import gap_agent
    from page_cite_rag.generation import ContextBudgetError

    workflow, result, _, first = _markdown_reading(
        tmp_path, "\n\n".join(f"## Part {i}\n\nFact {i}." for i in range(9)), "List", "Fact 0"
    )
    checked = []

    class PagedModel(_ScriptedModel):
        def prepare_budget(self, messages, schema):
            payload = json.loads(messages[1].content)
            if "unread_outline" in payload:
                assert schema["$defs"]
                assert payload["points"]
                checked.append(len(payload["unread_outline"]))
                if not single_fits or len(payload["unread_outline"]) > 2:
                    raise ContextBudgetError("complete prompt and schema exceed context")
            return super().prepare_budget(messages, schema)

    model = PagedModel(
        workflow,
        [lambda p: _section_ids(p, "Part 0"), _mark("Fact 0"), *([{}] * 4 if single_fits else [])],
        "S1",
    )
    workflow._model = model
    run = gap_agent.run_outline_reading(workflow, result.query, "en", result, (first,))
    actual = [p for p in model.gap_payloads if "unread_outline" in p]
    assert checked and not model.gap_replies
    if single_fits:
        assert len(actual) == 4
        assert all(len(p["unread_outline"]) <= 2 for p in actual)
        assert len({s["section_id"] for p in actual for s in p["unread_outline"]}) == 8
        assert not run.trace["navigation_degraded"]
    else:
        assert not actual
        assert run.trace["navigation_degraded"]
        assert run.trace["navigation_failure_type"] == "follow_up_prompt_budget"


def test_follow_up_reads_each_page_before_spending_calls_on_more_pages(tmp_path, monkeypatch):
    from page_cite_rag import gap_agent

    workflow, result, _, first = _markdown_reading(
        tmp_path, "## A\n\nAlpha.\n\n## B\n\nBeta.\n\n## C\n\nGamma.\n", "List", "Alpha"
    )
    monkeypatch.setattr(gap_agent, "OUTLINE_PAGE_SECTIONS", 1)
    monkeypatch.setattr(gap_agent, "MAX_SECTION_READ_CALLS", 2)
    model = _ScriptedModel(
        workflow,
        [
            lambda p: _section_ids(p, "A"),
            {},
            {},
            _mark("Alpha"),
            lambda p: _section_ids(p, "B"),
            _mark("Beta"),
        ],
        next(iter(_spans((first,)))),
    )
    workflow._model = model
    answer = _ask(workflow, result.query, result, first)
    trace = answer.trace["agent"]
    assert not model.gap_replies
    assert len([p for p in model.gap_payloads if "unread_outline" in p]) == 1
    assert len(model.writer_payloads) == 1
    assert len(trace["reading_points"]) == 2
    skipped = next(e for e in trace["tool_trace"] if e["status"] == "skipped_remaining_pages")
    assert skipped["reason"] == "read_call_limit"
    assert skipped["pages_skipped"] == 1
    assert len(skipped["section_ids"]) == 1


@pytest.mark.parametrize("stop_before_wait", [True, False])
def test_gpu_wait_preserves_writer_time_and_hands_off_existing_notes(tmp_path, stop_before_wait):
    from types import SimpleNamespace

    workflow, result, _, first = _markdown_reading(
        tmp_path, "## A\n\nAlpha.\n\n## B\n\nBeta.\n", "List", "Alpha"
    )
    model = _ScriptedModel(
        workflow,
        [] if stop_before_wait else [lambda p: _section_ids(p, "A"), _mark("Alpha")],
        next(iter(_spans((first,)))),
    )
    model.config = SimpleNamespace(
        num_ctx=workflow._budget_config.num_ctx,
        num_predict=workflow._budget_config.num_predict,
        template_reserve=workflow._budget_config.template_reserve,
        device="cuda",
    )
    remaining = [180.0 if stop_before_wait else 600.0]
    waits = []
    model.remaining_request_timeout_seconds = lambda: remaining[0]

    def wait(*, timeout_seconds):
        waits.append(timeout_seconds)
        if len(model.gap_payloads) == 2:
            remaining[0] = 50.0  # The wait itself crosses the writer reserve.
        return 120.0

    model.wait_for_gpu_headroom = wait
    workflow._model = model
    answer = _ask(workflow, result.query, result, first)
    assert len(model.writer_payloads) == 1
    assert not model.gap_replies
    assert answer.trace["agent"]["navigation_failure_type"] == "writer_reserve_deadline"
    assert answer.trace["agent"]["navigation_report_partial"]
    assert any("reserve time" in text for text in answer.result.limitations)
    assert answer.result.agent_usage.model_calls == (1 if stop_before_wait else 3)
    if stop_before_wait:
        assert waits == []
    else:
        assert waits == [120.0] * 3
        assert answer.trace["agent"]["writer_reading_point_count"] == 1


def test_gpu_safety_failure_during_wait_is_not_a_writer_time_handoff(tmp_path):
    from page_cite_rag.generation import ModelError

    workflow, result, _, first = _markdown_reading(
        tmp_path, "## A\n\nAlpha.\n\n## B\n\nBeta.\n", "List", "Alpha"
    )
    model = _ScriptedModel(workflow, [], next(iter(_spans((first,)))))

    def unsafe(**_kwargs):
        raise ModelError("GPU thermal slowdown")

    model.wait_for_gpu_headroom = unsafe
    workflow._model = model
    with pytest.raises(ModelError, match="GPU thermal slowdown"):
        _ask(workflow, result.query, result, first)
    assert model.gap_payloads == model.writer_payloads == []


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
