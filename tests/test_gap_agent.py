from __future__ import annotations

import json

from tests.test_agent_workflow import _Retriever, _setup

from page_cite_rag.agent import _spans
from page_cite_rag.application import ask_with_agent
from page_cite_rag.generation import (
    ContextBudgetError,
    GenerationResult,
    GenerationStats,
    PromptBudget,
    StructuredOutputError,
)
from page_cite_rag.retrieval import SearchHit, SearchResult


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
    """Answers the gap check from a script and the writer with a fixed cited answer."""

    identity = {"model": "scripted-gap-test"}

    def __init__(self, workflow, gap_replies, cite_source_id):
        self.config = workflow._budget_config
        self.gap_replies = list(gap_replies)
        self.cite_source_id = cite_source_id
        self.gap_payloads: list[dict] = []
        self.writer_payloads: list[dict] = []
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
        if response_model.__name__ in (
            "GapCheck",
            "FocusSelection",
            "OutlineSelection",
            "SectionSources",
            "FollowUpSelection",
        ):
            self.gap_payloads.append(payload)
            reply = self.gap_replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            value = response_model.model_validate(reply(payload) if callable(reply) else reply)
            return GenerationResult(value, _stats(), self._budget(), value.model_dump_json())
        self.writer_payloads.append(payload)
        value = response_model.model_validate(
            {
                "status": "answered",
                "answer_claims": (
                    {"text": "ok", "citations": ({"source_id": self.cite_source_id},)},
                ),
                "detail_claims": (),
                "limitations": (),
            }
        )
        return GenerationResult(value, _stats(100, 20), self._budget(), value.model_dump_json())


def _prepare(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    target = blocks[6]  # "Result paragraph." is not in the initial evidence
    workflow._retriever = _Retriever(
        SearchResult(
            document_id=workflow.document_id,
            index_id="synthetic-index",
            method="bm25",
            query="result",
            hits=(SearchHit(rank=1, score=1.0, block=target),),
            elapsed_ms=0.1,
        )
    )
    return workflow, result, blocks, target


def test_gap_reading_reads_missing_source_for_the_model_query_and_hands_over_ids_only(tmp_path):
    workflow, result, blocks, target = _prepare(tmp_path)
    initial = (blocks[1],)
    first_source = next(iter(_spans(initial)))
    target_source = next(iter(_spans((target,))))
    model = _ScriptedModel(
        workflow,
        [
            {
                "needs": [
                    {
                        "need_id": "need-01",
                        "status": "missing",
                        "query": "Result paragraph",
                        "mode": "exact",
                    }
                ]
            },
            lambda payload: {
                "needs": [
                    {
                        "need_id": "need-01",
                        "status": "supported",
                        "source_ids": [payload["sources"][-1]["source_id"]],
                    }
                ]
            },
        ],
        target_source,
    )
    workflow._model = model
    answer = ask_with_agent(
        workflow,
        result.query,
        language="en",
        initial_search=result,
        initial_blocks=initial,
        strategy="gap",
    )

    trace = answer.trace["agent"]
    assert trace["strategy"] == "gap_driven"
    assert len(model.gap_payloads) == 2
    # the program read the source the model asked for; the model never chose a tool
    events = [e for e in trace["tool_trace"] if e["tool"] == "read_source"]
    assert events[0]["actor"] == "program" and events[0]["chosen_by"] == "model_query"
    assert target.block_id in events[0]["returned_block_ids"]
    assert first_source in {s["source_id"] for s in model.gap_payloads[0]["sources"]}
    assert target_source in {s["source_id"] for s in model.gap_payloads[1]["sources"]}
    # writer receives the original text and an ID index, never free-form notes
    handoff = model.writer_payloads[0]["navigation_handoff"]["reading_report"]
    assert "observation" not in handoff["requirements"][0]
    assert handoff["requirements"][0]["source_ids"] == [target_source]
    assert handoff["unresolved"] == []
    assert any(item["source_id"] == target_source for item in model.writer_payloads[0]["evidence"])
    assert answer.result.status == "answered"
    assert answer.result.agent_usage.model_calls == 3  # two gap checks + the shared writer
    assert trace["navigation_degraded"] is False


def test_invalid_gap_check_degrades_to_the_shared_writer_without_notes(tmp_path):
    workflow, result, blocks, target = _prepare(tmp_path)
    initial = (blocks[1],)
    model = _ScriptedModel(workflow, [StructuredOutputError("schema")], next(iter(_spans(initial))))
    workflow._model = model
    answer = ask_with_agent(
        workflow,
        result.query,
        language="en",
        initial_search=result,
        initial_blocks=initial,
        strategy="gap",
    )

    assert answer.trace["agent"]["navigation_degraded"] is True
    assert answer.trace["agent"]["navigation_failure_type"] == "invalid_gap_check"
    assert answer.result.status == "answered"  # degradation never rewrites the writer status
    assert len(model.writer_payloads) == 1


def test_a_multi_block_chunk_counts_as_one_read_so_the_writer_is_not_flooded(tmp_path, monkeypatch):
    import page_cite_rag.gap_agent as gap
    from page_cite_rag.gap_agent import _read_for, _State
    from page_cite_rag.retrieval import SearchContextGroup

    workflow, _result, blocks, _target = _prepare(tmp_path)
    chunks = [blocks[1:3], blocks[3:5], blocks[5:7]]

    def group_for(_workflow, block_id):
        for index, members in enumerate(chunks):
            if any(item.block_id == block_id for item in members):
                return SearchContextGroup(
                    chunk_index=index,
                    text="chunk",
                    source_refs=("#/texts/0",),
                    source_block_ids=tuple(item.block_id for item in members),
                )
        return None

    monkeypatch.setattr(gap, "_group_for", group_for)
    monkeypatch.setattr(gap, "READS_PER_QUERY", 1)
    state = _State()
    taken = _read_for(workflow, state, "need-01", "paragraph", "exact")

    assert len(taken) == 2  # exactly one chunk of two blocks, not every matching chunk
    assert len(state.groups) == 1


def test_focus_reading_hands_the_writer_only_the_selected_sources(tmp_path):
    workflow, result, blocks, _target = _prepare(tmp_path)
    initial = (blocks[1], blocks[3], blocks[4])
    chosen = next(iter(_spans((blocks[3],))))
    model = _ScriptedModel(
        workflow, [{"needs": [{"need_id": "need-01", "source_ids": [chosen]}]}], chosen
    )
    workflow._model = model
    answer = ask_with_agent(
        workflow,
        result.query,
        language="en",
        initial_search=result,
        initial_blocks=initial,
        strategy="focus",
    )

    trace = answer.trace["agent"]
    assert trace["strategy"] == "focused"
    assert [item["source_id"] for item in model.writer_payloads[0]["evidence"]] == [chosen]
    assert set(trace["candidate_block_ids"]) == {block.block_id for block in initial}
    assert answer.result.included_block_ids == (blocks[3].block_id,)
    assert answer.result.agent_usage.model_calls == 2


def test_failed_focus_selection_keeps_the_full_initial_evidence(tmp_path):
    workflow, result, blocks, _target = _prepare(tmp_path)
    initial = (blocks[1], blocks[3])
    model = _ScriptedModel(
        workflow, [StructuredOutputError("schema")], next(iter(_spans((blocks[1],))))
    )
    workflow._model = model
    answer = ask_with_agent(
        workflow,
        result.query,
        language="en",
        initial_search=result,
        initial_blocks=initial,
        strategy="focus",
    )

    assert answer.trace["agent"]["navigation_failure_type"] == "invalid_focus_selection"
    assert len(model.writer_payloads[0]["evidence"]) == 2


def test_focus_handoff_lists_only_sources_that_survive_the_total_cap(tmp_path, monkeypatch):
    import page_cite_rag.gap_agent as gap

    workflow, result, blocks, _target = _prepare(tmp_path)
    initial = tuple(blocks[1:6])
    ids = [next(iter(_spans((block,)))) for block in initial]
    monkeypatch.setattr(gap, "MAX_FOCUS_SOURCES", 2)
    model = _ScriptedModel(
        workflow, [{"needs": [{"need_id": "need-01", "source_ids": ids[:4]}]}], ids[0]
    )
    workflow._model = model
    answer = ask_with_agent(
        workflow,
        result.query,
        language="en",
        initial_search=result,
        initial_blocks=initial,
        strategy="focus",
    )

    handoff = model.writer_payloads[0]["navigation_handoff"]["reading_report"]
    assert handoff["requirements"][0]["source_ids"] == ids[:2]
    assert len(model.writer_payloads[0]["evidence"]) == 2
    assert answer.result.status == "answered"


def test_quote_verification_tolerates_whitespace_and_script_but_not_invention():
    from page_cite_rag.gap_agent import quote_is_verbatim

    source = "教育领域的人工智能\n必须遵守人权标准。"
    assert quote_is_verbatim("教育領域的人工智能 必須遵守人權標準", source)
    assert not quote_is_verbatim("人工智能可以取代教師", source)
    assert not quote_is_verbatim("教育", source)  # too short to verify anything


def test_outline_reading_hands_writer_the_sources_marked_in_chosen_sections(tmp_path):
    workflow, result, blocks, target = _prepare(tmp_path)
    initial = (blocks[1],)
    target_source = next(iter(_spans((target,))))

    def choose(payload):
        ids = [e["section_id"] for e in payload["outline"] if "Result" in e["title"]]
        return {"section_ids": ids}

    def read(payload):
        assert payload["chosen_sections"] == ["Results"]
        source = next(
            item["source_id"]
            for part in payload["sections"]
            for item in part["sources"]
            if "Result paragraph" in item["text"]
        )
        return {"source_ids": [source, source]}

    def follow_up(payload):
        assert [(p["section"], p["text"].strip()) for p in payload["marked_passages"]] == [
            ("Results", "Result paragraph.")
        ]
        assert "Results" not in [e["title"] for e in payload["unread_outline"]]
        return {"section_ids": []}

    model = _ScriptedModel(workflow, [choose, read, follow_up], target_source)
    workflow._model = model
    answer = ask_with_agent(
        workflow,
        result.query,
        language="en",
        initial_search=result,
        initial_blocks=initial,
        strategy="outline",
    )

    trace = answer.trace["agent"]
    assert trace["strategy"] == "outline"
    read_event = next(e for e in trace["tool_trace"] if e.get("mode") == "section")
    assert read_event["sources_marked"] == 1
    evidence_ids = [item["source_id"] for item in model.writer_payloads[0]["evidence"]]
    assert target_source in evidence_ids
    assert blocks[1].block_id not in trace["initial_answer_block_ids"]
    assert trace["follow_up_section_ids"] == []
    # select + one section read + follow-up check + writer
    assert answer.result.agent_usage.model_calls == 4


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


def _mark(prefix: str):
    """Scripted reader that marks the sources whose text starts with ``prefix``."""

    def read(payload):
        return {
            "source_ids": [
                item["source_id"]
                for part in payload["sections"]
                for item in part["sources"]
                if item["text"].startswith(prefix)
            ]
        }

    return read


def _section_ids(payload, *titles: str) -> dict:
    entries = payload.get("outline") or payload["unread_outline"]
    return {"section_ids": [e["section_id"] for e in entries if e["title"] in titles]}


def _ask(workflow, question, result, initial):
    return ask_with_agent(
        workflow,
        question,
        language="en",
        initial_search=result,
        initial_blocks=(initial,),
        strategy="outline",
    )


def test_outline_reading_keeps_paragraphs_the_question_names(tmp_path):
    question = "Which strategies does paragraph 12 refer to?"
    workflow, result, blocks, strategy = _markdown_reading(
        tmp_path,
        "# Summary\n\n12. States should apply the strategies set out above.\n\n"
        "## Strategies\n\nStrategy text.\n\n## Other\n\nOther text.\n",
        question,
        "Strategy text",
    )
    named = next(block for block in blocks if block.text.startswith("12."))
    model = _ScriptedModel(
        workflow,
        [lambda p: _section_ids(p, "Strategies"), _mark("Strategy"), {"section_ids": []}],
        next(iter(_spans((strategy,)))),
    )
    workflow._model = model
    answer = _ask(workflow, question, result, strategy)

    trace = answer.trace["agent"]
    assert trace["pinned_block_ids"] == [named.block_id]
    assert named.block_id in trace["initial_answer_block_ids"]
    assert strategy.block_id in trace["initial_answer_block_ids"]


def test_long_outline_is_judged_in_parts_and_small_sections_share_one_read(tmp_path):
    question = "List every finding."
    parts = "".join(f"## Part {n}\n\nText of part {n}.\n\n" for n in range(1, 60))
    workflow, result, blocks, first = _markdown_reading(
        tmp_path, "# Report\n\nIntro.\n\n" + parts, question, "Intro"
    )
    late = next(block for block in blocks if block.text.startswith("Text of part 51."))
    payloads = []

    def choose(payload):
        payloads.append(payload)
        return _section_ids(payload, "Part 50", "Part 51")

    def read(payload):
        assert [part["title"] for part in payload["sections"]] == ["Part 50", "Part 51"]
        return _mark("Text of part 51")(payload)

    model = _ScriptedModel(
        workflow, [choose, choose, read, {"section_ids": []}], next(iter(_spans((late,))))
    )
    workflow._model = model
    answer = _ask(workflow, question, result, first)

    # 60 sections: two near-equal parts, each judged on its own (the first chose nothing).
    assert [(p["outline_part"], len(p["outline"])) for p in payloads] == [
        ("1 of 2", 30),
        ("2 of 2", 30),
    ]
    trace = answer.trace["agent"]
    assert trace["outline_pages"] == 2
    assert len(trace["selected_section_ids"]) == 2
    reads = [e for e in trace["tool_trace"] if e.get("mode") == "section"]
    assert [len(e["section_ids"]) for e in reads] == [2]
    assert late.block_id in trace["initial_answer_block_ids"]
    assert first.block_id not in trace["initial_answer_block_ids"]


def test_outline_reading_follows_up_where_a_marked_passage_points_back(tmp_path):
    question = "Trace what the discussion says about widgets back to their definition."
    workflow, result, blocks, discussion = _markdown_reading(
        tmp_path,
        "# Paper\n\n## Definitions\n\nA widget is a small gear.\n\n"
        "## Methods\n\nWe built things.\n\n"
        "## Discussion\n\nAs defined earlier, widgets failed under load.\n",
        question,
        "As defined",
    )
    definition = next(block for block in blocks if block.text.startswith("A widget"))

    def follow_up(payload):
        assert payload["sections_read"] == ["Discussion"]
        assert payload["marked_passages"][0]["text"].startswith("As defined earlier")
        return _section_ids(payload, "Definitions")

    def read_back(payload):
        assert payload["chosen_sections"] == ["Definitions"]
        return _mark("A widget")(payload)

    model = _ScriptedModel(
        workflow,
        [lambda p: _section_ids(p, "Discussion"), _mark("As defined"), follow_up, read_back],
        next(iter(_spans((discussion,)))),
    )
    workflow._model = model
    answer = _ask(workflow, question, result, discussion)

    trace = answer.trace["agent"]
    assert len(trace["follow_up_section_ids"]) == 1
    assert [e["chosen_by"] for e in trace["tool_trace"] if e.get("mode") == "section"] == [
        "model_section_selection",
        "model_follow_up",
    ]
    assert definition.block_id in trace["initial_answer_block_ids"]
    assert discussion.block_id in trace["initial_answer_block_ids"]


def test_outline_reading_keeps_one_source_per_section_before_seconds_when_writer_is_full(
    tmp_path,
):
    question = "List the finding of every part."
    text = "# Report\n\nIntro.\n\n" + "".join(
        f"## Part {n}\n\nFinding {n} first.\n\nFinding {n} second.\n\n" for n in (1, 2, 3)
    )
    workflow, result, blocks, first = _markdown_reading(tmp_path, text, question, "Finding 1 first")
    firsts = [b for b in blocks if b.text.startswith("Finding") and "first" in b.text]
    model = _ScriptedModel(
        workflow,
        [lambda p: _section_ids(p, "Part 1", "Part 2", "Part 3"), _mark("Finding"), {}],
        next(iter(_spans((first,)))),
    )
    model.evidence_limit = 3
    workflow._model = model
    answer = _ask(workflow, question, result, first)

    trace = answer.trace["agent"]
    assert trace["marked_block_count"] == 6
    assert trace["initial_answer_block_ids"] == [b.block_id for b in firsts]
    assert len(trace["writer_omitted_marked_block_ids"]) == 3
    # The shared handoff treats returned blocks as writer-eligible, so reads return only
    # what fits and record the rest separately.
    reads = [e for e in trace["tool_trace"] if e.get("mode") == "section"]
    assert {id for e in reads for id in e["returned_block_ids"]} == {b.block_id for b in firsts}
    assert {id for e in reads for id in e["writer_omitted_block_ids"]} == set(
        trace["writer_omitted_marked_block_ids"]
    )
    assert len(model.writer_payloads[0]["evidence"]) == 3
    assert answer.result.status == "answered"
