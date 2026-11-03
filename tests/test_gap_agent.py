from __future__ import annotations

import json

from tests.test_agent_workflow import _Retriever, _setup

from page_cite_rag.agent import _spans
from page_cite_rag.application import ask_with_agent
from page_cite_rag.generation import (
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

    def prepare_budget(self, _messages, _schema):
        return self._budget()

    def generate(self, messages, response_model):
        payload = json.loads(messages[1].content)
        if response_model.__name__ in ("GapCheck", "FocusSelection"):
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
