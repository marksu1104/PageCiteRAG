from __future__ import annotations

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import page_cite_rag.gpu_retrieval as gpu_retrieval
from page_cite_rag.agent import (
    AgentContext,
    AgentRunFailure,
    _search_in_section,
    fixed_section_supplement,
    run_bounded_reading_agent,
)
from page_cite_rag.full_answer import FullAnswerWorkflow
from page_cite_rag.generation import GenerationConfig
from page_cite_rag.gpu_retrieval import StagedGpuSearch
from page_cite_rag.hybrid import HybridRetriever
from page_cite_rag.ingest import prepare_source
from page_cite_rag.retrieval import SearchHit, SearchResult
from page_cite_rag.store import SQLiteDocumentStore


class _ChatModel(BaseChatModel):
    responses: list[AIMessage]
    call_count: int = 0
    seen_message_batches: list[list] = []

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted-agent-test"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.call_count += 1
        self.seen_message_batches.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


class _Retriever:
    def __init__(self, result):
        self.result = result
        self.queries = []
        self.top_k_values = []

    def search(self, query, *, top_k=5):
        self.queries.append(query)
        self.top_k_values.append(top_k)
        return self.result.model_copy(update={"query": query})

    def search_scoped(self, query, *, allowed_block_ids, candidate_k, top_k=5):
        self.queries.append(query)
        self.top_k_values.append(candidate_k)
        hits = tuple(hit for hit in self.result.hits if hit.block.block_id in allowed_block_ids)[
            :top_k
        ]
        return self.result.model_copy(
            update={
                "query": query,
                "hits": tuple(
                    hit.model_copy(update={"rank": rank}) for rank, hit in enumerate(hits, 1)
                ),
                "branches": (),
                "context_groups": (),
            }
        )


class _Model:
    identity = {"model": "scripted-test"}


def _setup(tmp_path):
    source = tmp_path / "paper.md"
    source.write_text(
        "# Intro\n\nIntro paragraph.\n\n## Method\n\n"
        "Method paragraph one.\n\nMethod paragraph two.\n\n"
        "## Results\n\nResult paragraph.\n",
        encoding="utf-8",
    )
    prepared = prepare_source(source, usage_scope="self-authored-test")
    store = SQLiteDocumentStore(tmp_path / "documents.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    blocks = tuple(store.get_blocks(prepared.document.document_id))
    result = SearchResult(
        document_id=prepared.document.document_id,
        index_id="synthetic-index",
        method="bm25",
        query="method details",
        hits=(SearchHit(rank=1, score=1.0, block=blocks[1]),),
        elapsed_ms=0.1,
    )
    workflow = FullAnswerWorkflow(
        SQLiteDocumentStore(store.database_path, read_only=True),
        prepared.document.document_id,
        _Retriever(result),
        _Model(),
        GenerationConfig(),
    )
    return workflow, result, blocks


def test_agent_tools_add_only_scoped_complete_blocks_and_obey_run_limits(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    outline = workflow.source_outline()
    method = next(section for section in outline.sections if section.title == "Method")
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_sources",
                        "args": {"query": "method details and conditions"},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_source",
                        "args": {
                            "source_id": method.section_id,
                            "cursor": method.start_ordinal,
                        },
                        "id": "2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_source",
                        "args": {
                            "source_id": method.section_id,
                            "cursor": method.start_ordinal + 1,
                        },
                        "id": "3",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="Evidence collected.",
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 9,
                    "total_tokens": 109,
                    "input_token_details": {"cache_read": 30},
                },
            ),
        ]
    )
    token_count_payloads = []

    def count_tokens(payload):
        token_count_payloads.append(payload)
        return 100

    run = run_bounded_reading_agent(
        workflow,
        "What details support the method?",
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=count_tokens,
    )

    assert model.call_count == 4
    assert run.trace["model_calls_completed"] == 4
    assert run.trace["tool_calls_recorded"] == 3
    assert len(token_count_payloads) == 4
    assert workflow._retriever.queries == ["method details and conditions"]
    assert run.trace["no_external_tools"] is True
    assert [block.ordinal for block in run.context.blocks] == [5, 4, 2]
    assert run.trace["tool_trace"][0]["query_sha256"]
    assert run.trace["tool_trace"][1]["source_id"] == method.section_id
    assert run.trace["tool_trace"][1]["next_cursor"] == method.start_ordinal + 1
    assert len(run.trace["model_call_records"]) == 4
    assert run.trace["model_call_records"][-1]["input_tokens_exact"] == 100
    assert run.trace["model_call_records"][-1]["output_tokens_observed"] == 9
    assert run.trace["model_call_records"][-1]["reported_usage"]["cache_read_tokens"] == 30
    assert "content" not in run.trace["model_call_records"][-1]


def test_agent_receives_initial_evidence_and_can_stop_without_searching(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    model = _ChatModel(responses=[AIMessage(content="The initial evidence is sufficient.")])

    run = run_bounded_reading_agent(
        workflow,
        "What does the introduction say?",
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=lambda _payload: 100,
    )

    initial_message = model.seen_message_batches[0][-1].content
    assert "initial_evidence" in initial_message
    assert "Intro paragraph." in initial_message
    assert workflow._retriever.queries == []
    assert run.trace["tool_calls_recorded"] == 0
    assert run.trace["model_calls_completed"] == 1


def test_search_preview_remains_readable_when_not_selected_for_final_context(tmp_path, monkeypatch):
    import page_cite_rag.agent as agent_module

    workflow, result, blocks = _setup(tmp_path)
    workflow._retriever.result = result.model_copy(
        update={"hits": (SearchHit(rank=1, score=2.0, block=blocks[6]),)}
    )

    def admit_first(_workflow, _question, _language, candidates, groups):
        selected = tuple(candidates[:1])
        return AgentContext(selected, tuple(groups), 100)

    monkeypatch.setattr(agent_module, "_admit", admit_first)
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_sources",
                        "args": {"query": "results"},
                        "id": "search",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="I inspected the search preview."),
        ]
    )

    run = run_bounded_reading_agent(
        workflow,
        "What is the result?",
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=lambda _payload: 100,
    )

    assert run.trace["tool_trace"][0]["newly_admitted_block_ids"] == []
    assert run.trace["tool_trace"][0]["replaced_block_ids"] == []
    assert run.trace["tool_trace"][0]["tool"] == "search_sources"
    tool_message = model.seen_message_batches[1][-1].content
    assert "Result paragraph." in tool_message
    assert '"admitted_for_final_answer":false' in tool_message
    assert [block.block_id for block in run.context.blocks] == [blocks[1].block_id]


def test_explicit_read_promotes_new_evidence_over_initial_context(tmp_path, monkeypatch):
    import page_cite_rag.agent as agent_module

    workflow, result, blocks = _setup(tmp_path)

    def admit_first(_workflow, _question, _language, candidates, groups):
        selected = tuple(candidates[:1])
        return AgentContext(selected, tuple(groups), 100)

    monkeypatch.setattr(agent_module, "_admit", admit_first)
    target = blocks[6]
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_source",
                        "args": {"source_id": target.block_id},
                        "id": "read",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The later source is more relevant."),
        ]
    )

    run = run_bounded_reading_agent(
        workflow,
        "What is the result?",
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=lambda _payload: 100,
    )

    assert [block.block_id for block in run.context.blocks] == [target.block_id]
    assert run.trace["read_source_ids"] == [target.block_id]
    assert run.trace["tool_trace"][0]["admitted_for_final_answer"] is True
    assert run.trace["tool_trace"][0]["replaced_block_ids"] == [blocks[1].block_id]
    assert "Result paragraph." in model.seen_message_batches[1][-1].content


def test_fixed_control_adds_at_most_one_neighboring_source_block(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    result = result.model_copy(
        update={
            "hits": (SearchHit(rank=1, score=1.0, block=blocks[2]),),
        }
    )
    context, trace = fixed_section_supplement(
        workflow,
        "What details support the method?",
        "en",
        result,
        (blocks[2],),
    )

    assert trace["policy"] == "top-hit-adjacent-block/v1"
    assert len(trace["admitted_block_ids"]) <= 1
    assert [block.ordinal for block in context.blocks] == [3, 4]


def test_fixed_control_reads_previous_neighbor_when_hit_is_last_in_section(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    result = result.model_copy(
        update={
            "hits": (SearchHit(rank=1, score=1.0, block=blocks[4]),),
        }
    )

    context, trace = fixed_section_supplement(
        workflow,
        "What details support the method?",
        "en",
        result,
        (blocks[4],),
    )

    assert trace["policy"] == "top-hit-adjacent-block/v1"
    assert len(trace["admitted_block_ids"]) <= 1
    assert {block.ordinal for block in context.blocks} == {4, 5}
    assert trace["candidate_block_id"] == blocks[3].block_id


def test_scoped_search_recovers_a_section_hit_beyond_global_final_k(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    ranked = tuple(
        SearchHit(rank=rank, score=1.0 / rank, block=block) for rank, block in enumerate(blocks, 1)
    )

    class Branch:
        def __init__(self, index_id, method):
            self.index_id = index_id
            self.method = method
            self.top_k_values = []

        def search(self, query, *, top_k):
            self.top_k_values.append(top_k)
            return SearchResult(
                document_id=result.document_id,
                index_id=self.index_id,
                method=self.method,
                query=query,
                hits=ranked[:top_k],
                elapsed_ms=0.1,
            )

    lexical = Branch("lexical-index", "bm25")
    dense = Branch("dense-index", "dense")
    workflow._retriever = HybridRetriever(lexical, dense, candidates=6)
    section = next(item for item in workflow.source_outline().sections if item.title == "Results")

    scoped, found = _search_in_section(workflow, "results", section)

    assert lexical.top_k_values == dense.top_k_values == [len(blocks)]
    assert {hit.block.block_id for hit in scoped.hits} == {
        blocks[5].block_id,
        blocks[6].block_id,
    }
    assert {block.block_id for block in found} == {blocks[5].block_id, blocks[6].block_id}
    assert scoped.hits[-1].block.block_id == blocks[6].block_id
    assert scoped.hits[-1].rank == len(scoped.hits)


def test_agent_preserves_cooldown_guard_failure_in_trace(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    model = _ChatModel(responses=[])

    def stop_for_safety():
        raise TimeoutError("cooldown guard stopped")

    with pytest.raises(AgentRunFailure) as failure:
        run_bounded_reading_agent(
            workflow,
            "What details support the method?",
            "en",
            result,
            (blocks[1],),
            model,
            count_chat_input_tokens=lambda _payload: 100,
            cooldown=stop_for_safety,
        )

    assert failure.value.trace["cooldown_error_type"] == "TimeoutError"
    assert failure.value.trace["model_calls_attempted"] == 1
    assert failure.value.trace["model_calls_completed"] == 0
    assert failure.value.trace["model_call_records"][0]["status"] == (
        "not_dispatched_cooldown_failed"
    )


def test_agent_stops_before_an_over_budget_call_and_keeps_admitted_context(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_sources",
                        "args": {"query": "method details"},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    counts = iter((100, 10_001))

    run = run_bounded_reading_agent(
        workflow,
        "What details support the method?",
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=lambda _payload: next(counts),
    )

    assert model.call_count == 1
    assert run.trace["budget_stop_reason"] == "per_call_input_limit"
    assert [block.ordinal for block in run.context.blocks] == [2]


def test_agent_stops_after_repeating_the_same_search_call(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    repeated_search = {
        "name": "search_sources",
        "args": {"query": "new method condition query"},
        "id": "repeat",
        "type": "tool_call",
    }
    model = _ChatModel(
        responses=[
            AIMessage(content="", tool_calls=[repeated_search]),
            AIMessage(content="", tool_calls=[repeated_search]),
        ]
    )

    run = run_bounded_reading_agent(
        workflow,
        "What details support the method?",
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=lambda _payload: 100,
    )

    assert model.call_count == 2
    assert workflow._retriever.queries == ["new method condition query"]
    assert run.trace["budget_stop_reason"] == "repeated_tool_call"
    assert [event["status"] for event in run.trace["tool_trace"]] == [
        "ok",
        "repeated_tool_call",
    ]
    assert {event["actor"] for event in run.trace["tool_trace"]} == {"parent"}


def test_delegated_reader_uses_a_scoped_nonrecursive_context_and_shared_budget(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    method = next(
        section for section in workflow.source_outline().sections if section.title == "Method"
    )
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "delegate_reading",
                        "args": {
                            "task": "Find the method conditions and quote the source.",
                            "source_scope": method.section_id,
                        },
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_source",
                        "args": {"source_id": blocks[1].block_id},
                        "id": "2",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_source",
                        "args": {"source_id": blocks[3].block_id},
                        "id": "3",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The reader found the method condition in the scoped source."),
            AIMessage(content="Delegated evidence collected."),
        ]
    )
    counted_payloads = []

    def count_tokens(payload):
        counted_payloads.append(payload)
        return 100

    run = run_bounded_reading_agent(
        workflow,
        "What details support the method?",
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=count_tokens,
        delegated=True,
    )

    assert model.call_count == 5
    assert len(counted_payloads) == 5
    assert run.trace["architecture"] == "delegated_readers"
    assert run.trace["reader_tasks_started"] == 1
    assert run.trace["tool_calls_recorded"] == 3
    assert run.trace["model_calls_attempted"] == 5
    assert run.trace["agent_output_tokens_reserved"] == 5 * 512
    assert [event["tool"] for event in run.trace["tool_trace"]] == [
        "delegate_reading",
        "read_source",
        "read_source",
    ]
    assert [event["actor"] for event in run.trace["tool_trace"]] == [
        "parent",
        "reader:1",
        "reader:1",
    ]
    assert run.trace["tool_trace"][1]["status"] == "outside_reader_scope"
    assert run.trace["tool_trace"][2]["block_id"] == blocks[3].block_id
    assert run.context.blocks


def test_delegated_reader_follows_section_next_cursor(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    method = next(
        section for section in workflow.source_outline().sections if section.title == "Method"
    )
    _section, candidates = workflow.read_outline_section(method.section_id, after_ordinal=0)
    assert len(candidates) > 1
    first_cursor = candidates[0].ordinal
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "delegate_reading",
                        "args": {
                            "task": "Read the complete Method section.",
                            "source_scope": method.section_id,
                        },
                        "id": "delegate-method-section",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_source",
                        "args": {"source_id": method.section_id, "cursor": 0},
                        "id": "read-method-first-page",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_source",
                        "args": {
                            "source_id": method.section_id,
                            "cursor": first_cursor,
                        },
                        "id": "read-method-next-page",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The Method section contains the requested evidence."),
            AIMessage(content="Delegated evidence collected."),
        ]
    )

    run = run_bounded_reading_agent(
        workflow,
        "What details support the method?",
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=lambda _payload: 100,
        delegated=True,
    )

    assert model.call_count == 5
    assert run.trace["reader_tasks_started"] == 1
    assert run.trace["budget_stop_reason"] is None
    assert [event["status"] for event in run.trace["tool_trace"]] == [
        "completed",
        "ok",
        "ok",
    ]
    assert [event["actor"] for event in run.trace["tool_trace"]] == [
        "parent",
        "reader:1",
        "reader:1",
    ]


def test_delegated_reader_stops_repeated_rejected_source_id(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    method = next(
        section for section in workflow.source_outline().sections if section.title == "Method"
    )
    outside_source_id = blocks[1].block_id
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "delegate_reading",
                        "args": {
                            "task": "Read only the Method section.",
                            "source_scope": method.section_id,
                        },
                        "id": "delegate-scoped-method",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_source",
                        "args": {"source_id": outside_source_id, "cursor": 0},
                        "id": "read-outside-source",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_source",
                        "args": {"source_id": outside_source_id, "cursor": 1},
                        "id": "repeat-outside-source",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    run = run_bounded_reading_agent(
        workflow,
        "What details support the method?",
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=lambda _payload: 100,
        delegated=True,
    )

    assert model.call_count == 3
    assert run.trace["reader_tasks_started"] == 1
    assert run.trace["budget_stop_reason"] == "repeated_reader_scope_violation"
    assert [event["status"] for event in run.trace["tool_trace"]] == [
        "budget_stopped",
        "outside_reader_scope",
        "cursor_out_of_range",
    ]
    assert run.trace["tool_trace"][-1]["repeated_rejected_source"] is True


def test_delegated_reader_search_uses_request_gpu_session_from_nested_worker_threads(
    tmp_path, monkeypatch
):
    workflow, result, blocks = _setup(tmp_path)
    method = next(
        section for section in workflow.source_outline().sections if section.title == "Method"
    )
    method_block = next(
        block for block in blocks if method.start_ordinal <= block.ordinal <= method.end_ordinal
    )
    scoped_result = result.model_copy(
        update={"hits": (SearchHit(rank=1, score=1.0, block=method_block),)}
    )

    class FakeEmbeddingServer:
        instances = []

        def __init__(self, *_args, **kwargs):
            self.startup_ms = 1.0
            self.closed = False
            self.temperature_limit = kwargs["max_gpu_temperature_c"]
            self.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.closed = True

        def diagnostics(self):
            return {"gpu_temperature_limit_c": self.temperature_limit}

    FakeEmbeddingServer.instances.clear()
    monkeypatch.setattr(gpu_retrieval, "verify_native_model", lambda _path: None)
    monkeypatch.setattr(gpu_retrieval, "NativeEmbeddingEncoder", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(gpu_retrieval, "NativeEmbeddingServer", FakeEmbeddingServer)
    monkeypatch.setattr(
        gpu_retrieval,
        "document_retriever",
        lambda *_args, **_kwargs: _Retriever(scoped_result),
    )
    retriever = StagedGpuSearch(
        workflow._store,
        workflow.document_id,
        "hybrid",
        tmp_path / "embedding.gguf",
        "GPU-test",
    )
    workflow._retriever = retriever
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "delegate_reading",
                        "args": {
                            "task": "Read the relevant Method evidence.",
                            "source_scope": method.section_id,
                        },
                        "id": "parent-delegate",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_sources",
                        "args": {"query": "method quantitative details"},
                        "id": "reader-search",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_source",
                        "args": {"source_id": method_block.block_id, "cursor": 0},
                        "id": "reader-read",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The scoped Method evidence was inspected."),
            AIMessage(content="The reader reported the relevant method evidence."),
        ]
    )

    with retriever.session():
        run = run_bounded_reading_agent(
            workflow,
            "What details support the method?",
            "en",
            result,
            (blocks[1],),
            model,
            count_chat_input_tokens=lambda _payload: 100,
            delegated=True,
        )

    assert [event["status"] for event in run.trace["tool_trace"]] == [
        "completed",
        "ok",
        "ok",
    ]
    assert [event["actor"] for event in run.trace["tool_trace"]] == [
        "parent",
        "reader:1",
        "reader:1",
    ]
    assert retriever.last_trace["query_count"] == 1
    assert len(FakeEmbeddingServer.instances) == 1
    assert FakeEmbeddingServer.instances[0].closed
    assert retriever._slot.active is None
