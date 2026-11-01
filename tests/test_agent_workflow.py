from __future__ import annotations

import json

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import page_cite_rag.gpu_retrieval as gpu_retrieval
from page_cite_rag.agent import (
    AgentContext,
    AgentRunFailure,
    _block_text_in_mapped_group,
    _search_in_section,
    _search_preview,
    fixed_section_supplement,
    run_bounded_reading_agent,
)
from page_cite_rag.full_answer import FullAnswerWorkflow
from page_cite_rag.generation import (
    GenerationConfig,
    GenerationResult,
    GenerationStats,
    PromptBudget,
)
from page_cite_rag.gpu_retrieval import StagedGpuSearch
from page_cite_rag.hybrid import HybridRetriever
from page_cite_rag.ingest import prepare_source
from page_cite_rag.retrieval import SearchContextGroup, SearchHit, SearchResult
from page_cite_rag.store import SQLiteDocumentStore
from page_cite_rag.tokenize import LexicalTokenizer


def _reading_report_message(
    block=None, *, source_id=None, source_ids=None, observation=None, call_id="reading-report"
):
    findings = (
        []
        if block is None
        else [
            {
                "need": "the requested fact",
                "observation": observation or "The source supplies this fact.",
                "source_ids": source_ids or [source_id or block.block_id],
            }
        ]
    )
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "ReadingReport",
                "args": {
                    "requirements": findings,
                    "unresolved": [] if findings else ["the requested fact"],
                },
                "id": call_id,
                "type": "tool_call",
            }
        ],
        usage_metadata={"input_tokens": 100, "output_tokens": 8, "total_tokens": 108},
    )


def test_writer_reserve_deadline_is_recoverable_only_before_dispatch():
    from page_cite_rag.application import _recoverable_navigation_failure

    trace = {
        "budget_stop_reason": "writer_reserve_deadline",
        "cooldown_error_type": None,
        "navigator_terminal_state": {
            "status": "budget_stopped",
            "pending_tool_call_count": 0,
            "cancelled_or_timed_out": False,
        },
        "tool_trace": [],
        "model_call_records": [
            {
                "status": "completed",
                "input_tokens_exact": 100,
                "output_tokens_observed": 30,
            },
            {
                "status": "blocked_before_dispatch",
                "budget_stop_reason": "writer_reserve_deadline",
            },
        ],
    }
    failure = AgentRunFailure("AgentBudgetExceeded", trace)

    assert _recoverable_navigation_failure(failure)
    trace["model_call_records"][-1]["status"] = "dispatched"
    assert not _recoverable_navigation_failure(failure)


class _ChatModel(BaseChatModel):
    responses: list[AIMessage]
    call_count: int = 0
    seen_message_batches: list[list] = []
    bound_tool_names: list[list[str]] = []

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names.append(
            [
                item.get("function", {}).get("name", item.get("name", ""))
                if isinstance(item, dict)
                else item.name
                for item in tools
            ]
        )
        return self

    @property
    def _llm_type(self) -> str:
        return "scripted-agent-test"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.call_count += 1
        self.seen_message_batches.append(list(messages))
        message = self.responses.pop(0)
        if "finish_reason" not in message.response_metadata:
            message = message.model_copy(
                update={
                    "response_metadata": {
                        **message.response_metadata,
                        "finish_reason": "tool_calls" if message.tool_calls else "stop",
                    }
                }
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


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
        GenerationConfig(num_predict=3072),
    )
    return workflow, result, blocks


@pytest.mark.parametrize("invalid", [None, "source", "second_source", "blank"])
def test_reading_report_keeps_long_findings_and_validates_originals(tmp_path, invalid):
    from page_cite_rag.agent import _AgentBudgetLedger

    workflow, result, blocks = _setup(tmp_path)
    block = blocks[1]
    model = _ChatModel(
        responses=[
            _reading_report_message(
                block,
                source_id=blocks[6].block_id if invalid == "source" else None,
                source_ids=[block.block_id, blocks[6].block_id]
                if invalid == "second_source"
                else None,
                observation=" " if invalid == "blank" else "A concise source observation. " * 10,
            )
        ]
    )
    ledger = _AgentBudgetLedger(
        model_call_limit=2,
        call_input_limit=12000,
        total_input_limit=24000,
        call_output_limit=3072,
        total_output_limit=6144,
    )
    kwargs = dict(
        count_chat_input_tokens=lambda _payload: 100, max_model_calls=1, shared_ledger=ledger
    )
    if invalid == "blank":
        with pytest.raises(AgentRunFailure) as failure:
            run_bounded_reading_agent(
                workflow, result.query, "en", result, (block,), model, **kwargs
            )
        assert failure.value.error_type == "InvalidStructuredAnswerSchema"
        assert failure.value.trace["model_calls_completed"] == 1
    else:
        run = run_bounded_reading_agent(
            workflow, result.query, "en", result, (block,), model, **kwargs
        )
        if invalid:
            assert run.trace["navigation_report_partial"] is True
            assert run.trace["navigation_handoff_note_status"] == "partial_source_validation"
            assert run.structured_response.requirements == []
            assert run.structured_response.unresolved == ["the requested fact"]
            assert run.trace["report_validation_events"][0]["invalid_source_ids"]
            assert run.trace["model_calls_completed"] == 1
        else:
            assert len(run.navigation_note) > 200
            assert run.trace["navigation_handoff_note_status"] == "present"
            assert run.structured_response.requirements[0].source_ids == [block.block_id]


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
                            "max_blocks": 1,
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
                            "max_blocks": 1,
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
    assert sum("tools" in payload for payload in token_count_payloads) == 4
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


def test_shared_agent_entry_rejects_mismatched_search_before_model_use(tmp_path):
    from page_cite_rag.application import ask_with_agent

    workflow, result, blocks = _setup(tmp_path)
    for changed in ({"query": "different question"}, {"document_id": "f" * 64}):
        with pytest.raises(ValueError, match="bound document and question"):
            ask_with_agent(
                workflow,
                "What does the method say?",
                initial_search=result.model_copy(update=changed),
                initial_blocks=blocks,
            )


@pytest.mark.parametrize("report_failure", ["length", "stop", "schema", "deadline"])
def test_shared_agent_recovers_navigation_without_report_from_verified_originals(
    tmp_path, monkeypatch, report_failure
):
    from contextlib import contextmanager

    import page_cite_rag.agent as agent_module
    from page_cite_rag.agent import _spans
    from page_cite_rag.application import ask_with_agent

    workflow, result, blocks = _setup(tmp_path)
    initial = (blocks[1],)

    class WriterModel:
        identity = {"model": "scripted-writer-test"}

        def __init__(self):
            self.config = workflow._budget_config
            self.calls = []

        def count_chat_input_tokens(self, _payload):
            return 100

        def prepare_budget(self, _messages, _schema):
            return PromptBudget(
                input_upper_estimate=100 + self.config.template_reserve,
                output_reserved=self.config.num_predict,
                context_limit=self.config.num_ctx,
                method="scripted-exact-token-test/v1",
                exact_input_tokens=100,
            )

        def generate(self, messages, response_model):
            self.calls.append(messages)
            source_id = next(iter(_spans(initial)))
            value = response_model.model_validate(
                {
                    "status": "answered",
                    "answer_claims": (
                        {
                            "text": "The source paragraph is available.",
                            "citations": ({"source_id": source_id},),
                        },
                    ),
                    "detail_claims": (),
                    "limitations": (),
                }
            )
            stats = GenerationStats(
                prompt_eval_count=100,
                eval_count=12,
                total_duration=100,
                load_duration=None,
                prompt_eval_duration=40,
                eval_duration=60,
                done_reason="stop",
                elapsed_ms=1.0,
            )
            return GenerationResult(
                value,
                stats,
                self.prepare_budget(messages, response_model.model_json_schema()),
                value.model_dump_json(),
            )

    writer = WriterModel()
    workflow._model = writer

    invalid_text = "INVALID_REPORT_MUST_NOT_REACH_WRITER"
    if report_failure == "deadline":
        message = _read_call(initial[0].block_id, max_blocks=1).model_copy(
            update={
                "usage_metadata": {"input_tokens": 100, "output_tokens": 8, "total_tokens": 108}
            }
        )
    elif report_failure == "schema":
        message = _reading_report_message(initial[0], observation=" ")
    else:
        message = AIMessage(
            content=invalid_text,
            response_metadata={"finish_reason": report_failure},
            usage_metadata={"input_tokens": 100, "output_tokens": 3072, "total_tokens": 3172},
        )
    navigator = _ChatModel(responses=[message])
    if report_failure == "deadline":
        writer.remaining_request_timeout_seconds = lambda: 60 if navigator.call_count else 600

    @contextmanager
    def provide_chat_model(_model, *, on_request=None):
        yield navigator

    monkeypatch.setattr(agent_module, "local_chat_model", provide_chat_model)
    answer = ask_with_agent(
        workflow,
        result.query,
        language="en",
        initial_search=result,
        initial_blocks=initial,
    )

    assert len(writer.calls) == 1
    user_payload = json.loads(writer.calls[0][1].content)
    assert user_payload["navigation_handoff"]["reading_report"]["requirements"] == []
    assert user_payload["navigation_handoff"]["reading_report"]["unresolved"] == [result.query]
    assert "failure_evidence" not in user_payload["navigation_handoff"]
    assert invalid_text not in writer.calls[0][1].content
    assert navigator.call_count == 1
    assert answer.result.agent_usage.model_calls == 2
    assert answer.result.agent_usage.output_usage_complete is True
    assert answer.trace["agent"]["navigator_terminal_state"]["valid"] is False
    assert answer.trace["agent"]["navigation_recovery"] == (
        "verified_originals_after_incomplete_structured_report"
    )
    assert answer.result.status == "partial"
    assert answer.result.stop_reason == "partial"
    assert answer.result.included_block_ids == (initial[0].block_id,)


def test_shared_agent_does_not_recover_a_source_scope_violation(tmp_path, monkeypatch):
    from contextlib import contextmanager

    import page_cite_rag.agent as agent_module
    from page_cite_rag.application import ask_with_agent

    workflow, result, blocks = _setup(tmp_path)
    context = (blocks[1],)
    workflow._model.count_chat_input_tokens = lambda _payload: 100

    @contextmanager
    def provide_chat_model(_model, *, on_request=None):
        yield object()

    def fail_navigation(*_args, **_kwargs):
        raise AgentRunFailure(
            "InvalidAgentToolExecution",
            {
                "navigator_terminal_state": {"reason": "invalid_tool_execution"},
                "tool_trace": [{"tool": "read_source", "status": "outside_reader_scope"}],
            },
        )

    monkeypatch.setattr(agent_module, "local_chat_model", provide_chat_model)
    monkeypatch.setattr(agent_module, "run_bounded_reading_agent", fail_navigation)
    with pytest.raises(AgentRunFailure, match="bounded local agent run failed"):
        ask_with_agent(
            workflow,
            result.query,
            language="en",
            initial_search=result,
            initial_blocks=context,
        )


@pytest.mark.parametrize(
    "failure_mode", ["pending_data_tool", "invalid_tool", "unknown_usage", "gpu_guard", "cancelled"]
)
def test_shared_agent_keeps_unsafe_navigation_failures_closed(tmp_path, monkeypatch, failure_mode):
    from contextlib import contextmanager

    import page_cite_rag.agent as agent_module
    from page_cite_rag.application import ask_with_agent
    from page_cite_rag.generation import ModelError, ModelTimeoutError

    workflow, result, blocks = _setup(tmp_path)
    writer_calls = []
    workflow._model.count_chat_input_tokens = lambda _payload: 100

    def unexpected_writer(*args, **kwargs):
        writer_calls.append(args)
        raise AssertionError("unsafe navigation must not dispatch the writer")

    workflow._model.generate = unexpected_writer
    message = AIMessage(content="Incomplete navigation.")
    if failure_mode == "pending_data_tool":
        report = _reading_report_message(blocks[1], observation=" ")
        message = report.model_copy(
            update={"tool_calls": [*report.tool_calls, *_read_call(blocks[3].block_id).tool_calls]}
        )
    elif failure_mode == "invalid_tool":
        message = AIMessage(
            content="",
            invalid_tool_calls=[
                {"name": "read_source", "args": "{", "id": "broken", "error": "Invalid JSON"}
            ],
            usage_metadata={"input_tokens": 100, "output_tokens": 8, "total_tokens": 108},
        )
    navigator = _ChatModel(responses=[message])
    if failure_mode in {"gpu_guard", "cancelled"}:

        def stop_before_dispatch():
            error = ModelError if failure_mode == "gpu_guard" else ModelTimeoutError
            raise error("GPU guard or cancellation stopped the request before dispatch")

        workflow._model.wait_for_gpu_headroom = stop_before_dispatch

    @contextmanager
    def provide_chat_model(_model, *, on_request=None):
        yield navigator

    monkeypatch.setattr(agent_module, "local_chat_model", provide_chat_model)
    with pytest.raises(AgentRunFailure) as failure:
        ask_with_agent(
            workflow,
            result.query,
            language="en",
            initial_search=result,
            initial_blocks=(blocks[1],),
        )

    assert not writer_calls
    trace = failure.value.trace
    if failure_mode in {"gpu_guard", "cancelled"}:
        assert navigator.call_count == 0
        assert trace["model_calls_completed"] == 0
        assert trace["agent_input_tokens_exact"] == 0
        assert trace["agent_output_tokens_unobserved_reserved"] == 0
        assert trace["model_call_records"][0]["status"] == "not_dispatched_cooldown_failed"
    elif failure_mode in {"pending_data_tool", "invalid_tool"}:
        assert navigator.call_count == 1
        assert trace["navigator_terminal_state"]["pending_data_tool_call_count"] == 1
    else:
        assert trace["agent_output_tokens_observed"] is None


def test_named_tool_token_count_matches_langchain_wire_request(tmp_path, monkeypatch):
    import httpx
    import langchain.agents
    from langchain.agents.middleware import wrap_model_call
    from langchain_openai import ChatOpenAI

    workflow, result, blocks = _setup(tmp_path)
    counted = []
    dispatched = []
    wire_mismatches = []
    original_factory = langchain.agents.create_agent

    @wrap_model_call
    def require_first_read(request, handler):
        if not dispatched:
            request = request.override(tool_choice="read_source")
        return handler(request)

    def create_agent(*args, **kwargs):
        kwargs["middleware"] = [require_first_read, *kwargs.get("middleware", ())]
        return original_factory(*args, **kwargs)

    monkeypatch.setattr(langchain.agents, "create_agent", create_agent)

    def count_tokens(payload):
        expected = (
            {"type": "function", "function": {"name": "read_source"}} if not dispatched else None
        )
        assert payload.get("tool_choice") == expected
        counted.append(payload)
        return 100

    def respond(request):
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        for field in ("messages", "tools", "tool_choice"):
            if payload.get(field) != counted[-1].get(field):
                wire_mismatches.append((field, counted[-1].get(field), payload.get(field)))
        dispatched.append(payload)
        first = len(dispatched) == 1
        message = (
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "read-1",
                        "type": "function",
                        "function": {
                            "name": "read_source",
                            "arguments": json.dumps(
                                {"source_id": blocks[2].block_id, "max_blocks": 1}
                            ),
                        },
                    }
                ],
            }
            if first
            else {"role": "assistant", "content": "Evidence collected."}
        )
        return httpx.Response(
            200,
            json={
                "id": f"chat-{len(dispatched)}",
                "object": "chat.completion",
                "model": "page-cite-rag-qwen35",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if first else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 8, "total_tokens": 108},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as client:
        model = ChatOpenAI(
            model="page-cite-rag-qwen35",
            base_url="http://localhost/v1",
            api_key="local-test",
            http_client=client,
            max_retries=0,
            temperature=0,
            use_responses_api=False,
        )
        try:
            run = run_bounded_reading_agent(
                workflow,
                "Read the method evidence.",
                "en",
                result,
                (blocks[1],),
                model,
                count_chat_input_tokens=count_tokens,
            )
        except AgentRunFailure as exc:
            pytest.fail(f"{exc.error_type}: {exc.trace['model_call_records']}")

    assert not wire_mismatches
    assert len(dispatched) == 2
    assert run.trace["tool_calls_recorded"] == 1
    assert run.trace["read_source_ids"] == [blocks[2].block_id]
    assert blocks[2].block_id in {block.block_id for block in run.context.blocks}
    assert "content" not in run.trace["model_call_records"][-1]


def test_local_chat_model_callback_captures_actual_http_payload(monkeypatch):
    import httpx

    from page_cite_rag.agent import local_chat_model

    observed = []
    captured = []
    original_client = httpx.Client

    def respond(request):
        payload = json.loads(request.content)
        observed.append(payload)
        return httpx.Response(
            200,
            json={
                "id": "wire-test",
                "object": "chat.completion",
                "created": 0,
                "model": "page-cite-rag-qwen35",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "captured"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14},
            },
        )

    class MockClient(original_client):
        def __init__(self, **kwargs):
            super().__init__(
                **kwargs,
                transport=httpx.MockTransport(respond),
            )

    monkeypatch.setattr(httpx, "Client", MockClient)

    class Model:
        def openai_compatible_credentials(self):
            return "http://127.0.0.1:1234/v1", "local-only"

    with local_chat_model(Model(), on_request=captured.append) as chat:
        result = chat.invoke("hello")

    assert result.content == "captured"
    assert len(observed) == len(captured) == 1
    assert captured[0]["method"] == "POST"
    assert captured[0]["path"] == "/v1/chat/completions"
    assert captured[0]["payload"] == observed[0]
    assert captured[0]["payload_sha256"]
    assert captured[0]["payload_bytes"] > 0
    assert "authorization" not in captured[0]


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
    assert run.trace["execution_mode"] == "sequential"
    assert run.trace["agent_budget_scope"] == (
        "navigation_parent_and_readers_only_final_writer_separate"
    )
    assert run.trace["navigator_terminal_state"]["status"] == "completed"
    assert run.trace["navigator_terminal_state"]["structured_response_expected"] is False
    assert run.trace["final_input_token_limit"] == 12_000
    assert run.trace["final_output_token_limit"] == workflow._budget_config.num_predict


def test_agent_rejects_terminal_state_with_pending_tool_call(tmp_path, monkeypatch):
    import langchain.agents

    workflow, result, blocks = _setup(tmp_path)

    class ReturningAgent:
        def invoke(self, _input, config=None):
            assert config["recursion_limit"] == 100_000
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_source",
                                "args": {"source_id": blocks[1].block_id},
                                "id": "unfinished-read",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            }

    monkeypatch.setattr(langchain.agents, "create_agent", lambda **_kwargs: ReturningAgent())

    with pytest.raises(AgentRunFailure) as failure:
        run_bounded_reading_agent(
            workflow,
            "Read the original source.",
            "en",
            result,
            (blocks[1],),
            object(),
            count_chat_input_tokens=lambda _payload: 100,
        )

    assert failure.value.error_type == "InvalidAgentTerminalState"
    terminal = failure.value.trace["navigator_terminal_state"]
    assert terminal["status"] == "invalid_terminal_state"
    assert terminal["reason"] == "terminal_tool_call_pending_or_invalid"
    assert terminal["terminal_tool_call_count"] == 1
    assert "messages" not in failure.value.trace


def test_integrated_agent_returns_verified_structured_answer_without_second_writer(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager

    import page_cite_rag.agent as agent_module
    from page_cite_rag.agent import _spans
    from page_cite_rag.application import ask_with_agent

    workflow, result, blocks = _setup(tmp_path)
    context = (blocks[1],)
    source_id = next(iter(_spans(context)))
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "IntegratedAgentAnswer",
                        "args": {
                            "status": "answered",
                            "answer_claims": [
                                {
                                    "text": "The method section contains a paragraph.",
                                    "citations": [{"source_id": source_id}],
                                }
                            ],
                            "detail_claims": [],
                            "limitations": [],
                        },
                        "id": "final-answer",
                        "type": "tool_call",
                    }
                ],
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 24,
                    "total_tokens": 124,
                },
            )
        ]
    )

    workflow._model.count_chat_input_tokens = lambda _payload: 100

    @contextmanager
    def provide_chat_model(_model):
        yield model

    monkeypatch.setattr(agent_module, "local_chat_model", provide_chat_model)
    try:
        answer = ask_with_agent(
            workflow,
            result.query,
            language="en",
            data_tools_enabled=False,
            initial_search=result,
            initial_blocks=context,
        )
    except AgentRunFailure as exc:
        pytest.fail(f"{exc.error_type}: {exc.trace['navigator_terminal_state']}")

    assert model.call_count == 1
    assert answer.trace["version"] == "integrated-reading-agent/v1"
    assert answer.trace["agent"]["architecture"] == "single_integrated_answer_agent"
    assert answer.trace["agent"]["response_strategy"] == "explicit_tool_strategy_no_retry"
    assert answer.trace["agent"]["agent_budget_scope"] == (
        "single_agent_tools_and_final_answer_shared_ledger"
    )
    assert answer.trace["agent"]["navigator_terminal_state"]["structured_response_present"] is True
    assert answer.trace["agent"]["final_model_call"]["output_token_limit"] == 3072
    assert answer.trace["agent"]["agent_output_tokens_observed"] == 24

    result = answer.result
    assert result.status == "answered"
    assert result.answer_claims[0].citations[0].block_id == context[0].block_id
    assert result.generation_stats is None
    assert result.agent_usage.model_calls == 1
    assert result.budget.exact_input_tokens == 100


@pytest.mark.parametrize(
    "read_new_source,cite_unread,repair_success,report_new_source",
    [
        (False, False, True, True),
        (True, False, True, True),
        (True, False, True, False),
        (False, True, True, True),
        (False, True, False, True),
    ],
)
def test_single_agent_uses_shared_writer_and_one_budget_ledger(
    tmp_path, monkeypatch, read_new_source, cite_unread, repair_success, report_new_source
):
    from contextlib import contextmanager

    import page_cite_rag.agent as agent_module
    from page_cite_rag.agent import _spans
    from page_cite_rag.application import ask_with_agent

    workflow, result, blocks = _setup(tmp_path)
    context = (blocks[1], blocks[3])
    source_id = next(iter(_spans(context)))
    searched_block = blocks[6]
    workflow._retriever = _Retriever(
        SearchResult(
            document_id=workflow.document_id,
            index_id="synthetic-index",
            method="bm25",
            query="result condition",
            hits=(SearchHit(rank=1, score=0.9, block=searched_block),),
            elapsed_ms=0.1,
        )
    )
    navigation_model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_sources",
                        "args": {
                            "query": "result condition",
                            "information_need": "the condition for the result",
                        },
                        "id": "search-condition",
                        "type": "tool_call",
                    }
                ],
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 8,
                    "total_tokens": 108,
                },
            ),
            *(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "read_source",
                                "args": {
                                    "source_id": f"S{searched_block.ordinal}",
                                    "information_need": "the condition for the result",
                                },
                                "id": "read-condition",
                                "type": "tool_call",
                            }
                        ],
                        usage_metadata={
                            "input_tokens": 100,
                            "output_tokens": 8,
                            "total_tokens": 108,
                        },
                    )
                ]
                if read_new_source
                else []
            ),
            _reading_report_message(
                context[0],
                source_ids=[context[0].block_id, searched_block.block_id]
                if (read_new_source and report_new_source) or cite_unread
                else None,
            ),
            *(
                [
                    _reading_report_message(
                        context[0],
                        source_ids=None if repair_success else [searched_block.block_id],
                        call_id="report-repair",
                    )
                ]
                if cite_unread
                else []
            ),
        ]
    )

    class WriterModel:
        identity = {"model": "scripted-writer-test"}

        def __init__(self):
            self.config = workflow._budget_config
            self.calls = []

        def count_chat_input_tokens(self, payload):
            user = next(message for message in payload["messages"] if message["role"] == "user")
            evidence = json.loads(user["content"]).get("initial_evidence", {}).get("evidence", [])
            return 13_000 if len(evidence) > 1 else 100

        def prepare_budget(self, _messages, _schema):
            evidence = json.loads(_messages[1].content).get("evidence", [])
            if not report_new_source and len(evidence) > 2:
                from page_cite_rag.generation import ContextBudgetError

                raise ContextBudgetError("only two complete originals fit this test budget")
            return PromptBudget(
                input_upper_estimate=100 + self.config.template_reserve,
                output_reserved=self.config.num_predict,
                context_limit=self.config.num_ctx,
                method="scripted-exact-token-test/v1",
                exact_input_tokens=100,
            )

        def generate(self, messages, response_model):
            self.calls.append(messages)
            value = response_model.model_validate(
                {
                    "status": "answered",
                    "answer_claims": (
                        {
                            "text": "The method section contains a paragraph.",
                            "citations": ({"source_id": source_id},),
                        },
                    ),
                    "detail_claims": (),
                    "limitations": (),
                }
            )
            stats = GenerationStats(
                prompt_eval_count=100,
                eval_count=20,
                total_duration=100,
                load_duration=None,
                prompt_eval_duration=40,
                eval_duration=60,
                done_reason="stop",
                elapsed_ms=1.0,
            )
            return GenerationResult(
                value,
                stats,
                self.prepare_budget(messages, response_model.model_json_schema()),
                value.model_dump_json(),
            )

    writer = WriterModel()
    workflow._model = writer
    workflow._model.count_chat_input_tokens = writer.count_chat_input_tokens

    @contextmanager
    def provide_navigation_model(_model):
        yield navigation_model

    monkeypatch.setattr(agent_module, "local_chat_model", provide_navigation_model)
    answer = ask_with_agent(
        workflow,
        result.query,
        language="en",
        initial_search=result,
        initial_blocks=context,
    )
    extra_navigation = read_new_source or cite_unread
    search_packet = json.loads(navigation_model.seen_message_batches[1][-1].content)
    assert search_packet["hits"][0]["source_id"] == f"S{searched_block.ordinal}"
    assert navigation_model.call_count == (3 if extra_navigation else 2)
    assert len(writer.calls) == 1
    assert answer.trace["version"] == "agentic-reading-v2"
    assert answer.trace["agent"]["model_calls_attempted"] == (4 if extra_navigation else 3)
    assert answer.trace["agent"]["tool_calls_recorded"] == (2 if read_new_source else 1)
    assert answer.result.agent_usage.model_calls == (4 if extra_navigation else 3)
    assert answer.result.agent_usage.input_tokens == (400 if extra_navigation else 300)
    assert answer.result.agent_usage.final_input_tokens == 100
    assert answer.result.generation_stats.eval_count == 20
    assert answer.result.answer_claims[0].citations[0].block_id == context[0].block_id
    writer_payload = json.loads(writer.calls[0][1].content)
    assert writer_payload["navigation_handoff"]["status"] == "unverified"
    lookups = writer_payload["navigation_handoff"]["lookups"]
    if read_new_source:
        assert f"S{searched_block.ordinal}" in {
            id for lookup in lookups for id in lookup["source_ids"]
        }
    else:
        assert lookups == []
    findings = writer_payload["navigation_handoff"]["reading_report"]["requirements"]
    if not repair_success:
        assert findings == []
        assert answer.result.status == "partial"
        assert answer.trace["agent"]["navigation_report_partial"] is True
        assert (
            answer.trace["agent"]["navigation_handoff_note_status"] == "partial_source_validation"
        )
        assert writer_payload["navigation_handoff"]["reading_report"]["unresolved"]
        assert answer.result.limitations
    else:
        assert findings
        assert findings[0]["observation"] == "The source supplies this fact."
        assert answer.trace["agent"]["navigation_handoff_note_status"] == "present"
        assert len(findings[0]["source_ids"]) == (2 if read_new_source and report_new_source else 1)
    if cite_unread:
        events = answer.trace["agent"]["report_validation_events"]
        assert len(events) == (1 if repair_success else 2)
        assert searched_block.block_id in events[0]["invalid_source_ids"]
        assert navigation_model.bound_tool_names[-1] == ["ReadingReport"]
    assert answer.trace["agent"]["bootstrap_omitted_block_ids"] == [context[1].block_id]
    shown = {item["source_id"] for item in writer_payload["evidence"]}
    assert set(_spans((context[0],))).issubset(shown)
    assert context[0].block_id in answer.result.included_block_ids
    assert (context[1].block_id in answer.result.included_block_ids) is report_new_source
    assert (searched_block.block_id in answer.result.included_block_ids) is read_new_source


def test_shared_navigation_reserves_the_last_model_call_for_handoff(tmp_path):
    from page_cite_rag.agent import _AgentBudgetLedger

    workflow, result, blocks = _setup(tmp_path)
    model = _ChatModel(
        responses=[
            *[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_outline",
                            "args": {"cursor": cursor},
                            "id": f"outline-{cursor}",
                            "type": "tool_call",
                        }
                    ],
                    usage_metadata={
                        "input_tokens": 100,
                        "output_tokens": 5,
                        "total_tokens": 105,
                    },
                )
                for cursor in range(4)
            ],
            _reading_report_message(),
        ]
    )
    ledger = _AgentBudgetLedger(
        model_call_limit=5,
        call_input_limit=12_000,
        total_input_limit=72_000,
        call_output_limit=3_072,
        total_output_limit=18_432,
    )
    counted_payloads = []

    def count_tokens(payload):
        counted_payloads.append(payload)
        return 100

    run = run_bounded_reading_agent(
        workflow,
        result.query,
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=count_tokens,
        max_model_calls=5,
        shared_ledger=ledger,
    )

    assert run.trace["model_calls_completed"] == 5
    assert run.trace["tool_calls_recorded"] == 4
    assert ledger.model_calls == 5
    assert len(counted_payloads) == 5
    assert all(payload["tool_choice"] == "required" for payload in counted_payloads)
    assert [tool["function"]["name"] for tool in counted_payloads[-1]["tools"]] == ["ReadingReport"]
    assert counted_payloads[-1]["tool_choice"] == "required"


def test_shared_navigation_hands_off_after_repeating_a_visible_tool_result(tmp_path):
    from page_cite_rag.agent import _AgentBudgetLedger

    workflow, result, blocks = _setup(tmp_path)
    target = blocks[3].block_id
    model = _ChatModel(
        responses=[
            _read_call(target, max_blocks=1),
            _read_call(target, max_blocks=1, call_id="read-again"),
            _reading_report_message(),
        ]
    )
    ledger = _AgentBudgetLedger(
        model_call_limit=5,
        call_input_limit=12_000,
        total_input_limit=None,
        call_output_limit=3_072,
        total_output_limit=None,
    )
    counted_payloads = []

    run = run_bounded_reading_agent(
        workflow,
        result.query,
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=lambda payload: counted_payloads.append(payload) or 100,
        max_model_calls=5,
        shared_ledger=ledger,
    )

    assert [event["status"] for event in run.trace["tool_trace"]] == [
        "ok",
        "repeated_tool_call",
    ]
    assert run.trace["navigation_handoff_requested"] is True
    assert run.trace["navigation_handoff_reason"] == "repeated_tool_no_progress"
    assert run.trace["model_calls_completed"] == 3
    assert run.structured_response is not None
    model_payloads = [payload for payload in counted_payloads if "tools" in payload]
    assert all(payload["tool_choice"] == "required" for payload in model_payloads)
    assert [tool["function"]["name"] for tool in model_payloads[-1]["tools"]] == ["ReadingReport"]


def test_shared_navigation_restores_an_evicted_read_once(tmp_path, monkeypatch):
    import page_cite_rag.agent as agent_module
    from page_cite_rag.agent import _AgentBudgetLedger

    workflow, result, blocks = _setup(tmp_path)
    first, second = blocks[3].block_id, blocks[4].block_id
    model = _ChatModel(
        responses=[
            _read_call(first, max_blocks=1),
            _read_call(second, max_blocks=1, call_id="read-second"),
            _read_call(first, max_blocks=1, call_id="restore-first"),
            _read_call(first, max_blocks=1, call_id="repeat-restored"),
            _reading_report_message(),
        ]
    )
    original_admit = agent_module._admit

    def admit_one_block(workflow, question, language, candidates, context_groups):
        admitted = original_admit(workflow, question, language, candidates, context_groups)
        if len(admitted.blocks) < 2:
            return admitted
        selected = admitted.blocks[0]
        return agent_module.AgentContext(
            (selected,),
            (),
            admitted.exact_input_tokens,
            tuple(
                dict.fromkeys(
                    (*admitted.omitted_block_ids, *(item.block_id for item in admitted.blocks[1:]))
                )
            ),
        )

    monkeypatch.setattr(agent_module, "_admit", admit_one_block)
    ledger = _AgentBudgetLedger(
        model_call_limit=6,
        call_input_limit=12_000,
        total_input_limit=None,
        call_output_limit=3_072,
        total_output_limit=None,
    )

    run = run_bounded_reading_agent(
        workflow,
        result.query,
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=lambda _payload: 100,
        max_model_calls=6,
        shared_ledger=ledger,
    )

    events = run.trace["tool_trace"]
    assert [event["status"] for event in events] == ["ok", "ok", "ok", "repeated_tool_call"]
    assert events[2]["restoration_of_tool_event"] == 0
    assert events[3]["previous_tool_event"] == 0
    assert run.trace["navigation_handoff_requested"] is True
    assert run.trace["model_calls_completed"] == 5


def test_shared_agent_has_no_low_per_case_call_tool_or_cumulative_token_caps(tmp_path):
    from page_cite_rag.agent import _AgentBudgetLedger, _spans

    workflow, result, blocks = _setup(tmp_path)
    source_id = next(iter(_spans((blocks[1],))))
    searches = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_sources",
                    "args": {
                        "query": f"distinct quality probe {index}",
                        "information_need": "locate relevant source context",
                    },
                    "id": f"search-{index}",
                    "type": "tool_call",
                }
            ],
            usage_metadata={
                "input_tokens": 7_000 if index == 0 else 10_000,
                "output_tokens": 8,
                "total_tokens": (7_000 if index == 0 else 10_000) + 8,
            },
        )
        for index in range(9)
    ]
    model = _ChatModel(responses=[*searches, _reading_report_message(blocks[1])])
    ledger = _AgentBudgetLedger(
        model_call_limit=None,
        call_input_limit=12_000,
        total_input_limit=None,
        call_output_limit=3_072,
        total_output_limit=None,
        tool_call_limit=None,
    )

    run = run_bounded_reading_agent(
        workflow,
        result.query,
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=lambda payload: (
            7_000
            if not any(message["role"] == "tool" for message in payload["messages"])
            else 10_000
        ),
        max_model_calls=None,
        shared_ledger=ledger,
    )

    assert model.call_count == 10
    assert run.trace["model_calls_completed"] == 10
    assert run.trace["tool_calls_recorded"] == 9
    assert run.trace["bootstrap_input_tokens_exact"] == 7_000
    assert run.trace["bootstrap_input_token_limit"] == 7_500
    assert run.trace["agent_input_tokens_exact"] == 97_000
    assert run.trace["model_call_limit"] is None
    assert run.trace["navigation_model_call_limit"] is None
    assert run.trace["tool_call_limit"] is None
    assert run.trace["agent_input_token_limit"] is None
    assert run.trace["agent_output_token_limit"] is None
    assert run.structured_response.requirements[0].source_ids == [blocks[1].block_id]
    assert source_id


def test_integrated_agent_can_answer_from_frozen_context_without_data_tools(tmp_path):
    from page_cite_rag.agent import _spans

    workflow, result, blocks = _setup(tmp_path)
    source_id = next(iter(_spans((blocks[1],))))
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "IntegratedAgentAnswer",
                        "args": {
                            "status": "answered",
                            "answer_claims": [
                                {
                                    "text": "The method section contains a paragraph.",
                                    "citations": [{"source_id": source_id}],
                                }
                            ],
                            "detail_claims": [],
                            "limitations": [],
                        },
                        "id": "diagnostic-final",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    )
    run = run_bounded_reading_agent(
        workflow,
        result.query,
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=lambda _payload: 100,
        max_model_calls=1,
        final_answer=True,
        data_tools_enabled=False,
    )

    assert model.call_count == 1
    assert model.bound_tool_names[-1] == ["IntegratedAgentAnswer"]
    assert run.trace["data_tools_enabled"] is False
    assert run.trace["data_tool_names"] == []
    assert run.trace["tool_calls_recorded"] == 0
    assert run.structured_response is not None


def test_integrated_answer_validation_failure_keeps_completed_model_usage(tmp_path, monkeypatch):
    from contextlib import contextmanager

    import page_cite_rag.agent as agent_module
    from page_cite_rag.agent import _spans
    from page_cite_rag.application import ask_with_agent

    workflow, result, blocks = _setup(tmp_path)
    source_id = next(iter(_spans((blocks[1],))))
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "IntegratedAgentAnswer",
                        "args": {
                            "status": "answered",
                            "answer_claims": [
                                {
                                    "text": "The method contains a paragraph.",
                                    "citations": [{"source_id": source_id}],
                                }
                            ],
                            "detail_claims": [],
                            "limitations": [],
                        },
                        "id": "final-answer",
                        "type": "tool_call",
                    }
                ],
                usage_metadata={
                    "input_tokens": 100,
                    "output_tokens": 24,
                    "total_tokens": 124,
                },
            )
        ]
    )
    workflow._model.count_chat_input_tokens = lambda _payload: 100

    @contextmanager
    def provide_chat_model(_model):
        yield model

    def fail_validation(*_args, **_kwargs):
        raise RuntimeError("answer contract rejected")

    monkeypatch.setattr(agent_module, "local_chat_model", provide_chat_model)
    monkeypatch.setattr(workflow, "answer_from_agent_draft", fail_validation)

    with pytest.raises(AgentRunFailure) as failure:
        ask_with_agent(
            workflow,
            result.query,
            language="en",
            data_tools_enabled=False,
            initial_search=result,
            initial_blocks=(blocks[1],),
        )

    assert failure.value.error_type == "RuntimeError"
    assert failure.value.trace["failure_stage"] == "answer_validation"
    assert failure.value.trace["failure_type"] == "RuntimeError"
    assert failure.value.trace["model_calls_completed"] == 1
    assert failure.value.trace["model_call_records"][-1]["output_tokens_observed"] == 24


def test_integrated_agent_reads_original_source_then_finishes_structured_answer(tmp_path):
    from page_cite_rag.agent import _spans

    workflow, result, blocks = _setup(tmp_path)
    anchor = blocks[2]
    source_id = next(iter(_spans((anchor,))))
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_source",
                        "args": {
                            "source_id": anchor.block_id,
                            "max_blocks": 1,
                            "information_need": "What does the method paragraph establish?",
                        },
                        "id": "read-1",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "IntegratedAgentAnswer",
                        "args": {
                            "status": "answered",
                            "answer_claims": [
                                {
                                    "text": "The method section contains the cited paragraph.",
                                    "citations": [{"source_id": source_id}],
                                }
                            ],
                            "detail_claims": [],
                            "limitations": [],
                        },
                        "id": "final-2",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    call_events = []
    run = run_bounded_reading_agent(
        workflow,
        result.query,
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=lambda _payload: 100,
        max_model_calls=6,
        final_answer=True,
        on_model_call=call_events.append,
    )

    assert model.call_count == 2
    assert run.trace["model_calls_completed"] == 2
    assert run.trace["tool_calls_recorded"] == 1
    assert run.trace["tool_trace"][0]["tool"] == "read_source"
    assert run.trace["tool_trace"][0]["information_need"] == (
        "What does the method paragraph establish?"
    )
    assert [event["status"] for event in call_events] == [
        "admitted",
        "ready_to_dispatch",
        "dispatched",
        "completed",
        "admitted",
        "ready_to_dispatch",
        "dispatched",
        "completed",
    ]
    initial = json.loads(model.seen_message_batches[0][1].content)
    following = json.loads(model.seen_message_batches[-1][1].content)
    assert initial["requested_output_language"] == "en"
    assert initial["remaining_model_calls"] == 6
    assert following["remaining_model_calls"] == 5
    assert anchor.block_id in run.trace["read_source_ids"]
    assert anchor.block_id in {item.block_id for item in run.context.blocks}
    last_messages = model.seen_message_batches[-1]
    read_result = next(
        message for message in last_messages if getattr(message, "type", None) == "tool"
    )
    assert "citation_handles" in read_result.content
    assert source_id in read_result.content
    assert run.trace["navigator_terminal_state"]["structured_response_present"] is True


def test_integrated_agent_resolves_citation_alias_when_reading_source(tmp_path):
    from page_cite_rag.agent import _spans

    workflow, result, blocks = _setup(tmp_path)
    initial_block = blocks[1]
    source_id = next(iter(_spans((initial_block,))))
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_source",
                        "args": {"source_id": source_id, "max_blocks": 1},
                        "id": "read-citation-alias",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "IntegratedAgentAnswer",
                        "args": {
                            "status": "answered",
                            "answer_claims": [
                                {
                                    "text": "The cited source contains the requested detail.",
                                    "citations": [{"source_id": source_id}],
                                }
                            ],
                            "detail_claims": [],
                            "limitations": [],
                        },
                        "id": "final-after-alias-read",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    run = run_bounded_reading_agent(
        workflow,
        result.query,
        "en",
        result,
        (initial_block,),
        model,
        count_chat_input_tokens=lambda _payload: 100,
        max_model_calls=2,
        final_answer=True,
    )

    assert run.trace["tool_trace"][0]["status"] == "ok"
    assert run.trace["tool_trace"][0]["source_id"] == source_id
    assert run.trace["tool_trace"][0]["canonical_source_id"] == initial_block.block_id
    assert initial_block.block_id in run.trace["read_source_ids"]
    assert model.bound_tool_names[-1] == ["IntegratedAgentAnswer"]


def test_integrated_agent_uses_the_full_configured_input_and_output_budgets(tmp_path):
    from page_cite_rag.agent import _spans

    workflow, result, blocks = _setup(tmp_path)
    source_id = next(iter(_spans((blocks[1],))))
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_outline",
                    "args": {"cursor": 0},
                    "id": f"outline-page-{call_index}",
                    "type": "tool_call",
                }
            ],
        )
        for call_index in range(5)
    ]
    responses.append(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "IntegratedAgentAnswer",
                    "args": {
                        "status": "answered",
                        "answer_claims": [
                            {
                                "text": "The method section contains the requested detail.",
                                "citations": [{"source_id": source_id}],
                            }
                        ],
                        "detail_claims": [],
                        "limitations": [],
                    },
                    "id": "answer-at-final-slot",
                    "type": "tool_call",
                }
            ],
        )
    )
    model = _ChatModel(responses=responses)

    run = run_bounded_reading_agent(
        workflow,
        result.query,
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=lambda _payload: 12_000,
        max_model_calls=6,
        final_answer=True,
    )

    assert model.call_count == 6
    assert run.trace["model_calls_completed"] == 6
    assert run.trace["agent_input_tokens_exact"] == 72_000
    assert run.trace["agent_output_token_limit"] == 18_432
    assert run.trace["agent_output_tokens_reserved"] == 18_432
    assert model.bound_tool_names[-1] == ["IntegratedAgentAnswer"]


@pytest.mark.parametrize("shared_writer", [False, True])
def test_agent_token_admission_matches_wire_schema_and_output_limit(tmp_path, shared_writer):
    import httpx
    from langchain_openai import ChatOpenAI

    from page_cite_rag.agent import _AgentBudgetLedger, _spans

    workflow, result, blocks = _setup(tmp_path)
    context = (blocks[1],)
    source_id = next(iter(_spans(context)))
    counted_payloads = []
    dispatched_payloads = []
    wire_mismatches = []

    def count_tokens(payload):
        counted_payloads.append(payload)
        return 100

    def respond(request):
        payload = json.loads(request.content)
        dispatched_payloads.append(payload)
        for field in ("messages", "tools", "tool_choice", "max_completion_tokens"):
            if payload.get(field) != counted_payloads[-1].get(field):
                wire_mismatches.append(
                    (
                        field,
                        counted_payloads[-1].get(field),
                        payload.get(field),
                        {key: value for key, value in payload.items() if "token" in key},
                    )
                )
        return httpx.Response(
            200,
            json={
                "id": "integrated-agent-call",
                "object": "chat.completion",
                "model": "page-cite-rag-qwen35",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "answer-call",
                                    "type": "function",
                                    "function": {
                                        "name": "ReadingReport"
                                        if shared_writer
                                        else "IntegratedAgentAnswer",
                                        "arguments": json.dumps(
                                            {
                                                "requirements": [],
                                                "unresolved": ["missing fact"],
                                            }
                                            if shared_writer
                                            else {
                                                "status": "answered",
                                                "answer_claims": [
                                                    {
                                                        "text": (
                                                            "The paragraph supports the answer."
                                                        ),
                                                        "citations": [{"source_id": source_id}],
                                                    }
                                                ],
                                                "detail_claims": [],
                                                "limitations": [],
                                            }
                                        ),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 24, "total_tokens": 124},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as client:
        model = ChatOpenAI(
            model="page-cite-rag-qwen35",
            base_url="http://localhost/v1",
            api_key="local-test",
            http_client=client,
            max_retries=0,
            max_tokens=512,
            temperature=0,
            use_responses_api=False,
        )
        try:
            run = run_bounded_reading_agent(
                workflow,
                result.query,
                "en",
                result,
                context,
                model,
                count_chat_input_tokens=count_tokens,
                max_model_calls=6,
                final_answer=not shared_writer,
                shared_ledger=_AgentBudgetLedger(
                    model_call_limit=6,
                    call_input_limit=12000,
                    total_input_limit=72000,
                    call_output_limit=3072,
                    total_output_limit=18432,
                )
                if shared_writer
                else None,
            )
        except AgentRunFailure as exc:
            pytest.fail(f"{exc.error_type}: {exc.trace['model_call_records']}")

    assert len(counted_payloads) == len(dispatched_payloads) == 1
    assert wire_mismatches == []
    assert dispatched_payloads[0]["max_completion_tokens"] == 3072
    assert dispatched_payloads[0]["tool_choice"] == "required"
    assert any(
        item["function"]["name"] == ("ReadingReport" if shared_writer else "IntegratedAgentAnswer")
        for item in dispatched_payloads[0]["tools"]
    )
    assert run.trace["model_calls_completed"] == 1
    assert run.trace["model_call_records"][-1]["input_tokens_exact"] == 100


def test_agent_bootstrap_trims_whole_blocks_without_shrinking_final_context(tmp_path):
    import json

    workflow, result, blocks = _setup(tmp_path)
    selected = (blocks[1], blocks[3])
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_outline", "args": {}, "id": "outline", "type": "tool_call"}
                ],
            ),
            AIMessage(content="Evidence inspected."),
        ]
    )

    def count_tokens(payload):
        data = json.loads(next(m["content"] for m in payload["messages"] if m["role"] == "user"))
        evidence = data["initial_evidence"]["evidence"]
        return 10_100 if len(evidence) > 1 else 7_000

    run = run_bounded_reading_agent(
        workflow,
        "What does the method say?",
        "en",
        result,
        selected,
        model,
        count_chat_input_tokens=count_tokens,
    )
    assert model.call_count == 2
    assert run.context.blocks == selected
    assert run.trace["bootstrap_input_tokens_exact"] == 7_000
    assert run.trace["bootstrap_shown_block_ids"] == [selected[0].block_id]
    assert run.trace["bootstrap_omitted_block_ids"] == [selected[1].block_id]
    data = json.loads(model.seen_message_batches[1][1].content)
    assert data["source_handles"][1] == {
        "citation_ids": [f"S{selected[1].ordinal}"],
        "source_id": f"S{selected[1].ordinal}",
        "not_shown": True,
    }
    assert run.trace["budget_stop_reason"] is None


def test_shared_writer_agent_reserves_first_navigation_context_for_source_reads(tmp_path):
    from page_cite_rag.agent import _AgentBudgetLedger

    workflow, result, blocks = _setup(tmp_path)
    selected = (blocks[1], blocks[3])
    model = _ChatModel(responses=[_reading_report_message()])

    def count_tokens(payload):
        user_content = next(
            message["content"] for message in payload["messages"] if message["role"] == "user"
        )
        evidence = json.loads(user_content)["initial_evidence"]["evidence"]
        return 8_100 if len(evidence) > 1 else 7_300

    run = run_bounded_reading_agent(
        workflow,
        "What does the method say?",
        "en",
        result,
        selected,
        model,
        count_chat_input_tokens=count_tokens,
        shared_ledger=_AgentBudgetLedger(
            model_call_limit=6,
            call_input_limit=12_000,
            total_input_limit=None,
            call_output_limit=3_072,
            total_output_limit=None,
            tool_call_limit=None,
        ),
    )

    assert run.trace["bootstrap_input_token_limit"] == 7_500
    assert run.trace["bootstrap_input_tokens_exact"] == 7_300
    assert len(run.trace["bootstrap_shown_block_ids"]) == 1
    assert run.trace["bootstrap_omitted_block_ids"]
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


def test_search_trace_marks_a_distinct_query_with_no_new_source_blocks(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    workflow._retriever.result = result.model_copy(
        update={"hits": (SearchHit(rank=1, score=2.0, block=blocks[6]),)}
    )
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_sources",
                        "args": {"query": "method result details"},
                        "id": "search-one",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_sources",
                        "args": {"query": "result of the method"},
                        "id": "search-two",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="The available result source is sufficient."),
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

    first, second = run.trace["tool_trace"]
    assert first["newly_returned_block_ids"] == [blocks[6].block_id]
    assert second["returned_block_ids"] == [blocks[6].block_id]
    assert second["newly_returned_block_ids"] == []
    first_packet = json.loads(model.seen_message_batches[1][-1].content)
    second_packet = json.loads(model.seen_message_batches[2][-1].content)
    assert first_packet["search_progress"] == "new_source_blocks"
    assert second_packet["search_progress"] == "no_new_source_blocks"
    assert second_packet["hits"][0]["new_search_result"] is False


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
    assert failure.value.trace["model_calls_attempted"] == 0
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


def test_agent_finishes_after_repeated_search_instead_of_aborting(tmp_path):
    from page_cite_rag.agent import _spans

    workflow, result, blocks = _setup(tmp_path)
    source_id = next(iter(_spans((blocks[1],))))
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
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "IntegratedAgentAnswer",
                        "args": {
                            "status": "answered",
                            "answer_claims": [
                                {
                                    "text": "The method section contains a paragraph.",
                                    "citations": [{"source_id": source_id}],
                                }
                            ],
                            "detail_claims": [],
                            "limitations": [],
                        },
                        "id": "final-answer",
                        "type": "tool_call",
                    }
                ],
                usage_metadata={"input_tokens": 100, "output_tokens": 24, "total_tokens": 124},
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
        final_answer=True,
        max_model_calls=3,
    )

    assert model.call_count == 3
    assert workflow._retriever.queries == ["new method condition query"]
    assert run.trace["budget_stop_reason"] is None
    assert [event["status"] for event in run.trace["tool_trace"]] == [
        "ok",
        "repeated_tool_call",
    ]
    assert run.trace["tool_trace"][1]["previous_status"] == "ok"
    assert run.trace["navigator_terminal_state"]["valid"] is True
    assert run.trace["final_model_call"]["status"] == "completed"
    assert run.structured_response.status == "answered"
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

    with pytest.raises(AgentRunFailure) as failure:
        run_bounded_reading_agent(
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
    assert sum("tools" in payload for payload in counted_payloads) == 5
    trace = failure.value.trace
    assert failure.value.error_type == "InvalidAgentTerminalState"
    assert trace["architecture"] == "delegated_readers"
    assert trace["reader_tasks_started"] == 1
    assert trace["tool_calls_recorded"] == 3
    assert trace["model_calls_attempted"] == 5
    assert trace["agent_output_tokens_reserved"] == 5 * 512
    assert trace["tool_trace"][0]["terminal"]["status"] == "completed"
    assert trace["tool_trace"][0]["terminal"]["structured_response_expected"] is False
    assert [event["tool"] for event in trace["tool_trace"]] == [
        "delegate_reading",
        "read_source",
        "read_source",
    ]
    assert [event["actor"] for event in trace["tool_trace"]] == [
        "parent",
        "reader:1",
        "reader:1",
    ]
    assert trace["tool_trace"][1]["status"] == "outside_reader_scope"
    assert trace["tool_trace"][2]["block_id"] == blocks[3].block_id
    assert trace["navigator_terminal_state"]["status"] == "invalid_tool_execution"


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
                        "args": {
                            "source_id": method.section_id,
                            "cursor": 0,
                            "max_blocks": 1,
                        },
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
                            "max_blocks": 1,
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


def _read_call(source_id, *, cursor=0, max_blocks=4, call_id="read"):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "read_source",
                "args": {
                    "source_id": source_id,
                    "cursor": cursor,
                    "max_blocks": max_blocks,
                },
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def _batch_read_call(source_ids, call_id="batch"):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "read_source",
                "args": {"source_ids": source_ids},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


def test_preview_uses_relevant_original_sentence_and_marks_cross_language_fallback():
    tokenizer = LexicalTokenizer()
    source = "Background. " + ("Unrelated text. " * 30) + "Accuracy is 92 percent."
    preview, kind = _search_preview(source, "accuracy 92", tokenizer)
    assert preview == "Accuracy is 92 percent."
    assert preview in source and kind == "lexical_match"
    preview, kind = _search_preview("A document in English.", "中文問題", tokenizer)
    assert preview == "A document in English." and kind == "prefix_fallback"


def test_block_text_is_not_elided_when_only_an_unrelated_group_contains_it(tmp_path):
    _workflow, _result, blocks = _setup(tmp_path)
    block = blocks[1]
    unrelated = SearchContextGroup(
        chunk_index=0,
        text=block.text,
        source_refs=("other",),
        source_block_ids=(blocks[2].block_id,),
    )
    mapped_without_text = SearchContextGroup(
        chunk_index=1,
        text="Different contextualized text.",
        source_refs=("target",),
        source_block_ids=(block.block_id,),
    )
    assert not _block_text_in_mapped_group(block, (unrelated, mapped_without_text))
    mapped_with_text = mapped_without_text.model_copy(update={"text": block.text})
    assert _block_text_in_mapped_group(block, (mapped_with_text,))


def test_batch_read_nonadjacent_sources_and_reject_unknown_before_disclosure(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    ids = [blocks[1].block_id, blocks[6].block_id]
    model = _ChatModel(
        responses=[
            _batch_read_call([ids[0], ids[1], ids[0], f"S{blocks[1].ordinal}"]),
            _batch_read_call([*ids, *ids, *ids, "f" * 64], "invalid"),
            AIMessage(content="Read completed."),
        ]
    )
    with pytest.raises(AgentRunFailure) as failure:
        run_bounded_reading_agent(
            workflow,
            "Compare sources.",
            "en",
            result,
            (blocks[1], blocks[6]),
            model,
            count_chat_input_tokens=lambda _payload: 100,
        )
    trace = failure.value.trace
    payload = json.loads(model.seen_message_batches[1][-1].content)
    rejected = json.loads(model.seen_message_batches[2][-1].content)
    assert payload["source_ids"] == ids and payload["pending_ids"] == []
    assert [item["block_id"] for item in payload["blocks"]] == ids
    assert [item["text"] for item in payload["blocks"]] == [blocks[1].text, blocks[6].text]
    assert "text" not in payload
    assert rejected == {"status": "source_unavailable"}
    assert [event["status"] for event in trace["tool_trace"]] == ["ok", "source_unavailable"]
    assert trace["navigator_terminal_state"]["status"] == "invalid_tool_execution"


@pytest.mark.parametrize("overflow", ["tokens", "count"])
def test_batch_read_reports_pending_ids_when_complete_source_exceeds_budget(tmp_path, overflow):
    workflow, result, blocks = _setup(tmp_path)
    ids = (
        [blocks[1].block_id, blocks[6].block_id]
        if overflow == "tokens"
        else [block.block_id for block in blocks[:7]]
    )
    model = _ChatModel(responses=[_batch_read_call(ids), AIMessage(content="Read partially.")])

    def count_tokens(payload):
        if len(payload["messages"]) == 1:
            contents = json.loads(payload["messages"][0]["content"])
            if overflow == "tokens" and len(contents.get("source_ids", [])) > 1:
                return 4100
        return 100

    run = run_bounded_reading_agent(
        workflow,
        "Compare sources.",
        "en",
        result,
        (blocks[1], blocks[6]) if overflow == "tokens" else blocks[:7],
        model,
        count_chat_input_tokens=count_tokens,
    )
    payload = json.loads(model.seen_message_batches[1][-1].content)
    assert payload["status"] == "partial"
    consumed = 1 if overflow == "tokens" else 5
    assert payload["source_ids"] == ids[:consumed]
    assert payload["pending_ids"] == ids[consumed:]
    assert len(payload["blocks"]) == consumed
    assert run.trace["tool_trace"][0]["status"] == "partial"


def test_repeated_read_can_recover_original_text_after_history_edit(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    source_id = blocks[3].block_id
    model = _ChatModel(
        responses=[
            _read_call(source_id, max_blocks=1),
            _read_call(source_id, max_blocks=1, call_id="again"),
            AIMessage(content="Read completed."),
        ]
    )
    run = run_bounded_reading_agent(
        workflow,
        "Explain the method.",
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=lambda _payload: 100,
    )
    first = json.loads(model.seen_message_batches[1][-1].content)
    second = json.loads(model.seen_message_batches[2][-1].content)
    assert first["blocks"][0]["text"] == blocks[3].text
    assert second["blocks"][0]["text"] == blocks[3].text
    assert [event["status"] for event in run.trace["tool_trace"]] == ["ok", "repeated_tool_call"]


def test_continuous_section_read_preserves_complete_blocks_and_cursor(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    method = next(
        section for section in workflow.source_outline().sections if section.title == "Method"
    )
    model = _ChatModel(
        responses=[
            _read_call(method.section_id, max_blocks=2),
            _read_call(
                method.section_id,
                cursor=method.start_ordinal + 1,
                max_blocks=2,
                call_id="next",
            ),
            AIMessage(content="Section inspected."),
        ]
    )

    run = run_bounded_reading_agent(
        workflow,
        "Explain the complete method.",
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=lambda _payload: 100,
    )

    first = json.loads(model.seen_message_batches[1][-1].content)
    second = json.loads(model.seen_message_batches[2][-1].content)
    assert [item["ordinal"] for item in first["blocks"]] == [3, 4]
    assert first["next_cursor"] == 4
    assert first["blocks"][1]["text"] == blocks[3].text
    assert [item["ordinal"] for item in second["blocks"]] == [5]
    assert second["next_cursor"] is None
    assert "Results" not in " ".join(item["text"] for item in first["blocks"] + second["blocks"])
    assert "text" not in first
    assert run.trace["read_source_ids"] == [block.block_id for block in blocks[2:5]]


def test_block_handle_reads_following_sources_without_crossing_section(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    model = _ChatModel(
        responses=[_read_call(blocks[3].block_id), AIMessage(content="Context inspected.")]
    )

    run = run_bounded_reading_agent(
        workflow,
        "Explain both method paragraphs.",
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=lambda _payload: 100,
    )

    returned = json.loads(model.seen_message_batches[1][-1].content)
    assert [item["ordinal"] for item in returned["blocks"]] == [4, 5]
    assert returned["next_cursor"] is None
    assert set(run.trace["read_source_ids"]) == {block.block_id for block in blocks[3:5]}


def test_continuous_read_stops_at_whole_source_token_budget(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    method = next(
        section for section in workflow.source_outline().sections if section.title == "Method"
    )
    model = _ChatModel(
        responses=[_read_call(method.section_id), AIMessage(content="Only the first source fit.")]
    )

    def count_tokens(payload):
        if len(payload["messages"]) == 1:
            packet = json.loads(payload["messages"][0]["content"])
            return 4_100 if len(packet["blocks"]) > 1 else 100
        return 100

    run = run_bounded_reading_agent(
        workflow,
        "Explain the method.",
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=count_tokens,
    )

    returned = json.loads(model.seen_message_batches[1][-1].content)
    assert [item["text"] for item in returned["blocks"]] == [blocks[2].text]
    assert returned["read_stop_reason"] == "source_read_token_budget"
    assert returned["next_cursor"] == blocks[2].ordinal
    assert blocks[3].block_id not in {block.block_id for block in run.context.blocks}


@pytest.mark.parametrize("ambiguous", [False, True])
def test_native_enumeration_lookup_distinguishes_footnotes_and_ambiguity(tmp_path, ambiguous):
    from page_cite_rag.docling_document import bundle_from_native

    originals = [
        {"label": "section_header", "text": "Recommendations"},
        {"label": "footnote", "text": "129 A reference to another author."},
        {
            "label": "list_item",
            "text": "Assess risks before deployment.",
            "orig": "129. Assess risks before deployment.",
            "marker": "129.",
            "enumerated": True,
        },
        {
            "label": "list_item",
            "text": "A separate third recommendation.",
            "orig": "3. A separate third recommendation.",
            "marker": "3.",
            "enumerated": True,
        },
        {
            "label": "list_item",
            "text": "Oversee decisions involving students.",
            "orig": "132. Oversee decisions involving students.",
            "marker": "132.",
            "enumerated": True,
        },
    ]
    if ambiguous:
        originals.append(
            {
                "label": "list_item",
                "text": "A different numbered item.",
                "orig": "129. A different numbered item.",
                "marker": "129.",
                "enumerated": True,
            }
        )
    texts = [
        {
            **item,
            "self_ref": f"#/texts/{index}",
            "prov": [{"page_no": 1, "bbox": {"l": 1, "t": 10, "r": 90, "b": 8}}],
        }
        for index, item in enumerate(originals)
    ]
    chunks = [
        {
            "chunk": {
                "meta": {
                    "headings": ["Recommendations"],
                    "doc_items": [{"self_ref": item["self_ref"]}],
                }
            },
            "search_text": item["text"],
            "search_tokens": 20,
        }
        for item in texts
    ]
    bundle = bundle_from_native(
        {"texts": texts, "tables": []},
        chunks,
        source_sha256="b" * 64,
        source_name="references.pdf",
        page_count=1,
        parser_version="self-authored-native-test",
    )
    store = SQLiteDocumentStore(tmp_path / "references.sqlite3")
    store.ingest(
        bundle.metadata,
        bundle.units(),
        block_factory=bundle.block_factory,
        source_manifests=bundle.source_manifests(),
    )
    result = SearchResult(
        document_id=bundle.metadata.document_id,
        index_id="native-test",
        method="bm25",
        query=(
            "Read paragraphs 129 and 132 in 3 points."
            if ambiguous
            else "請讀第129與132段，分3點說明。"
        ),
        hits=(),
        elapsed_ms=0,
    )
    workflow = FullAnswerWorkflow(
        store, bundle.metadata.document_id, _Retriever(result), _Model(), GenerationConfig()
    )
    stored_blocks = store.get_blocks(bundle.metadata.document_id)
    footnote_id = stored_blocks[1].block_id
    paragraph_id = stored_blocks[2].block_id
    writer_payload = json.loads(
        workflow._messages(result.query, "en", "local", stored_blocks[1:3], None)[1].content
    )
    assert [item["source_type"] for item in writer_payload["evidence"]] == [
        "footnote",
        "list_item",
    ]
    assert writer_payload["evidence"][0]["text"] == stored_blocks[1].text
    assert writer_payload["evidence"][1]["text"] == stored_blocks[2].text
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_source",
                        "args": {"enumeration_marker": "129.", "source_id": footnote_id},
                        "id": "conflicting-reference-arguments",
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
                            "enumeration_marker": "129.",
                            "source_id": paragraph_id,
                            "max_blocks": 1,
                        },
                        "id": "read-marker",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Read complete."),
        ]
    )
    run = run_bounded_reading_agent(
        workflow,
        result.query,
        "en",
        result,
        (),
        model,
        count_chat_input_tokens=lambda _payload: 100,
    )
    initial = json.loads(model.seen_message_batches[0][1].content)
    handles = initial["source_handles"]
    assert len(handles) == (3 if ambiguous else 2)
    assert {h["enumeration_marker"] for h in handles} == {"129.", "132."}
    assert all(h["not_shown"] for h in handles)
    assert all(h["source_type"] == "list_item" for h in handles)
    invalid = json.loads(model.seen_message_batches[1][-1].content)
    assert invalid["status"] == "invalid_reference_arguments"
    packet = json.loads(model.seen_message_batches[2][-1].content)
    if ambiguous:
        assert packet["status"] == "ambiguous_reference"
        assert len(packet["candidates"]) == 2
        assert run.trace["read_source_ids"] == []
    else:
        assert packet["status"] == "ok"
        assert packet["blocks"][0]["source_type"] == "list_item"
        assert packet["blocks"][0]["enumeration_marker"] == "129."
        assert "Assess risks before deployment." in str(packet)
        assert "A reference to another author." not in str(packet)
        assert run.trace["tool_trace"][1]["enumeration_marker"] == "129."


@pytest.mark.parametrize("narrow", [False, True])
def test_native_read_returns_entire_group_and_rejects_cross_unit_leak(tmp_path, narrow):
    from page_cite_rag.docling_document import bundle_from_native

    texts = [
        "Method",
        "Use 42 only under condition A.",
        "Do not use 42 under condition B.",
        "Condition C.",
        "Condition D.",
        "Condition E.",
        "Condition F.",
    ]
    bundle = bundle_from_native(
        {
            "texts": [
                {
                    "self_ref": f"#/texts/{index}",
                    "label": "section_header" if index == 0 else "text",
                    "text": text,
                    "prov": [{"page_no": 1, "bbox": {"l": 1, "t": 10, "r": 90, "b": 8}}],
                }
                for index, text in enumerate(texts)
            ],
            "tables": [],
        },
        [
            {
                "chunk": {
                    "meta": {
                        "headings": ["Method"],
                        "doc_items": [
                            {"self_ref": f"#/texts/{index}"} for index in range(len(texts))
                        ],
                    }
                },
                "search_text": " ".join(texts),
                "search_tokens": 24,
            }
        ],
        source_sha256="a" * 64,
        source_name="method.pdf",
        page_count=1,
        parser_version="self-authored-native-test",
    )
    store = SQLiteDocumentStore(tmp_path / "native.sqlite3")
    store.ingest(
        bundle.metadata,
        bundle.units(),
        block_factory=bundle.block_factory,
        source_manifests=bundle.source_manifests(),
    )
    blocks = tuple(store.get_blocks(bundle.metadata.document_id))
    result = SearchResult(
        document_id=bundle.metadata.document_id,
        index_id="native-test",
        method="bm25",
        query="method conditions",
        hits=(),
        elapsed_ms=0,
    )
    workflow = FullAnswerWorkflow(
        store, bundle.metadata.document_id, _Retriever(result), _Model(), GenerationConfig()
    )
    model = _ChatModel(
        responses=[
            _read_call(blocks[1 if narrow else 0].block_id, max_blocks=1),
            _read_call("unit:1", max_blocks=1, call_id="unit"),
            _batch_read_call([blocks[0].block_id, blocks[1].block_id], "native-batch"),
            AIMessage(content="Native context inspected."),
        ]
    )

    def count(payload):
        if narrow and len(payload["messages"]) == 1:
            packet = json.loads(payload["messages"][0]["content"])
            if packet.get("context_groups"):
                return 4100
        return 100

    run = run_bounded_reading_agent(
        workflow,
        "Under which conditions is 42 used?",
        "en",
        result,
        (),
        model,
        count_chat_input_tokens=count,
    )

    returned = json.loads(model.seen_message_batches[1][-1].content)
    if narrow:
        assert returned["context_groups"] == []
        assert len(returned["blocks"]) == 1
        assert returned["blocks"][0]["text"] == blocks[1].text
        assert returned["next_cursor"] == blocks[1].ordinal
        assert returned["read_stop_reason"] == "native_chunk_exceeds_read_budget"
        assert run.trace["read_source_ids"] == [blocks[1].block_id]
        return
    assert returned["next_cursor"] is None
    assert all("text" not in item for item in returned["blocks"])
    assert all(text in returned["context_groups"][0]["text"] for text in texts)
    assert returned["context_groups"][0]["source_block_ids"] == [block.block_id for block in blocks]
    assert [group.chunk_index for group in run.context.context_groups] == [0]
    assert run.trace["tool_trace"][1]["status"] == "native_chunk_crosses_source_scope"
    batch = json.loads(model.seen_message_batches[3][-1].content)
    assert len(batch["context_groups"]) == 1
    assert batch["context_groups"][0]["source_block_ids"] == [block.block_id for block in blocks]


def test_feedback_is_visible_in_every_parent_turn_and_shared_call_limit_stops_readers(tmp_path):
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
                            "task": "Read the method qualifications.",
                            "source_scope": method.section_id,
                        },
                        "id": "delegate",
                        "type": "tool_call",
                    }
                ],
            ),
            _read_call(method.section_id, max_blocks=2),
            AIMessage(content="The source qualifications were read."),
        ]
    )
    feedback = "The draft omitted the method's qualification; inspect the complete passage."
    observed_payloads = []

    def count_tokens(payload):
        observed_payloads.append(payload)
        return 100

    run = run_bounded_reading_agent(
        workflow,
        "Explain the method.",
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=count_tokens,
        delegated=True,
        reading_feedback=feedback,
        max_model_calls=3,
    )

    initial = json.loads(model.seen_message_batches[0][-1].content)
    assert initial["reading_feedback"] == feedback
    parent_inputs = [
        json.loads(message["content"])
        for payload in observed_payloads
        for message in payload["messages"]
        if message["role"] == "user" and '"reading_feedback"' in message["content"]
    ]
    assert len(parent_inputs) == 2
    assert all(item["reading_feedback"] == feedback for item in parent_inputs)
    assert run.trace["model_call_limit"] == 3
    assert run.trace["model_calls_completed"] == model.call_count == 3
    assert run.trace["budget_stop_reason"] == "model_call_limit"
    assert run.trace["reader_tasks_started"] == 1
    assert run.trace["tool_trace"][1]["actor"] == "reader:1"


def test_unit_read_recovers_from_out_of_scope_cursor(tmp_path):
    workflow, result, blocks = _setup(tmp_path)
    model = _ChatModel(
        responses=[
            _read_call("unit:1", max_blocks=2),
            _read_call("unit:1", cursor=2, max_blocks=2, call_id="next"),
            _read_call("unit:1", cursor=999, max_blocks=2, call_id="invalid"),
            AIMessage(content="Unit inspected."),
        ]
    )

    run = run_bounded_reading_agent(
        workflow,
        "Read the original unit.",
        "en",
        result,
        (blocks[1],),
        model,
        count_chat_input_tokens=lambda _payload: 100,
    )

    first = json.loads(model.seen_message_batches[1][-1].content)
    second = json.loads(model.seen_message_batches[2][-1].content)
    assert [item["ordinal"] for item in first["blocks"]] == [1, 2]
    assert first["next_cursor"] == 2
    assert [item["ordinal"] for item in second["blocks"]] == [3, 4]
    assert second["next_cursor"] == 4
    assert run.trace["tool_trace"][-1]["status"] == "cursor_out_of_range"
    assert run.trace["navigator_terminal_state"]["valid"] is True


@pytest.mark.parametrize("answer_format", ["general", "short_answer", "multiple_choice"])
def test_agent_status_contract_exposes_one_tool_and_preserves_validation_failure(
    tmp_path, answer_format
):
    from langchain.agents.structured_output import ToolStrategy

    workflow, result, blocks = _setup(tmp_path)
    workflow.answer_format = answer_format
    strategy = ToolStrategy(workflow.agent_tool_model(), handle_errors=False)
    assert [spec.name for spec in strategy.schema_specs] == ["IntegratedAgentAnswer"]
    assert strategy.schema_specs[0].schema_kind == "pydantic"
    assert "oneOf" not in strategy.schema_specs[0].json_schema
    assert "allOf" not in strategy.schema_specs[0].json_schema
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "IntegratedAgentAnswer",
                        "args": {"status": "answered", "limitations": []},
                        "id": "invalid-answer",
                        "type": "tool_call",
                    }
                ],
                response_metadata={"finish_reason": "tool_calls"},
                usage_metadata={"input_tokens": 100, "output_tokens": 12, "total_tokens": 112},
            )
        ]
    )
    with pytest.raises(AgentRunFailure) as failure:
        run_bounded_reading_agent(
            workflow,
            result.query,
            "en",
            result,
            (blocks[1],),
            model,
            count_chat_input_tokens=lambda _payload: 100,
            final_answer=True,
            max_model_calls=6,
        )
    assert failure.value.error_type == "InvalidStructuredAnswerSchema"
    evidence = failure.value.trace["failure_evidence"]
    assert len(evidence["raw_json"].encode("utf-8")) <= 65_536
    payload = json.loads(evidence["raw_json"])
    assert payload["validation_errors"]
    assert payload["last_assistant"]["response_metadata"]["finish_reason"] == "tool_calls"
    assert failure.value.trace["model_calls_completed"] == model.call_count == 1


def test_shared_reader_compacts_seen_reads_without_hiding_new_originals(tmp_path):
    from page_cite_rag.agent import _AgentBudgetLedger

    workflow, search, blocks = _setup(tmp_path)
    model = _ChatModel(
        responses=[
            _read_call(blocks[3].block_id, max_blocks=1, call_id="first"),
            _read_call(blocks[4].block_id, max_blocks=1, call_id="second"),
            _reading_report_message(blocks[3], source_ids=[blocks[3].block_id, blocks[4].block_id]),
        ]
    )

    def count(payload):
        if len(payload["messages"]) == 1:
            return 100
        full_reads = sum(
            message["role"] == "tool" and json.loads(message["content"]).get("status") == "ok"
            for message in payload["messages"]
        )
        if full_reads == 0:
            return 7_000
        return 12034 if full_reads > 1 else 11000

    ledger = _AgentBudgetLedger(
        model_call_limit=4,
        call_input_limit=12000,
        total_input_limit=72000,
        call_output_limit=3072,
        total_output_limit=18432,
    )
    run = run_bounded_reading_agent(
        workflow,
        search.query,
        "en",
        search,
        (blocks[1],),
        model,
        count_chat_input_tokens=count,
        shared_ledger=ledger,
        max_model_calls=3,
    )
    first = json.loads(model.seen_message_batches[1][-1].content)
    assert first["blocks"][0]["text"] == blocks[3].text
    packets = [
        json.loads(message.content)
        for message in model.seen_message_batches[-1]
        if message.type == "tool"
    ]
    assert packets[0]["status"] == "previously_read"
    assert packets[0]["source_ids"]
    assert packets[1]["status"] == "ok"
    assert packets[1]["blocks"][0]["text"] == blocks[4].text
    assert set(run.trace["read_source_ids"]) == {blocks[3].block_id, blocks[4].block_id}
    assert run.trace["tool_context_edits"][-1]["input_tokens_before"] == 12034
    assert run.trace["tool_context_edits"][-1]["input_tokens_after"] == 11000
    assert run.trace["navigation_report_partial"] is False
    assert run.structured_response.requirements[0].source_ids == [
        blocks[3].block_id,
        blocks[4].block_id,
    ]


def test_agent_removes_duplicate_bootstrap_blocks_before_budgeting_the_next_call(tmp_path):
    from page_cite_rag.agent import _spans

    workflow, search, blocks = _setup(tmp_path)
    context = tuple(blocks[1:4])
    anchor = context[-1]
    citation = next(iter(_spans((anchor,))))
    model = _ChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_source",
                        "args": {"source_id": anchor.block_id, "max_blocks": 1},
                        "id": "read",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "IntegratedAgentAnswer",
                        "args": {
                            "status": "answered",
                            "answer_claims": [
                                {
                                    "text": "The method contains the cited paragraph.",
                                    "citations": [{"source_id": citation}],
                                }
                            ],
                            "detail_claims": [],
                            "limitations": [],
                        },
                        "id": "answer",
                        "type": "tool_call",
                    }
                ],
            ),
        ]
    )

    def count(payload):
        user = next(m for m in payload["messages"] if m["role"] == "user")
        data = json.loads(user["content"])
        if "initial_evidence" not in data:
            return 100
        evidence = data["initial_evidence"]["evidence"]
        has_read = any(m["role"] == "tool" for m in payload["messages"])
        return 5000 + 2000 * len(evidence) + (2000 if has_read else 0)

    run = run_bounded_reading_agent(
        workflow,
        search.query,
        "en",
        search,
        context,
        model,
        count_chat_input_tokens=count,
        final_answer=True,
        max_model_calls=6,
    )
    assert run.trace["model_calls_completed"] == 2
    assert run.trace["bootstrap_shown_block_ids"] == [b.block_id for b in context]
    assert anchor.block_id not in run.trace["current_bootstrap_block_ids"]
    edit = run.trace["bootstrap_context_edits"][-1]
    assert edit["input_tokens_before"] == 13000
    assert edit["input_tokens_after"] == 11000
    assert edit["tool_results_retained"] is True
    tool = next(m for m in model.seen_message_batches[-1] if m.type == "tool")
    assert anchor.text.strip() in tool.content
