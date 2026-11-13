"""Evidence-agent integration using synthetic sources and mock HTTP only."""

import hashlib
import json
import math
from types import SimpleNamespace

import httpx
import pytest

from page_cite_rag import evidence_agent as agent_module
from page_cite_rag.evidence_agent import (
    LIMITS,
    EvidenceTools,
    GuardedTransport,
    run_evidence_agent,
)
from page_cite_rag.full_answer import FullAnswerWorkflow
from page_cite_rag.generation import GenerationResult, ModelError, check_budget
from page_cite_rag.ingest import prepare_source
from page_cite_rag.ollama_backend import OllamaGgufClient
from page_cite_rag.quote_first import quote_first_messages
from page_cite_rag.store import SQLiteDocumentStore
from page_cite_rag.workflows import AnswerResult
from tests import test_workflows as helpers


@pytest.fixture
def setup(tmp_path, monkeypatch):
    store, doc, blocks, retriever, model, _ = helpers.setup.__wrapped__(tmp_path)

    def generate(messages, response_model):
        model.calls.append((messages, response_model))
        assert response_model is None
        text = (
            f'Evidence:\n[S{blocks[0].ordinal}] "Children were excluded."\n'
            "Reasoning: The passage states it.\nAnswer: Children were excluded."
        )
        return GenerationResult(text, helpers.stats(), check_budget(messages, None, model.config))

    monkeypatch.setattr(model, "generate", generate)
    workflow = FullAnswerWorkflow(
        store,
        doc,
        retriever,
        model,
        model.config,
        writer="quote_first",
        answer_format="short_answer",
    )
    checks = []
    client = SimpleNamespace(
        chat_completions_base_url=lambda: "http://synthetic/v1",
        chat_model_name="synthetic",
        thinking_control="reasoning_effort_none",
        before_agent_dispatch=lambda: checks.append("dispatch"),
        chat_sampling=lambda: {"temperature": 0.0, "seed": 42},
    )
    return workflow, client, blocks, retriever, model, checks


CAPPED = {"reasoning_effort": "none", "max_completion_tokens": LIMITS.max_tokens}


def response(request, name="search_passages", arguments=None, **message_extra):
    call = {
        "id": "tool-1",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments or {"query": "entity relation"}),
        },
    }
    message = {"role": "assistant", "content": None, "tool_calls": [call], **message_extra}
    return httpx.Response(
        200,
        json={
            "id": "synthetic",
            "object": "chat.completion",
            "created": 0,
            "model": "synthetic",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        },
        request=request,
    )


def mock_agent_http(monkeypatch, client, handler):
    guards = []
    transport_type = GuardedTransport

    def make_transport(actual_client, **kwargs):
        assert actual_client is client
        guard = transport_type(client, inner=httpx.MockTransport(handler), **kwargs)
        guards.append(guard)
        return guard

    monkeypatch.setattr(agent_module, "GuardedTransport", make_transport)
    return guards


def test_search_search_finish(setup, monkeypatch):
    workflow, client, blocks, retriever, model, checks = setup
    pids = [f"P{b.ordinal}" for b in blocks]
    requests = []
    request_bodies = []
    initial = workflow.search_sources("Who was excluded?")[0]
    initial = initial.model_copy(update={"hits": initial.hits[:1]})
    searches = iter([blocks[1:2], blocks[2:]])
    retriever.change = lambda result: result.model_copy(
        update={
            "hits": tuple(result.hits[0].model_copy(update={"block": b}) for b in next(searches))
        }
    )
    sequence = [
        ("search_passages", {"query": "first entity relation"}),
        ("search_passages", {"query": "second entity relation"}),
        ("finish", {"passage_ids": [pids[2], pids[0], pids[2]]}),
    ]

    def handler(request):
        raw_body = request.read()
        request_bodies.append(raw_body)
        requests.append(json.loads(raw_body))
        name, arguments = sequence[len(requests) - 1]
        return response(request, name, arguments)

    mock_agent_http(monkeypatch, client, handler)
    before, after = [], []
    result = run_evidence_agent(
        workflow,
        client,
        "Who was excluded?",
        language="en",
        initial_search=initial,
        before_model_call=lambda: before.append(True),
        after_model_call=after.append,
    )
    assert len(requests) == len(checks) == len(before) == len(after) == 3
    assert all(r["reasoning_effort"] == "none" and not r["parallel_tool_calls"] for r in requests)
    assert requests[0]["messages"][0]["content"] == agent_module.SYSTEM_PROMPT
    assert requests[0]["messages"][1]["content"] == (
        f"Question: Who was excluded?\n\nInitial search results for the question:\n"
        f"{pids[0]}: {blocks[0].text.replace(chr(10), chr(32))}\n\nSteps used: 0 of 8."
    )
    trace = result.trace
    assert trace["version"] == "evidence-agent/v5"
    assert trace["tool_call_repairs"] == 0
    assert trace["sampling"] == {"temperature": 0.0, "seed": 42}
    assert all(
        r["max_tokens"] == LIMITS.max_tokens and "max_completion_tokens" not in r for r in requests
    )
    assert request_bodies == [json.dumps(request).encode("utf-8") for request in requests]
    # Captured before adding the API backend: all three local requests remain byte-identical.
    assert [hashlib.sha256(body).hexdigest() for body in request_bodies] == [
        "2fc9df0963d9c8400df80961c90521141d798e0a9a2f3ed909652709f5363408",
        "518f885f030566f06728b4e9a22c4f1bb52036887b8dbc617a003d870ba8e49f",
        "c0cc60b28576dfa902751600c02d35acb9d6142f2c13916d66707cf1b2e70553",
    ]
    assert trace["selected_passage_ids"] == [pids[2], pids[0]]
    assert trace["handoff_passage_ids"] == [pids[2], pids[0], pids[1]]
    assert trace["handoff_reason"] == "finish_called"
    assert [r["status"] for r in trace["tool_calls"]] == ["ok"] * 3
    timing = trace["timing"]
    assert set(timing) == {
        "total_ms",
        "agent_loop_ms",
        "writer_ms",
        "model_gate_ms",
        "model_http_ms",
        "tool_ms",
        "unattributed_ms",
    }
    assert all(
        isinstance(timing[key], (int, float)) and math.isfinite(timing[key]) and timing[key] >= 0
        for key in timing
        if key != "unattributed_ms"
    )
    assert math.isfinite(timing["unattributed_ms"])
    assert timing["model_gate_ms"] == sum(r["gate_ms"] for r in trace["model_calls"])
    assert timing["model_http_ms"] == sum(r["http_ms"] for r in trace["model_calls"])
    assert timing["tool_ms"] == sum(r["elapsed_ms"] for r in trace["tool_calls"])
    assert all(
        isinstance(record[key], (int, float)) and math.isfinite(record[key]) and record[key] >= 0
        for record in trace["model_calls"]
        for key in ("gate_ms", "hook_ms", "http_ms", "elapsed_ms")
    )
    assert all(
        isinstance(record[key], (int, float)) and math.isfinite(record[key]) and record[key] >= 0
        for record in trace["tool_calls"]
        for key in ("elapsed_ms", "search_ms")
    )
    assert all(record["search_ms"] <= record["elapsed_ms"] for record in trace["tool_calls"])
    assert all(r["prompt_eval_count"] == 10 and r["eval_count"] == 2 for r in after)
    expected = (
        "\n\nSearch steps that found these passages (unverified hints, not evidence):\n"
        "1. first entity relation\n2. second entity relation\n\nQuestion: Who was excluded?"
    )
    assert model.calls[-1][0][1].content.endswith(expected)
    assert workflow._search_steps == ()
    assert result.answer.status == "answered"
    AnswerResult.model_validate_json(result.answer.model_dump_json(), strict=True)
    assert all(b.text not in json.dumps(trace) for b in blocks)


def test_eight_searches_no_ninth_and_all_passages(setup, tmp_path, monkeypatch):
    workflow, client, _, retriever, model, _ = setup
    source = tmp_path / "many.txt"
    source.write_text(
        "\n\n".join(f"Passage {i} contains synthetic evidence." for i in range(45)),
        encoding="utf-8",
    )
    prepared = prepare_source(source, usage_scope="self-authored-test")
    store = SQLiteDocumentStore(tmp_path / "many.sqlite3")
    store.ingest(prepared.document, prepared.iter_units())
    blocks = store.get_blocks(prepared.document.document_id)
    retriever = helpers.FakeRetriever(prepared.document.document_id, blocks)
    workflow = FullAnswerWorkflow(
        store, prepared.document.document_id, retriever, model, model.config, writer="quote_first"
    )
    batches = iter([blocks[i : i + 5] for i in range(0, 45, 5)])
    retriever.change = lambda result: result.model_copy(
        update={
            "hits": tuple(
                result.hits[0].model_copy(update={"rank": i, "block": b})
                for i, b in enumerate(next(batches), 1)
            )
        }
    )
    requests = []
    handed = []
    original = workflow.answer_from_context

    def writer(question, context, **kwargs):
        handed.extend(context)
        return original(question, context, **kwargs)

    monkeypatch.setattr(workflow, "answer_from_context", writer)

    def handler(request):
        requests.append(json.loads(request.read()))
        assert len(requests) <= 8
        return response(request)

    mock_agent_http(monkeypatch, client, handler)
    result = run_evidence_agent(workflow, client, "Evidence?", language="en")
    assert len(requests) == 8
    assert any("call finish now" in str(m) for m in requests[7]["messages"])
    assert not any("call finish now" in str(m) for m in requests[6]["messages"])
    assert result.trace["handoff_reason"] == "finish_not_called"
    assert result.trace["selected_passage_ids"] == []
    assert result.trace["handoff_passage_ids"] == [f"P{b.ordinal}" for b in blocks]
    assert handed == blocks
    assert len(handed) > 20


@pytest.mark.parametrize(
    "body", [{}, {"reasoning_effort": "low"}, {"reasoning_effort": "none", "stream": True}]
)
def test_request_guard_precedes_dispatch(body):
    calls = []
    client = SimpleNamespace(before_agent_dispatch=lambda: calls.append("dispatch"))
    guard = GuardedTransport(client, inner=httpx.MockTransport(lambda r: calls.append("http")))
    with pytest.raises(ModelError):
        guard.handle_request(httpx.Request("POST", "http://synthetic", json=body))
    assert calls == []


@pytest.mark.parametrize("field", ["reasoning", "reasoning_content", "parallel"])
def test_response_guards_preserve_usage(field):
    after = []

    def handler(request):
        if field == "parallel":
            first = response(request).json()["choices"][0]["message"]["tool_calls"][0]
            return response(request, tool_calls=[first, {**first, "id": "tool-2"}])
        return response(request, **{field: "hidden reasoning"})

    guard = GuardedTransport(
        SimpleNamespace(before_agent_dispatch=lambda: None),
        inner=httpx.MockTransport(handler),
        after_model_call=after.append,
    )
    with pytest.raises(ModelError):
        guard.handle_request(httpx.Request("POST", "http://synthetic", json=CAPPED))
    assert len(after) == 1
    assert after[0]["status"] == "failed"
    assert after[0]["error_type"] == "ModelError"
    assert after[0]["prompt_eval_count"] == 10 and after[0]["eval_count"] == 2


def test_before_hook_failure_sends_nothing():
    calls, after = [], []

    def before():
        calls.append("before")
        raise ValueError("ledger exhausted")

    guard = GuardedTransport(
        SimpleNamespace(before_agent_dispatch=lambda: calls.append("dispatch")),
        inner=httpx.MockTransport(lambda r: calls.append("http")),
        before_model_call=before,
        after_model_call=after.append,
    )
    with pytest.raises(ValueError, match="ledger exhausted"):
        guard.handle_request(httpx.Request("POST", "http://synthetic", json=CAPPED))
    assert calls == ["dispatch", "before"]
    assert after[0]["status"] == "failed" and after[0]["eval_count"] is None


def test_transport_model_call_ceiling():
    sent = []

    def handler(request):
        sent.append(True)
        return response(request)

    guard = GuardedTransport(
        SimpleNamespace(before_agent_dispatch=lambda: None), inner=httpx.MockTransport(handler)
    )
    request = httpx.Request("POST", "http://synthetic", json=CAPPED)
    for _ in range(LIMITS.model_calls):
        guard.handle_request(request)
    with pytest.raises(ModelError, match="ceiling"):
        guard.handle_request(request)
    assert len(sent) == 8


@pytest.mark.parametrize("steps", [["x"] * 17, ["x" * 301], [""], ["  "], [123], "query", None])
def test_invalid_search_steps(setup, steps):
    workflow, _, blocks, _, model, _ = setup
    with pytest.raises(ValueError):
        workflow.answer_from_context("Question?", blocks, search_steps=steps)
    assert not model.calls


def test_structured_rejects_hints_and_agent(setup):
    workflow, client, blocks, _, _, _ = setup
    workflow.writer = "structured"
    with pytest.raises(ValueError):
        workflow.answer_from_context("Question?", blocks, search_steps=["entity relation"])
    with pytest.raises(ValueError):
        run_evidence_agent(workflow, client, "Question?", language="en")


def test_empty_steps_preserve_messages(setup):
    workflow, _, blocks, _, model, _ = setup
    workflow.answer_from_context("Who was excluded?", blocks, language="en", search_steps=())
    first = model.calls[-1][0]
    workflow.answer_from_context("Who was excluded?", blocks, language="en")
    assert model.calls[-1][0] == first
    assert quote_first_messages("Q", "en", "general", ()) == quote_first_messages(
        "Q", "en", "general", (), search_steps=()
    )
    assert first[1].content.endswith("\n\nQuestion: Who was excluded?")
    assert "Search steps" not in first[1].content


def test_tool_validation_and_empty_finish(setup):
    workflow, _, blocks, _, _, _ = setup
    initial = workflow.search_sources("Question?")[0]
    state = EvidenceTools(
        blocks, initial.model_copy(update={"hits": initial.hits[:1]}), lambda q: initial
    )
    for name, args in [
        ("search_passages", {"query": " "}),
        ("search_passages", {"query": "x" * 301}),
        ("finish", {"passage_ids": ["P01"]}),
        ("finish", {"passage_ids": ["P999"]}),
        ("finish", {"passage_ids": [f"P{blocks[1].ordinal}"]}),
        ("finish", {"passage_ids": [f"P{blocks[0].ordinal}"] * 13}),
    ]:
        assert state.invoke(name, args).startswith("Rejected:")
        assert state.calls[-1]["status"] == "rejected"
    assert "(seen)" in state.passages(state.initial_ids)
    state.invoke("finish", {"passage_ids": []})
    assert state.handoff() == ([], state.initial_ids, "finish_empty")


def test_client_public_endpoints_and_order(monkeypatch):
    client = object.__new__(OllamaGgufClient)
    calls = []
    client._http = SimpleNamespace(base_url=httpx.URL("http://synthetic/"))
    client._ollama_model = "pinned-local"
    monkeypatch.setattr(client, "_ensure_running", lambda: calls.append("running"))
    monkeypatch.setattr(client, "wait_for_gpu_headroom", lambda: calls.append("headroom"))
    monkeypatch.setattr(client, "_check_gpu", lambda: calls.append("gpu"))
    assert client.chat_completions_base_url() == "http://synthetic/v1"
    assert calls == ["running"]
    assert client.ollama_model_name == client.chat_model_name == "pinned-local"
    assert client.thinking_control == "reasoning_effort_none"
    with pytest.raises(AttributeError):
        client.ollama_model_name = "other"
    calls.clear()
    client.before_agent_dispatch()
    assert calls == ["running", "headroom", "gpu"]


@pytest.mark.parametrize("failure", ["reasoning", "before", "http"])
def test_agent_failure_propagates_without_writer_or_retry(setup, monkeypatch, failure):
    workflow, client, _, _, model, checks = setup
    requests, after = [], []

    def handler(request):
        requests.append(True)
        if failure == "http":
            raise httpx.ReadError("synthetic read failure")
        return response(request, reasoning_content="hidden reasoning")

    def before():
        if failure == "before":
            raise ValueError("synthetic ledger failure")

    mock_agent_http(monkeypatch, client, handler)
    error = {"reasoning": ModelError, "before": ValueError, "http": httpx.ReadError}[failure]
    with pytest.raises(error):
        run_evidence_agent(
            workflow,
            client,
            "Who was excluded?",
            language="en",
            before_model_call=before,
            after_model_call=after.append,
        )
    assert len(checks) == len(after) == 1
    assert len(requests) == (0 if failure == "before" else 1)
    assert after[0]["status"] == "failed"
    assert model.calls == []


def test_search_step_boundaries_are_preserved_and_do_not_leak(setup):
    workflow, _, blocks, _, model, _ = setup
    steps = ["q" * 300] * 16
    workflow.answer_from_context("Who was excluded?", blocks, language="en", search_steps=steps)
    user = model.calls[-1][0][1].content
    assert f"16. {steps[-1]}\n\nQuestion:" in user
    workflow.answer_from_context("Who was excluded?", blocks, language="en")
    assert "Search steps" not in model.calls[-1][0][1].content


def test_passage_namespace_rejects_foreign_search_and_keeps_full_text(setup):
    workflow, _, blocks, _, _, _ = setup
    initial = workflow.search_sources("Question?")[0]
    foreign = blocks[0].model_copy(update={"document_id": "f" * 64})
    bad = initial.model_copy(
        update={"hits": (initial.hits[0].model_copy(update={"block": foreign}),)}
    )
    with pytest.raises(ValueError, match="outside this document"):
        EvidenceTools(blocks, bad, lambda q: initial)
    state = EvidenceTools(blocks, initial, lambda q: bad)
    assert state.invoke("search_passages", {"query": "entity relation"}).startswith("Rejected:")
    long = blocks[0].model_copy(update={"text": "x" * 3000 + "\nlast fact"})
    long_initial = initial.model_copy(
        update={"hits": (initial.hits[0].model_copy(update={"block": long}),)}
    )
    state = EvidenceTools([long], long_initial, lambda q: long_initial)
    assert state.initial_passages.endswith("x" * 3000 + " last fact")


def test_cap_is_rewritten_for_ollama_and_required():
    sent = []

    def handler(request):
        sent.append(json.loads(request.read()))
        return response(request)

    guard = GuardedTransport(
        SimpleNamespace(before_agent_dispatch=lambda: None), inner=httpx.MockTransport(handler)
    )
    guard.handle_request(httpx.Request("POST", "http://synthetic", json=CAPPED))
    assert sent[0]["max_tokens"] == LIMITS.max_tokens and "max_completion_tokens" not in sent[0]
    for body in ({"reasoning_effort": "none"}, {**CAPPED, "max_completion_tokens": 4096}):
        with pytest.raises(ModelError, match="token cap"):
            guard.handle_request(httpx.Request("POST", "http://synthetic", json=body))
    assert len(sent) == 1


@pytest.mark.parametrize("completion_tokens", [LIMITS.max_tokens + 1, None])
def test_uncapped_or_unaccounted_output_fails(completion_tokens):
    after = []

    def handler(request):
        reply = response(request).json()
        reply["usage"]["completion_tokens"] = completion_tokens
        return httpx.Response(200, json=reply, request=request)

    guard = GuardedTransport(
        SimpleNamespace(before_agent_dispatch=lambda: None),
        inner=httpx.MockTransport(handler),
        after_model_call=after.append,
    )
    with pytest.raises(ModelError, match="token cap"):
        guard.handle_request(httpx.Request("POST", "http://synthetic", json=CAPPED))
    assert after[0]["status"] == "failed"


def test_chat_sampling_follows_client_profile():
    client = object.__new__(OllamaGgufClient)
    client._sampling_seed = 7
    client._sampling_profile = "greedy"
    assert client.chat_sampling() == {"temperature": 0.0, "seed": 7}
    client._sampling_profile = "qwen-general"
    assert client.chat_sampling() == {
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "seed": 7,
    }
    client._sampling_profile = "longbench-official"
    with pytest.raises(ModelError):
        client.chat_sampling()


REJECTED_FINISH = (
    'llama-server returned invalid tool call arguments for "finish": unexpected end of JSON input'
)


def server_error(request, message):
    return httpx.Response(
        500, json={"error": {"message": message, "type": "api_error"}}, request=request
    )


def test_rejected_tool_call_is_corrected_once(setup, monkeypatch):
    workflow, client, blocks, _, model, checks = setup
    pid = f"P{blocks[0].ordinal}"
    requests, before, after = [], [], []

    def handler(request):
        requests.append(json.loads(request.read()))
        if len(requests) == 1:
            return server_error(request, REJECTED_FINISH)
        return response(request, "finish", {"passage_ids": [pid]})

    initial = workflow.search_sources("Who was excluded?")[0]
    mock_agent_http(monkeypatch, client, handler)
    result = run_evidence_agent(
        workflow,
        client,
        "Who was excluded?",
        language="en",
        initial_search=initial.model_copy(update={"hits": initial.hits[:1]}),
        before_model_call=lambda: before.append(True),
        after_model_call=after.append,
    )
    assert len(requests) == len(checks) == len(before) == len(after) == 2
    assert requests[1]["messages"][:-1] == requests[0]["messages"]
    correction = requests[1]["messages"][-1]
    assert correction["role"] == "user" and "finish call" in correction["content"]
    assert "512 tokens" in correction["content"] and "only one tool call" in correction["content"]
    rejected, repaired = result.trace["model_calls"]
    assert rejected["status"] == "tool_call_rejected" and rejected["http_status"] == 500
    assert rejected["error_message"] == REJECTED_FINISH and rejected["eval_count"] is None
    assert repaired["status"] == "completed" and repaired["repair"] is True
    assert result.trace["tool_call_repairs"] == 1
    assert result.trace["handoff_reason"] == "finish_called"
    assert result.answer.status == "answered" and len(model.calls) == 1


def test_rejections_beyond_the_repair_budget_fail_without_writer(setup, monkeypatch):
    workflow, client, _, _, model, _ = setup
    after = []
    mock_agent_http(monkeypatch, client, lambda request: server_error(request, REJECTED_FINISH))
    with pytest.raises(agent_module.AgentToolCallError, match="finish"):
        run_evidence_agent(
            workflow, client, "Who was excluded?", language="en", after_model_call=after.append
        )
    assert issubclass(agent_module.AgentToolCallError, ModelError)
    assert [r["status"] for r in after] == ["tool_call_rejected"] * (LIMITS.tool_call_repairs + 1)
    assert [r["repair"] for r in after] == [False] + [True] * LIMITS.tool_call_repairs
    assert model.calls == []


def test_other_server_errors_are_recorded_and_not_resent():
    sent, after = [], []

    def handler(request):
        sent.append(True)
        return server_error(request, "llama runner process has terminated")

    guard = GuardedTransport(
        SimpleNamespace(before_agent_dispatch=lambda: None),
        inner=httpx.MockTransport(handler),
        after_model_call=after.append,
    )
    with pytest.raises(httpx.HTTPStatusError):
        guard.handle_request(httpx.Request("POST", "http://synthetic", json=CAPPED))
    assert len(sent) == 1 and guard.repairs == 0
    assert after[0]["status"] == "failed" and after[0]["http_status"] == 500
    assert after[0]["error_message"] == "llama runner process has terminated"


def test_correction_does_not_use_an_agent_step():
    replies = iter(["reject"] + ["ok"] * LIMITS.model_calls)

    def handler(request):
        if next(replies) == "reject":
            return server_error(request, REJECTED_FINISH)
        return response(request)

    guard = GuardedTransport(
        SimpleNamespace(before_agent_dispatch=lambda: None), inner=httpx.MockTransport(handler)
    )
    request = httpx.Request("POST", "http://synthetic", json={**CAPPED, "messages": []})
    for _ in range(LIMITS.model_calls):
        guard.handle_request(request)
    assert len(guard.records) == LIMITS.model_calls + 1 and guard.repairs == 1
    with pytest.raises(ModelError, match="ceiling"):
        guard.handle_request(request)


def cut_reply(request, content="Let me think about which passages matter most"):
    message = {"role": "assistant", "content": content, "tool_calls": None}
    return httpx.Response(
        200,
        json={
            "id": "synthetic",
            "object": "chat.completion",
            "created": 0,
            "model": "synthetic",
            "choices": [{"index": 0, "finish_reason": "length", "message": message}],
            "usage": {"prompt_tokens": 10, "completion_tokens": LIMITS.max_tokens},
        },
        request=request,
    )


def test_reply_cut_before_any_tool_call_ends_search_without_repair(setup, monkeypatch):
    workflow, client, _, _, model, _ = setup
    requests = []

    def handler(request):
        requests.append(True)
        return cut_reply(request)

    mock_agent_http(monkeypatch, client, handler)
    result = run_evidence_agent(workflow, client, "Who was excluded?", language="en")
    (call,) = result.trace["model_calls"]
    assert call["status"] == "cut_without_tool_call" and call["finish_reason"] == "length"
    assert len(requests) == 1 and result.trace["tool_call_repairs"] == 0
    assert result.trace["handoff_reason"] == "finish_not_called"
    assert len(model.calls) == 1


def test_token_usage_bounds_calls_without_reported_usage():
    usage = agent_module.agent_token_usage(
        [
            {"prompt_eval_count": 100, "eval_count": 20},
            {"prompt_eval_count": None, "eval_count": None, "status": "tool_call_rejected"},
            {"prompt_eval_count": 180, "eval_count": 30},
        ]
    )
    assert usage == {
        "exact": False,
        "input_tokens": 280,
        "output_tokens": 50,
        "input_tokens_upper_bound": 460,
        "output_tokens_upper_bound": 50 + LIMITS.max_tokens,
    }
    last_unknown = agent_module.agent_token_usage([{"prompt_eval_count": None, "eval_count": None}])
    assert last_unknown["input_tokens_upper_bound"] is None


def test_api_agent_omits_reasoning_authenticates_and_accounts(setup, monkeypatch):
    from page_cite_rag.api_backend import ApiChatClient
    from page_cite_rag.generation import GenerationConfig

    workflow, _, blocks, _, _, _ = setup
    monkeypatch.setenv("PAGECITE_API_KEY", "agent-key-canary")
    monkeypatch.setattr(ApiChatClient, "_process_calls", 0)
    monkeypatch.setattr(ApiChatClient, "_process_prompt_tokens", 0)
    monkeypatch.setattr(ApiChatClient, "_process_completion_tokens", 0)
    requests = []
    with ApiChatClient(
        GenerationConfig(),
        api_base="https://provider.invalid/v1",
        model="cloud-model",
        transport=httpx.MockTransport(lambda r: pytest.fail("writer uses workflow's fake client")),
    ) as client:

        def handler(request):
            body = json.loads(request.read())
            requests.append(body)
            assert "reasoning_effort" not in body
            assert body["model"] == "cloud-model"
            assert body["max_tokens"] == 512
            assert body["parallel_tool_calls"] is False
            assert request.headers["authorization"] == "Bearer agent-key-canary"
            return response(request, "finish", {"passage_ids": [f"P{blocks[0].ordinal}"]})

        mock_agent_http(monkeypatch, client, handler)
        result = run_evidence_agent(workflow, client, "Who was excluded?", language="en")
        assert result.answer.status == "answered"
        assert len(requests) == client.calls_used == 1
        assert client.token_totals == {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        }
        assert "agent-key-canary" not in json.dumps(result.trace)


@pytest.mark.parametrize("failure", ["reasoning", "reasoning_content", "http", "limit"])
def test_api_agent_failures_preserve_guards_and_safe_errors(monkeypatch, failure):
    from page_cite_rag.api_backend import ApiChatClient
    from page_cite_rag.generation import GenerationConfig

    secret = "api-agent-secret-canary"
    monkeypatch.setenv("PAGECITE_API_KEY", secret)
    monkeypatch.setattr(ApiChatClient, "_process_calls", 1 if failure == "limit" else 0)
    sent, after = [], []

    def handler(request):
        sent.append(request)
        if failure == "http":
            return server_error(request, secret)
        return response(request, **{failure: "hidden reasoning"})

    with ApiChatClient(
        GenerationConfig(),
        api_base="https://provider.invalid/v1",
        model="cloud",
        max_calls=1,
        transport=httpx.MockTransport(lambda r: pytest.fail("unused")),
    ) as client:
        guard = GuardedTransport(
            client,
            inner=httpx.MockTransport(handler),
            after_model_call=after.append,
        )
        with pytest.raises(ModelError) as caught:
            guard.handle_request(
                httpx.Request(
                    "POST",
                    "https://provider.invalid/v1/chat/completions",
                    json={
                        "max_tokens": 512,
                        "parallel_tool_calls": False,
                    },
                )
            )
        assert len(sent) == (0 if failure == "limit" else 1)
        assert len(after) == 1
        assert secret not in str(caught.value) + repr(caught.value) + json.dumps(after)
        if sent:
            assert "authorization" not in sent[0].headers


def test_findings_handoff_validates_individually_and_maps_sources(setup, monkeypatch):
    workflow, client, blocks, _, model, _ = setup
    pid = f"P{blocks[0].ordinal}"
    good = {"fact": "Children were excluded", "passage_id": pid}
    bad = [
        {"fact": "", "passage_id": pid},
        {"fact": " " * 3, "passage_id": pid},
        {"fact": "x" * 201, "passage_id": pid},
        {"fact": 3, "passage_id": pid},
        {"fact": "unknown", "passage_id": "P999"},
        "malformed",
    ]
    requests = []

    def handler(request):
        requests.append(json.loads(request.read()))
        return response(request, "finish", {"passage_ids": [pid], "findings": [good, *bad]})

    mock_agent_http(monkeypatch, client, handler)
    result = run_evidence_agent(
        workflow, client, "Who was excluded?", language="en", handoff_findings=True
    )
    schema = requests[0]["tools"][-1]["function"]["parameters"]
    assert schema["properties"]["findings"]["maxItems"] == 8
    assert "findings" not in schema["required"]
    assert "Zentro phone is made by Kalvex Corp" in requests[0]["messages"][0]["content"]
    assert result.trace["findings"] == [good]
    assert [r["finding"] for r in result.trace["findings_rejected"]] == bad
    assert result.trace["handoff_findings"] is True
    assert result.trace["model_call_limit"] == 8
    assert result.answer.status == "answered"
    user = model.calls[-1][0][1].content
    assert (
        "Findings reported by the search (unverified hints, not evidence):\n"
        f"Children were excluded [S{blocks[0].ordinal}]" in user
    )
    assert workflow._search_findings == ()
    workflow.answer_from_context(
        "Who was excluded?",
        blocks,
        language="en",
        search_steps=["entity"],
        search_findings=[(good["fact"], pid)],
    )
    user = model.calls[-1][0][1].content
    assert user.index("Search steps") < user.index("Findings reported") < user.index("Question:")
    workflow.answer_from_context("Who was excluded?", blocks, language="en")
    assert "Findings reported" not in model.calls[-1][0][1].content


def test_findings_schema_disabled_and_bounds(setup):
    workflow, _, blocks, _, _, _ = setup
    initial = workflow.search_sources("Question?")[0]
    state = EvidenceTools(blocks, initial, lambda q: initial)
    assert "findings" not in state.tools()[-1].args_schema.model_json_schema()["properties"]
    state = EvidenceTools(blocks, initial, lambda q: initial, handoff_findings=True)
    pid = state.initial_ids[0]
    good = {"fact": "f" * 200, "passage_id": pid}
    state.tools()[-1].invoke({"passage_ids": [], "findings": [good] * 9})
    assert state.findings == [good] * 8
    assert len(state.findings_rejected) == 1
    assert state.handoff()[2] == "finish_empty"


def test_twelve_model_calls_and_thirteenth_blocked(setup, monkeypatch):
    workflow, client, _, _, model, _ = setup
    requests = []

    def handler(request):
        requests.append(json.loads(request.read()))
        return response(request)

    guards = mock_agent_http(monkeypatch, client, handler)
    result = run_evidence_agent(
        workflow, client, "Who was excluded?", language="en", model_calls=12
    )
    assert len(requests) == 12
    assert result.trace["model_call_limit"] == 12
    assert "at most 12 steps" in requests[0]["messages"][0]["content"]
    assert "Steps used: 0 of 12." in requests[0]["messages"][1]["content"]
    assert (
        "Steps used: 11 of 12. Your next step is the last one"
        in requests[11]["messages"][-1]["content"]
    )
    with pytest.raises(ModelError, match="ceiling"):
        guards[0].handle_request(httpx.Request("POST", "http://synthetic", json=CAPPED))
    assert len(requests) == 12
    assert len(model.calls) == 1


@pytest.mark.parametrize("limit", [7, 13, 8.5, True, None])
def test_invalid_model_call_limit_sends_nothing(setup, limit):
    workflow, client, _, _, model, checks = setup
    with pytest.raises(ValueError):
        run_evidence_agent(workflow, client, "Question?", language="en", model_calls=limit)
    assert not model.calls and not checks


@pytest.mark.parametrize(
    "findings",
    [
        None,
        "fact",
        [("", "P1")],
        [("x" * 201, "P1")],
        [("fact", "P999")],
        [123],
        [("fact",)],
        [("fact", [])],
    ],
)
def test_invalid_writer_findings(setup, findings):
    workflow, _, blocks, _, model, _ = setup
    with pytest.raises(ValueError):
        workflow.answer_from_context("Question?", blocks, search_findings=findings)
    assert not model.calls


def test_optional_findings_omitted_preserves_writer_and_trace(setup, monkeypatch):
    workflow, client, blocks, _, model, _ = setup
    pid = f"P{blocks[0].ordinal}"
    mock_agent_http(
        monkeypatch,
        client,
        lambda request: response(request, "finish", {"passage_ids": [pid]}),
    )
    baseline = run_evidence_agent(workflow, client, "Who was excluded?", language="en")
    baseline_messages = model.calls[-1][0]
    candidate = run_evidence_agent(
        workflow,
        client,
        "Who was excluded?",
        language="en",
        handoff_findings=True,
    )
    assert baseline.trace["handoff_findings"] is False
    assert baseline.trace["findings"] == baseline.trace["findings_rejected"] == []
    assert baseline.trace["model_call_limit"] == LIMITS.model_calls
    assert candidate.trace["findings"] == candidate.trace["findings_rejected"] == []
    assert len(candidate.trace["tool_calls"]) == 1
    assert candidate.trace["tool_calls"][0]["status"] == "ok"
    assert model.calls[-1][0] == baseline_messages


def test_findings_unseen_passage_and_invalid_container_do_not_break_finish(setup):
    workflow, _, blocks, _, _, _ = setup
    initial = workflow.search_sources("Question?")[0]
    initial = initial.model_copy(update={"hits": initial.hits[:1]})
    for findings in ([{"fact": "unseen", "passage_id": f"P{blocks[1].ordinal}"}], None):
        state = EvidenceTools(blocks, initial, lambda q: initial, handoff_findings=True)
        assert (
            state.tools()[-1].invoke({"passage_ids": state.initial_ids, "findings": findings})
            == "Evidence selection complete."
        )
        assert not state.findings and len(state.findings_rejected) == 1
        assert state.handoff()[2] == "finish_called"
