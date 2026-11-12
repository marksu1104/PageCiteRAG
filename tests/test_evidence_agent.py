"""Evidence-agent integration using synthetic sources and mock HTTP only."""

import json
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
        ollama_model_name="synthetic",
        before_agent_dispatch=lambda: checks.append("dispatch"),
    )
    return workflow, client, blocks, retriever, model, checks


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
        requests.append(json.loads(request.read()))
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
    assert trace["version"] == "evidence-agent/v2"
    assert trace["selected_passage_ids"] == [pids[2], pids[0]]
    assert trace["handoff_passage_ids"] == [pids[2], pids[0], pids[1]]
    assert trace["handoff_reason"] == "finish_called"
    assert [r["status"] for r in trace["tool_calls"]] == ["ok"] * 3
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
    source.write_text("\n\n".join(f"Passage {i} contains synthetic evidence." for i in range(45)))
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
        guard.handle_request(
            httpx.Request("POST", "http://synthetic", json={"reasoning_effort": "none"})
        )
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
        guard.handle_request(
            httpx.Request("POST", "http://synthetic", json={"reasoning_effort": "none"})
        )
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
    request = httpx.Request("POST", "http://synthetic", json={"reasoning_effort": "none"})
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
    assert client.ollama_model_name == "pinned-local"
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
