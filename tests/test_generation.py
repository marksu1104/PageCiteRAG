from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from page_cite_rag.generation import (
    ChatMessage,
    ContextBudgetError,
    GenerationConfig,
    ModelError,
    ModelTimeoutError,
    OllamaModelClient,
    StructuredOutputError,
    check_budget,
)


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str


MESSAGES = [ChatMessage(role="user", content="A synthetic question.")]


def response():
    return {
        "model": GenerationConfig().model,
        "done": True,
        "done_reason": "stop",
        "message": {"role": "assistant", "content": '{"answer":"not significant unless n>64"}'},
        "prompt_eval_count": 30,
        "eval_count": 12,
        "total_duration": 20000,
        "load_duration": 1000,
        "prompt_eval_duration": 1000,
        "eval_duration": 18000,
    }


def backend(*, chat=None, digest=None, runtime="0.35.0", metadata=None):
    calls = []
    config = GenerationConfig()

    def handler(request):
        assert request.url.host == "127.0.0.1"
        assert "authorization" not in request.headers
        path = request.url.path
        calls.append((path, json.loads(request.content) if request.content else None))
        if path == "/api/version":
            return httpx.Response(200, json={"version": runtime})
        if path == "/api/tags":
            return httpx.Response(
                200, json={"models": [{"name": config.model, "digest": digest or config.digest}]}
            )
        if path == "/api/show":
            return httpx.Response(
                200,
                json=metadata
                or {
                    "capabilities": ["completion", "thinking"],
                    "details": {"quantization_level": "Q4_K_M"},
                    "model_info": {
                        "general.architecture": "qwen35",
                        "qwen35.context_length": 262144,
                    },
                    "template": "a pinned synthetic template",
                },
            )
        if path == "/api/ps":
            return httpx.Response(200, json={"models": []})
        if path == "/api/chat":
            if callable(chat):
                return chat(request)
            return httpx.Response(200, json=chat if chat is not None else response())
        raise AssertionError(f"unexpected endpoint {path}")

    return httpx.MockTransport(handler), calls


def test_structured_generation_pins_controls_and_reuses_client(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "https://remote.invalid")
    monkeypatch.setenv("HTTP_PROXY", "http://remote.invalid:3128")
    monkeypatch.setenv("OLLAMA_API_KEY", "private-env-canary")
    transport, calls = backend()
    with OllamaModelClient(GenerationConfig(), transport=transport) as client:
        for _ in range(2):
            result = client.generate(MESSAGES, Output)
            assert result.value.answer == "not significant unless n>64"
            assert result.stats.prompt_eval_count == 30
        assert client.loaded_state() == []
    chats = [payload for path, payload in calls if path == "/api/chat"]
    assert len(chats) == 2
    assert sum(path == "/api/show" for path, _ in calls) == 1
    assert sum(path == "/api/tags" for path, _ in calls) == 3
    payload = chats[0]
    assert payload["think"] is False and payload["truncate"] is False
    assert payload["stream"] is False and payload["keep_alive"] == "30m"
    assert payload["options"]["num_ctx"] == 16384
    assert payload["options"]["num_predict"] == 2048
    assert payload["options"]["num_gpu"] == 0
    assert payload["format"] == Output.model_json_schema()
    assert "tools" not in payload
    with pytest.raises(ModelError, match="closed"):
        client.loaded_state()


@pytest.mark.parametrize(
    "host",
    [
        "https://127.0.0.1:11435",
        "http://example.com:11435",
        "http://localhost:11435",
        "http://127.0.0.1",
        "http://127.0.0.1:99999",
        "http://u:p@127.0.0.1:11435",
        "http://127.0.0.1:11435/path",
        "http://127.0.0.1:11435?key=secret",
        "http://127.0.0.1:11435#fragment",
        "http://192.168.0.2:11435",
    ],
)
def test_remote_ambiguous_or_credential_hosts_rejected(host):
    with pytest.raises(ValidationError):
        GenerationConfig(host=host)


def test_ipv6_loopback_and_invalid_settings():
    GenerationConfig(host="http://[::1]:11435")
    for settings in (
        {"num_ctx": True},
        {"timeout_seconds": float("inf")},
        {"num_predict": -1},
        {"keep_alive": "0"},
        {"digest": "latest"},
        {"model": "qwen:cloud", "num_ctx": 2048},
    ):
        with pytest.raises(ValidationError):
            GenerationConfig(**settings)


def test_utf8_budget_rejects_before_any_new_network_request():
    transport, calls = backend()
    with OllamaModelClient(GenerationConfig(), transport=transport) as client:
        previous_calls = len(calls)
        with pytest.raises(ContextBudgetError):
            client.generate([ChatMessage(role="user", content="中文" * 3000)], Output)
        assert len(calls) == previous_calls
    budget = check_budget([ChatMessage(role="user", content="中")], {}, GenerationConfig())
    assert budget.input_upper_estimate == 3 + 64 + 2 + 1024
    with pytest.raises(ValueError):
        check_budget([], {}, GenerationConfig())


def test_runtime_and_digest_drift_fail_without_chat():
    for settings in ({"digest": "f" * 64}, {"runtime": "0.34.0"}):
        transport, calls = backend(**settings)
        with pytest.raises(ModelError):
            OllamaModelClient(GenerationConfig(), transport=transport)
        assert all(path != "/api/chat" for path, _ in calls)


def test_digest_verified_again_before_each_generation():
    transport, calls = backend()
    with OllamaModelClient(GenerationConfig(), transport=transport) as client:
        # A changed model list must not cause a pull or automatic fallback.
        original_request = client._request

        def drift(path, payload=None):
            return {"models": []} if path == "/api/tags" else original_request(path, payload)

        client._request = drift
        with pytest.raises(ModelError, match="missing"):
            client.generate(MESSAGES, Output)
    assert all(path != "/api/chat" for path, _ in calls)


@pytest.mark.parametrize(
    "metadata",
    [
        {"capabilities": ["completion"]},
        {"capabilities": "thinking"},
        {"capabilities": ["completion", "thinking"], "details": [], "model_info": {}},
        {"capabilities": ["completion", "thinking"], "model_info": {"x.context_length": 999}},
        {"remote_host": "https://ollama.com"},
    ],
)
def test_incompatible_metadata_stops(metadata):
    transport, _ = backend(metadata=metadata)
    with pytest.raises(ModelError):
        OllamaModelClient(GenerationConfig(), transport=transport)


@pytest.mark.parametrize(
    "change",
    [
        {"done": False},
        {"done_reason": "length"},
        {"model": "other:latest"},
        {"message": {"role": "assistant", "content": '{"answer":"x"}', "thinking": "secret"}},
        {"message": {"role": "assistant", "content": '{"answer":"x"}', "tool_calls": [{}]}},
        {"message": {"role": "user", "content": '{"answer":"x"}'}},
        {"message": {"role": "assistant", "content": '{"answer":12}'}},
        {"message": {"role": "assistant", "content": '{"answer":"x","extra":true}'}},
        {"message": {"role": "assistant", "content": '```json\n{"answer":"x"}\n```'}},
        {"eval_count": "12"},
        {"load_duration": -1},
        {"load_duration": None},
    ],
)
def test_invalid_incomplete_or_unexpected_output_is_not_repaired(change):
    transport, calls = backend(chat=response() | change)
    with OllamaModelClient(GenerationConfig(), transport=transport) as client:
        with pytest.raises(StructuredOutputError):
            client.generate(MESSAGES, Output)
    assert sum(path == "/api/chat" for path, _ in calls) == 1


@pytest.mark.parametrize(
    "change",
    [
        {"prompt_eval_count": 0},
        {"prompt_eval_count": 10000},
        {"eval_count": 2049},
    ],
)
def test_invalid_runtime_token_counts(change):
    transport, _ = backend(chat=response() | change)
    with OllamaModelClient(GenerationConfig(), transport=transport) as client:
        with pytest.raises(ContextBudgetError):
            client.generate(MESSAGES, Output)


@pytest.mark.parametrize("kind", ["timeout", "connection", "redirect", "http", "json", "large"])
def test_transport_failures_have_safe_errors_and_no_retry(kind):
    secret = "raw-document-and-server-secret"

    def failed(request):
        if kind == "timeout":
            raise httpx.ReadTimeout(secret, request=request)
        if kind == "connection":
            raise httpx.ConnectError(secret, request=request)
        if kind == "redirect":
            return httpx.Response(307, headers={"location": "https://remote.invalid"})
        if kind == "http":
            return httpx.Response(500, json={"error": secret})
        if kind == "json":
            return httpx.Response(200, content=secret)
        return httpx.Response(200, content=b"x" * (2 * 1024 * 1024 + 1))

    transport, calls = backend(chat=failed)
    with OllamaModelClient(GenerationConfig(), transport=transport) as client:
        with pytest.raises(ModelTimeoutError if kind == "timeout" else ModelError) as error:
            client.generate(MESSAGES, Output)
        assert secret not in str(error.value)
    assert sum(path == "/api/chat" for path, _ in calls) == 1
