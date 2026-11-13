"""Cloud option contracts: synthetic content, MockTransport, no network or paid calls."""

import json
import time

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from page_cite_rag.api_backend import ApiChatClient, validate_api_base
from page_cite_rag.generation import (
    ChatMessage,
    ContextBudgetError,
    GenerationConfig,
    ModelError,
    ModelTimeoutError,
    OutputValidationError,
    budget_for_model,
)

KEY = "private-api-key-canary"
MESSAGES = [ChatMessage(role="user", content="Synthetic document question")]


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str


@pytest.fixture(autouse=True)
def isolated_api_process(monkeypatch):
    monkeypatch.setenv("PAGECITE_API_KEY", KEY)
    for attr in ("_process_calls", "_process_prompt_tokens", "_process_completion_tokens"):
        monkeypatch.setattr(ApiChatClient, attr, 0)


def completion(content="plain answer", finish="stop", **changes):
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": finish}
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 3},
        **changes,
    }


def make_client(handler, **kwargs):
    return ApiChatClient(
        GenerationConfig(),
        api_base="https://provider.invalid/v1",
        model="cloud-model",
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


@pytest.mark.parametrize("structured", [False, True])
def test_success_request_usage_budget_and_key_privacy(structured, monkeypatch):
    requests = []
    monkeypatch.setenv("HTTPS_PROXY", "https://unwanted.invalid")

    def handler(request):
        assert request.headers["authorization"] == f"Bearer {KEY}"
        assert str(request.url) == "https://provider.invalid/v1/chat/completions"
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=completion('{"answer":"yes"}' if structured else "yes"))

    with make_client(handler) as client:
        result = client.generate(MESSAGES, Output if structured else None)
        assert (result.value.answer if structured else result.value) == "yes"
        assert result.stats.prompt_eval_count == 20
        assert result.stats.eval_count == 3
        assert result.stats.total_duration == int(result.stats.elapsed_ms * 1_000_000)
        assert client.calls_used == 1
        assert client.token_totals == {
            "prompt_tokens": 20,
            "completion_tokens": 3,
            "total_tokens": 23,
        }
        assert not hasattr(client, "prepare_budget")
        assert budget_for_model(client, MESSAGES, None, client.config).exact_input_tokens is None
        assert client.identity == {
            "backend": "api",
            "api_host": "provider.invalid",
            "model": "cloud-model",
        }
        assert KEY not in repr(client) + json.dumps(client.identity)
    body = requests[0]
    assert body == {
        "model": "cloud-model",
        "messages": [m.model_dump() for m in MESSAGES],
        "max_tokens": 3072,
        "temperature": 0.0,
        "stream": False,
        **(
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "Output",
                        "schema": Output.model_json_schema(),
                        "strict": False,
                    },
                }
            }
            if structured
            else {}
        ),
    }


@pytest.mark.parametrize(
    "content,finish,reason",
    [
        ('{"answer":1}', "stop", "schema_invalid"),
        ('{"answer":"yes","extra":true}', "stop", "schema_invalid"),
        ('{"answer":', "length", "output_limit"),
    ],
)
def test_output_failures_keep_raw_usage_without_repair(content, finish, reason):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=completion(content, finish))

    with make_client(handler) as client:
        with pytest.raises(OutputValidationError) as caught:
            client.generate(MESSAGES, Output)
        assert caught.value.reason == reason
        assert caught.value.raw_model_output == content
        assert caught.value.accounting.prompt_eval_count == 20
        assert caught.value.accounting.eval_count == 3
        assert client.token_totals["total_tokens"] == 23
        assert KEY not in str(caught.value) + repr(caught.value)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "kind", ["http", "http_secret_type", "usage", "timeout", "connect", "json", "redirect"]
)
def test_failures_are_safe_and_never_retried(kind):
    calls = []

    def handler(request):
        calls.append(True)
        if kind in ("timeout", "connect"):
            cls = httpx.ReadTimeout if kind == "timeout" else httpx.ConnectError
            raise cls(KEY + " document content", request=request)
        if kind == "json":
            return httpx.Response(200, content=KEY)
        if kind == "redirect":
            return httpx.Response(307, headers={"location": "https://other.invalid"})
        if kind == "usage":
            return httpx.Response(200, json=completion(usage={"prompt_tokens": 20}))
        return httpx.Response(
            401,
            json={
                "error": {
                    "type": "authentication_error" if kind == "http" else KEY,
                    "message": KEY + " document content",
                }
            },
        )

    with make_client(handler) as client:
        with pytest.raises(ModelError) as caught:
            client.generate(MESSAGES, None)
        assert KEY not in str(caught.value) + repr(caught.value) + json.dumps(
            client.identity
        ) + repr(client)
        if kind == "http":
            assert str(caught.value) == "API HTTP 401: authentication_error"
        assert client.calls_used == 1
    assert len(calls) == 1


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": True, "completion_tokens": 2},
        {"prompt_tokens": 20, "completion_tokens": -1},
    ],
)
def test_invalid_usage_fails(usage):
    with make_client(lambda r: httpx.Response(200, json=completion(usage=usage))) as client:
        with pytest.raises(ModelError, match="usage"):
            client.generate(MESSAGES, None)


def test_missing_key_and_custom_environment(monkeypatch):
    monkeypatch.delenv("PAGECITE_API_KEY")
    with pytest.raises(ModelError, match="environment variable is missing"):
        make_client(lambda r: pytest.fail("no dispatch"))
    monkeypatch.setenv("CUSTOM_KEY", KEY)
    with make_client(
        lambda r: httpx.Response(200, json=completion()), api_key_env="CUSTOM_KEY"
    ) as client:
        client.generate(MESSAGES, None)


def test_process_limit_covers_writer_agent_failures_and_new_clients():
    sent = []

    def handler(request):
        sent.append(True)
        return httpx.Response(503, json={"error": {"type": "server_error"}})

    with make_client(handler, max_calls=2) as client:
        with pytest.raises(ModelError, match="503"):
            client.generate(MESSAGES, None)
        client.before_agent_dispatch()
        assert client.calls_used == 1  # The guard alone spends nothing.
        with pytest.raises(ModelError, match="503"):
            client.dispatch_agent_request(
                httpx.MockTransport(handler), httpx.Request("POST", "https://provider.invalid/v1")
            )
        assert client.calls_used == 2
        with pytest.raises(ModelError, match="call limit"):
            client.generate(MESSAGES, None)
    with make_client(handler, max_calls=2) as replacement:
        with pytest.raises(ModelError, match="call limit"):
            replacement.before_agent_dispatch()
        assert replacement.calls_used == 2
    assert len(sent) == 2


def test_budget_and_deadline_reject_before_dispatch():
    with make_client(lambda r: pytest.fail("must not dispatch")) as client:
        with pytest.raises(ContextBudgetError):
            client.generate([ChatMessage(role="user", content="中" * 10000)], None)
        client.set_request_deadline(time.monotonic() - 1)
        with pytest.raises(ModelTimeoutError):
            client.generate(MESSAGES, None)
        assert client.calls_used == 0


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.invalid/v1",
        "https://u:secret@provider.invalid/v1",
        "https://provider.invalid/v1?key=secret",
        "https://provider.invalid/#x",
        "https://",
        "ftp://provider.invalid",
        "https://provider.invalid:99999",
    ],
)
def test_unsafe_api_base_rejected(url):
    with pytest.raises(ModelError, match="HTTPS URL"):
        validate_api_base(url)


@pytest.mark.parametrize(
    "url", ["https://provider.invalid/v1", "http://127.0.0.1:8000/v1", "http://localhost:8000/v1"]
)
def test_valid_bases(url):
    assert validate_api_base(url + "/") == url


def test_transport_failure_scrubs_retained_request():
    retained = []

    def handler(request):
        retained.append(request)
        raise httpx.ReadTimeout(KEY, request=request)

    with make_client(handler) as client:
        with pytest.raises(ModelError):
            client.generate(MESSAGES, None)
    assert "authorization" not in retained[0].headers


def test_schema_is_budgeted_even_without_prompt_schema():
    with make_client(lambda r: pytest.fail("schema must be budgeted before dispatch")) as client:

        class LargeSchema(BaseModel):
            answer: str

            @classmethod
            def model_json_schema(cls, **kwargs):
                return {"description": "x" * client.config.num_ctx}

        with pytest.raises(ContextBudgetError):
            client.generate(MESSAGES, LargeSchema, include_schema_in_prompt=False)
        assert client.calls_used == 0


def test_agent_preserves_timeout_and_does_not_spend_call_on_ledger_failure():
    from page_cite_rag.evidence_agent import GuardedTransport

    requests = []

    def handler(request):
        requests.append(request)
        assert request.extensions["timeout"] == {"read": 7.0}
        assert request.headers["authorization"] == f"Bearer {KEY}"
        return httpx.Response(200, json=completion())

    def failed_ledger():
        raise ModelError("ledger exhausted")

    with make_client(handler) as client:
        request = httpx.Request(
            "POST",
            "https://provider.invalid/v1/chat/completions",
            json={"max_tokens": 512},
            extensions={"timeout": {"read": 7.0}},
        )
        guard = GuardedTransport(
            client,
            inner=httpx.MockTransport(handler),
            before_model_call=failed_ledger,
        )
        with pytest.raises(ModelError, match="ledger exhausted"):
            guard.handle_request(request)
        assert client.calls_used == len(requests) == 0
        guard.before_model_call = None
        guard.handle_request(request)
        assert client.calls_used == len(requests) == 1


def test_closed_client_does_not_consume_call():
    client = make_client(lambda r: pytest.fail("no dispatch"))
    client.close()
    with pytest.raises(ModelError, match="closed"):
        client.generate(MESSAGES, None)
    assert client.calls_used == 0
