from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

import page_cite_rag.llama as llama
from page_cite_rag.generation import (
    ChatMessage,
    ContextBudgetError,
    ModelError,
    ModelTimeoutError,
    OutputValidationError,
    StructuredOutputError,
)


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str


MESSAGES = [ChatMessage(role="user", content="A synthetic question.")]


def completion():
    return {
        "model": "page-cite-rag-qwen35",
        "content": '{"answer":"not significant"}',
        "stop": True,
        "stop_type": "eos",
        "truncated": False,
        "generation_settings": {"n_predict": 1024},
        "tokens_evaluated": 30,
        "tokens_predicted": 12,
        "timings": {"prompt_ms": 20.0, "predicted_ms": 30.0},
    }


def make_client(
    tmp_path,
    monkeypatch,
    *,
    result=None,
    tokens=None,
    startup_failure=False,
    budget_policy="conservative",
    recover_output_errors=False,
    sampling_profile="greedy",
    sampling_seed=42,
    config_updates=None,
):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "fake-model").write_bytes(b"fake weights")
    (bundle / "manifest.json").write_text("{}")
    monkeypatch.setattr(llama, "verify_bundle", lambda *_: {"files": {"fake-model": "test"}})
    calls = []
    lifecycle = {"starts": 0, "stops": 0, "returncode": None}

    def handler(request):
        assert request.url.host == "127.0.0.1"
        assert "secret-env" not in str(request.headers)
        payload = json.loads(request.content) if request.content else None
        calls.append((request.url.path, payload))
        if request.url.path == "/props":
            return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 16384}})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": tokens if tokens is not None else [7] * 30})
        if request.url.path == "/v1/chat/completions/input_tokens":
            return httpx.Response(
                200,
                json={
                    "input_tokens": (len(tokens) if tokens is not None else 30)
                    if (config_updates or {}).get("thinking")
                    else 128
                },
            )
        if request.url.path == "/apply-template":
            return httpx.Response(200, json={"prompt": "<|im_start|>assistant\n<think>\n"})
        if request.url.path in ("/completion", "/v1/chat/completions"):
            return (
                result(request)
                if callable(result)
                else httpx.Response(
                    200, content=json.dumps(result if result is not None else completion()).encode()
                )
            )
        raise AssertionError("unexpected endpoint")

    def start(self, **kwargs):
        lifecycle["starts"] += 1
        self._process = SimpleNamespace(pid=12345, poll=lambda: lifecycle["returncode"])
        self._http = httpx.Client(
            base_url="http://127.0.0.1:12345",
            trust_env=False,
            follow_redirects=False,
            transport=httpx.MockTransport(handler),
        )
        if startup_failure:
            raise ModelError("synthetic startup failure")

    def terminate(self):
        if self._process is not None and lifecycle["returncode"] is None:
            lifecycle["stops"] += 1
            lifecycle["returncode"] = 0

    monkeypatch.setattr(llama.LlamaModelClient, "_start", start)
    monkeypatch.setattr(llama.LlamaModelClient, "_terminate_process", terminate)
    config = llama.LlamaConfig(bundle_dir=bundle, **(config_updates or {}))
    return (
        llama.LlamaModelClient(
            config,
            budget_policy=budget_policy,
            recover_output_errors=recover_output_errors,
            sampling_profile=sampling_profile,
            sampling_seed=sampling_seed,
        ),
        calls,
        lifecycle,
    )


def test_sequential_requests_reuse_owned_server_and_close_once(tmp_path, monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://remote.invalid")
    monkeypatch.setenv("OLLAMA_API_KEY", "secret-env")
    client, calls, lifecycle = make_client(tmp_path, monkeypatch)
    with client:
        for _ in range(2):
            result = client.generate(MESSAGES, Output)
            assert result.value.answer == "not significant"
            assert result.stats.load_duration is None
            assert result.stats.total_duration == 50_000_000
            assert client.observations()["server_pid"] == 12345
    assert lifecycle == {"starts": 1, "stops": 1, "returncode": 0}
    client.close()
    assert lifecycle["stops"] == 1
    with pytest.raises(ModelError, match="closed"):
        client.generate(MESSAGES, Output)
    payload = next(payload for path, payload in calls if path == "/completion")
    assert payload["prompt"] == [7] * 30
    assert payload["json_schema"] == Output.model_json_schema()
    assert payload["stream"] is False and payload["cache_prompt"] is True
    assert payload["temperature"] == 0 and payload["n_predict"] == 1024
    assert "tools" not in payload


@pytest.mark.parametrize("budget_policy", ["exact", "conservative"])
def test_explicit_plain_text_uses_same_runtime_budget_and_usage(
    tmp_path, monkeypatch, budget_policy
):
    text = "The result is 42% [S1]."
    client, calls, lifecycle = make_client(
        tmp_path, monkeypatch, result=completion() | {"content": text}, budget_policy=budget_policy
    )
    with client:
        result = client.generate(MESSAGES, None)
        assert result.value == result.raw_model_output == text
        assert result.stats.prompt_eval_count == 30
        assert result.stats.eval_count == 12
        assert result.budget.output_reserved == 1024
    request = next(payload for path, payload in calls if path == "/completion")
    assert "json_schema" not in request
    assert request["n_predict"] == 1024 and request["temperature"] == 0
    plain_prompt = llama.render_prompt(MESSAGES, None)
    assert "Return only JSON" not in plain_prompt
    assert plain_prompt.endswith("<think>\n\n</think>\n\n")
    assert lifecycle["starts"] == lifecycle["stops"] == 1


@pytest.mark.parametrize("truncated", [False, True])
def test_plain_text_does_not_bypass_output_or_context_limits(tmp_path, monkeypatch, truncated):
    raw = completion() | {"content": "Partial text", "truncated": truncated}
    if not truncated:
        raw.update(stop_type="limit", tokens_predicted=1024)
    client, calls, lifecycle = make_client(tmp_path, monkeypatch, result=raw)
    with pytest.raises(ContextBudgetError if truncated else OutputValidationError):
        client.generate(MESSAGES, None)
    assert sum(path == "/completion" for path, _ in calls) == 1
    assert lifecycle["stops"] == 1


def test_qwen_general_sampling_is_explicit_and_server_confirmed(tmp_path, monkeypatch):
    expected = {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repeat_penalty": 1.0,
        "seed": 2026,
    }

    def reply(request):
        payload = json.loads(request.content)
        settings = {"n_predict": 1024, **{key: payload[key] for key in expected}}
        return httpx.Response(200, json={**completion(), "generation_settings": settings})

    client, calls, _ = make_client(
        tmp_path,
        monkeypatch,
        result=reply,
        sampling_profile="qwen-general",
        sampling_seed=2026,
    )
    with client:
        assert client.generate(MESSAGES, Output).value.answer == "not significant"
    payload = next(payload for path, payload in calls if path == "/completion")
    assert {key: payload[key] for key in expected} == expected


def test_longbench_sampling_profile_uses_official_temperature_and_is_confirmed(
    tmp_path, monkeypatch
):
    expected = {"temperature": 0.1, "seed": 42, "repeat_penalty": 1.0}

    def reply(request):
        payload = json.loads(request.content)
        settings = {"n_predict": 1024, **{key: payload[key] for key in expected}}
        return httpx.Response(200, json={**completion(), "generation_settings": settings})

    client, calls, _ = make_client(
        tmp_path,
        monkeypatch,
        result=reply,
        sampling_profile="longbench-official",
    )
    with client:
        result = client.generate(MESSAGES, None)

    payload = next(payload for path, payload in calls if path == "/completion")
    assert {key: payload[key] for key in expected} == expected
    assert result.value == '{"answer":"not significant"}'


def test_longbench_sampling_profile_rejects_ignored_temperature(tmp_path, monkeypatch):
    response = {
        **completion(),
        "generation_settings": {
            "n_predict": 1024,
            "temperature": 0.0,
            "seed": 42,
            "repeat_penalty": 1.0,
        },
    }
    client, _, lifecycle = make_client(
        tmp_path,
        monkeypatch,
        result=response,
        sampling_profile="longbench-official",
    )

    with pytest.raises(ModelError, match="server did not confirm sampling setting"):
        client.generate(MESSAGES, None)

    assert lifecycle["stops"] == 1


def test_qwen_general_rejects_silently_ignored_setting(tmp_path, monkeypatch):
    response = {**completion(), "generation_settings": {"n_predict": 1024}}
    client, _, lifecycle = make_client(
        tmp_path,
        monkeypatch,
        result=response,
        sampling_profile="qwen-general",
    )
    with pytest.raises(ModelError, match="server did not confirm sampling setting"):
        client.generate(MESSAGES, Output)
    assert lifecycle["stops"] == 1


def thinking_completion(*, limit=False, final='{"answer":"not significant"}'):
    raw = "Evidence reasoning.\n</think>\n\n" + final
    native = completion() | {
        "content": raw,
        "tokens_predicted": 6144 if limit else 52,
        "stop_type": "limit" if limit else "eos",
        "generation_settings": {
            "n_predict": 6144,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repeat_penalty": 1.0,
            "seed": 42,
            "reasoning_format": "deepseek",
            "reasoning_in_content": False,
            "generation_prompt": "<|im_start|>assistant\n<think>\n",
            "grammar": 'root ::= "reasoning followed by final JSON"',
        },
    }
    return {
        "model": "page-cite-rag-qwen35",
        "choices": [
            {
                "index": 0,
                "finish_reason": "length" if limit else "stop",
                "message": {
                    "role": "assistant",
                    "content": final,
                    "reasoning_content": "Evidence reasoning.",
                },
            }
        ],
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": native["tokens_predicted"],
            "total_tokens": 30 + native["tokens_predicted"],
        },
        "__verbose": native,
    }


def thinking_client(tmp_path, monkeypatch, **kwargs):
    return make_client(
        tmp_path,
        monkeypatch,
        budget_policy="exact",
        sampling_profile="qwen-general",
        config_updates={"thinking": True, "reasoning_budget_tokens": 4096, "num_predict": 6144},
        **kwargs,
    )


def test_thinking_uses_native_parser_and_accounts_for_all_generated_tokens(tmp_path, monkeypatch):
    response = thinking_completion()
    client, calls, lifecycle = thinking_client(tmp_path, monkeypatch, result=response)
    with client:
        result = client.generate(MESSAGES, Output)
        assert result.value.answer == "not significant"
        assert result.raw_model_output == response["choices"][0]["message"]["content"]
        assert result.stats.eval_count == 52  # Includes reasoning, not only final JSON.
        assert result.budget.output_reserved == 6144
        assert client.completion_trace() == response
        trace = client.completion_trace()
        trace["usage"]["completion_tokens"] = -1
        assert client.completion_trace()["usage"]["completion_tokens"] == 52
        assert client.identity["thinking"] is True
    request = next(payload for path, payload in calls if path == "/v1/chat/completions")
    assert request["chat_template_kwargs"] == {"enable_thinking": True}
    assert request["reasoning_format"] == "deepseek"
    assert request["reasoning_budget_tokens"] == 4096 and request["max_tokens"] == 6144
    assert request["response_format"]["json_schema"]["schema"] == Output.model_json_schema()
    assert "json_schema" not in request and "grammar" not in request and "prompt" not in request
    assert all(path != "/completion" for path, _ in calls)
    assert lifecycle["starts"] == lifecycle["stops"] == 1


def test_thinking_length_failure_preserves_full_raw_reasoning_and_cost(tmp_path, monkeypatch):
    response = thinking_completion(limit=True, final="")
    response["choices"][0]["message"]["content"] = None
    client, calls, lifecycle = thinking_client(
        tmp_path, monkeypatch, result=response, recover_output_errors=True
    )
    with client:
        with pytest.raises(OutputValidationError) as caught:
            client.generate(MESSAGES, Output)
        assert caught.value.reason == "output_limit"
        assert caught.value.accounting.eval_count == 6144
        assert caught.value.raw_model_output == response["__verbose"]["content"]
        assert client.completion_trace() == response
        assert lifecycle["stops"] == 0
    assert sum(path == "/v1/chat/completions" for path, _ in calls) == 1


@pytest.mark.parametrize(
    "kind", ["usage", "reasoning", "empty_reasoning", "limit", "final", "grammar"]
)
def test_thinking_rejects_inconsistent_runtime_metadata(tmp_path, monkeypatch, kind):
    response = thinking_completion()
    if kind == "usage":
        response["usage"]["completion_tokens"] = 12
    elif kind == "reasoning":
        response["__verbose"]["generation_settings"]["reasoning_in_content"] = True
    elif kind == "empty_reasoning":
        response["choices"][0]["message"]["reasoning_content"] = ""
    elif kind == "limit":
        response["choices"][0]["finish_reason"] = "length"
    elif kind == "grammar":
        response["__verbose"]["generation_settings"]["grammar"] = ""
    else:
        response["choices"][0]["message"]["content"] = None
    client, _, lifecycle = thinking_client(tmp_path, monkeypatch, result=response)
    with pytest.raises(StructuredOutputError):
        client.generate(MESSAGES, Output)
    assert client.completion_trace() == response
    assert lifecycle["stops"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_predict": 6144},
        {"reasoning_budget_tokens": 100},
        {"thinking": True},
        {"thinking": True, "reasoning_budget_tokens": 1024},
    ],
)
def test_extended_output_requires_explicit_bounded_thinking(kwargs):
    with pytest.raises(ValidationError):
        llama.LlamaConfig(**kwargs)


def test_nonthinking_allows_planned_3072_output_budget():
    config = llama.LlamaConfig(num_predict=3072)
    assert config.thinking is False
    assert config.num_predict == 3072


def test_thinking_budget_rejects_chat_tokenizer_disagreement_before_generation(
    tmp_path, monkeypatch
):
    client, calls, lifecycle = thinking_client(tmp_path, monkeypatch)
    monkeypatch.setattr(client, "count_chat_input_tokens", lambda _: 31)
    with pytest.raises(ContextBudgetError, match="counts disagree"):
        client.generate(MESSAGES, Output)
    assert not any(path == "/v1/chat/completions" for path, _ in calls)
    assert lifecycle["stops"] == 1


def test_thinking_schema_failure_keeps_native_reasoning_and_never_repairs(tmp_path, monkeypatch):
    response = thinking_completion(final='{"answer":123}')
    client, calls, lifecycle = thinking_client(
        tmp_path, monkeypatch, result=response, recover_output_errors=True
    )
    with client:
        with pytest.raises(OutputValidationError) as caught:
            client.generate(MESSAGES, Output)
        assert caught.value.reason == "schema_invalid"
        assert caught.value.raw_model_output == response["__verbose"]["content"]
        assert caught.value.accounting.eval_count == 52
        assert lifecycle["stops"] == 0
    assert sum(path == "/v1/chat/completions" for path, _ in calls) == 1


def test_model_size_selects_an_explicit_identity_not_manifest_claims():
    assert llama.model_spec()["sha256"] == llama.MODEL_SHA256
    small = llama.model_spec("4b")
    assert small["file"] == "Qwen3.5-4B-Q4_K_M.gguf" and small["bytes"] == 2740937888
    assert small["sha256"] != llama.model_spec()["sha256"]
    reference = llama.model_spec("27b")
    assert reference == {
        "repo": "unsloth/Qwen3.5-27B-GGUF",
        "revision": "3221f178a6b842d04f1fb42f1c413534adcc0a6a",
        "file": "Qwen3.5-27B-Q4_K_M.gguf",
        "bytes": 16740812704,
        "sha256": "84b5f7f112156d63836a01a69dc3f11a6ba63b10a23b8ca7a7efaf52d5a2d806",
        "layers": 64,
    }
    assert llama.LlamaConfig().model_size == "9b"
    assert llama.LlamaConfig(model_size="27b").model_size == "27b"
    with pytest.raises(ValueError, match="unsupported"):
        llama.model_spec("other")


@pytest.mark.parametrize("text", ["中" * 6000, "padding " * 10000], ids=["chinese", "english"])
def test_oversized_input_rejected_without_request_and_client_remains_usable(
    tmp_path, monkeypatch, text
):
    client, calls, lifecycle = make_client(tmp_path, monkeypatch)
    with client:
        with pytest.raises(ContextBudgetError):
            client.generate([ChatMessage(role="user", content=text)], Output)
        assert not calls and lifecycle["returncode"] is None
        client.generate(MESSAGES, Output)


def test_exact_tokenization_rejects_before_completion(tmp_path, monkeypatch):
    client, calls, _ = make_client(tmp_path, monkeypatch, tokens=[5] * 16384)
    with client:
        with pytest.raises(ContextBudgetError, match="exact"):
            client.generate(MESSAGES, Output)
    assert [path for path, _ in calls] == ["/props", "/tokenize"]


def test_exact_tokenization_enforces_the_twelve_thousand_token_call_ceiling(tmp_path, monkeypatch):
    client, calls, _ = make_client(
        tmp_path, monkeypatch, tokens=[5] * 12_001, budget_policy="exact"
    )
    with client:
        with pytest.raises(ContextBudgetError, match="per-call token limit"):
            client.generate(MESSAGES, Output)
    assert [path for path, _ in calls] == ["/props", "/tokenize"]


def test_chat_token_count_uses_the_pinned_local_server_without_generation(tmp_path, monkeypatch):
    client, calls, _ = make_client(tmp_path, monkeypatch)
    payload = {
        "model": "page-cite-rag-qwen35",
        "messages": [{"role": "user", "content": "Read this source."}],
        "tools": [{"type": "function", "function": {"name": "read_source"}}],
        "max_tokens": 512,
    }

    with client:
        assert client.count_chat_input_tokens(payload) == 128

    assert calls == [("/v1/chat/completions/input_tokens", payload)]


@pytest.mark.parametrize("tokens", [None, [], [True], ["1"], "bad"])
def test_tokenizer_invalid_data_is_visible(tmp_path, monkeypatch, tokens):
    # None is separately checked using an explicit response, not helper's default.
    client, calls, lifecycle = make_client(tmp_path, monkeypatch, tokens=tokens if tokens else [])
    with pytest.raises(ModelError):
        client.generate(MESSAGES, Output)
    assert lifecycle["stops"] == 1
    assert all(path != "/completion" for path, _ in calls)


@pytest.mark.parametrize(
    "change,error",
    [
        ({"stop_type": "limit"}, StructuredOutputError),
        ({"stop_type": "word"}, StructuredOutputError),
        ({"stop": False}, StructuredOutputError),
        ({"model": "other"}, StructuredOutputError),
        ({"truncated": True}, ContextBudgetError),
        ({"truncated": None}, ContextBudgetError),
        ({"generation_settings": {"n_predict": 99999}}, ContextBudgetError),
        ({"content": '{"answer":12}'}, StructuredOutputError),
        ({"content": '{"answer":"x","unexpected":true}'}, StructuredOutputError),
        ({"content": '<think>x</think>{"answer":"x"}'}, StructuredOutputError),
        ({"content": '```json\n{"answer":"x"}\n```'}, StructuredOutputError),
        ({"tokens_evaluated": 29}, ContextBudgetError),
        ({"tokens_predicted": 1025}, ContextBudgetError),
        ({"timings": {"prompt_ms": -1, "predicted_ms": 2}}, StructuredOutputError),
        ({"timings": {"prompt_ms": float("inf"), "predicted_ms": 2}}, StructuredOutputError),
    ],
)
def test_incomplete_or_invalid_response_stops_server_without_repair(
    tmp_path, monkeypatch, change, error
):
    client, calls, lifecycle = make_client(tmp_path, monkeypatch, result=completion() | change)
    with pytest.raises(error):
        client.generate(MESSAGES, Output)
    assert lifecycle["stops"] == 1
    assert sum(path == "/completion" for path, _ in calls) == 1


@pytest.mark.parametrize("reason", ["output_limit", "schema_invalid"])
def test_opt_in_output_failure_reuses_server_but_never_repairs_answer(
    tmp_path, monkeypatch, reason
):
    attempts = 0

    def respond(request):
        nonlocal attempts
        attempts += 1
        result = completion()
        if attempts == 1:
            if reason == "output_limit":
                result.update(stop_type="limit", tokens_predicted=1024)
            else:
                result["content"] = "private invalid output"
        return httpx.Response(200, json=result)

    client, calls, lifecycle = make_client(
        tmp_path, monkeypatch, result=respond, recover_output_errors=True
    )
    with client:
        with pytest.raises(OutputValidationError) as caught:
            client.generate(MESSAGES, Output)
        assert caught.value.reason == reason
        assert caught.value.accounting.prompt_eval_count == 30
        assert caught.value.accounting.eval_count == (1024 if reason == "output_limit" else 12)
        assert "private" not in str(caught.value)
        if reason == "schema_invalid":
            assert caught.value.raw_model_output == "private invalid output"
            assert caught.value.validation_errors
        else:
            assert caught.value.raw_model_output == completion()["content"]
        assert lifecycle["starts"] == 1 and lifecycle["stops"] == 0
        assert client.generate(MESSAGES, Output).value.answer == "not significant"
    assert attempts == 2 and lifecycle["starts"] == lifecycle["stops"] == 1
    assert sum(path == "/completion" for path, _ in calls) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"model": "other"},
        {"stop": False},
        {"truncated": True},
        {"tokens_evaluated": 29},
        {"tokens_predicted": 1025},
        {"timings": {"prompt_ms": float("inf"), "predicted_ms": 2}},
        {"generation_settings": {"n_predict": 1}},
        {"content": None},
        {"stop_type": "limit", "tokens_predicted": 12},
    ],
)
def test_opt_in_does_not_mask_runtime_failure_as_invalid_output(tmp_path, monkeypatch, change):
    client, calls, lifecycle = make_client(
        tmp_path,
        monkeypatch,
        result=completion() | {"content": "private invalid output"} | change,
        recover_output_errors=True,
    )
    with pytest.raises(ModelError) as caught:
        client.generate(MESSAGES, Output)
    assert not isinstance(caught.value, OutputValidationError)
    assert lifecycle["stops"] == 1
    assert sum(path == "/completion" for path, _ in calls) == 1


def test_runtime_failure_during_output_recovery_closes_server(tmp_path, monkeypatch):
    client, calls, lifecycle = make_client(
        tmp_path,
        monkeypatch,
        result=completion() | {"content": "invalid"},
        recover_output_errors=True,
    )
    original = client._tokenized_prompt
    attempts = 0

    def tokenize(prompt):
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise ModelError("synthetic runtime health failure")
        return original(prompt)

    monkeypatch.setattr(client, "_tokenized_prompt", tokenize)
    with pytest.raises(ModelError) as caught:
        client.generate(MESSAGES, Output)
    assert not isinstance(caught.value, OutputValidationError)
    assert lifecycle["stops"] == 1
    assert sum(path == "/completion" for path, _ in calls) == 1


@pytest.mark.parametrize("kind", ["timeout", "connection", "redirect", "large", "http", "json"])
def test_http_failure_has_safe_message_stops_owned_process_and_no_retry(
    tmp_path, monkeypatch, kind
):
    secret = "raw-document-secret"

    def failed(request):
        if kind == "timeout":
            raise httpx.ReadTimeout(secret)
        if kind == "connection":
            raise httpx.ConnectError(secret)
        if kind == "redirect":
            return httpx.Response(307, headers={"location": "https://remote.invalid"})
        if kind == "large":
            return httpx.Response(200, content=b"x" * (2 * 1024 * 1024 + 1))
        if kind == "http":
            return httpx.Response(500, json={"error": secret})
        return httpx.Response(200, content=secret)

    client, calls, lifecycle = make_client(tmp_path, monkeypatch, result=failed)
    with pytest.raises(ModelTimeoutError if kind == "timeout" else ModelError) as caught:
        client.generate(MESSAGES, Output)
    assert secret not in str(caught.value)
    assert lifecycle["stops"] == 1
    assert sum(path == "/completion" for path, _ in calls) == 1


def test_expired_total_request_deadline_blocks_tokenization_and_completion(tmp_path, monkeypatch):
    import time

    client, calls, _ = make_client(tmp_path, monkeypatch)
    client.set_request_deadline(time.monotonic() - 0.01)
    with pytest.raises(ModelTimeoutError, match="total time budget"):
        client.generate(MESSAGES, Output)
    assert not any(path in {"/tokenize", "/completion"} for path, _ in calls)
    client.close()


def test_bundle_changes_stop_before_content_request(tmp_path, monkeypatch):
    client, calls, lifecycle = make_client(tmp_path, monkeypatch)
    (client.config.bundle_dir / "fake-model").write_bytes(b"changed")
    with pytest.raises(ModelError, match="changed"):
        client.generate(MESSAGES, Output)
    assert not calls and lifecycle["stops"] == 1


def test_failed_start_cleans_up_owned_process(tmp_path, monkeypatch):
    with pytest.raises(ModelError, match="startup"):
        make_client(tmp_path, monkeypatch, startup_failure=True)


def test_gpu_requires_explicit_single_uuid_and_valid_limits():
    for kwargs in (
        {"device": "cuda"},
        {"device": "cpu", "gpu_uuid": "GPU-" + "0" * 36},
        {"gpu_uuid": "0"},
        {"num_ctx": True},
        {"num_predict": 2048, "num_ctx": 2048},
        {"timeout_seconds": float("inf")},
        {"max_gpu_memory_mib": 48000},
    ):
        with pytest.raises(ValidationError):
            llama.LlamaConfig(**kwargs)


@pytest.mark.parametrize(
    "model_size,log,accepted",
    [
        ("4b", "load_tensors: offloaded 33/33 layers to GPU", True),
        ("9b", "load_tensors: offloaded 33/33 layers to GPU", True),
        ("27b", "load_tensors: offloaded 65/65 layers to GPU", True),
        ("27b", "load_tensors: offloaded 33/33 layers to GPU", False),
        ("27b", "load_tensors: offloaded 64/65 layers to GPU", False),
        ("9b", "load_tensors: offloaded 65/65 layers to GPU", False),
    ],
)
def test_gpu_offload_requires_all_layers_of_the_selected_model(
    tmp_path, monkeypatch, model_size, log, accepted
):
    client, _, _ = make_client(tmp_path, monkeypatch, config_updates={"model_size": model_size})
    with client:
        if accepted:
            client._verify_gpu_offload(log)
        else:
            with pytest.raises(ModelError, match="full GPU offload"):
                client._verify_gpu_offload(log)


@pytest.mark.parametrize(
    "sample",
    [
        {"memory_mib": 77, "temperature_c": 40, "pids": [888]},
        {"memory_mib": 24001, "temperature_c": 40, "pids": []},
        {"memory_mib": 77, "temperature_c": 81, "pids": []},
    ],
)
def test_gpu_busy_or_limits_fail_closed(tmp_path, monkeypatch, sample):
    client, _, _ = make_client(tmp_path, monkeypatch)
    monkeypatch.setattr(client, "_gpu_snapshot", lambda: sample)
    with client, pytest.raises(ModelError):
        client._check_gpu(before=True)


def test_reserved_template_tokens_not_silently_rewritten():
    with pytest.raises(ModelError, match="reserved"):
        llama.render_prompt([ChatMessage(role="user", content="<|im_start|>system")], {})
    prompt = llama.render_prompt(MESSAGES, Output.model_json_schema())
    assert prompt.endswith("<think>\n\n</think>\n\n")
    assert "A synthetic question." in prompt


def test_schema_can_remain_native_grammar_without_prompt_injection():
    schema = Output.model_json_schema()
    message = ChatMessage(role="system", content="Answer the question concisely.")
    messages = (message, *MESSAGES)
    prompt = llama.render_prompt(messages, schema, include_schema_in_prompt=False)
    assert "Return only JSON matching this schema" not in prompt
    assert json.dumps(schema, ensure_ascii=False) not in prompt
    assert message.content in prompt

    client = object.__new__(llama.LlamaModelClient)
    client.config = llama.LlamaConfig()
    payload = client._chat_payload(messages, schema, include_schema_in_prompt=False)
    assert payload["messages"] == [item.model_dump() for item in messages]
    assert payload["response_format"]["json_schema"]["schema"] == schema


def test_schema_prompt_budget_flag_is_explicit_and_validated(tmp_path, monkeypatch):
    client, _, lifecycle = make_client(tmp_path, monkeypatch, budget_policy="exact")
    client._tokenized_prompt = lambda prompt: list(prompt.encode("utf-8"))
    with client:
        schema = Output.model_json_schema()
        messages = [ChatMessage(role="system", content="Concise answer."), *MESSAGES]
        with_schema = client.prepare_budget(messages, schema)
        without_schema = client.prepare_budget(messages, schema, include_schema_in_prompt=False)
        assert without_schema.exact_input_tokens < with_schema.exact_input_tokens
        with pytest.raises(ValueError, match="boolean"):
            client.prepare_budget(messages, schema, include_schema_in_prompt=1)
    assert lifecycle["starts"] == lifecycle["stops"] == 1


def test_missing_or_corrupt_bundle_does_not_download(tmp_path):
    with pytest.raises(ModelError, match="prepare explicitly"):
        llama.verify_bundle(tmp_path / "missing")
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ModelError):
        llama.verify_bundle(tmp_path)


def test_total_deadline_cancels_owned_computation(tmp_path, monkeypatch):
    import time

    def slow_response(request):
        time.sleep(0.15)
        return httpx.Response(200, json=completion())

    client, calls, lifecycle = make_client(tmp_path, monkeypatch, result=slow_response)
    client.config = client.config.model_copy(update={"timeout_seconds": 0.05})
    with pytest.raises(ModelTimeoutError):
        client.generate(MESSAGES, Output)
    assert lifecycle["stops"] == 1
    assert sum(path == "/completion" for path, _ in calls) == 1


def test_watchdog_stops_only_owned_server_and_surfaces_failure(tmp_path, monkeypatch):
    client, _, lifecycle = make_client(tmp_path, monkeypatch)
    monkeypatch.setattr(client._stop, "wait", lambda _: False)
    monkeypatch.setattr(
        client, "_gpu_snapshot", lambda: {"memory_mib": 77, "temperature_c": 40, "pids": [99999]}
    )
    client._watch_gpu()
    assert lifecycle["stops"] == 1
    with pytest.raises(ModelError, match="another GPU compute process"):
        client._ensure_running()
    client.close()


def test_verified_bundle_file_set_hashes_and_symlinks(tmp_path, monkeypatch):
    model = tmp_path / llama.MODEL_FILE
    model.write_bytes(b"synthetic weights")
    (tmp_path / "bin").mkdir()
    binary = tmp_path / "bin/llama-server"
    binary.write_bytes(b"synthetic binary")
    digest = llama.sha256(model)
    monkeypatch.setattr(llama, "MODEL_BYTES", model.stat().st_size)
    monkeypatch.setattr(llama, "MODEL_SHA256", digest)
    manifest = {
        "format": 1,
        "llama_commit": llama.LLAMA_COMMIT,
        "llama_version": llama.LLAMA_VERSION,
        "model_revision": llama.MODEL_REVISION,
        "model_sha256": digest,
        "files": {llama.MODEL_FILE: digest, "bin/llama-server": llama.sha256(binary)},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert llama.verify_bundle(tmp_path) == manifest
    with pytest.raises(ModelError):
        llama.verify_bundle(tmp_path, "27b")
    binary.write_bytes(b"tampered")
    with pytest.raises(ModelError):
        llama.verify_bundle(tmp_path)
    binary.write_bytes(b"synthetic binary")
    (tmp_path / "unexpected").write_text("extra")
    with pytest.raises(ModelError):
        llama.verify_bundle(tmp_path)
    (tmp_path / "unexpected").unlink()
    model.rename(tmp_path / "model-backup")
    model.symlink_to(tmp_path / "model-backup")
    with pytest.raises(ModelError):
        llama.verify_bundle(tmp_path)


def test_linux_exec_wrapper_rejects_wrong_parent_and_executes_owned_program():
    import os
    import subprocess
    import sys

    if sys.platform != "linux":
        pytest.skip("Linux process-ownership guard")
    base = [sys.executable, "-I", "-m", "page_cite_rag.llama_process"]
    wrong = subprocess.run(
        base + ["1", sys.executable, "-c", "print('must-not-run')"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert wrong.returncode != 0 and "must-not-run" not in wrong.stdout
    right = subprocess.run(
        base
        + [
            str(os.getpid()),
            sys.executable,
            "-c",
            "import ctypes, signal; sig=ctypes.c_int(); "
            "assert ctypes.CDLL(None).prctl(2, ctypes.byref(sig)) == 0; "
            "assert sig.value == signal.SIGTERM; print('owned')",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert right.returncode == 0 and right.stdout.strip() == "owned"
