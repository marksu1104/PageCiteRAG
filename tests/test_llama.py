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
):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "fake-model").write_bytes(b"fake weights")
    (bundle / "manifest.json").write_text("{}")
    monkeypatch.setattr(llama, "verify_bundle", lambda _: {"files": {"fake-model": "test"}})
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
            return httpx.Response(200, json={"input_tokens": 128})
        if request.url.path == "/completion":
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
    config = llama.LlamaConfig(bundle_dir=bundle)
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
    "sample",
    [
        {"memory_mib": 77, "temperature_c": 40, "pids": [888]},
        {"memory_mib": 12001, "temperature_c": 40, "pids": []},
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
