from __future__ import annotations

import io
import json
import signal
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

import page_cite_rag.llama as llama
from page_cite_rag.generation import (
    ChatMessage,
    ContextBudgetError,
    ModelError,
    ModelTimeoutError,
    StructuredOutputError,
)


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str


def completion():
    return {
        "model": "page-cite-rag-qwen35",
        "content": '{"answer":"private generated text"}',
        "stop": True,
        "stop_type": "eos",
        "truncated": False,
        "generation_settings": {"n_predict": 1024},
        "tokens_evaluated": 30,
        "tokens_predicted": 12,
        "timings": {"prompt_ms": 20.0, "predicted_ms": 30.0},
    }


def make_client(tmp_path, monkeypatch, *, result=None, log=b"", startup_failure=False):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "fake-model").write_bytes(b"synthetic weights")
    (bundle / "manifest.json").write_text("{}")
    monkeypatch.setattr(llama, "verify_bundle", lambda _: {"files": {"fake-model": "test"}})
    state = SimpleNamespace(code=None, signals=[], created=None, requests=[])

    def handler(request):
        state.requests.append(request.url.path)
        if request.url.path == "/props":
            return httpx.Response(200, json={"default_generation_settings": {"n_ctx": 16384}})
        if request.url.path == "/tokenize":
            return httpx.Response(200, json={"tokens": [7] * 30})
        if request.url.path == "/completion":
            if callable(result):
                return result(request)
            return httpx.Response(
                200, content=json.dumps(completion() if result is None else result).encode("utf-8")
            )
        raise AssertionError("unexpected endpoint")

    def start(self, **kwargs):
        state.created = self
        self._process = SimpleNamespace(
            pid=12345, poll=lambda: state.code, wait=lambda **_: state.code
        )
        self._http = httpx.Client(
            base_url="http://127.0.0.1:12345", transport=httpx.MockTransport(handler)
        )
        self._auth_token = "a" * 64
        self._log = io.BytesIO(log)
        if startup_failure:
            raise ModelError("synthetic startup failure")

    def killpg(pid, sig):
        assert pid == 12345
        state.signals.append((pid, sig))
        state.code = -sig

    monkeypatch.setattr(llama.LlamaModelClient, "_start", start)
    monkeypatch.setattr(llama.os, "killpg", killpg)
    return lambda: llama.LlamaModelClient(llama.LlamaConfig(bundle_dir=bundle)), state


def generate(client):
    return client.generate([ChatMessage(role="user", content="private document text")], Output)


def test_closed_snapshot_is_detached_safe_and_never_probes(tmp_path, monkeypatch):
    factory, state = make_client(tmp_path, monkeypatch)
    client = factory()
    generate(client)
    client.close()

    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostics must not probe a closed model or process")

    client._process.poll = forbidden
    monkeypatch.setattr(client, "_gpu_snapshot", forbidden)
    monkeypatch.setattr(client, "_start", forbidden)
    snapshot = client.diagnostics()
    assert snapshot["closed"] is True
    assert snapshot["server_pid"] == 12345
    assert snapshot["exit_code_before_termination"] is None
    assert snapshot["exit_code_after_termination"] == -signal.SIGTERM
    assert snapshot["termination_requested"] is True
    assert snapshot["last_request"] == {
        "endpoint": "/completion",
        "error_category": None,
        "http_status": 200,
    }
    assert snapshot["last_completion"] == {
        "stop_type": "eos",
        "truncated": False,
        "model": "page-cite-rag-qwen35",
        "tokens_evaluated": 30,
        "tokens_predicted": 12,
        "timings": {"prompt_ms": 20.0, "predicted_ms": 30.0},
        "generation_settings": {"n_predict": 1024},
    }
    text = json.dumps(snapshot, allow_nan=False)
    assert "private" not in text and client._auth_token not in text
    assert "server_log_tail" not in snapshot
    snapshot["last_completion"]["timings"]["prompt_ms"] = -1
    assert client.diagnostics()["last_completion"]["timings"]["prompt_ms"] == 20.0
    assert state.signals == [(12345, signal.SIGTERM)]


def test_natural_exit_is_distinct_from_our_termination(tmp_path, monkeypatch):
    factory, state = make_client(tmp_path, monkeypatch)
    client = factory()
    state.code = 17
    with pytest.raises(ModelError, match="exited"):
        generate(client)
    client.close()
    snapshot = client.diagnostics()
    assert snapshot["exit_code_before_termination"] == 17
    assert snapshot["exit_code_after_termination"] == 17
    assert snapshot["termination_requested"] is False
    assert not state.signals


@pytest.mark.parametrize(
    "kind,category,status,error",
    [
        ("timeout", "timeout", None, ModelTimeoutError),
        ("connection", "connection", None, ModelError),
        ("json", "invalid_json", 200, ModelError),
        ("http", "http_status", 500, ModelError),
        ("size", "response_size", 200, ModelError),
        ("invalid", "invalid_response", 200, ModelError),
    ],
)
def test_request_errors_retain_only_safe_categories(
    tmp_path, monkeypatch, kind, category, status, error
):
    secret = "private response and credential"

    def fail(request):
        if kind == "timeout":
            raise httpx.ReadTimeout(secret)
        if kind == "connection":
            raise httpx.ConnectError(secret)
        if kind == "http":
            return httpx.Response(500, json={"error": secret})
        if kind == "size":
            return httpx.Response(200, content=b"x" * (2 * 1024 * 1024 + 1))
        if kind == "invalid":
            return httpx.Response(200, json={"error": secret})
        return httpx.Response(200, content=secret)

    factory, state = make_client(tmp_path, monkeypatch, result=fail)
    client = factory()
    with pytest.raises(error) as caught:
        generate(client)
    snapshot = client.diagnostics()
    assert snapshot["closed"] is True
    assert snapshot["last_request"] == {
        "endpoint": "/completion",
        "error_category": category,
        "http_status": status,
    }
    assert secret not in str(caught.value)
    assert secret not in json.dumps(snapshot, allow_nan=False)
    assert state.requests.count("/completion") == 1


def test_failed_schema_keeps_returned_counts_without_response_text(tmp_path, monkeypatch):
    result = completion() | {"content": "private invalid content"}
    factory, _ = make_client(tmp_path, monkeypatch, result=result)
    client = factory()
    with pytest.raises(StructuredOutputError):
        generate(client)
    recorded = client.diagnostics()["last_completion"]
    assert recorded["tokens_evaluated"] == 30 and recorded["tokens_predicted"] == 12
    assert recorded["timings"] == {"prompt_ms": 20.0, "predicted_ms": 30.0}
    assert "private" not in json.dumps(client.diagnostics(), allow_nan=False)


@pytest.mark.parametrize("failure", ["timeout", "budget"])
def test_new_failed_generation_does_not_borrow_previous_completion(tmp_path, monkeypatch, failure):
    attempts = 0

    def respond(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(200, json=completion())
        raise httpx.ReadTimeout("private failure detail")

    factory, _ = make_client(tmp_path, monkeypatch, result=respond)
    client = factory()
    generate(client)
    assert client.diagnostics()["last_completion"]["tokens_predicted"] == 12
    if failure == "timeout":
        with pytest.raises(ModelTimeoutError):
            generate(client)
    else:
        with pytest.raises(ContextBudgetError):
            client.generate([ChatMessage(role="user", content="中" * 6000)], Output)
    assert client.diagnostics()["last_completion"] is None
    client.close()


def test_untrusted_metadata_strings_and_nonfinite_metrics_are_not_retained(tmp_path, monkeypatch):
    result = completion() | {
        "model": "private model text",
        "stop_type": "private stop text",
        "tokens_evaluated": "private count text",
        "tokens_predicted": True,
        "timings": {"prompt_ms": "private timing text", "predicted_ms": float("inf")},
    }
    factory, _ = make_client(tmp_path, monkeypatch, result=result)
    client = factory()
    with pytest.raises(StructuredOutputError):
        generate(client)
    recorded = client.diagnostics()["last_completion"]
    assert recorded["model"] == recorded["stop_type"] == "unexpected"
    assert recorded["tokens_evaluated"] is None and recorded["tokens_predicted"] is None
    assert recorded["timings"] == {}
    assert "private" not in json.dumps(client.diagnostics(), allow_nan=False)


def test_gpu_guard_retains_bounded_timestamped_samples_before_termination(tmp_path, monkeypatch):
    factory, state = make_client(tmp_path, monkeypatch)
    client = factory()
    client._diagnostic_started = 100.0
    ticks = iter(range(101, 228))
    monkeypatch.setattr(llama.time, "monotonic", lambda: float(next(ticks)))
    sample = {"memory_mib": 77, "temperature_c": 40, "pids": []}
    monkeypatch.setattr(client, "_gpu_snapshot", lambda: sample)
    for _ in range(125):
        client._check_gpu(before=True)
    sample = {"memory_mib": 12001, "temperature_c": 40, "pids": []}
    monkeypatch.setattr(client._stop, "wait", lambda _: False)
    client._watch_gpu()
    client.close()
    snapshot = client.diagnostics()
    assert snapshot["guard_error"] == "gpu_memory_limit"
    samples = snapshot["monitor_samples"]
    assert len(samples) == 120
    assert samples[-1]["memory_mib"] == 12001
    assert [item["elapsed_seconds"] for item in samples] == list(range(7, 127))
    samples[-1]["pids"].append(99)
    assert client.diagnostics()["monitor_samples"][-1]["pids"] == []
    assert state.signals == [(12345, signal.SIGTERM)]
    assert client.config.max_gpu_memory_mib == 12000
    assert client.config.max_gpu_temperature_c == 80


@pytest.mark.parametrize(
    ("sample", "expected"),
    [
        ({"memory_mib": 77, "temperature_c": 40, "pids": [99999]}, "unexpected_gpu_process"),
        ({"memory_mib": 12001, "temperature_c": 40, "pids": []}, "gpu_memory_limit"),
        ({"memory_mib": 77, "temperature_c": 81, "pids": []}, "gpu_temperature_limit"),
    ],
)
def test_gpu_guard_records_the_specific_stopping_reason(tmp_path, monkeypatch, sample, expected):
    factory, _state = make_client(tmp_path, monkeypatch)
    client = factory()
    monkeypatch.setattr(client, "_gpu_snapshot", lambda: sample)

    with pytest.raises(ModelError):
        client._check_gpu(before=True)
    client.close()

    assert client.diagnostics()["guard_error"] == expected


def test_gpu_unavailable_classification_is_safe(tmp_path, monkeypatch):
    factory, _ = make_client(tmp_path, monkeypatch)
    client = factory()

    def unavailable():
        raise ModelError("GPU status unavailable; stop rather than assume it is safe")

    monkeypatch.setattr(client, "_gpu_snapshot", unavailable)
    with pytest.raises(ModelError, match="unavailable"):
        client._check_gpu(before=True)
    client.close()
    assert client.diagnostics()["guard_error"] == "gpu_status_unavailable"


@pytest.mark.parametrize("connection_error", [httpx.ReadError, OSError])
def test_gpu_guard_killed_connection_preserves_the_guard_error(
    tmp_path, monkeypatch, connection_error
):
    guard_error = ModelError("GPU busy or observed resource limit exceeded; own server stopped")

    def killed_connection(request):
        client = state.created
        client._record_guard_error("gpu_resource_limit")
        client._guard_error = guard_error
        client._stop_owned_process()
        raise connection_error("private connection failure after guard termination")

    factory, state = make_client(tmp_path, monkeypatch, result=killed_connection)
    client = factory()
    with pytest.raises(ModelError) as caught:
        generate(client)
    assert caught.value is guard_error
    snapshot = client.diagnostics()
    assert snapshot["closed"]
    assert snapshot["guard_error"] == "gpu_resource_limit"
    assert snapshot["last_request"] == {
        "endpoint": "/completion",
        "error_category": "gpu_guard",
        "http_status": None,
    }
    assert snapshot["exit_code_before_termination"] is None
    assert snapshot["exit_code_after_termination"] == -signal.SIGTERM
    assert "private" not in json.dumps(snapshot, allow_nan=False)
    assert state.requests.count("/completion") == 1
    assert state.signals == [(12345, signal.SIGTERM)]


def test_opt_in_log_is_redacted_byte_bounded_and_retained_after_cleanup(tmp_path, monkeypatch):
    token = "a" * 64
    raw = ("前" * 5000 + f"\nBearer {token}\n{token}\nprivate backend log").encode("utf-8")
    factory, _ = make_client(tmp_path, monkeypatch, log=raw)
    client = factory()
    assert "server_log_tail" not in client.diagnostics()
    assert token not in client.diagnostics(include_log=True)["server_log_tail"]
    client.close()
    snapshot = client.diagnostics(include_log=True)
    tail = snapshot["server_log_tail"]
    assert len(tail.encode("utf-8")) <= 8192
    assert "private backend log" in tail
    assert "Bearer [REDACTED]" in tail
    assert token not in json.dumps(snapshot)
    assert "server_log_tail" not in client.diagnostics()


def test_real_log_tail_read_does_not_move_the_server_file_offset(tmp_path, monkeypatch):
    factory, _ = make_client(tmp_path, monkeypatch)
    client = factory()
    client._log.close()
    with (tmp_path / "owned-server.log").open("w+b") as stream:
        stream.write(b"synthetic log")
        stream.flush()
        stream.seek(3)
        client._log = stream
        assert client.diagnostics(include_log=True)["server_log_tail"] == "synthetic log"
        assert stream.tell() == 3
        client.close()


def test_redaction_handles_a_key_crossing_the_tail_boundary(tmp_path, monkeypatch):
    token = "a" * 64
    # Later redactions shorten the tail, so a partial key at its start must be removed too.
    raw = token.encode() + b"x" * 8008 + ("Bearer " + token + "\n").encode() * 3
    factory, _ = make_client(tmp_path, monkeypatch, log=raw)
    client = factory()
    client.close()
    tail = client.diagnostics(include_log=True)["server_log_tail"]
    assert "a" * 20 not in tail
    assert len(tail.encode("utf-8")) <= 8192


def test_redaction_handles_an_incomplete_key_at_log_tail_end(tmp_path, monkeypatch):
    token = "a" * 64
    factory, _ = make_client(
        tmp_path, monkeypatch, log=("diagnostic Bearer " + token[:32]).encode()
    )
    client = factory()
    client.close()
    tail = client.diagnostics(include_log=True)["server_log_tail"]
    assert tail == "diagnostic Bearer [REDACTED]"


def test_broken_telemetry_does_not_mask_original_error_or_cleanup(tmp_path, monkeypatch):
    factory, state = make_client(
        tmp_path, monkeypatch, result={"content": "private invalid output"}
    )
    client = factory()

    class BrokenLog:
        closed = False

        def fileno(self):
            raise RuntimeError("private telemetry error")

        def close(self):
            self.closed = True

    client._log.close()
    client._log = BrokenLog()
    with pytest.raises(ModelError, match="truncation") as caught:
        generate(client)
    assert "private" not in str(caught.value)
    assert client._log.closed and client._http.is_closed
    assert client.diagnostics(include_log=True)["server_log_tail"] == ""
    assert state.signals == [(12345, signal.SIGTERM)]


def test_startup_failure_retains_diagnostics_after_cleanup(tmp_path, monkeypatch):
    factory, state = make_client(
        tmp_path, monkeypatch, startup_failure=True, log=b"startup details"
    )
    with pytest.raises(ModelError, match="synthetic startup"):
        factory()
    snapshot = state.created.diagnostics(include_log=True)
    assert snapshot["closed"] and snapshot["server_pid"] == 12345
    assert snapshot["exit_code_before_termination"] is None
    assert snapshot["exit_code_after_termination"] == -signal.SIGTERM
    assert snapshot["server_log_tail"] == "startup details"
