from __future__ import annotations

import json
import subprocess
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel

from page_cite_rag import llama, ollama_backend
from page_cite_rag.generation import ChatMessage, ContextBudgetError, ModelError

MESSAGES = (
    ChatMessage(role="system", content="Answer from the source."),
    ChatMessage(role="user", content="Is it significant?"),
)
NAME = ollama_backend.ollama_model_name("9b")
SPEC = llama.model_spec("9b")


class Output(BaseModel):
    answer: str


def make_ollama(
    tmp_path, monkeypatch, *, blob_sha=None, ps=None, generated=None, tokens=30, exit_on_start=False
):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "fake-model").write_bytes(b"fake weights")
    (bundle / "manifest.json").write_text("{}")
    monkeypatch.setattr(llama, "verify_bundle", lambda *_: {"files": {"fake-model": "test"}})
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin" / "ollama").write_bytes(b"fake ollama")
    blob = tmp_path / "models" / "blobs" / f"sha256-{blob_sha or SPEC['sha256']}"
    blob.parent.mkdir(parents=True)
    with blob.open("wb") as handle:
        handle.truncate(SPEC["bytes"])  # sparse: the size check needs the pinned size only
    monkeypatch.setattr(
        ollama_backend.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout="", stderr="Warning: client version is 0.35.0"
        ),
    )
    lifecycle = {"returncode": None, "stops": 0}
    monkeypatch.setattr(
        ollama_backend.socket,
        "socket",
        lambda: nullcontext(
            SimpleNamespace(bind=lambda _: None, getsockname=lambda: ("127.0.0.1", 12345))
        ),
    )

    def launch(*args, **kwargs):
        if exit_on_start:
            kwargs["stdout"].write(
                b'time=now level=ERROR source=server.go:1 msg="runner startup failed"\n'
                b"private document continuation\n"
            )
            kwargs["stdout"].flush()
            lifecycle["returncode"] = 17
        return SimpleNamespace(pid=12345, poll=lambda: lifecycle["returncode"])

    monkeypatch.setattr(ollama_backend, "launch_owned_llama_process", launch)
    monkeypatch.setattr(ollama_backend, "register_owned_gpu_process", lambda process: None)

    def terminate(self):
        lifecycle["stops"] += 1
        lifecycle["returncode"] = 0

    monkeypatch.setattr(llama.LlamaModelClient, "_terminate_process", terminate)
    monkeypatch.setattr(ollama_backend, "gguf_tokenizer", lambda path: (None, "qwen35"))
    monkeypatch.setattr(ollama_backend, "gguf_token_ids", lambda path, text: [7] * tokens)
    calls = []

    def handler(request):
        payload = json.loads(request.content) if request.content else None
        calls.append((request.url.path, payload))
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.35.0"})
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"modelfile": f"# generated\nFROM {blob}\n"})
        if request.url.path == "/api/ps":
            return httpx.Response(
                200,
                json={
                    "models": [
                        ps or {"name": NAME, "context_length": 16384, "size": 100, "size_vram": 0}
                    ]
                },
            )
        if request.url.path == "/api/generate" and payload["prompt"] == "":
            return httpx.Response(200, json={"model": NAME, "done": True, "done_reason": "load"})
        if request.url.path == "/api/generate":
            return httpx.Response(
                200,
                json=generated
                or {
                    "model": NAME,
                    "response": '{"answer":"not significant"}',
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": tokens,
                    "eval_count": 9,
                    "prompt_eval_duration": 20_000_000,
                    "eval_duration": 30_000_000,
                },
            )
        raise AssertionError("unexpected endpoint")

    config = llama.LlamaConfig(bundle_dir=bundle, device="cpu")
    client = ollama_backend.OllamaGgufClient(
        config,
        runtime_dir=runtime,
        models_dir=tmp_path / "models",
        transport=httpx.MockTransport(handler),
        budget_policy="exact",
    )
    return client, calls, lifecycle


def test_ollama_serves_the_pinned_blob_with_the_rendered_raw_prompt(tmp_path, monkeypatch):
    client, calls, lifecycle = make_ollama(tmp_path, monkeypatch)
    with client:
        result = client.generate(MESSAGES, Output)
        assert client.identity["backend"] == "ollama-owned-server"
        assert client.identity["model_sha256"] == SPEC["sha256"]
        assert client.identity["tokenizer"] == "gguf-bpe-qwen35/v1"
        assert result.budget.method == ollama_backend.OLLAMA_BUDGET_VERSION
    assert result.value.answer == "not significant"
    assert result.stats.prompt_eval_count == 30
    request = [payload for path, payload in calls if path == "/api/generate"][-1]
    schema = Output.model_json_schema()
    assert request["prompt"] == llama.render_prompt(MESSAGES, schema)
    assert request["raw"] is True and request["truncate"] is False
    assert request["format"] == schema
    assert request["options"] == {
        "num_ctx": 16384,
        "num_predict": 1024,
        "num_batch": 128,
        "num_gpu": 0,
        "num_thread": 2,
        "temperature": 0.0,
        "seed": 42,
        "repeat_penalty": 1.0,
    }
    assert lifecycle["stops"] == 1


def test_ollama_refuses_a_model_that_is_not_the_pinned_blob(tmp_path, monkeypatch):
    with pytest.raises(ModelError, match="not the pinned GGUF"):
        make_ollama(tmp_path, monkeypatch, blob_sha="0" * 64)


def test_ollama_server_prompt_count_must_equal_the_gguf_count(tmp_path, monkeypatch):
    generated = {
        "model": NAME,
        "response": '{"answer":"x"}',
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 31,
        "eval_count": 3,
        "prompt_eval_duration": 1,
        "eval_duration": 1,
    }
    client, _calls, _lifecycle = make_ollama(tmp_path, monkeypatch, generated=generated)
    with client, pytest.raises(ContextBudgetError, match="token counts"):
        client.generate(MESSAGES, Output)


def test_ollama_cpu_configuration_rejects_gpu_offload(tmp_path, monkeypatch):
    ps = {"name": NAME, "context_length": 16384, "size": 100, "size_vram": 100}
    with pytest.raises(ModelError, match="unexpectedly offloaded"):
        make_ollama(tmp_path, monkeypatch, ps=ps)


def test_ollama_rejects_a_resident_context_other_than_configured(tmp_path, monkeypatch):
    ps = {"name": NAME, "context_length": 4096, "size": 100, "size_vram": 0}
    with pytest.raises(ModelError, match="context differs"):
        make_ollama(tmp_path, monkeypatch, ps=ps)


def test_ollama_startup_exit_keeps_safe_log_after_temporary_directory_cleanup(
    tmp_path, monkeypatch
):
    with pytest.raises(ModelError, match="runner startup failed") as caught:
        make_ollama(tmp_path, monkeypatch, exit_on_start=True)
    snapshot = caught.value.diagnostics
    assert snapshot["closed"]
    assert snapshot["exit_code_before_termination"] == 17
    assert "runner startup failed" in snapshot["server_failure_log_tail"]
    assert "private" not in json.dumps(snapshot)
    assert "private" not in str(caught.value)


def test_ollama_exit_before_first_dispatch_keeps_log_without_restart(tmp_path, monkeypatch):
    client, calls, lifecycle = make_ollama(tmp_path, monkeypatch)
    client._log.write(b"ggml_cuda_init: CUDA driver initialization failed\n")
    client._log.flush()
    lifecycle["returncode"] = 17
    with pytest.raises(ModelError, match="CUDA driver initialization failed") as caught:
        client.generate(MESSAGES, Output)
    client.close()
    assert "CUDA driver initialization failed" in client.diagnostics()["server_failure_log_tail"]
    assert caught.value.diagnostics["exit_code_before_termination"] == 17
    assert not [p for path, p in calls if path == "/api/generate" and p.get("prompt")]
    assert client.diagnostics()["closed"]
