"""Owned, version-pinned Ollama server serving the same pinned GGUF as llama.cpp.

Ollama runs llama.cpp underneath, so the prompt contract stays identical: the pinned
non-thinking Qwen prompt is rendered here and sent raw, its tokens are counted from the
GGUF's own tokenizer before dispatch, and the server's prompt count must match exactly.
GPU safeguards, budgets and output validation are inherited from ``LlamaModelClient``.
"""

from __future__ import annotations

import hashlib
import os
import socket
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from page_cite_rag.generation import ModelError, ModelTimeoutError, StructuredOutputError
from page_cite_rag.gguf_tokenizer import gguf_token_ids, gguf_tokenizer
from page_cite_rag.gpu_processes import gpu_process_lifecycle, register_owned_gpu_process
from page_cite_rag.llama import (
    DIAGNOSTICS_VERSION,
    PROMPT_VERSION,
    LlamaConfig,
    LlamaModelClient,
)
from page_cite_rag.owned_process import launch_owned_llama_process

OLLAMA_VERSION = "0.35.0"
DEFAULT_OLLAMA_RUNTIME = Path("models/ollama-runtime") / f"v{OLLAMA_VERSION}" / "gpu"
DEFAULT_OLLAMA_MODELS = Path("models/ollama")
OLLAMA_BUDGET_VERSION = "gguf-tokenized-rendered-prompt-plus-template-reserve-12k/v1"
# Mirrors the llama.cpp server flags (-b/-ub 128, flash attention off, one sequence).
BATCH_TOKENS = 128


def ollama_model_name(model_size: str) -> str:
    """Local name of a pinned GGUF imported by scripts/prepare_ollama.py."""
    return f"pagecite-qwen35-{model_size}-pinned:q4km"


class OllamaGgufClient(LlamaModelClient):
    """The llama client contract on an owned Ollama server; no pull, no remote model."""

    _budget_version = OLLAMA_BUDGET_VERSION

    def __init__(
        self,
        config: LlamaConfig,
        *,
        runtime_dir: Path = DEFAULT_OLLAMA_RUNTIME,
        models_dir: Path = DEFAULT_OLLAMA_MODELS,
        **kwargs,
    ) -> None:
        if config.thinking:
            raise ValueError("the Ollama backend serves the pinned non-thinking prompt only")
        self._runtime_dir = Path(runtime_dir).absolute()
        self._models_dir = Path(models_dir).absolute()
        self._ollama_model = ollama_model_name(config.model_size)
        self._gguf_path: Path | None = None
        self._tokenizer_name: str | None = None
        super().__init__(config, **kwargs)
        sampling = (
            {"sampling_profile": self._sampling_profile, "sampling_seed": self._sampling_seed}
            if self._sampling_profile != "greedy" or self._sampling_seed != 42
            else {}
        )
        self.identity = {
            "backend": "ollama-owned-server",
            "config": config.model_dump(mode="json"),
            "ollama_version": OLLAMA_VERSION,
            "ollama_runtime_sha256": self._runtime_sha256,
            "ollama_model": self._ollama_model,
            "model_sha256": self._model_spec["sha256"],
            "model_revision": self._model_spec["revision"],
            "prompt_version": PROMPT_VERSION,
            "thinking": False,
            **sampling,
            "request_options": {"num_batch": BATCH_TOKENS, "flash_attention": False, "raw": True},
            "tokenizer": f"gguf-bpe-{self._tokenizer_name}/v1",
            "settings_confirmation": "request-side; Ollama does not echo sampling settings",
            "duration_source": "Ollama prompt_eval_duration + eval_duration",
            "diagnostics_version": DIAGNOSTICS_VERSION,
            **(
                {"budget_policy": "exact", "budget_version": OLLAMA_BUDGET_VERSION}
                if self._budget_policy == "exact"
                else {}
            ),
        }

    def _start(self, *, transport=None) -> None:
        import httpx

        binary = self._runtime_dir / "bin" / "ollama"
        self._temporary = tempfile.TemporaryDirectory(prefix="page-cite-rag-ollama-session-")
        temporary = Path(self._temporary.name)
        # Ollama requires HOME and writes its keys there; keep both inside the session.
        env = {"PATH": os.defpath, "LANG": "C.UTF-8", "HOME": str(temporary)}
        try:
            version = subprocess.run(
                [str(binary), "--version"],
                # An unused port: report the client version without reaching any server.
                env={**env, "OLLAMA_HOST": "127.0.0.1:1"},
                capture_output=True,
                text=True,
                timeout=10,
            )
            self._runtime_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
        except (OSError, UnicodeError, subprocess.SubprocessError):
            raise ModelError("pinned Ollama runtime could not be started") from None
        if f"version is {OLLAMA_VERSION}" not in version.stdout + version.stderr:
            raise ModelError("Ollama executable version differs from the pinned runtime")
        if self.config.device == "cuda":
            self._check_gpu(before=True)
        with socket.socket() as port_probe:
            port_probe.bind(("127.0.0.1", 0))
            port = port_probe.getsockname()[1]
        env.update(
            {
                "OLLAMA_HOST": f"127.0.0.1:{port}",
                "OLLAMA_MODELS": str(self._models_dir),
                "OLLAMA_NUM_PARALLEL": "1",
                "OLLAMA_MAX_LOADED_MODELS": "1",
                "OLLAMA_KEEP_ALIVE": "-1",
                "OLLAMA_FLASH_ATTENTION": "0",
                "OLLAMA_NOPRUNE": "1",
                # No cloud models, model recommendations or ollama.com lookups.
                "OLLAMA_NO_CLOUD": "1",
                "OLLAMA_CONTEXT_LENGTH": str(self.config.num_ctx),
                "OMP_NUM_THREADS": str(self.config.num_thread),
                "CUDA_VISIBLE_DEVICES": self.config.gpu_uuid or "-1",
                "CUDA_CACHE_PATH": str(temporary / "cuda-cache"),
            }
        )
        log_path = temporary / "server.log"
        log_path.touch(mode=0o600)
        self._log = log_path.open("w+b")
        with gpu_process_lifecycle():
            self._process = launch_owned_llama_process(
                [str(binary), "serve"], env=env, cwd=temporary, stdout=self._log
            )
            register_owned_gpu_process(self._process)
        self._capture_process_status()
        self._http = httpx.Client(
            base_url=f"http://127.0.0.1:{port}",
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(self.config.timeout_seconds, connect=2),
            transport=transport,
        )
        if self.config.device == "cuda":
            self._watcher = threading.Thread(target=self._watch_gpu, daemon=True)
            self._watcher.start()
        start = time.monotonic()
        while True:
            self._ensure_running()
            try:
                runtime = self._request("/api/version", allow_loading=True).get("version")
            except ModelError:
                runtime = None
            if runtime is not None:
                break
            if time.monotonic() - start >= self.config.startup_seconds:
                raise ModelTimeoutError("local Ollama startup exceeded its budget")
            time.sleep(0.1)
        if runtime != OLLAMA_VERSION:
            raise ModelError("Ollama server version differs from the pinned runtime")
        self._verify_model()
        loaded = self._request(
            "/api/generate",
            {
                "model": self._ollama_model,
                "prompt": "",
                "keep_alive": -1,
                "options": self._options({}),
            },
        )
        if loaded.get("done_reason") != "load":
            raise ModelError("Ollama did not confirm loading the pinned model")
        self._verify_gpu_offload("")

    def _verify_model(self) -> None:
        """The served file must be the pinned GGUF: Ollama names blobs by content hash."""
        shown = self._request("/api/show", {"model": self._ollama_model})
        if shown.get("remote_model") or shown.get("remote_host"):
            raise ModelError("remote models are not permitted by the owned client")
        sources = [
            line.split(" ", 1)[1].strip()
            for line in str(shown.get("modelfile", "")).splitlines()
            if line.startswith("FROM ")
        ]
        expected = f"sha256-{self._model_spec['sha256']}"
        if len(sources) != 1 or Path(sources[0]).name != expected:
            raise ModelError("served model is not the pinned GGUF; prepare it separately")
        self._gguf_path = Path(sources[0])
        if (
            not self._gguf_path.is_file()
            or self._gguf_path.stat().st_size != (self._model_spec["bytes"])
        ):
            raise ModelError("pinned GGUF blob is missing or has the wrong size")
        try:
            _tokenizer, self._tokenizer_name = gguf_tokenizer(self._gguf_path)
        except ValueError as error:
            raise ModelError(f"no verified tokenizer for the served model: {error}") from None

    def _verify_gpu_offload(self, log: str) -> None:
        models = self._request("/api/ps").get("models")
        loaded = [m for m in models or () if isinstance(m, dict)]
        if len(loaded) != 1 or loaded[0].get("name") != self._ollama_model:
            raise ModelError("Ollama is not serving exactly the pinned model")
        if loaded[0].get("context_length") != self.config.num_ctx:
            raise ModelError("resident Ollama context differs from the explicit configuration")
        on_gpu = loaded[0].get("size_vram")
        if self.config.device == "cuda" and on_gpu != loaded[0].get("size"):
            raise ModelError("requested full GPU offload was not confirmed")
        if self.config.device == "cpu" and on_gpu:
            raise ModelError("CPU configuration unexpectedly offloaded to a GPU")

    def _options(self, sampling: dict) -> dict:
        return {
            "num_ctx": self.config.num_ctx,
            "num_predict": self.config.num_predict,
            "num_batch": BATCH_TOKENS,
            "num_gpu": 999 if self.config.device == "cuda" else 0,
            "num_thread": self.config.num_thread,
            **sampling,
        }

    def _tokenized_prompt(self, prompt):
        self._ensure_running()
        if self._gguf_path is None:
            raise ModelError("tokenizer is unavailable before the pinned model is verified")
        return gguf_token_ids(self._gguf_path, prompt)

    def _complete(
        self, prompt: str, tokens: list[int], schema: dict | None, sampling: dict
    ) -> dict:
        response = self._request(
            "/api/generate",
            {
                "model": self._ollama_model,
                "prompt": prompt,
                "raw": True,
                "stream": False,
                "truncate": False,
                "keep_alive": -1,
                **({"format": schema} if schema is not None else {}),
                "options": self._options(sampling),
            },
        )
        if response.get("model") != self._ollama_model or response.get("done") is not True:
            raise StructuredOutputError("Ollama generation incomplete or wrong model")
        try:
            timings = {
                "prompt_ms": response["prompt_eval_duration"] / 1e6,
                "predicted_ms": response["eval_duration"] / 1e6,
            }
        except (KeyError, TypeError):
            raise StructuredOutputError("Ollama runtime metrics are missing") from None
        completion = {
            "content": response.get("response"),
            "stop": True,
            "stop_type": {"stop": "eos", "length": "limit"}.get(response.get("done_reason")),
            # With truncate=false Ollama rejects an over-long prompt instead of cutting it,
            # and generate() still requires the server's prompt count to equal ``tokens``.
            "truncated": False,
            "model": "page-cite-rag-qwen35",
            "generation_settings": {"n_predict": self.config.num_predict, **sampling},
            "timings": timings,
            "tokens_evaluated": response.get("prompt_eval_count"),
            "tokens_predicted": response.get("eval_count"),
        }
        self._record_completion(completion)
        return completion

    def count_chat_input_tokens(self, payload: dict) -> int:
        raise ModelError("native chat-template token counting is unavailable on Ollama")
