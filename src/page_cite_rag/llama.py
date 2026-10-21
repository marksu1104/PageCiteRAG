"""Explicitly prepared llama.cpp runtime; importing this module starts nothing."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from page_cite_rag.generation import (
    ChatMessage,
    ContextBudgetError,
    GenerationResult,
    GenerationStats,
    ModelError,
    ModelTimeoutError,
    OutputFailureAccounting,
    OutputValidationError,
    PromptBudget,
    StructuredOutputError,
    T,
    check_budget,
)
from page_cite_rag.gpu_processes import (
    gpu_process_lifecycle,
    register_owned_gpu_process,
    snapshot_gpu_processes,
    unregister_owned_gpu_process,
)

LLAMA_COMMIT = "7fe450e19305b828c199d602c23a8337aaa1f03b"
LLAMA_VERSION = "0.5.0"
SOURCE_SHA256 = "ab71e8f46a7bee2fd0d273cd997a4b283debf7623adc1c5f28f31b6c6f68ac83"
MODEL_REPO = "unsloth/Qwen3.5-9B-GGUF"
MODEL_REVISION = "3885219b6810b007914f3a7950a8d1b469d598a5"
MODEL_FILE = "Qwen3.5-9B-Q4_K_M.gguf"
MODEL_BYTES = 5680522464
MODEL_SHA256 = "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8"
DEFAULT_BUNDLE = Path("models/llama-qwen35-9b") / f"{LLAMA_COMMIT[:8]}-{MODEL_REVISION[:8]}"
DIAGNOSTICS_VERSION = "llama-diagnostics/v1"
DIAGNOSTIC_LOG_BYTES = 8192
EXACT_BUDGET_VERSION = "llama-rendered-token-count-plus-template-reserve-12k/v3"
MAX_EXACT_INPUT_TOKENS = 12_000
PROMPT_VERSION = "qwen35/nonthinking/merged-first-system/v2"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_bundle(directory: Path) -> dict:
    """Offline integrity check, not authentication of someone else's manifest."""
    try:
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError("missing bundle")
        manifest = directory / "manifest.json"
        if manifest.is_symlink() or manifest.stat().st_size > 65536:
            raise ValueError("invalid manifest")
        value = json.loads(manifest.read_text())
        if (
            value["format"] != 1
            or value["llama_commit"] != LLAMA_COMMIT
            or value["llama_version"] != LLAMA_VERSION
            or value["model_revision"] != MODEL_REVISION
            or value["model_sha256"] != MODEL_SHA256
        ):
            raise ValueError("identity mismatch")
        hashes = value["files"]
        if not isinstance(hashes, dict) or not {MODEL_FILE, "bin/llama-server"} <= hashes.keys():
            raise ValueError("missing required files")
        actual = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
        if actual != set(hashes) | {"manifest.json"}:
            raise ValueError("file set mismatch")
        for name, expected in hashes.items():
            path = directory / name
            if (
                not isinstance(name, str)
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or any(p.is_symlink() for p in (path, *path.parents[: len(Path(name).parts)]))
                or sha256(path) != expected
            ):
                raise ValueError("checksum or path mismatch")
        model = directory / MODEL_FILE
        if model.stat().st_size != MODEL_BYTES or hashes[MODEL_FILE] != MODEL_SHA256:
            raise ValueError("model changed")
        return value
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        raise ModelError("local llama bundle is missing or changed; prepare explicitly") from None


class LlamaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    bundle_dir: Path = DEFAULT_BUNDLE
    device: Literal["cpu", "cuda"] = "cpu"  # Explicit opt-in; never auto fallback.
    gpu_uuid: str | None = Field(default=None, pattern=r"^GPU-[0-9a-fA-F-]{36}$")
    num_ctx: int = Field(default=16384, ge=2048, le=16384)
    num_predict: int = Field(default=1024, ge=1, le=2048)
    template_reserve: int = Field(default=1024, ge=256, le=4096)
    num_thread: int = Field(default=2, ge=1, le=6)
    timeout_seconds: float = Field(default=120.0, gt=0, le=120, allow_inf_nan=False)
    startup_seconds: float = Field(default=60.0, gt=0, le=90, allow_inf_nan=False)
    max_gpu_memory_mib: int = Field(default=12000, ge=1024, le=12000)
    # Higher limits require explicit CUDA opt-in and verified driver thresholds.
    max_gpu_temperature_c: int = Field(default=80, ge=40, le=88)

    @model_validator(mode="after")
    def validate_controls(self) -> LlamaConfig:
        if self.num_predict + self.template_reserve >= self.num_ctx:
            raise ValueError("context must reserve input space")
        if (self.device == "cuda") != (self.gpu_uuid is not None):
            raise ValueError("CUDA requires one explicit GPU UUID; CPU must not select a GPU")
        if self.max_gpu_temperature_c > 80 and self.device != "cuda":
            raise ValueError("higher temperature guards require explicit CUDA")
        return self


def render_prompt(messages: Sequence[ChatMessage], schema: dict) -> str:
    """Pinned Qwen text-only non-thinking format, checked against /apply-template.

    The model permits a system turn only at the beginning. Merge the schema
    instruction into that turn rather than silently creating a second system
    turn. This is not a universal tool/multimodal/reasoning-history renderer.
    """
    if any("<|" in message.content or "|>" in message.content for message in messages):
        raise ModelError("reserved chat-token delimiters in input; no silent text rewriting")
    if any(message.role == "system" for message in messages[1:]):
        raise ModelError("system message must be first; no silent message reordering")
    prompt = "<|im_start|>system\nReturn only JSON matching this schema:\n"
    prompt += json.dumps(schema, ensure_ascii=False)
    if messages and messages[0].role == "system":
        prompt += "\n\n" + messages[0].content
        messages = messages[1:]
    prompt += "<|im_end|>\n"
    for message in messages:
        prompt += f"<|im_start|>{message.role}\n{message.content}<|im_end|>\n"
    return prompt + "<|im_start|>assistant\n<think>\n\n</think>\n\n"


class LlamaModelClient:
    """Own one resident, authenticated loopback server; close it on exit/failure.

    Sequential requests only. No download, external server, tools, HTTP retry or
    automatic backend fallback. GPU observations are safeguards, NOT isolation.
    """

    def __init__(
        self,
        config: LlamaConfig,
        *,
        transport=None,
        budget_policy="conservative",
        recover_output_errors=False,
        sampling_profile: Literal["greedy", "qwen-general"] = "greedy",
        sampling_seed: int = 42,
    ) -> None:
        if budget_policy not in ("conservative", "exact"):
            raise ValueError("budget policy must be conservative or exact")
        if type(recover_output_errors) is not bool:
            raise ValueError("output recovery must be an explicit boolean")
        if sampling_profile not in ("greedy", "qwen-general"):
            raise ValueError("unsupported sampling profile")
        if type(sampling_seed) is not int or not 0 <= sampling_seed <= 2_147_483_647:
            raise ValueError("sampling seed must be a nonnegative 32-bit integer")
        self._sampling_profile = sampling_profile
        self._sampling_seed = sampling_seed
        self._recover_output_errors = recover_output_errors
        self._budget_policy = budget_policy
        self._token_cache = None  # One prompt hash + token sequence, never persisted.
        self.config = config
        self._closed = False
        self._process = None
        self._http = None
        self._temporary = None
        self._log = None
        self._watcher = None
        self._stop = threading.Event()
        self._process_lock = threading.Lock()
        self._diagnostic_lock = threading.RLock()
        self._diagnostic_started = time.monotonic()
        self._diagnostic = {
            "diagnostics_version": DIAGNOSTICS_VERSION,
            "server_pid": None,
            "exit_code_before_termination": None,
            "exit_code_after_termination": None,
            "termination_requested": False,
            "last_request": None,
            "last_completion": None,
            "guard_error": None,
            "last_gpu_check": None,
            "monitor_sample_count": 0,
            "observed_gpu_peak": {"memory_mib": None, "temperature_c": None},
        }
        self._exit_before_captured = False
        self._server_log_tail = ""
        self._auth_token = None
        self._guard_error = None
        self._monitor_samples = []
        self._generation_deadline = None
        self._gpu_temperature_limits = None
        try:
            manifest = verify_bundle(config.bundle_dir)
            self._bundle = config.bundle_dir.resolve()
            self._fingerprints = self._file_state(manifest)
            self._start(transport=transport)
            self._capture_process_status()
            self.identity = {
                "backend": "llama-cpp-owned-server",
                "config": config.model_dump(mode="json"),
                "llama_commit": LLAMA_COMMIT,
                "model_sha256": MODEL_SHA256,
                "model_revision": MODEL_REVISION,
                "runtime_files": manifest["files"],
                "prompt_version": PROMPT_VERSION,
                "thinking": False,
                **(
                    {"sampling_profile": sampling_profile, "sampling_seed": sampling_seed}
                    if sampling_profile != "greedy" or sampling_seed != 42
                    else {}
                ),
                "duration_source": "server prompt_ms + predicted_ms; load_duration unavailable",
                "diagnostics_version": DIAGNOSTICS_VERSION,
                **(
                    {"budget_policy": "exact", "budget_version": EXACT_BUDGET_VERSION}
                    if budget_policy == "exact"
                    else {}
                ),
            }
        except BaseException:
            self.close()
            raise

    def _file_state(self, manifest: dict) -> dict:
        return {
            name: (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns)
            for name in (*manifest["files"], "manifest.json")
            for p in [self._bundle / name]
        }

    def _gpu_snapshot(self) -> dict:
        try:
            values = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    self.config.gpu_uuid,
                    "--query-gpu=memory.used,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=3,
            ).strip()
            memory, temperature = map(int, values.split(","))
            pids = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    self.config.gpu_uuid,
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=3,
            ).strip()
            sample = {
                "memory_mib": memory,
                "temperature_c": temperature,
                "pids": [int(pid) for pid in pids.splitlines() if pid],
            }
            if self.config.max_gpu_temperature_c > 80:
                sample.update(self._gpu_health())
            return sample
        except (OSError, ValueError, subprocess.SubprocessError):
            raise ModelError("GPU status unavailable; stop rather than assume it is safe") from None

    def _gpu_health(self) -> dict:
        """Read-only telemetry for an explicit, bounded higher-temperature policy.

        A five-degree margin below the driver-reported software thermal threshold
        is our test policy, not a vendor safety certification. Never set GPU state.
        """
        if self._gpu_temperature_limits is None:
            text = subprocess.check_output(
                ["nvidia-smi", "-q", "-i", self.config.gpu_uuid, "-d", "TEMPERATURE"],
                text=True,
                timeout=3,
            )
            limits = {}
            for name, label in (
                ("max_operating_c", "GPU Max Operating Temp"),
                ("target_temperature_c", "GPU Target Temperature"),
            ):
                matches = re.findall(rf"^\s*{label}\s*:\s*(\d+)\s*C\s*$", text, re.MULTILINE)
                if len(matches) != 1 or not 40 <= int(matches[0]) <= 125:
                    raise ValueError("missing or invalid driver thermal threshold")
                limits[name] = int(matches[0])
            if self.config.max_gpu_temperature_c > limits["max_operating_c"] - 5:
                raise ModelError("temperature guard must leave 5 C below GPU thermal threshold")
            self._gpu_temperature_limits = limits
        values = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    self.config.gpu_uuid,
                    "--query-gpu=fan.speed,power.draw,power.limit,utilization.gpu,"
                    "clocks_throttle_reasons.hw_thermal_slowdown,"
                    "clocks_throttle_reasons.sw_thermal_slowdown",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=3,
            )
            .strip()
            .split(",")
        )
        if len(values) != 6:
            raise ValueError("unexpected GPU health fields")
        health = dict(self._gpu_temperature_limits)
        for name, raw in zip(
            ("fan_speed_percent", "power_draw_w", "power_limit_w", "utilization_gpu_percent"),
            values[:4],
            strict=True,
        ):
            value = None if raw.strip() == "N/A" else float(raw)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1_000_000):
                raise ValueError("invalid GPU telemetry number")
            health[name] = value
        for name, raw in zip(
            ("hw_thermal_slowdown", "sw_thermal_slowdown"),
            values[4:],
            strict=True,
        ):
            if raw.strip() not in ("Active", "Not Active"):
                raise ValueError("GPU thermal slowdown state unavailable")
            health[name] = raw.strip() == "Active"
        return health

    def _check_gpu(self, *, before: bool = False) -> None:
        try:
            sample, process_state = snapshot_gpu_processes(
                self._gpu_snapshot,
                expected_process=self._process if not before else None,
            )
        except (ModelError, OSError, ValueError) as exc:
            self._record_guard_error("gpu_status_unavailable")
            if isinstance(exc, ModelError):
                raise
            raise ModelError("GPU status unavailable; stop rather than assume it is safe") from None
        phase = "startup" if before else "monitor"
        self._record_gpu_sample(sample, process_state=process_state, phase=phase)
        if process_state["owned_service_exited"]:
            self._record_guard_error("owned_service_exited")
            raise ModelError("owned llama service exited during GPU monitoring")
        if self.config.max_gpu_temperature_c > 80:
            if (
                type(sample.get("max_operating_c")) is not int
                or self.config.max_gpu_temperature_c > sample["max_operating_c"] - 5
                or any(
                    type(sample.get(key)) is not bool
                    for key in ("hw_thermal_slowdown", "sw_thermal_slowdown")
                )
            ):
                self._record_guard_error("gpu_status_unavailable")
                raise ModelError("GPU thermal safeguards unavailable; own server stopped")
            if sample["hw_thermal_slowdown"] or sample["sw_thermal_slowdown"]:
                self._record_guard_error("gpu_thermal_slowdown")
                raise ModelError("GPU thermal slowdown observed; own server stopped")
        if process_state["unexpected_pids"]:
            self._record_guard_error("unexpected_gpu_process")
            raise ModelError("another GPU compute process is present; own server stopped")
        if sample["memory_mib"] > self.config.max_gpu_memory_mib:
            self._record_guard_error("gpu_memory_limit")
            raise ModelError("GPU memory limit exceeded; own server stopped")
        if sample["temperature_c"] > self.config.max_gpu_temperature_c:
            self._record_guard_error("gpu_temperature_limit")
            raise ModelError("GPU temperature limit exceeded; own server stopped")

    def _watch_gpu(self) -> None:
        while not self._stop.wait(0.5):
            try:
                self._check_gpu()
            except ModelError as error:
                self._guard_error = error
                self._stop_owned_process()
                return

    def _start(self, *, transport=None) -> None:
        import httpx

        binary = self._bundle / "bin/llama-server"
        env = {
            "PATH": os.defpath,
            "LANG": "C.UTF-8",
            "LD_LIBRARY_PATH": str(binary.parent),
            "OMP_NUM_THREADS": str(self.config.num_thread),
            "CUDA_VISIBLE_DEVICES": self.config.gpu_uuid or "-1",
        }
        try:
            version = subprocess.check_output(
                [str(binary), "--version"], env=env, stderr=subprocess.STDOUT, timeout=10
            ).decode()
        except (OSError, UnicodeError, subprocess.SubprocessError):
            raise ModelError("prepared llama runtime could not be started") from None
        if LLAMA_COMMIT[:7] not in version or LLAMA_VERSION not in version:
            raise ModelError("llama executable version differs from the pinned bundle")
        if self.config.device == "cuda":
            self._check_gpu(before=True)
        self._temporary = tempfile.TemporaryDirectory(prefix="page-cite-rag-llama-session-")
        temporary = Path(self._temporary.name)
        token = secrets.token_hex(32)
        self._auth_token = token
        key_file = temporary / "api-key"
        key_file.touch(mode=0o600)
        key_file.write_text(token + "\n")
        env["CUDA_CACHE_PATH"] = str(temporary / "cuda-cache")
        with socket.socket() as port_probe:
            port_probe.bind(("127.0.0.1", 0))
            port = port_probe.getsockname()[1]
        argv = [
            str(binary),
            "-m",
            str(self._bundle / MODEL_FILE),
            "--alias",
            "page-cite-rag-qwen35",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--api-key-file",
            str(key_file),
            "-ngl",
            "99" if self.config.device == "cuda" else "0",
            "-c",
            str(self.config.num_ctx),
            "-np",
            "1",
            "-t",
            str(self.config.num_thread),
            "-b",
            "128",
            "-ub",
            "128",
            "--flash-attn",
            "off",
            "--no-warmup",
            "--no-context-shift",
            "--fit",
            "off",
            "--cache-ram",
            "0",
            "--ctx-checkpoints",
            "0",
            "--no-ui",
            "--jinja",
            "--no-agent",
            "--no-slots",
            "--no-mmproj",
            "--offline",
            "--verbosity",
            "4",
            "--threads-http",
            "2",
        ]
        log_path = temporary / "server.log"
        log_path.touch(mode=0o600)
        self._log = log_path.open("w+b")
        with gpu_process_lifecycle():
            self._process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-m",
                    "page_cite_rag.llama_process",
                    str(os.getpid()),
                    *argv,
                ],
                env=env,
                cwd=temporary,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            register_owned_gpu_process(self._process)
        self._capture_process_status()
        self._http = httpx.Client(
            base_url=f"http://127.0.0.1:{port}",
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(self.config.timeout_seconds, connect=2),
            headers={"Authorization": f"Bearer {token}"},
            transport=transport,
        )
        if self.config.device == "cuda":
            self._watcher = threading.Thread(target=self._watch_gpu, daemon=True)
            self._watcher.start()
        start = time.monotonic()
        while True:
            self._ensure_running()
            try:
                health = self._request("/health", allow_loading=True)
            except ModelError:
                health = None
            if health == {"status": "ok"}:
                break
            if time.monotonic() - start >= self.config.startup_seconds:
                raise ModelTimeoutError("local llama startup exceeded its budget")
            time.sleep(0.1)
        if self.config.device == "cuda":
            self._log.flush()
            log = (temporary / "server.log").read_text(errors="replace")
            if "offloaded 33/33 layers to GPU" not in log:
                raise ModelError("requested full GPU offload was not confirmed")
        props = self._request("/props")
        if props.get("default_generation_settings", {}).get("n_ctx") != self.config.num_ctx:
            raise ModelError("resident server context differs from the explicit configuration")

    def openai_compatible_credentials(self) -> tuple[str, str]:
        """Expose only this owned loopback server to an optional local agent adapter."""
        self._ensure_running()
        if self._http is None or self._auth_token is None:
            raise ModelError("local chat endpoint is unavailable")
        return str(self._http.base_url).rstrip("/") + "/v1", self._auth_token

    def count_chat_input_tokens(self, payload: dict) -> int:
        """Count the exact pinned chat-template input without generating a completion."""
        if not isinstance(payload, dict) or payload.get("model") != "page-cite-rag-qwen35":
            raise ValueError("an explicit local chat-completion payload is required")
        response = self._request("/v1/chat/completions/input_tokens", payload)
        token_count = response.get("input_tokens")
        if type(token_count) is not int or token_count < 1:
            raise ModelError("local chat tokenizer returned invalid input-token metadata")
        return token_count

    def _ensure_running(self) -> None:
        if self._closed:
            raise ModelError("generation client is closed")
        if self._guard_error:
            raise self._guard_error
        if self._process is None or self._process.poll() is not None:
            self._capture_process_status()
            raise ModelError("owned llama server exited; no automatic restart")

    def _request(self, path: str, payload: dict | None = None, *, allow_loading=False) -> dict:
        import httpx

        with self._diagnostic_lock:
            self._diagnostic["last_request"] = {
                "endpoint": path
                if path
                in {
                    "/health",
                    "/props",
                    "/tokenize",
                    "/completion",
                    "/v1/chat/completions/input_tokens",
                }
                else "other",
                "error_category": None,
                "http_status": None,
            }
        try:
            self._ensure_running()
        except ModelError:
            self._record_request_error("gpu_guard" if self._guard_error else "server_exited")
            raise
        start = time.monotonic()
        budget = (
            min(2.0, self.config.timeout_seconds) if allow_loading else self.config.timeout_seconds
        )
        if self._generation_deadline is not None:
            budget = min(budget, self._generation_deadline - start)
        if budget <= 0:
            self._record_request_error("timeout")
            raise ModelTimeoutError("local llama generation exceeded its total time budget")
        expired = threading.Event()

        def cancel_owned_computation():
            expired.set()
            self._stop_owned_process()

        watchdog = threading.Timer(budget, cancel_owned_computation)
        watchdog.daemon = True
        watchdog.start()
        try:
            with self._http.stream(
                "GET" if payload is None else "POST",
                path,
                json=payload,
                timeout=budget,
            ) as r:
                with self._diagnostic_lock:
                    self._diagnostic["last_request"]["http_status"] = (
                        r.status_code if type(r.status_code) is int else None
                    )
                if r.status_code == 503 and allow_loading:
                    return {}
                if not 200 <= r.status_code < 300:
                    self._record_request_error("http_status")
                    raise ModelError(f"local llama request failed with HTTP {r.status_code}")
                body = bytearray()
                for chunk in r.iter_bytes():
                    body.extend(chunk)
                    if len(body) > 2 * 1024 * 1024:
                        self._record_request_error("response_size")
                        raise ModelError("local llama response exceeds its size budget")
                    if time.monotonic() - start > budget:
                        self._record_request_error("timeout")
                        raise ModelTimeoutError("local llama response exceeded its time budget")
                value = json.loads(body)
        except httpx.TimeoutException:
            self._record_request_error("timeout")
            raise ModelTimeoutError("local llama request timed out; no retry") from None
        except (httpx.HTTPError, OSError):
            if self._guard_error is not None:
                self._record_request_error("gpu_guard")
                raise self._guard_error from None
            if expired.is_set():
                self._record_request_error("timeout")
                raise ModelTimeoutError("local llama generation exceeded its time budget") from None
            self._record_request_error("connection")
            raise ModelError("local llama response or connection failed; no retry") from None
        except (ValueError, UnicodeError):
            self._record_request_error("timeout" if expired.is_set() else "invalid_json")
            if expired.is_set():
                raise ModelTimeoutError("local llama generation exceeded its time budget") from None
            raise ModelError("local llama response or connection failed; no retry") from None
        finally:
            watchdog.cancel()
            watchdog.join(timeout=6)
        if expired.is_set():
            self._record_request_error("timeout")
            raise ModelTimeoutError("local llama generation exceeded its time budget")
        if path == "/completion" and isinstance(value, dict):
            self._record_completion(value)
        if not isinstance(value, dict) or "error" in value:
            self._record_request_error("invalid_response")
            raise ModelError("local llama returned an invalid response")
        try:
            self._ensure_running()
        except ModelError:
            self._record_request_error("gpu_guard" if self._guard_error else "server_exited")
            raise
        return value

    def _tokenized_prompt(self, prompt):
        self._ensure_running()
        for name, expected in self._fingerprints.items():
            path = self._bundle / name
            if (
                path.is_symlink()
                or (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns) != expected
            ):
                raise ModelError("prepared runtime/model changed during this session")
        props = self._request("/props")
        if props.get("default_generation_settings", {}).get("n_ctx") != self.config.num_ctx:
            raise ContextBudgetError("resident server context changed")
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if self._budget_policy == "exact" and self._token_cache is not None:
            prior_digest, tokens = self._token_cache
            if prior_digest == digest:
                return list(tokens)
        tokens = self._request(
            "/tokenize", {"content": prompt, "add_special": True, "parse_special": True}
        ).get("tokens")
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(t) is not int or t < 0 for t in tokens)
        ):
            raise ModelError("local tokenizer returned invalid tokens")
        if self._budget_policy == "exact" and len(tokens) < self.config.num_ctx:
            self._token_cache = (digest, tuple(tokens))
        return tokens

    def _max_input_tokens(self) -> int:
        context_capacity = (
            self.config.num_ctx - self.config.num_predict - self.config.template_reserve - 1
        )
        return min(MAX_EXACT_INPUT_TOKENS, context_capacity)

    def prepare_budget(self, messages, schema) -> PromptBudget:
        """Count the actual pinned rendered prompt; no completion/model reload.

        The optional exact mode also retains the configured template margin as
        extra headroom. Tokenizer/identity failure never degrades to estimation.
        """
        self._ensure_running()
        if self._budget_policy == "conservative":
            return check_budget(messages, schema, self.config)
        if not messages or any(not isinstance(m, ChatMessage) for m in messages):
            raise ValueError("nonempty validated ChatMessage sequence required")
        prompt = render_prompt(messages, schema)
        if len(prompt.encode("utf-8")) > 1_000_000:
            raise ContextBudgetError("input exceeds the tokenizer request size limit")
        prior_deadline = self._generation_deadline
        self._generation_deadline = prior_deadline or time.monotonic() + self.config.timeout_seconds
        input_overflow = False
        try:
            tokens = self._tokenized_prompt(prompt)
            if len(tokens) > self._max_input_tokens():
                input_overflow = True
                raise ContextBudgetError("exact input exceeds the per-call token limit")
            upper = len(tokens) + self.config.template_reserve
            if upper + self.config.num_predict >= self.config.num_ctx:
                input_overflow = True
                raise ContextBudgetError("exact input plus safety reserve exceeds context")
            return PromptBudget(
                upper,
                self.config.num_predict,
                self.config.num_ctx,
                EXACT_BUDGET_VERSION,
                len(tokens),
            )
        except ContextBudgetError:
            # A full input is recoverable; changed resident configuration is not.
            if not input_overflow:
                self.close()
            raise
        except BaseException:
            self.close()
            raise
        finally:
            self._generation_deadline = prior_deadline

    def generate(
        self, messages: Sequence[ChatMessage], response_model: type[T]
    ) -> GenerationResult[T]:
        with self._diagnostic_lock:
            self._diagnostic["last_completion"] = None
        self._ensure_running()
        messages = tuple(messages)
        schema = response_model.model_json_schema()
        budget = self.prepare_budget(messages, schema)
        prompt = render_prompt(messages, schema)
        recoverable_budget_rejection = False
        self._generation_deadline = time.monotonic() + self.config.timeout_seconds
        # Metadata guards avoid rehashing 5.7 GB for every request. They are not
        # protection against deliberate same-size/timestamp tampering by the owner.
        try:
            start = time.perf_counter()
            tokens = self._tokenized_prompt(prompt)
            if (
                len(tokens) > self._max_input_tokens()
                or len(tokens) + self.config.num_predict + self.config.template_reserve
                >= self.config.num_ctx
            ):
                recoverable_budget_rejection = True
                raise ContextBudgetError("exact tokenized input exceeds reserved context")
            sampling = {
                "temperature": 0.0,
                "seed": self._sampling_seed,
                "repeat_penalty": 1.0,
            }
            if self._sampling_profile == "qwen-general":
                sampling.update(
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    min_p=0.0,
                    presence_penalty=1.5,
                )
            response = self._request(
                "/completion",
                {
                    "prompt": tokens,
                    "json_schema": schema,
                    "n_predict": self.config.num_predict,
                    "stream": False,
                    "cache_prompt": True,
                    **sampling,
                    "n_keep": -1,
                },
            )
            if response.get("truncated") is not False:
                raise ContextBudgetError("server reported missing/true truncation status")
            if (
                response.get("stop") is not True
                or response.get("stop_type") not in ("eos", "limit")
                or response.get("model") != "page-cite-rag-qwen35"
            ):
                raise StructuredOutputError("llama generation incomplete or wrong model")
            settings = response.get("generation_settings", {})
            if (
                not isinstance(settings, dict)
                or settings.get("n_predict") != self.config.num_predict
            ):
                raise ContextBudgetError("server generation limit differs from configuration")
            if self._sampling_profile == "qwen-general":
                for key, expected in sampling.items():
                    observed = settings.get(key)
                    if (
                        type(observed) not in (int, float)
                        or not math.isfinite(observed)
                        or not math.isclose(observed, expected, rel_tol=1e-5, abs_tol=1e-6)
                    ):
                        raise ModelError(f"server did not confirm sampling setting: {key}")
            try:
                timing = response["timings"]
                prompt_ms, predicted_ms = timing["prompt_ms"], timing["predicted_ms"]
                if (
                    type(prompt_ms) not in (int, float)
                    or type(predicted_ms) not in (int, float)
                    or prompt_ms < 0
                    or predicted_ms < 0
                ):
                    raise ValueError("invalid phase timing")
                stats = GenerationStats(
                    prompt_eval_count=response["tokens_evaluated"],
                    eval_count=response["tokens_predicted"],
                    total_duration=int((prompt_ms + predicted_ms) * 1e6),
                    load_duration=None,
                    prompt_eval_duration=int(prompt_ms * 1e6),
                    eval_duration=int(predicted_ms * 1e6),
                    done_reason="stop",
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                )
            except (ValidationError, KeyError, TypeError, ValueError, OverflowError):
                raise StructuredOutputError("llama runtime metrics violate the schema") from None
            if (
                stats.prompt_eval_count != len(tokens)
                or stats.prompt_eval_count > budget.input_upper_estimate
                or stats.eval_count > self.config.num_predict
            ):
                raise ContextBudgetError("runtime token counts violate the configured budget")
            if type(response.get("content")) is not str:
                raise StructuredOutputError("llama completion content is missing or invalid")
            reason = None
            validation_errors = ()
            if response["stop_type"] == "limit":
                if stats.eval_count != self.config.num_predict:
                    raise StructuredOutputError("llama output-limit accounting is inconsistent")
                reason = "output_limit"
            else:
                try:
                    value = response_model.model_validate_json(response["content"], strict=True)
                except ValidationError as exc:
                    reason = "schema_invalid"
                    validation_errors = tuple(
                        {
                            "location": [str(part)[:120] for part in error.get("loc", ())[:12]],
                            "type": str(error.get("type", "validation_error"))[:120],
                            "message": str(error.get("msg", "invalid output"))[:300],
                        }
                        for error in exc.errors(include_input=False)[:20]
                    )
                except ValueError:
                    reason = "schema_invalid"
            if reason is not None:
                # No restart or repair: verify that the same resident runtime is
                # still healthy before classifying this as output-only failure.
                self._tokenized_prompt(prompt)
                if self.config.device == "cuda":
                    self._check_gpu()
                raise OutputValidationError(
                    reason,
                    OutputFailureAccounting(
                        prompt_eval_count=stats.prompt_eval_count,
                        eval_count=stats.eval_count,
                        elapsed_ms=stats.elapsed_ms,
                        stop_type=response["stop_type"],
                    ),
                    raw_model_output=response["content"],
                    validation_errors=validation_errors,
                )
            return GenerationResult(value, stats, budget)
        except ContextBudgetError:
            # Pre-generation budget failures are recoverable; truncated responses aren't.
            if not recoverable_budget_rejection:
                self.close()
            raise
        except OutputValidationError:
            if not self._recover_output_errors:
                self.close()
            raise
        except BaseException:
            self.close()
            raise
        finally:
            self._generation_deadline = None

    def observations(self) -> dict:
        self._ensure_running()
        return {"server_pid": self._process.pid, "monitor_samples": list(self._monitor_samples)}

    @staticmethod
    def _diagnostic_number(value):
        if type(value) in (int, float) and abs(value) <= 2**63 - 1 and math.isfinite(value):
            return value
        return None

    def _record_request_error(self, category: str) -> None:
        with self._diagnostic_lock:
            if self._diagnostic["last_request"] is not None:
                self._diagnostic["last_request"]["error_category"] = category

    def _record_guard_error(self, category: str) -> None:
        with self._diagnostic_lock:
            self._diagnostic["guard_error"] = category

    def _record_gpu_sample(
        self, sample: dict, *, process_state: dict | None = None, phase: str = "monitor"
    ) -> None:
        """Telemetry must not change the result of the existing GPU safeguard."""
        try:
            recorded = {
                "elapsed_seconds": max(0.0, time.monotonic() - self._diagnostic_started),
                "phase": phase,
                "memory_mib": self._diagnostic_number(sample.get("memory_mib")),
                "temperature_c": self._diagnostic_number(sample.get("temperature_c")),
                "pids": [
                    pid
                    for pid in sample.get("pids", [])[:120]
                    if type(pid) is int and 0 < pid <= 2**63 - 1
                ],
            }
            for name in (
                "fan_speed_percent",
                "power_draw_w",
                "power_limit_w",
                "utilization_gpu_percent",
                "max_operating_c",
                "target_temperature_c",
            ):
                if name in sample:
                    recorded[name] = self._diagnostic_number(sample[name])
            for name in ("hw_thermal_slowdown", "sw_thermal_slowdown"):
                if name in sample:
                    recorded[name] = sample[name] if type(sample[name]) is bool else None
            with self._diagnostic_lock:
                if process_state is not None:
                    recorded["process_state"] = copy.deepcopy(process_state)
                    self._diagnostic["last_gpu_check"] = copy.deepcopy(recorded)
                self._monitor_samples.append(recorded)
                self._monitor_samples = self._monitor_samples[-120:]
                self._diagnostic["monitor_sample_count"] += 1
                peak = self._diagnostic["observed_gpu_peak"]
                for name in ("memory_mib", "temperature_c"):
                    value = recorded[name]
                    if peak[name] is None or value > peak[name]:
                        peak[name] = value
        except Exception:
            pass

    def _record_completion(self, response: dict) -> None:
        try:
            timings = response.get("timings")
            settings = response.get("generation_settings")
            timing_names = (
                "prompt_ms",
                "predicted_ms",
                "prompt_n",
                "predicted_n",
                "cache_n",
                "prompt_per_token_ms",
                "predicted_per_token_ms",
                "prompt_per_second",
                "predicted_per_second",
            )
            recorded = {
                "stop_type": response.get("stop_type")
                if response.get("stop_type") in ("eos", "limit", "word")
                else "unexpected",
                "truncated": response.get("truncated")
                if type(response.get("truncated")) is bool
                else None,
                "model": "page-cite-rag-qwen35"
                if response.get("model") == "page-cite-rag-qwen35"
                else "unexpected",
                "tokens_evaluated": self._diagnostic_number(response.get("tokens_evaluated")),
                "tokens_predicted": self._diagnostic_number(response.get("tokens_predicted")),
                "generation_settings": {
                    name: value
                    for name in (
                        "n_predict",
                        "temperature",
                        "top_p",
                        "top_k",
                        "min_p",
                        "presence_penalty",
                        "repeat_penalty",
                        "seed",
                    )
                    if isinstance(settings, dict)
                    and (value := self._diagnostic_number(settings.get(name))) is not None
                },
                "timings": {
                    name: number
                    for name in timing_names
                    if isinstance(timings, dict)
                    and (number := self._diagnostic_number(timings.get(name))) is not None
                },
            }
            with self._diagnostic_lock:
                self._diagnostic["last_completion"] = recorded
        except Exception:
            pass

    def _capture_process_status(self, *, stage: str | None = None) -> int | None:
        try:
            process = self._process
            if process is None:
                return None
            pid = process.pid
            with self._diagnostic_lock:
                if type(pid) is int and pid > 0:
                    self._diagnostic["server_pid"] = pid
            code = process.poll()
            if type(code) is not int and code is not None:
                return None
            with self._diagnostic_lock:
                if not self._exit_before_captured and (stage == "before" or code is not None):
                    self._diagnostic["exit_code_before_termination"] = code
                    self._exit_before_captured = True
                if stage == "after":
                    self._diagnostic["exit_code_after_termination"] = code
            return code
        except Exception:
            return None

    def _capture_log_tail(self) -> None:
        try:
            if self._log is None or self._log.closed:
                return
            # pread leaves the inherited server file offset untouched.
            token = self._auth_token or ""
            limit = DIAGNOSTIC_LOG_BYTES + len(token.encode("utf-8"))
            try:
                descriptor = self._log.fileno()
                size = os.fstat(descriptor).st_size
                raw = os.pread(descriptor, limit, max(0, size - limit))
            except (AttributeError, OSError, ValueError):
                position = self._log.tell()
                try:
                    self._log.seek(0, os.SEEK_END)
                    self._log.seek(max(0, self._log.tell() - limit))
                    raw = self._log.read(limit)
                finally:
                    self._log.seek(position)
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            if token:
                token_bytes = token.encode("utf-8")
                raw = raw.replace(token_bytes, b"[REDACTED]")
                # The retained read may begin halfway through a logged key.
                for length in range(min(len(token_bytes) - 1, len(raw)), 0, -1):
                    if raw.startswith(token_bytes[-length:]):
                        raw = b"[REDACTED]" + raw[length:]
                        break
                # A size snapshot can also catch an in-progress key write.
                for length in range(min(len(token_bytes) - 1, len(raw)), 0, -1):
                    if raw.endswith(token_bytes[:length]):
                        raw = raw[:-length] + b"[REDACTED]"
                        break
            text = raw.decode("utf-8", errors="replace")
            tail = text.encode("utf-8")[-DIAGNOSTIC_LOG_BYTES:].decode("utf-8", errors="ignore")
            with self._diagnostic_lock:
                self._server_log_tail = tail
        except Exception:
            pass

    def diagnostics(self, *, include_log: bool = False) -> dict:
        """Detached bounded metadata, available after close without any model probe.

        Logs are opt-in: a backend log can include document or generated text.
        This method retains its tail in memory and never writes a diagnostic file.
        """
        if include_log and not self._closed:
            self._capture_log_tail()
        with self._diagnostic_lock:
            result = copy.deepcopy(self._diagnostic)
            result["closed"] = self._closed
            result["monitor_samples"] = copy.deepcopy(self._monitor_samples)
            if include_log:
                result["server_log_tail"] = self._server_log_tail
            return result

    def _stop_owned_process(self) -> None:
        self._capture_process_status(stage="before")
        try:
            self._terminate_process()
        finally:
            self._capture_process_status(stage="after")

    def _terminate_process(self) -> None:
        with self._process_lock, gpu_process_lifecycle():
            if self._process is not None and self._process.poll() is None:
                with self._diagnostic_lock:
                    self._diagnostic["termination_requested"] = True
                try:
                    os.killpg(self._process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass  # The child may have exited between poll() and signal.
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self._process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self._process.wait(timeout=5)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._token_cache = None
        self._stop.set()
        self._stop_owned_process()
        if self._watcher is not None:
            self._watcher.join(timeout=12)
        with gpu_process_lifecycle():
            unregister_owned_gpu_process(self._process)
        self._capture_log_tail()
        if self._http is not None:
            self._http.close()
        if self._log is not None:
            self._log.close()
        if self._temporary is not None:
            self._temporary.cleanup()

    def __enter__(self) -> LlamaModelClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()
