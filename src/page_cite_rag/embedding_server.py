"""Owned, authenticated CUDA embedding server for explicit staged query requests."""

from __future__ import annotations

import hashlib
import math
import os
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import httpx
import numpy as np

from page_cite_rag.embedding_native import GGUF_REVISION, GGUF_SHA256
from page_cite_rag.gpu_processes import (
    gpu_process_lifecycle,
    register_owned_gpu_process,
    snapshot_gpu_processes,
    unregister_owned_gpu_process,
)
from page_cite_rag.llama import DEFAULT_BUNDLE, LLAMA_COMMIT, verify_bundle
from page_cite_rag.retrieval import RetrievalError

_MODEL_BYTES = 1_197_629_632
MODEL_FILENAME = "Qwen3-Embedding-0.6B-f16.gguf"
DEFAULT_GGUF_PATH = Path("models/qwen3-embedding-0.6b-gguf") / GGUF_REVISION / MODEL_FILENAME


class _GpuGuardTriggered(RetrievalError):
    def __init__(self, category: str, sample: dict) -> None:
        super().__init__(category)
        self.category = category
        self.sample = _sample_record(sample, "guard")


def _sample_record(sample: dict, phase: str) -> dict:
    record = dict(sample)
    record["pids"] = sorted(record.get("pids", ()))
    record["phase"] = phase
    return record


def verify_native_model(path: Path) -> None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size != _MODEL_BYTES:
        raise RetrievalError("prepared native embedding GGUF is missing or has wrong size")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != GGUF_SHA256:
        raise RetrievalError("prepared native embedding GGUF checksum changed")


def prepare_native_model(path: Path = DEFAULT_GGUF_PATH) -> dict:
    """Explicit public download, atomic verified install; no inference side effect."""
    path = path.expanduser()
    if path.exists() or path.is_symlink():
        verify_native_model(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".native-model-", dir=path.parent) as temporary:
            staged = Path(temporary) / MODEL_FILENAME
            url = (
                "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF/resolve/"
                f"{GGUF_REVISION}/{MODEL_FILENAME}"
            )
            with urllib.request.urlopen(url, timeout=60) as response, staged.open("wb") as output:
                received = 0
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > _MODEL_BYTES:
                        raise RetrievalError("native embedding download exceeded expected size")
                    output.write(chunk)
            verify_native_model(staged)
            staged.rename(path)
    return {"path": str(path), "sha256": GGUF_SHA256, "bytes": _MODEL_BYTES}


def _temperature_limits(uuid: str, limit_c: int) -> dict:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "-q", "-i", uuid, "-d", "TEMPERATURE"],
            text=True,
            timeout=3,
        )
        limits = {}
        for key, label in (
            ("max_operating_c", "GPU Max Operating Temp"),
            ("target_temperature_c", "GPU Target Temperature"),
        ):
            matches = re.findall(rf"^\s*{label}\s*:\s*(\d+)\s*C\s*$", output, re.MULTILINE)
            if len(matches) != 1 or not 40 <= int(matches[0]) <= 125:
                raise ValueError("missing or invalid driver thermal threshold")
            limits[key] = int(matches[0])
        if limit_c > limits["max_operating_c"] - 5:
            raise ValueError("temperature guard must leave 5 C below GPU thermal threshold")
        return limits
    except (OSError, ValueError, subprocess.SubprocessError):
        raise RetrievalError("GPU thermal thresholds unavailable for embedding") from None


def _gpu_sample(
    uuid: str,
    *,
    max_gpu_temperature_c: int = 80,
    temperature_limits: dict | None = None,
) -> dict:
    try:
        if (
            type(max_gpu_temperature_c) is not int
            or not 40 <= max_gpu_temperature_c <= 88
            or (max_gpu_temperature_c > 80 and not isinstance(temperature_limits, dict))
        ):
            raise ValueError("invalid GPU temperature policy")
        if max_gpu_temperature_c > 80 and (
            type(temperature_limits.get("max_operating_c")) is not int
            or max_gpu_temperature_c > temperature_limits["max_operating_c"] - 5
        ):
            raise ValueError("GPU temperature guard lacks the required headroom")
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                uuid,
                "--query-gpu=memory.used,temperature.gpu,"
                "clocks_throttle_reasons.hw_thermal_slowdown,"
                "clocks_throttle_reasons.sw_thermal_slowdown",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        ).strip()
        memory, temperature, hardware, software = [value.strip() for value in raw.split(",")]
        raw_pids = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                uuid,
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        ).strip()
        result = {
            "memory_mib": int(memory),
            "temperature_c": int(temperature),
            "pids": {int(pid) for pid in raw_pids.splitlines() if pid},
            "hw_thermal_slowdown": hardware == "Active",
            "sw_thermal_slowdown": software == "Active",
        }
        if temperature_limits is not None:
            result.update(temperature_limits)
        if hardware not in ("Active", "Not Active") or software not in (
            "Active",
            "Not Active",
        ):
            raise RetrievalError("GPU thermal telemetry unavailable for embedding")
        if hardware == "Active" or software == "Active":
            raise _GpuGuardTriggered("gpu_thermal_slowdown", result)
        if result["memory_mib"] > 12000:
            raise _GpuGuardTriggered("gpu_memory_limit", result)
        if result["temperature_c"] > max_gpu_temperature_c:
            raise _GpuGuardTriggered("gpu_temperature_limit", result)
        return result
    except (OSError, ValueError, subprocess.SubprocessError):
        raise RetrievalError("GPU telemetry unavailable for embedding") from None


class NativeEmbeddingServer:
    """One owned model; shares GPU only with other registered local services."""

    def __init__(
        self,
        model_path: Path,
        gpu_uuid: str,
        *,
        bundle_dir: Path = DEFAULT_BUNDLE,
        cancel_event: threading.Event | None = None,
        max_gpu_temperature_c: int = 80,
    ):
        if type(max_gpu_temperature_c) is not int or not 40 <= max_gpu_temperature_c <= 88:
            raise ValueError("GPU temperature guard must be an integer from 40 to 88 C")
        self.model_path = model_path.expanduser().absolute()
        self.gpu_uuid = gpu_uuid
        self.bundle_dir = bundle_dir.expanduser().absolute()
        self._cancel_event = cancel_event
        self.max_gpu_temperature_c = max_gpu_temperature_c
        self._temperature_limits = None
        self._http: httpx.Client | None = None
        self._process: subprocess.Popen | None = None
        self._temporary: tempfile.TemporaryDirectory | None = None
        self._log = None
        self._stop = threading.Event()
        self._watcher: threading.Thread | None = None
        self._guard_error: str | None = None
        self._last_gpu_check: dict | None = None
        self._last_gpu_sample: dict | None = None
        self.samples: list[dict] = []
        self.startup_ms = 0.0
        try:
            self._start()
        except BaseException:
            self.close()
            raise

    def _start(self) -> None:
        started = time.perf_counter()
        self._verify_model()
        verify_bundle(self.bundle_dir)
        self._check_cancelled()
        if self.max_gpu_temperature_c > 80:
            self._temperature_limits = _temperature_limits(
                self.gpu_uuid, self.max_gpu_temperature_c
            )
        try:
            initial, process_state = snapshot_gpu_processes(
                lambda: _gpu_sample(
                    self.gpu_uuid,
                    max_gpu_temperature_c=self.max_gpu_temperature_c,
                    temperature_limits=self._temperature_limits,
                )
            )
        except _GpuGuardTriggered as exc:
            self._guard_error = exc.category
            self._last_gpu_sample = _sample_record(exc.sample, "startup")
            self.samples.append(self._last_gpu_sample)
            raise
        self._last_gpu_sample = _sample_record(initial, "startup")
        self.samples.append(self._last_gpu_sample)
        self._last_gpu_check = process_state
        if process_state["unexpected_pids"]:
            self._guard_error = "unexpected_gpu_process"
            raise _GpuGuardTriggered("unexpected_gpu_process", initial)
        binary = self.bundle_dir / "bin/llama-server"
        self._temporary = tempfile.TemporaryDirectory(prefix="page-cite-rag-embedding-session-")
        session = Path(self._temporary.name)
        token = secrets.token_hex(32)
        key_file = session / "api-key"
        key_file.touch(mode=0o600)
        key_file.write_text(token + "\n", encoding="utf-8")
        with socket.socket() as port_probe:
            port_probe.bind(("127.0.0.1", 0))
            port = port_probe.getsockname()[1]
        env = {
            "PATH": os.defpath,
            "LANG": "C.UTF-8",
            "LD_LIBRARY_PATH": str(binary.parent),
            "CUDA_VISIBLE_DEVICES": self.gpu_uuid,
            "CUDA_CACHE_PATH": str(session / "cuda-cache"),
            "OMP_NUM_THREADS": "2",
        }
        try:
            version = subprocess.check_output(
                [str(binary), "--version"], env=env, stderr=subprocess.STDOUT, timeout=10
            ).decode()
        except (OSError, UnicodeError, subprocess.SubprocessError):
            raise RetrievalError("prepared native embedding server could not start") from None
        if LLAMA_COMMIT[:7] not in version:
            raise RetrievalError("native embedding runtime differs from pinned bundle")
        self._check_cancelled()
        argv = [
            str(binary),
            "-m",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--api-key-file",
            str(key_file),
            "-ngl",
            "99",
            "-c",
            str(self._context_size()),
            "-np",
            "1",
            "-t",
            "2",
            "-b",
            str(self._batch_size()),
            "-ub",
            str(self._batch_size()),
            "--flash-attn",
            "off",
            *self._mode_options(),
            "--no-warmup",
            "--no-ui",
            "--no-agent",
            "--no-slots",
            "--no-mmproj",
            "--offline",
            "--verbosity",
            "4",
            "--threads-http",
            "2",
        ]
        self._log = (session / "server.log").open("w+b")
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
                cwd=session,
                env=env,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            register_owned_gpu_process(self._process)
        self._http = httpx.Client(
            base_url=f"http://127.0.0.1:{port}",
            headers={"Authorization": f"Bearer {token}"},
            trust_env=False,
            follow_redirects=False,
            timeout=self._request_timeout(),
        )
        self._watcher = threading.Thread(target=self._watch, daemon=True)
        self._watcher.start()
        deadline = time.monotonic() + 60
        while True:
            self._check()
            try:
                response = self._http.get("/health")
                if response.status_code == 200 and response.json() == {"status": "ok"}:
                    break
            except (httpx.HTTPError, ValueError):
                pass
            if time.monotonic() >= deadline:
                raise RetrievalError("native embedding server startup exceeded 60 seconds")
            time.sleep(0.2)
        self._log.flush()
        self._verify_offload((session / "server.log").read_text(errors="replace"))
        self.startup_ms = (time.perf_counter() - started) * 1000

    def _verify_model(self) -> None:
        verify_native_model(self.model_path)

    def _context_size(self) -> int:
        return 1024

    def _batch_size(self) -> int:
        return 128

    def _mode_options(self) -> list[str]:
        return ["--embedding", "--pooling", "last", "--embd-normalize", "2"]

    def _request_timeout(self) -> int:
        return 15

    def _verify_offload(self, log: str) -> None:
        if "offloaded 29/29 layers to GPU" not in log:
            raise RetrievalError("native embedding full GPU offload was not confirmed")

    def diagnostics(self) -> dict:
        """Return resource observations without process arguments or source text."""
        return {
            "gpu_uuid": self.gpu_uuid,
            "gpu_temperature_limit_c": self.max_gpu_temperature_c,
            "max_operating_c": (self._temperature_limits or {}).get("max_operating_c"),
            "target_temperature_c": (self._temperature_limits or {}).get("target_temperature_c"),
            "monitor_sample_count": len(self.samples),
            "observed_gpu_peak": {
                "memory_mib": max((sample["memory_mib"] for sample in self.samples), default=None),
                "temperature_c": max(
                    (sample["temperature_c"] for sample in self.samples), default=None
                ),
            },
            "hw_thermal_slowdown": any(sample["hw_thermal_slowdown"] for sample in self.samples),
            "sw_thermal_slowdown": any(sample["sw_thermal_slowdown"] for sample in self.samples),
            "guard_error": self._guard_error,
            "last_gpu_check": self._last_gpu_check,
            "last_gpu_sample": self._last_gpu_sample,
        }

    def _watch(self) -> None:
        while not self._stop.wait(0.5):
            try:
                self._check_cancelled()
                sample, process_state = snapshot_gpu_processes(
                    lambda: _gpu_sample(
                        self.gpu_uuid,
                        max_gpu_temperature_c=self.max_gpu_temperature_c,
                        temperature_limits=self._temperature_limits,
                    ),
                    expected_process=self._process,
                )
                self._last_gpu_check = process_state
                if process_state["owned_service_exited"]:
                    raise _GpuGuardTriggered("owned_service_exited", sample)
                if process_state["unexpected_pids"]:
                    raise _GpuGuardTriggered("unexpected_gpu_process", sample)
                self._last_gpu_sample = _sample_record(sample, "monitor")
                self.samples.append(self._last_gpu_sample)
            except (RetrievalError, OSError, ValueError) as exc:
                if isinstance(exc, _GpuGuardTriggered):
                    self._guard_error = exc.category
                    self._last_gpu_sample = _sample_record(exc.sample, "monitor")
                    self.samples.append(self._last_gpu_sample)
                else:
                    self._guard_error = str(exc)
                with gpu_process_lifecycle():
                    if self._process is not None and self._process.poll() is None:
                        self._process.terminate()
                return

    def _check(self) -> None:
        self._check_cancelled()
        if self._guard_error is not None:
            raise RetrievalError(self._guard_error)
        if self._process is None or self._process.poll() is not None:
            raise RetrievalError("owned native embedding server exited")

    def _check_cancelled(self) -> None:
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise RetrievalError("owned native embedding request was cancelled")

    def _post(self, path: str, payload: dict) -> dict:
        self._check()
        try:
            response = self._http.post(path, json=payload)
            response.raise_for_status()
            if len(response.content) > 2 * 1024 * 1024:
                raise RetrievalError("native embedding response is too large")
            value = response.json()
        except httpx.TimeoutException:
            raise RetrievalError(f"native local request timed out at {path}") from None
        except httpx.HTTPStatusError as exc:
            raise RetrievalError(
                f"native local endpoint returned HTTP {exc.response.status_code} at {path}"
            ) from None
        except httpx.HTTPError as exc:
            raise RetrievalError(
                f"native local transport failed at {path} ({type(exc).__name__})"
            ) from None
        except ValueError:
            raise RetrievalError(f"native local endpoint returned invalid JSON at {path}") from None
        self._check()
        if not isinstance(value, dict):
            raise RetrievalError("native embedding response is invalid")
        return value

    def token_ids(self, text: str) -> list[int]:
        value = self._post(
            "/tokenize", {"content": text, "add_special": True, "parse_special": False}
        )
        return value.get("tokens")

    def embed(self, texts: list[str]) -> np.ndarray:
        value = self._post("/v1/embeddings", {"input": texts, "model": "qwen"})
        entries = value.get("data")
        if not isinstance(entries, list) or len(entries) != len(texts):
            raise RetrievalError("native embedding returned a wrong row count")
        entries = sorted(entries, key=lambda item: item.get("index", -1))
        if [item.get("index") for item in entries] != list(range(len(texts))):
            raise RetrievalError("native embedding returned invalid row indexes")
        vectors = [item.get("embedding") for item in entries]
        if any(
            not isinstance(vector, list)
            or len(vector) != 1024
            or any(
                not isinstance(number, (int, float)) or not math.isfinite(number)
                for number in vector
            )
            for vector in vectors
        ):
            raise RetrievalError("native embedding returned invalid vectors")
        return np.asarray(vectors, dtype=np.float32)

    def close(self) -> None:
        self._stop.set()
        if self._http is not None:
            self._http.close()
            self._http = None
        with gpu_process_lifecycle():
            if self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
            unregister_owned_gpu_process(self._process)
        if self._watcher is not None and self._watcher is not threading.current_thread():
            self._watcher.join(timeout=5)
            self._watcher = None
        if self._log is not None:
            self._log.close()
            self._log = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None

    def __enter__(self) -> NativeEmbeddingServer:
        return self

    def __exit__(self, *args) -> None:
        self.close()
