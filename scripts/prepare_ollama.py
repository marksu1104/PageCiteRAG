"""Import a pinned Qwen3.5 GGUF into the private Ollama store.

Nothing is downloaded: Ollama copies the local file into a blob named by its SHA-256,
which must equal the pinned model hash. A temporary CPU-only server is started for the
import and stopped afterwards. Run before any evaluation that uses ``--generator-backend
ollama``; the evaluation itself never imports or pulls models. ``--source-model`` can
select the model-only artifact created by ``prepare_llama.py --gguf-only``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from page_cite_rag.llama import LLAMA_COMMIT, model_spec, sha256  # noqa: E402
from page_cite_rag.ollama_backend import (  # noqa: E402
    DEFAULT_OLLAMA_MODELS,
    DEFAULT_OLLAMA_RUNTIME,
    OLLAMA_VERSION,
    ollama_model_name,
)


def bundle_gguf(model_size: str, source_model: Path | None = None) -> Path:
    spec = model_spec(model_size)
    if source_model is not None:
        candidate = Path(source_model).expanduser()
        model_root = ROOT / "models"
        if model_root.is_symlink() or candidate.is_symlink():
            raise SystemExit("source model must be the pinned GGUF inside this checkout's models")
        try:
            model_root = model_root.resolve(strict=True)
            path = candidate.resolve(strict=True)
            inside_models = path.is_relative_to(model_root)
        except (OSError, RuntimeError):
            inside_models = False
            path = candidate
        if (
            not inside_models
            or not path.is_file()
            or path.stat().st_size != spec["bytes"]
            or sha256(path) != spec["sha256"]
        ):
            raise SystemExit("source model must be the pinned GGUF inside this checkout's models")
        return path
    bundle = (
        ROOT
        / "models"
        / f"llama-qwen35-{model_size}"
        / (f"{LLAMA_COMMIT[:8]}-{spec['revision'][:8]}")
    )
    path = bundle / spec["file"]
    if not path.is_file() or path.stat().st_size != spec["bytes"]:
        raise SystemExit(f"pinned GGUF is missing or has the wrong size: {path}")
    return path


def modelfile_contents(source: Path) -> str:
    """Render an absolute, quoted GGUF path for Ollama's Modelfile parser."""
    resolved = Path(source).resolve(strict=True)
    value = str(resolved)
    if "\n" in value or "\r" in value:
        raise ValueError("model path cannot contain a newline")
    return f"FROM {json.dumps(value, ensure_ascii=False)}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=["4b", "9b", "27b"], required=True)
    parser.add_argument(
        "--source-model",
        type=Path,
        help="verified local GGUF under models/ (for example, the CPU-only preparation route)",
    )
    args = parser.parse_args()
    spec = model_spec(args.model_size)
    source = bundle_gguf(args.model_size, args.source_model)
    name = ollama_model_name(args.model_size)
    runtime = ROOT / DEFAULT_OLLAMA_RUNTIME
    models = ROOT / DEFAULT_OLLAMA_MODELS
    with tempfile.TemporaryDirectory(prefix="page-cite-rag-ollama-import-") as temporary:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = {
            "PATH": os.defpath,
            "LANG": "C.UTF-8",
            "HOME": temporary,
            "OLLAMA_HOST": f"127.0.0.1:{port}",
            "OLLAMA_MODELS": str(models),
            "OLLAMA_NOPRUNE": "1",
            "OLLAMA_NO_CLOUD": "1",
            "CUDA_VISIBLE_DEVICES": "-1",
        }
        modelfile = Path(temporary) / "Modelfile"
        modelfile.write_text(modelfile_contents(source))
        log = open(Path(temporary) / "server.log", "wb")
        server = subprocess.Popen(
            [str(runtime / "bin" / "ollama"), "serve"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", trust_env=False) as client:
                for _ in range(120):
                    try:
                        version = client.get("/api/version").json()["version"]
                        break
                    except httpx.TransportError:
                        time.sleep(0.5)
                else:
                    raise SystemExit("temporary Ollama server did not start")
                if version != OLLAMA_VERSION:
                    raise SystemExit("Ollama runtime version differs from the pinned version")
                created = subprocess.run(
                    [str(runtime / "bin" / "ollama"), "create", name, "-f", str(modelfile)],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
                if created.returncode != 0:
                    raise SystemExit(f"ollama create failed: {created.stderr[-500:]}")
                shown = client.post("/api/show", json={"model": name}, timeout=60).json()
        finally:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=30)
            log.close()
    sources = [
        line.split(" ", 1)[1].strip()
        for line in shown.get("modelfile", "").splitlines()
        if line.startswith("FROM ")
    ]
    if len(sources) != 1 or Path(sources[0]).name != f"sha256-{spec['sha256']}":
        raise SystemExit("imported blob is not the pinned GGUF")
    print(
        json.dumps(
            {"model": name, "sha256": spec["sha256"], "blob": sources[0], "source": str(source)}
        )
    )


if __name__ == "__main__":
    main()
