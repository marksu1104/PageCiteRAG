"""Explicit local CUDA build/model download; no sudo, inference or system changes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

from page_cite_rag.llama import (
    DEFAULT_BUNDLE,
    LLAMA_COMMIT,
    LLAMA_VERSION,
    MODEL_BYTES,
    MODEL_FILE,
    MODEL_REPO,
    MODEL_REVISION,
    MODEL_SHA256,
    SOURCE_SHA256,
    sha256,
    verify_bundle,
)


def download(url: str, path: Path, cap: int, timeout: int, expected: str) -> None:
    start = time.monotonic()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=30) as response, path.open("xb") as output:
        received = 0
        while chunk := response.read(1024 * 1024):
            received += len(chunk)
            if received > cap or time.monotonic() - start > timeout:
                raise RuntimeError("download exceeded its bounded preparation budget")
            output.write(chunk)
    if sha256(path) != expected:
        raise RuntimeError("download checksum differs from the fixed source")


def run(argv: list[str], log: Path, env: dict, timeout: int) -> None:
    print(f"start {log.stem}", flush=True)
    with log.open("xb") as output:
        process = subprocess.Popen(
            argv, env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = process.wait(timeout=timeout)
        finally:
            if process.poll() is None:
                import signal

                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
        if code:
            raise RuntimeError(f"{log.stem} failed; inspect the local log, no automatic retry")
    print(f"passed {log.stem}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cuda-root", type=Path, required=True)
    parser.add_argument("--gcc", type=Path, required=True)
    parser.add_argument("--gxx", type=Path, required=True)
    parser.add_argument("--architecture", default="86-real", choices=["86-real"])
    args = parser.parse_args()
    os.umask(0o077)
    target = args.bundle_dir.absolute()
    root = Path.cwd().resolve()
    if (root / "models").is_symlink() or not target.resolve().is_relative_to(root / "models"):
        raise ValueError("bundle must be inside this checkout's models directory")
    if target.exists() or target.is_symlink():
        verify_bundle(target)
        print("Existing pinned bundle verified; no download or build.")
        return
    # Keep preparation artifacts inside the designated output directories.
    output = args.output_dir.absolute()
    if (
        (root / "outputs").is_symlink()
        or not output.resolve().is_relative_to(root / "outputs")
        or output.exists()
        or output.is_symlink()
    ):
        raise ValueError("use a new output directory inside this checkout's outputs")
    for tool in (args.gcc, args.gxx, args.cuda_root / "bin/nvcc"):
        if not tool.is_absolute() or not tool.is_file():
            raise ValueError("provide existing absolute compiler/toolkit paths; no auto install")
    output.mkdir(parents=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    report = {"scope": "explicit pinned local preparation, not GPU inference validation"}
    try:
        with tempfile.TemporaryDirectory(prefix="page-cite-rag-llama-build-") as temporary:
            work = Path(temporary)
            report["temporary_directory"] = str(work)
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("LLAMA_", "OLLAMA_", "HF_", "CMAKE_", "CUDA_"))
            }
            env.update(
                {
                    "UV_CACHE_DIR": str(work / "uv-cache"),
                    "CC": str(args.gcc),
                    "CXX": str(args.gxx),
                    "CUDACXX": str(args.cuda_root / "bin/nvcc"),
                }
            )
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            for package, version in (("cmake", "3.31.10"), ("ninja", "1.13.0")):
                request = urllib.request.Request(
                    "https://api.osv.dev/v1/query",
                    data=json.dumps(
                        {"package": {"name": package, "ecosystem": "PyPI"}, "version": version}
                    ).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with opener.open(request, timeout=30) as response:
                    advisory = json.load(response)
                if advisory.get("vulns"):
                    raise RuntimeError("temporary build tool has known advisories; stop for review")
            run(
                ["uv", "venv", "--python", "3.11", str(work / "venv")], output / "venv.log", env, 60
            )
            run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(work / "venv/bin/python"),
                    "cmake==3.31.10",
                    "ninja==1.13.0",
                ],
                output / "build-tools.log",
                env,
                120,
            )
            archive = work / "source.tar.gz"
            download(
                f"https://api.github.com/repos/ggml-org/llama.cpp/tarball/{LLAMA_COMMIT}",
                archive,
                50 * 1024 * 1024,
                90,
                SOURCE_SHA256,
            )
            with tarfile.open(archive) as source:
                for member in source.getmembers():
                    if (
                        not (member.isfile() or member.isdir())
                        or member.name.startswith("/")
                        or ".." in Path(member.name).parts
                    ):
                        raise RuntimeError("source archive has an unexpected path/type")
                source.extractall(work / "source", filter="data")
            (source_dir,) = (work / "source").iterdir()
            cmake = str(work / "venv/bin/cmake")
            flags = [
                f"-DCMAKE_MAKE_PROGRAM={work}/venv/bin/ninja",
                f"-DCMAKE_C_COMPILER={args.gcc}",
                f"-DCMAKE_CXX_COMPILER={args.gxx}",
                f"-DCMAKE_CUDA_COMPILER={args.cuda_root}/bin/nvcc",
                f"-DCMAKE_CUDA_HOST_COMPILER={args.gxx}",
                f"-DCUDAToolkit_ROOT={args.cuda_root}",
                f"-DCMAKE_CUDA_ARCHITECTURES={args.architecture}",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DGGML_CUDA=ON",
                "-DGGML_CUDA_FA=OFF",
                "-DGGML_CUDA_GRAPHS=OFF",
                "-DGGML_CUDA_NO_VMM=ON",
                "-DGGML_CUDA_CUB_3DOT2=OFF",
                "-DGGML_NATIVE=OFF",
                "-DLLAMA_BUILD_TESTS=OFF",
                "-DLLAMA_BUILD_APP=OFF",
                "-DLLAMA_BUILD_SERVER=ON",
                "-DLLAMA_BUILD_TOOLS=ON",
                "-DLLAMA_BUILD_EXAMPLES=OFF",
                "-DLLAMA_OPENSSL=OFF",
                "-DLLAMA_BUILD_UI=OFF",
                "-DLLAMA_USE_PREBUILT_UI=OFF",
                "-DLLAMA_BUILD_IS_DEV=OFF",
                f"-DLLAMA_BUILD_COMMIT={LLAMA_COMMIT}",
                "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
            ]
            run(
                [cmake, "-S", str(source_dir), "-B", str(work / "build"), "-G", "Ninja", *flags],
                output / "configure.log",
                env,
                90,
            )
            run(
                [
                    cmake,
                    "--build",
                    str(work / "build"),
                    "--target",
                    "llama-server",
                    "--parallel",
                    "2",
                ],
                output / "build.log",
                env,
                900,
            )
            # Stage atomically; failed preparation never replaces an existing bundle.
            with tempfile.TemporaryDirectory(prefix=".prepare-", dir=target.parent) as staging:
                bundle = Path(staging) / "bundle"
                (bundle / "bin").mkdir(parents=True)
                for path in (work / "build/bin").iterdir():
                    if path.name == "llama-server" or ".so" in path.name:
                        shutil.copy2(path, bundle / "bin" / path.name, follow_symlinks=True)
                shutil.copy2(source_dir / "LICENSE", bundle / "LLAMA-LICENSE")
                download(
                    f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_FILE}",
                    bundle / MODEL_FILE,
                    MODEL_BYTES,
                    600,
                    MODEL_SHA256,
                )
                manifest = {
                    "format": 1,
                    "llama_commit": LLAMA_COMMIT,
                    "llama_version": LLAMA_VERSION,
                    "source_sha256": SOURCE_SHA256,
                    "model_repo": MODEL_REPO,
                    "model_revision": MODEL_REVISION,
                    "model_sha256": MODEL_SHA256,
                    "model_license": "Apache-2.0",
                    "publisher": "Unsloth third-party quantization",
                    "build_flags": flags,
                    "files": {
                        str(p.relative_to(bundle)): sha256(p)
                        for p in bundle.rglob("*")
                        if p.is_file()
                    },
                }
                (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                verify_bundle(bundle)
                bundle.rename(target)
            report["prepared"] = True
            report["bundle_dir"] = str(target)
    finally:
        report["temporary_directory_removed"] = not Path(
            report.get("temporary_directory", "/path-not-created")
        ).exists()
        (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Pinned bundle prepared; temporary build environment removed.", flush=True)


if __name__ == "__main__":
    main()
