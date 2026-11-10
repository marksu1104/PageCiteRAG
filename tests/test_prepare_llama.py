import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PREPARE_PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_llama.py"
_PREPARE_SPEC = importlib.util.spec_from_file_location("prepare_llama", _PREPARE_PATH)
prepare_llama = importlib.util.module_from_spec(_PREPARE_SPEC)
_PREPARE_SPEC.loader.exec_module(prepare_llama)

_OLLAMA_PREPARE_PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_ollama.py"
_OLLAMA_PREPARE_SPEC = importlib.util.spec_from_file_location(
    "prepare_ollama", _OLLAMA_PREPARE_PATH
)
prepare_ollama = importlib.util.module_from_spec(_OLLAMA_PREPARE_SPEC)
_OLLAMA_PREPARE_SPEC.loader.exec_module(prepare_ollama)


def _fake_runtime(tmp_path):
    runtime = tmp_path / "models" / "runtime-9b"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "bin" / "llama-server").write_bytes(b"server")
    (runtime / "LLAMA-LICENSE").write_text("runtime license")
    (runtime / "original-9b.gguf").write_bytes(b"original weights")
    return runtime


def test_4b_preparation_reuses_verified_runtime_and_only_downloads_selected_model(
    monkeypatch, tmp_path
):
    runtime = _fake_runtime(tmp_path)
    target = tmp_path / "models" / "new-4b"
    output = tmp_path / "outputs" / "prepare-4b"
    spec = prepare_llama.model_spec("4b")
    verified = []

    def verify(path, model_size="9b"):
        verified.append((path, model_size))
        if model_size == "4b":
            manifest = json.loads((path / "manifest.json").read_text())
            assert manifest["model_revision"] == spec["revision"]
            assert manifest["model_sha256"] == spec["sha256"]
            assert set(manifest["files"]) == {
                spec["file"],
                "bin/llama-server",
                "LLAMA-LICENSE",
            }
            assert (path / spec["file"]).read_bytes() == b"downloaded 4B weights"
        return {"build_flags": ["-DGGML_CUDA=ON"]}

    def download(url, path, cap, timeout, expected):
        assert url == (
            f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{spec['file']}"
        )
        assert (cap, timeout, expected) == (2_740_937_888, 600, spec["sha256"])
        path.write_bytes(b"downloaded 4B weights")

    monkeypatch.setattr(prepare_llama, "verify_bundle", verify)
    monkeypatch.setattr(prepare_llama, "download", download)
    prepare_llama.prepare_4b(target, output, runtime)

    assert [model_size for _, model_size in verified] == ["9b", "4b"]
    assert (target / "bin" / "llama-server").read_bytes() == b"server"
    assert (runtime / "original-9b.gguf").read_bytes() == b"original weights"
    assert not (target / "original-9b.gguf").exists()
    report = json.loads((output / "preparation.json").read_text())
    assert report["prepared"] is True
    assert report["runtime_rebuilt"] is False
    assert not list(target.parent.glob(".prepare-*"))


def test_4b_failed_download_does_not_publish_bundle_or_change_runtime(monkeypatch, tmp_path):
    runtime = _fake_runtime(tmp_path)
    target = tmp_path / "models" / "new-4b"
    output = tmp_path / "outputs" / "prepare-4b"
    monkeypatch.setattr(
        prepare_llama, "verify_bundle", lambda *_: {"build_flags": ["-DGGML_CUDA=ON"]}
    )

    def fail_download(_url, path, *_args):
        path.write_bytes(b"partial")
        raise RuntimeError("download checksum differs from the fixed source")

    monkeypatch.setattr(prepare_llama, "download", fail_download)
    with pytest.raises(RuntimeError, match="checksum"):
        prepare_llama.prepare_4b(target, output, runtime)

    assert not target.exists()
    assert not list(target.parent.glob(".prepare-*"))
    assert (runtime / "original-9b.gguf").read_bytes() == b"original weights"
    assert json.loads((output / "preparation.json").read_text())["prepared"] is False


def test_4b_cli_uses_pinned_default_without_requiring_cuda_compilers(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_llama.py", "--model-size", "4b", "--output-dir", "outputs/prepare-4b"],
    )
    calls = []
    monkeypatch.setattr(prepare_llama, "prepare_4b", lambda *args: calls.append(args))

    prepare_llama.main()

    target, output, runtime = calls[0]
    assert target == tmp_path / "models/llama-qwen35-4b/7fe450e1-e87f1764"
    assert output == tmp_path / "outputs/prepare-4b"
    assert runtime == tmp_path / prepare_llama.DEFAULT_BUNDLE


def test_4b_cli_rejects_runtime_outside_models(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_llama.py",
            "--model-size",
            "4b",
            "--output-dir",
            "outputs/prepare-4b",
            "--runtime-bundle",
            str(tmp_path / "outside-runtime"),
        ],
    )
    with pytest.raises(ValueError, match="inside this checkout's models"):
        prepare_llama.main()
    assert not (tmp_path / "outputs").exists()


def test_27b_preparation_links_verified_local_weights_without_download_or_build(
    monkeypatch, tmp_path
):
    runtime = _fake_runtime(tmp_path)
    target = tmp_path / "models" / "reference-27b"
    output = tmp_path / "outputs" / "prepare-27b"
    source = tmp_path / "models" / "downloaded-27b.gguf"
    source.write_bytes(b"synthetic 27B weights")
    spec = {
        **prepare_llama.model_spec("27b"),
        "bytes": source.stat().st_size,
        "sha256": prepare_llama.sha256(source),
    }
    verified = []

    def verify(path, model_size="9b"):
        verified.append(model_size)
        if model_size == "27b":
            manifest = json.loads((path / "manifest.json").read_text())
            assert manifest["llama_commit"] == prepare_llama.LLAMA_COMMIT
            assert manifest["model_revision"] == spec["revision"]
            assert manifest["model_sha256"] == spec["sha256"]
            assert manifest["files"][spec["file"]] == spec["sha256"]
            assert (path / spec["file"]).stat().st_ino == source.stat().st_ino
        return {"build_flags": ["-DGGML_CUDA=ON"]}

    monkeypatch.setattr(prepare_llama, "verify_bundle", verify)
    monkeypatch.setattr(prepare_llama, "model_spec", lambda size: spec if size == "27b" else None)
    monkeypatch.setattr(prepare_llama, "download", lambda *_: pytest.fail("unexpected download"))
    monkeypatch.setattr(prepare_llama, "run", lambda *_: pytest.fail("unexpected build"))
    prepare_llama.prepare_with_runtime(
        target, output, runtime, model_size="27b", source_model=source
    )

    assert verified == ["9b", "27b"]
    assert (runtime / "original-9b.gguf").read_bytes() == b"original weights"
    assert (target / spec["file"]).read_bytes() == source.read_bytes()
    assert source.stat().st_nlink == 2
    assert not (target / "original-9b.gguf").exists()
    report = json.loads((output / "preparation.json").read_text())
    assert report["prepared"] is True and report["runtime_rebuilt"] is False
    assert report["local_source_model"] == str(source)
    assert not list(target.parent.glob(".prepare-*"))


def test_27b_preparation_rejects_unverified_weights_before_publishing(monkeypatch, tmp_path):
    runtime = _fake_runtime(tmp_path)
    target = tmp_path / "models" / "reference-27b"
    output = tmp_path / "outputs" / "prepare-27b"
    source = tmp_path / "models" / "bad-27b.gguf"
    source.write_bytes(b"bad weights")
    spec = {
        **prepare_llama.model_spec("27b"),
        "bytes": source.stat().st_size,
    }
    monkeypatch.setattr(prepare_llama, "verify_bundle", lambda *_: {"build_flags": []})
    monkeypatch.setattr(prepare_llama, "model_spec", lambda _size: spec)
    monkeypatch.setattr(prepare_llama, "download", lambda *_: pytest.fail("unexpected download"))
    with pytest.raises(ValueError, match="local weights"):
        prepare_llama.prepare_with_runtime(target, output, runtime, model_size="27b")
    with pytest.raises(ValueError, match="checksum"):
        prepare_llama.prepare_with_runtime(
            target, output, runtime, model_size="27b", source_model=source
        )
    assert not target.exists() and not output.exists()
    assert source.read_bytes() == b"bad weights"
    assert (runtime / "original-9b.gguf").read_bytes() == b"original weights"


def test_9b_gguf_only_preparation_downloads_without_runtime_or_compiler(monkeypatch, tmp_path):
    weights = b"pinned 9B weights"
    spec = {
        "repo": "example/qwen35-9b",
        "revision": "a" * 40,
        "file": "Qwen3.5-9B-Q4_K_M.gguf",
        "bytes": len(weights),
        "sha256": hashlib.sha256(weights).hexdigest(),
    }
    monkeypatch.setattr(prepare_llama, "model_spec", lambda _size: spec)
    downloads = []

    def download(url, path, cap, timeout, expected):
        downloads.append((url, cap, timeout, expected))
        path.write_bytes(weights)

    monkeypatch.setattr(prepare_llama, "download", download)
    target = tmp_path / "models/gguf-qwen35-9b" / spec["revision"][:8]
    output = tmp_path / "outputs/prepare-9b-cpu"

    prepare_llama.prepare_gguf_only(target, output)

    assert downloads == [
        (
            f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{spec['file']}",
            len(weights),
            600,
            spec["sha256"],
        )
    ]
    assert (target / spec["file"]).read_bytes() == weights
    assert prepare_llama.verify_gguf_only(target)["model_sha256"] == spec["sha256"]
    report = json.loads((output / "preparation.json").read_text())
    assert report["prepared"] is True
    assert "no llama.cpp build or CUDA toolkit" in report["scope"]
    assert not list(target.parent.glob(".gguf-only-*"))


def test_9b_gguf_only_preparation_hardlinks_verified_local_model_without_download(
    monkeypatch, tmp_path
):
    weights = b"pinned local 9B weights"
    spec = {
        "repo": "example/qwen35-9b",
        "revision": "d" * 40,
        "file": "Qwen3.5-9B-Q4_K_M.gguf",
        "bytes": len(weights),
        "sha256": hashlib.sha256(weights).hexdigest(),
    }
    script_root = tmp_path / "checkout"
    model_root = script_root / "models"
    source = model_root / "llama-bundle" / spec["file"]
    source.parent.mkdir(parents=True)
    source.write_bytes(weights)
    monkeypatch.chdir(script_root)
    monkeypatch.setattr(prepare_llama, "__file__", str(script_root / "scripts/prepare_llama.py"))
    monkeypatch.setattr(prepare_llama, "model_spec", lambda _size: spec)
    monkeypatch.setattr(
        prepare_llama, "download", lambda *_: pytest.fail("local preparation must not download")
    )

    target = model_root / "gguf-qwen35-9b" / spec["revision"][:8]
    output = script_root / "outputs/prepare-9b-cpu"
    prepare_llama.prepare_gguf_only(target, output, Path("models/llama-bundle") / spec["file"])

    prepared_model = target / spec["file"]
    assert prepared_model.stat().st_ino == source.stat().st_ino
    assert prepare_llama.verify_gguf_only(target)["model_sha256"] == spec["sha256"]
    assert json.loads((output / "preparation.json").read_text())["prepared"] is True


def test_9b_gguf_only_failed_download_publishes_no_partial_model(monkeypatch, tmp_path):
    weights = b"pinned 9B weights"
    spec = {
        "repo": "example/qwen35-9b",
        "revision": "b" * 40,
        "file": "Qwen3.5-9B-Q4_K_M.gguf",
        "bytes": len(weights),
        "sha256": hashlib.sha256(weights).hexdigest(),
    }
    monkeypatch.setattr(prepare_llama, "model_spec", lambda _size: spec)

    def fail_download(_url, path, *_args):
        path.write_bytes(b"partial")
        raise RuntimeError("download checksum differs from the fixed source")

    monkeypatch.setattr(prepare_llama, "download", fail_download)
    target = tmp_path / "models/gguf-qwen35-9b" / spec["revision"][:8]
    output = tmp_path / "outputs/prepare-9b-cpu"

    with pytest.raises(RuntimeError, match="checksum"):
        prepare_llama.prepare_gguf_only(target, output)

    assert not target.exists()
    assert not list(target.parent.glob(".gguf-only-*"))
    assert json.loads((output / "preparation.json").read_text())["prepared"] is False


def test_9b_gguf_only_cli_skips_cuda_compiler_arguments(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_llama.py",
            "--model-size",
            "9b",
            "--gguf-only",
            "--source-model",
            "models/source.gguf",
            "--output-dir",
            "outputs/prepare-9b-cpu",
        ],
    )
    spec = {**prepare_llama.model_spec("9b"), "revision": "c" * 40}
    monkeypatch.setattr(prepare_llama, "model_spec", lambda _size: spec)
    calls = []
    monkeypatch.setattr(prepare_llama, "prepare_gguf_only", lambda *args: calls.append(args))

    prepare_llama.main()

    assert calls == [
        (
            tmp_path / "models/gguf-qwen35-9b" / ("c" * 8),
            tmp_path / "outputs/prepare-9b-cpu",
            Path("models/source.gguf"),
        )
    ]


def test_gguf_only_cli_rejects_non_9b_before_creating_output(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_llama.py",
            "--model-size",
            "4b",
            "--gguf-only",
            "--output-dir",
            "outputs/prepare-cpu",
        ],
    )

    with pytest.raises(ValueError, match="pinned 9B"):
        prepare_llama.main()

    assert not (tmp_path / "outputs").exists()


def test_ollama_import_accepts_verified_model_only_gguf_under_models(monkeypatch, tmp_path):
    weights = b"pinned 9B weights"
    source = tmp_path / "models/model weights/Qwen3.5-9B-Q4_K_M.gguf"
    source.parent.mkdir(parents=True)
    source.write_bytes(weights)
    spec = {
        "bytes": len(weights),
        "sha256": hashlib.sha256(weights).hexdigest(),
    }
    monkeypatch.setattr(prepare_ollama, "ROOT", tmp_path)
    monkeypatch.setattr(prepare_ollama, "model_spec", lambda _size: spec)
    monkeypatch.chdir(tmp_path)

    resolved = prepare_ollama.bundle_gguf("9b", Path("models/model weights/Qwen3.5-9B-Q4_K_M.gguf"))
    assert resolved == source.resolve()
    assert prepare_ollama.modelfile_contents(resolved) == f'FROM "{resolved}"\n'


def test_ollama_import_rejects_model_only_gguf_outside_models(monkeypatch, tmp_path):
    weights = b"pinned 9B weights"
    source = tmp_path / "outside/Qwen3.5-9B-Q4_K_M.gguf"
    source.parent.mkdir()
    source.write_bytes(weights)
    spec = {
        "bytes": len(weights),
        "sha256": hashlib.sha256(weights).hexdigest(),
    }
    monkeypatch.setattr(prepare_ollama, "ROOT", tmp_path)
    monkeypatch.setattr(prepare_ollama, "model_spec", lambda _size: spec)

    with pytest.raises(SystemExit, match="inside this checkout's models"):
        prepare_ollama.bundle_gguf("9b", source)
