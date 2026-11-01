import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PREPARE_PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_llama.py"
_PREPARE_SPEC = importlib.util.spec_from_file_location("prepare_llama", _PREPARE_PATH)
prepare_llama = importlib.util.module_from_spec(_PREPARE_SPEC)
_PREPARE_SPEC.loader.exec_module(prepare_llama)


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
