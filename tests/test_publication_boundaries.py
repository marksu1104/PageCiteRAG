from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_generated_paths_are_ignored() -> None:
    paths = [
        "database/index.sqlite",
        "custom.sqlite3.indexes/example/data.csc.index.npy",
        "outputs/report.json",
        "notebooks/example.ipynb",
        ".env",
        "models/example.safetensors",
    ]

    for path in paths:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", "--", path],
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        assert result.returncode == 0, f"expected generated path to be ignored: {path}"


def test_runtime_artifacts_are_not_tracked() -> None:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--",
            "database",
            "outputs",
            "models",
            "data",
            "artifacts",
            "indexes",
            "storage",
            ".env",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""


def test_build_configuration_has_explicit_package_allowlist() -> None:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as config_file:
        config = tomllib.load(config_file)

    targets = config["tool"]["hatch"]["build"]["targets"]
    assert targets["wheel"]["packages"] == ["src/page_cite_rag"]
    assert set(targets["sdist"]["only-include"]) == {
        "README.md",
        "pyproject.toml",
        "src/page_cite_rag",
    }
