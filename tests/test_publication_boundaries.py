from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_generated_and_reserved_paths_are_excluded() -> None:
    paths = ["database/x.db", "outputs/x.json", "data/x", ".env", "models/x"]

    for path in paths:
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", "--", path],
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        assert result.returncode == 0, f"expected reserved path to be excluded: {path}"


def test_nonrelease_paths_are_not_tracked() -> None:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--",
            "AGENTS.md",
            "docs",
            "reference",
            "dataset",
            "database",
            "legacy",
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
