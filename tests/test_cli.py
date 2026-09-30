from __future__ import annotations

import json
import shutil
import subprocess
import sys

import pytest

from doc_rag.cli import main


def test_help_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as help_exit:
        main(["--help"])
    assert help_exit.value.code == 0
    assert "doctor" in capsys.readouterr().out

    with pytest.raises(SystemExit) as version_exit:
        main(["--version"])
    assert version_exit.value.code == 0
    assert capsys.readouterr().out.startswith("doc-rag ")


def test_unknown_option_has_clear_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error_exit:
        main(["--unknown-option"])

    assert error_exit.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_doctor_json_is_allowlisted_and_omits_environment_secrets(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "doc-rag-test-secret-must-not-appear"
    monkeypatch.setenv("DOC_RAG_TEST_SECRET", secret)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)

    assert main(["doctor", "--json"]) == 0
    output = capsys.readouterr().out
    report = json.loads(output)

    assert set(report) == {"package", "python", "sqlite"}
    assert set(report["package"]) == {"name", "version"}
    assert set(report["python"]) == {"implementation", "version"}
    assert set(report["sqlite"]) == {"available", "version"}
    assert report["package"]["name"] == "doc-rag"
    assert isinstance(report["sqlite"]["available"], bool)
    assert secret not in output


def test_module_entry_point_works_outside_repository(tmp_path) -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-m", "doc_rag", "--help"],
        cwd=tmp_path,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0
    assert "usage: doc-rag" in result.stdout


def test_console_entry_point_exists_in_development_environment() -> None:
    assert shutil.which("doc-rag") is not None
