from __future__ import annotations

import subprocess
import sys


def test_package_and_cli_do_not_import_legacy_or_heavy_modules() -> None:
    script = """
import importlib.abc
import sys

blocked = {
    "Utils", "llm", "torch", "transformers", "llama_index", "paddleocr",
    "ckip_transformers", "chromadb",
}

class ImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in blocked:
            raise AssertionError(f"forbidden import attempted: {fullname}")
        return None

sys.meta_path.insert(0, ImportBlocker())
import doc_rag
from doc_rag.cli import main

assert main(["doctor", "--json"]) == 0
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"sqlite"' in result.stdout
