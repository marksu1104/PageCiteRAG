from __future__ import annotations

import subprocess
import sys


def test_package_and_cli_do_not_import_legacy_or_heavy_modules() -> None:
    script = """
import importlib.abc
import sys

blocked = {
    "Utils", "llm", "torch", "transformers", "llama_index", "paddleocr",
    "ckip_transformers", "chromadb", "bm25s", "jieba", "numpy", "scipy", "pyarrow",
    "sentence_transformers", "huggingface_hub",
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


def test_dense_and_hybrid_modules_do_not_initialize_model_libraries():
    script = """
import importlib.abc
import sys
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.', 1)[0] in {'torch', 'transformers', 'sentence_transformers'}:
            raise AssertionError('model dependency imported in core')
sys.meta_path.insert(0, Blocker())
import doc_rag.embedding
import doc_rag.dense
import doc_rag.hybrid
import doc_rag.evaluation
"""
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
