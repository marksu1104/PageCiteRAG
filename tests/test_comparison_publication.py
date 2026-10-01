from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from doc_rag import evaluation, qasper
from doc_rag.evaluation import run_qasper
from doc_rag.qasper import Annotation, Paper, Passage, Question


def publisher():
    path = Path(__file__).resolve().parents[1] / "scripts/publish_qasper_comparison.py"
    spec = importlib.util.spec_from_file_location("publication", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.publish


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    def snapshot(data, split):
        return iter(
            [
                Paper(
                    split,
                    (Passage(split + "/a", 0, 0, "Synthetic alpha."),),
                    (
                        Question(
                            split + "-q", "alpha?", (Annotation(False, ("Synthetic alpha.",)),)
                        ),
                    ),
                )
            ]
        )

    monkeypatch.setattr(qasper, "read_snapshot", snapshot)
    original = evaluation.version
    monkeypatch.setattr(
        evaluation, "version", lambda name: "synthetic" if name == "pyarrow" else original(name)
    )

    class Encoder:
        dimension = 2
        identity = {"model": "synthetic"}

        def split_document(self, text):
            return [(0, len(text))]

        def encode_documents(self, texts):
            return np.ones((len(texts), 2))

        encode_queries = encode_documents

    source = tmp_path / "runs"
    for method in ("bm25", "dense", "hybrid"):
        run_qasper(
            tmp_path,
            source,
            method=method,
            encoder=Encoder(),
            dev_questions=1,
            validation_questions=1,
            index_database=tmp_path / "shared.sqlite3",
        )
    return source


def test_publication_recomputes_metrics_and_exports_only_query_record_fields(tmp_path, comparison):
    destination = tmp_path / "public"
    publisher()(comparison, destination)
    assert {p.name for p in destination.iterdir()} == {
        "summary.json",
        "bm25.jsonl",
        "dense.jsonl",
        "hybrid.jsonl",
    }
    for method in ("bm25", "dense", "hybrid"):
        rows = [
            json.loads(line)
            for line in (destination / (method + ".jsonl")).read_text().splitlines()
        ]
        assert len(rows) == 2
        assert all(set(row) == set(evaluation.QueryRecord.model_fields) for row in rows)
        assert "Synthetic alpha." not in (destination / (method + ".jsonl")).read_text()
    with pytest.raises(ValueError, match="overwrite"):
        publisher()(comparison, destination)


@pytest.mark.parametrize("damage", ["text", "metric", "scope"])
def test_publication_refuses_unexpected_text_or_inconsistent_results(tmp_path, comparison, damage):
    directory = next(p for p in comparison.iterdir() if p.is_dir() and len(p.name) == 64)
    if damage == "metric":
        path = directory / "summary.json"
        summary = json.loads(path.read_text())
        summary["results"]["validation"]["metrics_at_k"]["5"]["macro_evidence_recall"] = 0.1
        path.write_text(json.dumps(summary))
    else:
        path = directory / "results.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if damage == "text":
            rows[0]["question_text"] = "must not be published"
        else:
            rows[0]["retrieved_passage_ids"] = ["other-paper/a"]
        path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError):
        publisher()(comparison, tmp_path / "public")
    assert not (tmp_path / "public").exists()
