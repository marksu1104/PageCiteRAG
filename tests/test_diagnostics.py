from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from page_cite_rag import evaluation, qasper
from page_cite_rag.diagnostics import (
    ExistingIndexStore,
    audit_candidates,
    calibrate,
    load_comparison,
    write_diagnostic,
)
from page_cite_rag.evaluation import EvaluationError, run_qasper
from page_cite_rag.qasper import Annotation, Paper, Passage, Question
from page_cite_rag.retrieval import BM25Retriever, RetrievalError


class Encoder:
    identity = {"model": "synthetic", "version": 1}
    dimension = 2

    def __init__(self):
        self.document_calls = self.query_calls = 0

    def split_document(self, text):
        return [(0, len(text))]

    def encode_documents(self, texts):
        self.document_calls += 1
        return np.ones((len(texts), 2))

    def encode_queries(self, texts):
        self.query_calls += 1
        return np.ones((len(texts), 2))


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    papers = {}
    for split in ("train", "validation"):
        paragraphs = tuple(
            Passage(f"{split}/s0000/p{i:04d}", 0, i, f"Alpha sample number {i}.") for i in range(25)
        )
        questions = (
            Question(
                split + "-q1",
                "Alpha samples?",
                (
                    Annotation(False, (paragraphs[0].text,)),
                    Annotation(False, (paragraphs[-1].text,)),
                ),
            ),
            Question(split + "-q2", "Alpha samples?", (Annotation(False, (paragraphs[-1].text,)),)),
        )
        papers[split] = [Paper(split, paragraphs, questions)]
    monkeypatch.setattr(qasper, "read_snapshot", lambda data, split: iter(papers[split]))
    original = evaluation.version
    monkeypatch.setattr(
        evaluation, "version", lambda n: "synthetic" if n == "pyarrow" else original(n)
    )
    encoder = Encoder()
    source = tmp_path / "runs"
    database = tmp_path / "shared.sqlite3"
    for method in ("bm25", "dense", "hybrid"):
        run_qasper(
            tmp_path,
            source,
            method=method,
            encoder=encoder,
            dev_questions=2,
            validation_questions=2,
            index_database=database,
        )
    script = Path(__file__).resolve().parents[1] / "scripts/publish_qasper_comparison.py"
    spec = importlib.util.spec_from_file_location("publisher", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    published = tmp_path / "baseline"
    module.publish(source, published)
    return published, database, encoder


def test_calibration_uses_frozen_ranks_and_changes_no_baseline_files(
    tmp_path, monkeypatch, baseline
):
    source, _, encoder = baseline
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    from page_cite_rag.embedding import QwenEmbeddingEncoder

    monkeypatch.setattr(
        QwenEmbeddingEncoder, "__init__", lambda *a, **kw: pytest.fail("calibration loaded a model")
    )
    monkeypatch.setattr(
        qasper.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("diagnostics used network")
    )
    comparison = load_comparison(source, tmp_path)
    monkeypatch.setattr(
        BM25Retriever, "search", lambda *a, **kw: pytest.fail("calibration searched")
    )
    report, rows = calibrate(comparison)
    assert len(rows) == 12
    assert report["retrieval_queries_executed"] == 0
    for method in ("bm25", "dense", "hybrid"):
        result = report["results"][method]["validation"]
        assert result["groups"]["all"]["10"]["union_complete"] == 0
        assert result["groups"]["all"]["10"]["any_reference_complete"] == 1
        assert result["union_multiple_with_single_reference"] == 1
        assert result["groups"]["all_references_multiple"]["10"]["questions"] == 0
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / "calibration"
    write_diagnostic(output, report, rows)
    assert "Alpha sample" not in (output / "records.jsonl").read_text()
    assert "Alpha samples?" not in (output / "report.json").read_text()
    with pytest.raises(EvaluationError, match="exists"):
        write_diagnostic(output, report, rows)
    assert encoder.document_calls == 2


@pytest.mark.parametrize(
    "damage", ["gold", "extra", "duplicate", "score", "summary", "selection", "revision"]
)
def test_calibration_rejects_corrupt_or_unexpected_baseline(tmp_path, baseline, damage):
    source, _, _ = baseline
    path = source / "dense.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text())
    if damage == "gold":
        records[0]["gold_passage_ids"] = ["unknown"]
    elif damage == "extra":
        records[0]["question_text"] = "must not enter diagnostic exports"
    elif damage == "duplicate":
        records[-1] = records[0]
    elif damage == "score":
        records[0]["scores"][0] = float("nan")
    elif damage == "summary":
        summary["methods"]["dense"]["results"]["validation"]["metrics_at_k"]["10"][
            "macro_evidence_recall"
        ] = 999
    elif damage == "selection":
        summary["selection_sha256"] = "0" * 64
    else:
        summary["methods"]["bm25"]["configuration"]["revision"] = "not-pinned"
    path.write_text("\n".join(json.dumps(row) for row in records))
    summary_path.write_text(json.dumps(summary))
    with pytest.raises((EvaluationError, ValueError)):
        load_comparison(source, tmp_path)


def test_candidate_audit_is_train_only_readonly_and_reuses_model(tmp_path, baseline):
    source, database, encoder = baseline
    comparison = load_comparison(source, tmp_path)
    before = database.read_bytes()
    encoded = encoder.document_calls
    calls = encoder.query_calls
    report, rows = audit_candidates(comparison, ExistingIndexStore(database), encoder)
    assert len(rows) == 2
    assert all(row["split"] == "train" and row["paper_id"] == "train" for row in rows)
    assert report["branch_queries_executed"] == 4
    assert report["index_builds"] == 0
    assert report["papers_loaded"] == 1
    assert encoder.document_calls == encoded
    assert encoder.query_calls - calls == 2
    assert database.read_bytes() == before
    for method in ("bm25", "dense", "hybrid"):
        assert report["results"]["20"][method]["diagnoses"] == {
            "complete_reference": 1,
            "candidate_miss": 1,
        }
        assert report["results"]["50"][method]["diagnoses"] == {
            "complete_reference": 1,
            "selection_miss": 1,
        }
        assert report["results"]["50"][method]["whole_paper_pools"] == 2


def test_existing_store_refuses_creation_migration_and_writes(tmp_path, baseline):
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(EvaluationError, match="missing"):
        ExistingIndexStore(missing)
    assert not missing.exists()
    empty = tmp_path / "old.sqlite3"
    with sqlite3.connect(empty) as connection:
        connection.execute("PRAGMA user_version=1")
    with pytest.raises(EvaluationError, match="schema"):
        ExistingIndexStore(empty)
    _, database, _ = baseline
    store = ExistingIndexStore(database)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        store.save_retrieval_manifest("x", "bm25", "x")


def test_missing_index_stops_without_rebuilding(tmp_path, baseline):
    source, database, encoder = baseline
    comparison = load_comparison(source, tmp_path)
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM retrieval_indexes WHERE kind='dense'")
    with pytest.raises(RetrievalError, match="no dense index"):
        audit_candidates(comparison, ExistingIndexStore(database), encoder)


@pytest.mark.parametrize("drift", ["code", "encoder", "ranking", "package"])
def test_candidate_audit_stops_on_baseline_drift(tmp_path, baseline, drift):
    source, database, encoder = baseline
    comparison = load_comparison(source, tmp_path)
    if drift == "code":
        comparison.summaries["dense"]["configuration"]["code_sha256"]["dense.py"] = "0" * 64
    elif drift == "encoder":
        encoder.identity = {"model": "changed"}
    elif drift == "package":
        comparison.summaries["dense"]["configuration"]["versions"]["numpy"] = "changed"
    else:
        records = comparison.records["dense"]
        index = next(i for i, r in enumerate(records) if r.split == "train")
        records[index] = records[index].model_copy(update={"retrieved_passage_ids": ()})
    with pytest.raises(EvaluationError, match="differ|drifted"):
        audit_candidates(comparison, ExistingIndexStore(database), encoder)


def test_failed_queries_stay_in_reference_metric_denominator(tmp_path, baseline):
    source, _, _ = baseline
    comparison = load_comparison(source, tmp_path)
    records = comparison.records["dense"]
    index = next(i for i, record in enumerate(records) if record.split == "validation")
    records[index] = records[index].model_copy(
        update={
            "status": "error",
            "error_type": "RetrievalError",
            "retrieved_passage_ids": (),
            "scores": (),
        }
    )
    report, _ = calibrate(comparison)
    result = report["results"]["dense"]["validation"]
    assert result["errors"] == 1
    assert result["groups"]["all"]["10"]["questions"] == 2
    assert result["groups"]["all"]["10"]["any_reference_complete"] == 0
    assert result["groups"]["all"]["10"]["best_reference_recall"] == 0
