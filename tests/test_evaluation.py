from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_rag import evaluation, qasper
from doc_rag.cli import main
from doc_rag.evaluation import EvaluationError, recall_at, run_qasper, select_questions
from doc_rag.qasper import Annotation, Paper, Passage, Question
from doc_rag.retrieval import BM25Retriever, RetrievalError


def paper(paper_id, *, questions=3):
    passages = (
        Passage(paper_id + "/a", 0, 0, "Alpha samples."),
        Passage(paper_id + "/b", 0, 1, "Beta controls."),
    )
    return Paper(
        paper_id,
        passages,
        tuple(
            Question(paper_id + f"-q{i}", "Alpha?", (Annotation(False, ("Alpha samples.",)),))
            for i in range(questions)
        ),
    )


@pytest.fixture
def synthetic_snapshot(monkeypatch):
    papers = {
        "train": [paper("dev-a"), paper("dev-b")],
        "validation": [paper("val-a"), paper("val-b")],
    }
    monkeypatch.setattr(qasper, "read_snapshot", lambda data, split: iter(papers[split]))
    # Synthetic runner unit tests do not require installing optional Parquet support.
    original_version = evaluation.version
    monkeypatch.setattr(
        evaluation,
        "version",
        lambda name: "synthetic-reader" if name == "pyarrow" else original_version(name),
    )
    return papers


def test_document_seeded_selection_is_deterministic_and_reports_all_exclusions():
    papers = [paper("p" + str(i)) for i in range(8)]
    papers.append(Paper("excluded", (), (Question("unanswerable", "q", (Annotation(True, ()),)),)))
    selected, counts = select_questions(papers, limit=7, seed=42)
    reordered, _ = select_questions(list(reversed(papers)), limit=7, seed=42)
    assert [item[1].question_id for item in selected] == [item[1].question_id for item in reordered]
    assert len(selected) == 7
    assert counts["eligible_questions"] == 24
    assert counts["total_questions"] == 25
    assert sum(counts["excluded_questions"].values()) == 1
    with pytest.raises(EvaluationError, match="only"):
        select_questions(papers, limit=100, seed=42)


def test_metric_is_fraction_of_distinct_evidence_and_duplicates_do_not_inflate():
    assert recall_at(("a", "a", "c"), ("a", "b"), 5) == 0.5
    assert recall_at(("z", "b", "a"), ("a", "b"), 1) == 0
    assert recall_at(("z", "b", "a"), ("a", "b"), 3) == 1
    with pytest.raises(ValueError):
        recall_at((), (), 5)


def test_offline_runner_outputs_ids_only_and_reuses_completed_records(
    tmp_path, monkeypatch, synthetic_snapshot
):
    monkeypatch.setattr(
        qasper.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("evaluation attempted a download"),
    )
    result = run_qasper(tmp_path, tmp_path / "out", dev_questions=4, validation_questions=4)
    for split in ("train", "validation"):
        assert result["results"][split]["metrics_at_k"]["5"]["macro_evidence_recall"] == 1
        assert result["results"][split]["questions"] == 4
    assert result["this_invocation"]["index_builds"] == 4
    assert result["this_invocation"]["index_loads"] == 4
    out = Path(result["run_directory"])
    rows = [json.loads(line) for line in (out / "results.jsonl").read_text().splitlines()]
    assert len(rows) == 8
    assert all("question" not in row and "text" not in row for row in rows)
    assert len(list((out / "queries").glob("*.json"))) == 8

    def forbidden(*args, **kwargs):
        raise AssertionError("completed run unexpectedly executed a query")

    monkeypatch.setattr(BM25Retriever, "search", forbidden)
    resumed = run_qasper(
        tmp_path, tmp_path / "out", dev_questions=4, validation_questions=4, resume=True
    )
    assert resumed["results"] == result["results"]
    assert resumed["this_invocation"]["questions_executed"] == 0
    assert resumed["this_invocation"]["index_loads"] == 0
    assert resumed["this_invocation"]["resumed_results"] == 8


def test_partial_interruption_resumes_missing_queries(tmp_path, monkeypatch, synthetic_snapshot):
    original_search = BM25Retriever.search
    calls = []

    def interrupt(self, query, **kwargs):
        calls.append(query)
        if len(calls) == 2:
            raise KeyboardInterrupt("synthetic interruption")
        return original_search(self, query, **kwargs)

    monkeypatch.setattr(BM25Retriever, "search", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_qasper(tmp_path, tmp_path / "out", dev_questions=3, validation_questions=3)
    assert len(list((tmp_path / "out").glob("*/queries/*.json"))) == 1
    monkeypatch.setattr(BM25Retriever, "search", original_search)
    resumed = run_qasper(
        tmp_path, tmp_path / "out", dev_questions=3, validation_questions=3, resume=True
    )
    assert resumed["this_invocation"]["resumed_results"] == 1
    assert resumed["this_invocation"]["questions_executed"] == 5


def test_query_failures_remain_in_denominator_and_can_be_retried(
    tmp_path, monkeypatch, synthetic_snapshot
):
    original_search = BM25Retriever.search
    monkeypatch.setattr(
        BM25Retriever,
        "search",
        lambda *args, **kwargs: (_ for _ in ()).throw(RetrievalError("synthetic failure")),
    )
    result = run_qasper(tmp_path, tmp_path / "out", dev_questions=2, validation_questions=2)
    assert result["results"]["validation"]["errors"] == 2
    assert result["results"]["validation"]["metrics_at_k"]["5"]["macro_evidence_recall"] == 0
    monkeypatch.setattr(BM25Retriever, "search", original_search)
    retried = run_qasper(
        tmp_path,
        tmp_path / "out",
        dev_questions=2,
        validation_questions=2,
        resume=True,
        retry_errors=True,
    )
    assert retried["results"]["validation"]["errors"] == 0
    assert retried["this_invocation"]["questions_executed"] == 4


def test_cannot_overwrite_existing_run_or_reuse_foreign_results(tmp_path, synthetic_snapshot):
    result = run_qasper(tmp_path, tmp_path / "out", dev_questions=2, validation_questions=2)
    with pytest.raises(EvaluationError, match="exists"):
        run_qasper(tmp_path, tmp_path / "out", dev_questions=2, validation_questions=2)
    query_path = next((Path(result["run_directory"]) / "queries").glob("*.json"))
    record = json.loads(query_path.read_text())
    record["retrieved_passage_ids"] = ["foreign-paper/paragraph"]
    record["scores"] = [1]
    query_path.write_text(json.dumps(record))
    with pytest.raises(EvaluationError, match="frozen selection"):
        run_qasper(tmp_path, tmp_path / "out", dev_questions=2, validation_questions=2, resume=True)


def test_overlapping_splits_and_duplicate_questions_are_rejected(tmp_path, synthetic_snapshot):
    synthetic_snapshot["validation"].append(synthetic_snapshot["train"][0])
    with pytest.raises(EvaluationError, match="overlap"):
        run_qasper(tmp_path, tmp_path / "out", dev_questions=2, validation_questions=2)
    with pytest.raises(EvaluationError, match="duplicate paper"):
        select_questions([paper("same"), paper("same")], limit=1, seed=42)
    with pytest.raises(EvaluationError, match="requires resume"):
        run_qasper(tmp_path, tmp_path / "out", retry_errors=True)


def test_cli_evaluation_reports_results_and_missing_data(tmp_path, capsys, synthetic_snapshot):
    assert (
        main(
            [
                "eval",
                "qasper",
                "--data-dir",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--dev-questions",
                "2",
                "--validation-questions",
                "2",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["results"]["validation"]["questions"] == 2
    assert (
        main(
            [
                "eval",
                "qasper",
                "--data-dir",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "out"),
                "--dev-questions",
                "2",
                "--validation-questions",
                "2",
            ]
        )
        == 1
    )
    assert "exists" in capsys.readouterr().err
