"""Offline, document-scoped QASPER retrieval evaluation with resumable results."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import sqlite3
import tempfile
import time
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, ValidationError

from doc_rag import qasper
from doc_rag.embedding import EmbeddingEncoder
from doc_rag.qasper import Alignment, Paper, Question
from doc_rag.retrieval import BM25Retriever, RetrievalError, build_bm25_index
from doc_rag.store import SQLiteDocumentStore, StoreError
from doc_rag.tokenize import tokenizer_version

_BENCHMARK_VERSION = "qasper-document-retrieval-v2"
_KS = (5, 10)


class EvaluationError(RuntimeError):
    """The selected evaluation cannot be reproduced or safely resumed."""


class QueryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    split: Literal["train", "validation"]
    paper_id: str
    question_id: str
    gold_passage_ids: tuple[str, ...]
    retrieved_passage_ids: tuple[str, ...] = ()
    scores: tuple[FiniteFloat, ...] = ()
    elapsed_ms: float = Field(ge=0, allow_inf_nan=False)
    status: Literal["ok", "error"]
    error_type: str | None = None


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    """Atomically replace only this run's generated result file."""
    with tempfile.TemporaryDirectory(prefix=".write-", dir=path.parent) as temporary:
        staging = Path(temporary) / "result.json"
        staging.write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        staging.replace(path)


def select_questions(
    papers: list[Paper], *, limit: int, seed: int
) -> tuple[list[tuple[Paper, Question, Alignment]], dict]:
    """Shuffle documents before scoring; all eligibility counts cover the full split."""
    if type(limit) is not int or limit < 1:
        raise ValueError("question limit must be a positive integer")
    eligible = defaultdict(list)
    excluded: Counter = Counter()
    issues: Counter = Counter()
    total = 0
    seen_papers: set[str] = set()
    seen_questions: set[str] = set()
    for paper in papers:
        if paper.paper_id in seen_papers:
            raise EvaluationError("duplicate paper ID in evaluation input")
        seen_papers.add(paper.paper_id)
        for question in paper.questions:
            if question.question_id in seen_questions:
                raise EvaluationError("duplicate question ID in evaluation input")
            seen_questions.add(question.question_id)
            total += 1
            aligned = qasper.align_question(paper, question)
            issues.update(aligned.annotation_issues)
            if aligned.exclusion_reason:
                excluded[aligned.exclusion_reason] += 1
            else:
                eligible[paper.paper_id].append((paper, question, aligned))
    eligible_count = sum(map(len, eligible.values()))
    if eligible_count < limit:
        raise EvaluationError(f"only {eligible_count} eligible questions; requested {limit}")
    order = sorted(eligible)
    random.Random(seed).shuffle(order)
    selected = []
    for paper_id in order:
        selected.extend(sorted(eligible[paper_id], key=lambda item: item[1].question_id))
        if len(selected) >= limit:
            selected = selected[:limit]
            break
    return selected, {
        "total_papers": len(papers),
        "total_questions": total,
        "eligible_questions": eligible_count,
        "excluded_questions": dict(sorted(excluded.items())),
        "annotation_issue_counts": dict(sorted(issues.items())),
        "selected_questions": len(selected),
        "selected_papers": len({item[0].paper_id for item in selected}),
        "eligible_not_selected": eligible_count - len(selected),
    }


def recall_at(retrieved: tuple[str, ...], gold: tuple[str, ...], k: int) -> float:
    if not gold:
        raise ValueError("evidence recall requires a nonempty gold set")
    return len(set(retrieved[:k]).intersection(gold)) / len(set(gold))


def _validate_record(
    record: QueryRecord,
    split: str,
    paper: Paper,
    question: Question,
    aligned: Alignment,
    method: str = "bm25",
) -> None:
    if (
        record.split != split
        or record.paper_id != paper.paper_id
        or record.question_id != question.question_id
        or record.gold_passage_ids != aligned.gold_passage_ids
        or len(record.retrieved_passage_ids) != len(record.scores)
        or len(record.retrieved_passage_ids) > max(_KS)
        or len(set(record.retrieved_passage_ids)) != len(record.retrieved_passage_ids)
        or (method != "dense" and any(score <= 0 for score in record.scores))
        or any(a < b for a, b in zip(record.scores, record.scores[1:]))
        or not set(record.retrieved_passage_ids).issubset(p.passage_id for p in paper.passages)
        or (record.status == "error" and (record.retrieved_passage_ids or not record.error_type))
        or (record.status == "ok" and record.error_type is not None)
    ):
        raise EvaluationError("cached query result does not match the frozen selection")


def _coverage_metrics(records: list[QueryRecord], k: int) -> dict:
    values = [recall_at(r.retrieved_passage_ids, r.gold_passage_ids, k) for r in records]
    return {
        "macro_evidence_recall": sum(values) / len(values) if values else None,
        "complete": sum(v == 1 for v in values),
        "partial": sum(0 < v < 1 for v in values),
        "none": sum(v == 0 for v in values),
    }


def _summarize(records: list[QueryRecord]) -> dict:
    summary = {}
    for split in ("train", "validation"):
        subset = [record for record in records if record.split == split]
        timings = [record.elapsed_ms for record in subset if record.status == "ok"]
        metrics = {}
        for k in _KS:
            recalls = [
                recall_at(record.retrieved_passage_ids, record.gold_passage_ids, k)
                for record in subset
            ]
            matched = sum(
                len(set(record.retrieved_passage_ids[:k]).intersection(record.gold_passage_ids))
                for record in subset
            )
            gold_count = sum(len(record.gold_passage_ids) for record in subset)
            metrics[str(k)] = {
                "macro_evidence_recall": sum(recalls) / len(subset),
                "micro_evidence_recall": matched / gold_count,
                "questions_with_any_evidence": sum(value > 0 for value in recalls),
                "questions_with_all_evidence": sum(value == 1 for value in recalls),
                "matched_evidence": matched,
                "gold_evidence": gold_count,
            }
        summary[split] = {
            "questions": len(subset),
            "papers": len({record.paper_id for record in subset}),
            "errors": sum(record.status == "error" for record in subset),
            "zero_results": sum(
                record.status == "ok" and not record.retrieved_passage_ids for record in subset
            ),
            "metrics_at_k": metrics,
            "query_p50_ms": float(np.percentile(timings, 50)) if timings else None,
            "query_p95_ms": float(np.percentile(timings, 95)) if timings else None,
            "evidence_groups": {
                group: {
                    "questions": len(group_records),
                    "metrics_at_k": {str(k): _coverage_metrics(group_records, k) for k in _KS},
                }
                for group, group_records in (
                    ("single", [r for r in subset if len(r.gold_passage_ids) == 1]),
                    ("multiple", [r for r in subset if len(r.gold_passage_ids) > 1]),
                    ("three_or_more", [r for r in subset if len(r.gold_passage_ids) >= 3]),
                )
            },
        }
    return summary


def _configuration(
    *,
    dev_questions: int,
    validation_questions: int,
    seed: int,
    method: str = "bm25",
    encoder: EmbeddingEncoder | None = None,
    candidates: int = 20,
    rrf_constant: int = 60,
) -> dict:
    module_files = (
        Path(__file__),
        Path(qasper.__file__),
        Path(__file__).with_name("retrieval.py"),
        Path(__file__).with_name("tokenize.py"),
        Path(__file__).with_name("store.py"),
        Path(__file__).with_name("models.py"),
        Path(__file__).with_name("embedding.py"),
        Path(__file__).with_name("dense.py"),
        Path(__file__).with_name("hybrid.py"),
    )
    return {
        "benchmark_version": _BENCHMARK_VERSION,
        "dataset": qasper.DATASET,
        "revision": qasper.REVISION,
        "license": "CC-BY-4.0",
        "source_files": qasper.SOURCES,
        "dev_questions": dev_questions,
        "validation_questions": validation_questions,
        "seed": seed,
        "python": platform.python_version(),
        "selection_policy": "seeded-sorted-document-order/then-sorted-question-id/v1",
        "alignment_policy": qasper.ALIGNMENT_VERSION,
        "metrics_policy": "question-macro-recall/union-of-complete-text-annotations/errors-zero/v1",
        "retriever": {
            "method": method,
            "lexical": {"method": "lucene", "k1": 1.5, "b": 0.75, "backend": "numpy"},
            "encoder": encoder.identity if encoder is not None else None,
            "top_k": 10,
            "candidates": candidates if method == "hybrid" else None,
            "rrf_constant": rrf_constant if method == "hybrid" else None,
        },
        "tokenizer": tokenizer_version(),
        "versions": {
            name: version(name) for name in ("bm25s", "jieba", "numpy", "scipy", "pyarrow")
        },
        "code_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in module_files
        },
        "scope": (
            "English structured full_text paragraphs, known-paper retrieval; "
            "no PDF parsing or answer generation."
        ),
    }


def run_qasper(
    data_dir: Path,
    output_dir: Path,
    *,
    dev_questions: int = 50,
    validation_questions: int = 200,
    seed: int = 42,
    resume: bool = False,
    retry_errors: bool = False,
    method: Literal["bm25", "dense", "hybrid"] = "bm25",
    encoder: EmbeddingEncoder | None = None,
    candidates: int = 20,
    rrf_constant: int = 60,
    model_load_ms: float = 0,
    index_database: Path | None = None,
) -> dict:
    """Prepare each selected document once and persist each question result atomically."""
    if retry_errors and not resume:
        raise EvaluationError("retry_errors requires resume")
    if method not in ("bm25", "dense", "hybrid") or (method != "bm25" and encoder is None):
        raise EvaluationError("dense/hybrid evaluation requires an explicitly loaded encoder")
    if type(candidates) is not int or candidates < max(_KS):
        raise EvaluationError("candidate budget must cover the evaluated top-k")
    if type(rrf_constant) is not int or rrf_constant < 1:
        raise EvaluationError("RRF constant must be a positive integer")
    papers = {
        split: list(qasper.read_snapshot(data_dir, split)) for split in ("train", "validation")
    }
    if {paper.paper_id for paper in papers["train"]}.intersection(
        paper.paper_id for paper in papers["validation"]
    ):
        raise EvaluationError("development and validation documents overlap")
    train_question_ids = {q.question_id for paper in papers["train"] for q in paper.questions}
    if train_question_ids.intersection(
        q.question_id for paper in papers["validation"] for q in paper.questions
    ):
        raise EvaluationError("development and validation question IDs overlap")
    selected = {}
    coverage = {}
    for split, count in (("train", dev_questions), ("validation", validation_questions)):
        selected[split], coverage[split] = select_questions(papers[split], limit=count, seed=seed)
    config = _configuration(
        dev_questions=dev_questions,
        validation_questions=validation_questions,
        seed=seed,
        method=method,
        encoder=encoder if method != "bm25" else None,
        candidates=candidates,
        rrf_constant=rrf_constant,
    )
    selection = [
        {
            "split": split,
            "paper_id": paper.paper_id,
            "question_id": question.question_id,
            "gold_passage_ids": list(aligned.gold_passage_ids),
        }
        for split, items in selected.items()
        for paper, question, aligned in items
    ]
    run_id = _json_hash({"configuration": config, "selection": selection})
    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / run_id
    if run_dir.exists() or run_dir.is_symlink():
        if not resume or run_dir.is_symlink():
            raise EvaluationError("run directory exists; use --resume or a new output directory")
        try:
            if (
                json.loads((run_dir / "configuration.json").read_text()) != config
                or json.loads((run_dir / "selection.json").read_text()) != selection
            ):
                raise EvaluationError("run metadata does not match this configuration")
        except (OSError, ValueError) as exc:
            raise EvaluationError("run metadata is damaged") from exc
    else:
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=output_dir) as temporary:
            staging = Path(temporary) / "run"
            staging.mkdir()
            _write_json(staging / "configuration.json", config)
            _write_json(staging / "selection.json", selection)
            (staging / "queries").mkdir()
            staging.rename(run_dir)

    records: list[QueryRecord] = []
    pending = defaultdict(list)
    resumed = 0
    for split, items in selected.items():
        for paper, question, aligned in items:
            path = (
                run_dir
                / "queries"
                / (_json_hash([split, paper.paper_id, question.question_id]) + ".json")
            )
            if path.exists():
                try:
                    record = QueryRecord.model_validate_json(path.read_text(encoding="utf-8"))
                    _validate_record(record, split, paper, question, aligned, method)
                except (OSError, ValidationError) as exc:
                    raise EvaluationError("saved query record is damaged") from exc
                if record.status == "ok" or not retry_errors:
                    records.append(record)
                    resumed += 1
                    continue
            pending[(split, paper.paper_id)].append((paper, question, aligned, path))

    builds = loads = executed = 0
    build_ms = load_ms = 0.0
    store = (
        SQLiteDocumentStore(index_database or run_dir / "documents.sqlite3") if pending else None
    )
    handled_errors = (RetrievalError, StoreError, OSError, sqlite3.Error, ValidationError)
    for (split, _), items in pending.items():
        preparation_error = None
        try:
            document_id, mapping = qasper.ingest_paper(store, items[0][0])
            branches = {}
            kinds = ("bm25", "dense") if method == "hybrid" else (method,)
            for kind in kinds:
                if store.get_retrieval_manifest(document_id, kind) is None:
                    start = time.perf_counter()
                    if kind == "bm25":
                        build_bm25_index(store, document_id)
                    else:
                        from doc_rag.dense import build_dense_index

                        build_dense_index(store, document_id, encoder)
                    build_ms += (time.perf_counter() - start) * 1000
                    builds += 1
                start = time.perf_counter()
                if kind == "bm25":
                    branches[kind] = BM25Retriever(store, document_id)
                else:
                    from doc_rag.dense import DenseRetriever

                    branches[kind] = DenseRetriever(store, document_id, encoder)
                load_ms += (time.perf_counter() - start) * 1000
                loads += 1
            if method == "hybrid":
                from doc_rag.hybrid import HybridRetriever

                retriever = HybridRetriever(
                    branches["bm25"],
                    branches["dense"],
                    candidates=candidates,
                    constant=rrf_constant,
                )
            else:
                retriever = branches[method]
        except handled_errors as exc:
            preparation_error = type(exc).__name__
        for paper, question, aligned, path in items:
            start = time.perf_counter()
            retrieved: tuple[str, ...] = ()
            scores: tuple[float, ...] = ()
            error = preparation_error
            if not error:
                try:
                    result = retriever.search(question.text, top_k=max(_KS))
                    retrieved = tuple(mapping[hit.block.block_id] for hit in result.hits)
                    scores = tuple(hit.score for hit in result.hits)
                except handled_errors as exc:
                    error = type(exc).__name__
            record = QueryRecord(
                split=split,
                paper_id=paper.paper_id,
                question_id=question.question_id,
                gold_passage_ids=aligned.gold_passage_ids,
                retrieved_passage_ids=retrieved,
                scores=scores,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                status="error" if error else "ok",
                error_type=error,
            )
            _validate_record(record, split, paper, question, aligned, method)
            _write_json(path, record.model_dump(mode="json"))
            records.append(record)
            executed += 1
    records.sort(key=lambda record: (record.split, record.paper_id, record.question_id))
    summary = {
        "run_id": run_id,
        "selection_sha256": _json_hash(selection),
        "platform": platform.platform(),
        "configuration": config,
        "coverage": coverage,
        "results": _summarize(records),
        "this_invocation": {
            "index_builds": builds,
            "index_loads": loads,
            "questions_executed": executed,
            "resumed_results": resumed,
            "index_build_ms_total": build_ms,
            "index_load_ms_total": load_ms,
            "model_load_ms": model_load_ms,
        },
    }
    with tempfile.TemporaryDirectory(prefix=".aggregate-", dir=run_dir) as temporary:
        staging = Path(temporary) / "results.jsonl"
        with staging.open("w", encoding="utf-8") as out:
            for record in records:
                row = record.model_dump(mode="json")
                row["recall_at_k"] = {
                    str(k): recall_at(record.retrieved_passage_ids, record.gold_passage_ids, k)
                    for k in _KS
                }
                out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        staging.replace(run_dir / "results.jsonl")
    _write_json(run_dir / "summary.json", summary)
    return {"run_directory": str(run_dir), **summary}
