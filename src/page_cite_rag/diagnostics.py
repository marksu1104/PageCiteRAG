"""Offline calibration of frozen ranks and train-only candidate diagnostics.

This module never changes retrievers, rebuilds an index, downloads a model, or
uses evidence labels to decide a production query. Original artifacts are read
only; each diagnostic publication needs a new destination directory.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from page_cite_rag import qasper
from page_cite_rag.embedding import EmbeddingEncoder
from page_cite_rag.evaluation import (
    EvaluationError,
    QueryRecord,
    _json_hash,
    _summarize,
    _validate_record,
    select_questions,
)
from page_cite_rag.evidence_metrics import (
    METRICS_VERSION,
    EvidenceReference,
    align_references,
    candidate_diagnosis,
    evidence_group,
    neighbor_opportunity,
    reference_metrics,
)
from page_cite_rag.hybrid import reciprocal_rank_fusion
from page_cite_rag.retrieval import BM25Retriever
from page_cite_rag.store import SQLiteDocumentStore
from page_cite_rag.tokenize import tokenizer_version

METHODS = ("bm25", "dense", "hybrid")
DEPTHS = (20, 50, 100)
DIAGNOSTICS_VERSION = "frozen-ranks/reference-calibration/train-candidate-audit/v1"


@dataclass
class FrozenComparison:
    summaries: dict
    records: dict[str, list[QueryRecord]]
    selected: dict[tuple[str, str], tuple[qasper.Paper, qasper.Question]]
    references: dict[tuple[str, str], tuple[EvidenceReference, ...]]
    provenance: dict


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_comparison(source: Path, data_dir: Path) -> FrozenComparison:
    """Validate ranks/selection/old scores and align references to the pinned source.

    Reconstruct the original seeded selection, not a new sample selected after
    observing scores. Questions and reference text never enter output artifacts.
    """
    artifact = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    if set(artifact) != {"selection_sha256", "methods"} or set(artifact["methods"]) != set(METHODS):
        raise EvaluationError("expected a complete frozen three-method comparison")
    summaries = artifact["methods"]
    config = summaries["bm25"]["configuration"]
    if (
        config["dataset"] != qasper.DATASET
        or config["revision"] != qasper.REVISION
        or config["source_files"] != qasper.SOURCES
        or config["alignment_policy"] != qasper.ALIGNMENT_VERSION
    ):
        raise EvaluationError("baseline data or alignment policy differs from the pinned source")
    papers = {
        split: list(qasper.read_snapshot(data_dir, split)) for split in ("train", "validation")
    }
    if {p.paper_id for p in papers["train"]} & {p.paper_id for p in papers["validation"]}:
        raise EvaluationError("development and validation papers overlap")
    if {q.question_id for p in papers["train"] for q in p.questions} & {
        q.question_id for p in papers["validation"] for q in p.questions
    }:
        raise EvaluationError("development and validation questions overlap")
    selected = {}
    references = {}
    alignments = {}
    selection = []
    for split, count_key in (("train", "dev_questions"), ("validation", "validation_questions")):
        choices, _ = select_questions(papers[split], limit=config[count_key], seed=config["seed"])
        for paper, question, aligned in choices:
            key = (split, question.question_id)
            if key in selected:
                raise EvaluationError("duplicate selected question")
            refs = align_references(paper, question)
            union = set().union(*(set(reference.passage_ids) for reference in refs))
            if not refs or union != set(aligned.gold_passage_ids):
                raise EvaluationError("per-reference alignment changed the original gold union")
            selected[key] = (paper, question)
            references[key] = refs
            alignments[key] = aligned
            selection.append(
                {
                    "split": split,
                    "paper_id": paper.paper_id,
                    "question_id": question.question_id,
                    "gold_passage_ids": list(aligned.gold_passage_ids),
                }
            )
    selection_sha = _json_hash(selection)
    if selection_sha != artifact["selection_sha256"]:
        raise EvaluationError("original seeded selection checksum changed")
    records = {}
    hashes = {"summary.json": _sha(source / "summary.json")}
    same_config = (
        "dataset",
        "revision",
        "source_files",
        "alignment_policy",
        "seed",
        "dev_questions",
        "validation_questions",
        "selection_policy",
        "metrics_policy",
    )
    for method in METHODS:
        summary = summaries[method]
        current = summary["configuration"]
        if (
            any(current[key] != config[key] for key in same_config)
            or current["retriever"]["method"] != method
            or summary["selection_sha256"] != selection_sha
            or _json_hash({"configuration": current, "selection": selection}) != summary["run_id"]
        ):
            raise EvaluationError("baseline methods disagree about provenance or selection")
        path = source / f"{method}.jsonl"
        hashes[path.name] = _sha(path)
        rows = []
        seen = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            record = QueryRecord.model_validate_json(line)
            key = (record.split, record.question_id)
            if key in seen or key not in selected:
                raise EvaluationError("duplicate or unexpected baseline query")
            paper, question = selected[key]
            _validate_record(record, record.split, paper, question, alignments[key], method)
            rows.append(record)
            seen.add(key)
        if seen != set(selected) or _summarize(rows) != summary["results"]:
            raise EvaluationError("baseline coverage or original aggregate metrics changed")
        records[method] = sorted(rows, key=lambda row: (row.split, row.paper_id, row.question_id))
    return FrozenComparison(
        summaries,
        records,
        selected,
        references,
        {
            "dataset": qasper.DATASET,
            "revision": qasper.REVISION,
            "source_files": qasper.SOURCES,
            "selection_sha256": selection_sha,
            "baseline_run_ids": {method: summaries[method]["run_id"] for method in METHODS},
            "baseline_file_sha256": hashes,
            "diagnostics_version": DIAGNOSTICS_VERSION,
            "metrics_version": METRICS_VERSION,
            "diagnostic_code_sha256": {
                path.name: _sha(path)
                for path in (Path(__file__), Path(__file__).with_name("evidence_metrics.py"))
            },
        },
    )


def _metric_summary(rows: list[dict], k: int) -> dict:
    scores = [row["metrics_at_k"][str(k)] for row in rows]
    averages = ("union_recall", "best_reference_recall", "best_reference_paragraph_f1")
    return {
        "questions": len(rows),
        **{
            name: sum(score[name] for score in scores) / len(scores) if scores else None
            for name in averages
        },
        "union_complete": sum(score["union_complete"] for score in scores),
        "any_reference_complete": sum(score["any_reference_complete"] for score in scores),
    }


def calibrate(comparison: FrozenComparison) -> tuple[dict, list[dict]]:
    """Rescore existing top-10 records; no retrieval or model is needed."""
    rows = []
    results = {}
    for method in METHODS:
        for record in comparison.records[method]:
            key = (record.split, record.question_id)
            refs = comparison.references[key]
            rows.append(
                {
                    "method": method,
                    "split": record.split,
                    "paper_id": record.paper_id,
                    "question_id": record.question_id,
                    "status": record.status,
                    "references": [
                        {"annotation_index": ref.annotation_index, "passage_ids": ref.passage_ids}
                        for ref in refs
                    ],
                    "evidence_group": evidence_group(refs),
                    "metrics_at_k": {
                        str(k): reference_metrics(record.retrieved_passage_ids, refs, k=k)
                        for k in (5, 10)
                    },
                }
            )
        results[method] = {}
        for split in ("train", "validation"):
            subset = [row for row in rows if row["method"] == method and row["split"] == split]
            groups = {
                "all": subset,
                "union_single": [
                    r for r in subset if r["evidence_group"]["union_evidence_count"] == 1
                ],
                "union_multiple": [
                    r for r in subset if r["evidence_group"]["union_evidence_count"] > 1
                ],
                "union_three_or_more": [
                    r for r in subset if r["evidence_group"]["union_evidence_count"] >= 3
                ],
                "all_references_multiple": [
                    r for r in subset if r["evidence_group"]["all_references_multiple"]
                ],
            }
            results[method][split] = {
                "errors": sum(row["status"] == "error" for row in subset),
                "alternative_evidence_questions": sum(
                    row["evidence_group"]["distinct_evidence_sets"] > 1 for row in subset
                ),
                "union_multiple_with_single_reference": sum(
                    row["evidence_group"]["union_multiple_with_single_reference"] for row in subset
                ),
                "groups": {
                    name: {str(k): _metric_summary(group, k) for k in (5, 10)}
                    for name, group in groups.items()
                },
            }
    return {
        "provenance": comparison.provenance,
        "scope": "Retrospective text-only known-paper ranks; not official QASPER Evidence-F1.",
        "retrieval_queries_executed": 0,
        "results": results,
    }, rows


class ExistingIndexStore(SQLiteDocumentStore):
    """Diagnostic-only read adapter: never initialize/migrate/write the source store."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.expanduser().resolve()
        if database_path.is_symlink() or not self.database_path.is_file():
            raise EvaluationError("existing baseline database is missing or a symlink")
        with self._read_connection() as connection:
            if connection.execute("PRAGMA user_version").fetchone()[0] != 2:
                raise EvaluationError("diagnostics require the existing schema-v2 baseline store")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path.as_uri() + "?mode=ro", uri=True, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection


def _existing_paper(store: ExistingIndexStore, paper: qasper.Paper) -> tuple[str, dict[str, str]]:
    with store._read_connection() as connection:
        rows = connection.execute(
            "SELECT document_id FROM documents WHERE source_name=? AND parser_name='qasper' "
            "AND parser_version=?",
            (paper.paper_id + ".qasper", qasper.PARSER_VERSION),
        ).fetchall()
    if len(rows) != 1:
        raise EvaluationError("paper needs one existing baseline source; not rebuilding")
    document_id = rows[0][0]
    blocks = store.get_blocks(document_id)
    if len(blocks) != len(paper.passages):
        raise EvaluationError("stored paragraph count differs from the pinned paper")
    mapping = {}
    for block, passage in zip(blocks, paper.passages, strict=True):
        expected = hashlib.sha256(f"{document_id}/{passage.passage_id}".encode()).hexdigest()
        if block.block_id != expected or block.text != passage.text:
            raise EvaluationError("stored paragraph identity/text differs from pinned source")
        mapping[block.block_id] = passage.passage_id
    return document_id, mapping


def audit_candidates(
    comparison: FrozenComparison, store: ExistingIndexStore, encoder: EmbeddingEncoder
) -> tuple[dict, list[dict]]:
    """Run only the original train selection, reusing indexes and one encoder.

    Hybrid's actual candidate pool is BM25@20 union dense@20. Larger branch
    unions are labeled diagnostic opportunities, NOT reconfigured hybrid runs.
    Ranking/config drift and missing indexes stop the audit rather than silently
    replacing a frozen baseline. No best-performing parameters are selected.
    """
    from page_cite_rag.dense import DenseRetriever

    for method in METHODS:
        config = comparison.summaries[method]["configuration"]
        if config["tokenizer"] != tokenizer_version():
            raise EvaluationError("tokenizer differs from frozen baseline")
        if any(
            version(name) != config["versions"][name]
            for name in ("bm25s", "jieba", "numpy", "scipy")
        ):
            raise EvaluationError("retrieval package versions differ from frozen baseline")
        for name, expected in config["code_sha256"].items():
            if _sha(Path(__file__).with_name(name)) != expected:
                raise EvaluationError("retrieval code differs from frozen baseline")
        if method != "bm25" and config["retriever"]["encoder"] != encoder.identity:
            raise EvaluationError("encoder differs from frozen baseline")
    hybrid_config = comparison.summaries["hybrid"]["configuration"]["retriever"]
    if hybrid_config["candidates"] != 20 or hybrid_config["top_k"] != 10:
        raise EvaluationError("candidate audit expects the frozen 20-candidate/10-final baseline")
    expected_rows = {
        method: {r.question_id: r for r in comparison.records[method] if r.split == "train"}
        for method in METHODS
    }
    by_paper = {}
    for (split, _), (paper, question) in comparison.selected.items():
        if split == "train":
            by_paper.setdefault(paper.paper_id, (paper, []))[1].append(question)
    output = []
    database_sha = _sha(store.database_path)
    index_sources = {}
    for paper, questions in by_paper.values():
        document_id, mapping = _existing_paper(store, paper)
        lexical = BM25Retriever(store, document_id)
        dense = DenseRetriever(store, document_id, encoder)
        index_sources[paper.paper_id] = {
            name: {
                "index_id": retriever.manifest.index_id,
                "manifest_sha256": _json_hash(retriever.manifest.model_dump(mode="json")),
            }
            for name, retriever in (("bm25", lexical), ("dense", dense))
        }
        for question in questions:
            print(f"Auditing train {paper.paper_id}/{question.question_id}", flush=True)
            branches = {
                "bm25": lexical.search(question.text, top_k=max(DEPTHS)),
                "dense": dense.search(question.text, top_k=max(DEPTHS)),
            }
            short = [
                result.model_copy(update={"hits": result.hits[:20]}) for result in branches.values()
            ]
            fused = reciprocal_rank_fusion(short, top_k=10, constant=hybrid_config["rrf_constant"])
            final = {
                **{
                    name: tuple(mapping[h.block.block_id] for h in r.hits[:10])
                    for name, r in branches.items()
                },
                "hybrid": tuple(mapping[h.block.block_id] for h in fused),
            }
            for method in METHODS:
                expected = expected_rows[method][question.question_id]
                if expected.status != "ok" or final[method] != expected.retrieved_passage_ids:
                    raise EvaluationError("development top-10 drifted from frozen rankings")
            refs = comparison.references[("train", question.question_id)]
            depths = {}
            for depth in DEPTHS:
                pools = {
                    name: tuple(mapping[h.block.block_id] for h in result.hits[:depth])
                    for name, result in branches.items()
                }
                pools["hybrid"] = tuple(sorted(set(pools["bm25"]) | set(pools["dense"])))
                depths[str(depth)] = {
                    name: {
                        "candidate_count": len(pool),
                        "covers_whole_paper": len(pool) == len(paper.passages),
                        "best_reference_recall": reference_metrics(pool, refs, k=len(pool) or 1)[
                            "best_reference_recall"
                        ],
                        "any_reference_complete": any(
                            set(r.passage_ids) <= set(pool) for r in refs
                        ),
                        "diagnosis": candidate_diagnosis(final[name], pool, refs),
                    }
                    for name, pool in pools.items()
                }
            output.append(
                {
                    "split": "train",
                    "paper_id": paper.paper_id,
                    "question_id": question.question_id,
                    "paper_paragraph_count": len(paper.passages),
                    "evidence_group": evidence_group(refs),
                    "branch_rankings": {
                        name: [mapping[h.block.block_id] for h in result.hits]
                        for name, result in branches.items()
                    },
                    "final_rankings": final,
                    "candidate_depths": depths,
                    "neighbor_opportunities": {
                        name: neighbor_opportunity(paper, ids, refs) for name, ids in final.items()
                    },
                }
            )
    if _sha(store.database_path) != database_sha:
        raise EvaluationError("baseline database changed during the read-only audit")
    results = {}
    for depth in DEPTHS:
        results[str(depth)] = {}
        for method in METHODS:
            entries = [row["candidate_depths"][str(depth)][method] for row in output]
            results[str(depth)][method] = {
                "questions": len(entries),
                "candidate_complete": sum(entry["any_reference_complete"] for entry in entries),
                "macro_best_reference_recall": sum(e["best_reference_recall"] for e in entries)
                / len(entries),
                "whole_paper_pools": sum(entry["covers_whole_paper"] for entry in entries),
                "mean_candidate_count": sum(e["candidate_count"] for e in entries) / len(entries),
                "diagnoses": dict(sorted(Counter(e["diagnosis"] for e in entries).items())),
            }
    return {
        "provenance": comparison.provenance,
        "database_sha256_before_and_after": database_sha,
        "index_sources": index_sources,
        "scope": "Original train subset only; gold-aware diagnostics, not live routing.",
        "index_builds": 0,
        "papers_loaded": len(by_paper),
        "branch_queries_executed": len(output) * 2,
        "candidate_depths": DEPTHS,
        "hybrid_candidate_note": (
            "depth=20 is actual pre-fusion pool; 50/100 are larger-pool opportunities only"
        ),
        "neighbor_note": (
            "same-section ±1 uses more context; "
            "not an equal-token comparison or a deployed improvement"
        ),
        "results": results,
        "neighbor_opportunities": {
            method: {
                "complete_with_neighbors": sum(
                    r["neighbor_opportunities"][method]["any_reference_complete_with_neighbors"]
                    for r in output
                ),
                "mean_expanded_paragraph_count": sum(
                    r["neighbor_opportunities"][method]["expanded_paragraph_count"] for r in output
                )
                / len(output),
            }
            for method in METHODS
        },
    }, output


def write_diagnostic(destination: Path, report: dict, rows: list[dict]) -> None:
    """Exclusive, atomic output; old runs and source text are never copied."""
    if destination.exists() or destination.is_symlink():
        raise EvaluationError("diagnostic destination exists; choose a new directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".diagnostic-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        (staging / "report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
        (staging / "records.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
        staging.rename(destination)
