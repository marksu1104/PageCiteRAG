"""Calibrate frozen QASPER ranks, or audit train candidates using existing local indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from page_cite_rag.diagnostics import (
    ExistingIndexStore,
    audit_candidates,
    calibrate,
    load_comparison,
    write_diagnostic,
)
from page_cite_rag.embedding import DEFAULT_MODEL_DIR
from page_cite_rag.evaluation import EvaluationError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("calibrate", "candidates"))
    parser.add_argument("--baseline", type=Path, default=Path("benchmarks/qasper-retrieval"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/qasper"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--index-db", type=Path, default=Path("outputs/qasper-comparison/indexes.sqlite3")
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise EvaluationError("output exists; refusing to overwrite a completed diagnostic")
    start = time.perf_counter()
    comparison = load_comparison(args.baseline, args.data_dir)
    preparation_ms = (time.perf_counter() - start) * 1000
    model_load_ms = 0.0
    if args.mode == "calibrate":
        start = time.perf_counter()
        report, rows = calibrate(comparison)
    else:
        from page_cite_rag.embedding import QwenEmbeddingEncoder

        store = ExistingIndexStore(args.index_db)
        start = time.perf_counter()
        encoder = QwenEmbeddingEncoder(args.model_dir)
        model_load_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        report, rows = audit_candidates(comparison, store, encoder)
    report["timing_ms"] = {
        "baseline_and_source_preparation": preparation_ms,
        "model_load": model_load_ms,
        "diagnostic_execution": (time.perf_counter() - start) * 1000,
    }
    report["provenance"]["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_diagnostic(args.output_dir, report, rows)
    print(
        json.dumps({"mode": args.mode, "output": str(args.output_dir), "records": len(rows)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
