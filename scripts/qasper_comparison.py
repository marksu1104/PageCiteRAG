"""Run fixed BM25/dense/RRF comparisons with one model and shared document indexes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from doc_rag.embedding import DEFAULT_MODEL_DIR, QwenEmbeddingEncoder
from doc_rag.evaluation import run_qasper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/qasper"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qasper-comparison"))
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dev-questions", type=int, default=50)
    parser.add_argument("--validation-questions", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    encoder = QwenEmbeddingEncoder(args.model_dir)
    model_load_ms = (time.perf_counter() - start) * 1000
    selection_sha = None
    errors = 0
    for method in ("bm25", "dense", "hybrid"):
        print(f"Starting {method}", flush=True)
        result = run_qasper(
            args.data_dir,
            args.output_dir,
            dev_questions=args.dev_questions,
            validation_questions=args.validation_questions,
            method=method,
            encoder=encoder if method != "bm25" else None,
            model_load_ms=model_load_ms if method == "dense" else 0,
            index_database=args.output_dir / "indexes.sqlite3",
            resume=args.resume,
        )
        if selection_sha is not None and selection_sha != result["selection_sha256"]:
            raise AssertionError("comparison selection changed between methods")
        selection_sha = result["selection_sha256"]
        errors += sum(split["errors"] for split in result["results"].values())
        print(
            json.dumps(
                {
                    "method": method,
                    "run_directory": result["run_directory"],
                    "selection_sha256": selection_sha,
                    "results": result["results"],
                    "this_invocation": result["this_invocation"],
                }
            ),
            flush=True,
        )
    if errors:
        raise SystemExit(f"Comparison retained {errors} errors; inspect before publication")


if __name__ == "__main__":
    main()
