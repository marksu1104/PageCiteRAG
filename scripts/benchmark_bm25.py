"""Reproducible resource-reuse smoke benchmark using generated English paragraphs.

Run with: uv run --locked python scripts/benchmark_bm25.py
This is a latency smoke test, not an evidence-recall or bilingual benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np

from doc_rag.ingest import prepare_source
from doc_rag.retrieval import BM25Retriever, build_bm25_index, index_root
from doc_rag.store import SQLiteDocumentStore


def run_benchmark(*, paragraphs: int = 1000, queries: int = 50) -> dict:
    if paragraphs < 1 or queries < 1:
        raise ValueError("paragraphs and queries must be positive")
    source_text = "\n\n".join(
        f"Trial marker{i:06d} measured {i + 1} samples. "
        "The reported result does not imply causation."
        for i in range(paragraphs)
    )
    query_indexes = [(i * 7919) % paragraphs for i in range(queries)]
    with tempfile.TemporaryDirectory(prefix="doc-rag-benchmark-") as temporary:
        root = Path(temporary)
        source = root / "synthetic.txt"
        source.write_text(source_text, encoding="utf-8")
        store = SQLiteDocumentStore(root / "documents.sqlite3")
        prepared = prepare_source(source, language="en", usage_scope="synthetic-benchmark")
        store.ingest(prepared.document, prepared.iter_units())
        start = time.perf_counter()
        manifest = build_bm25_index(store, prepared.document.document_id)
        build_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        retriever = BM25Retriever(store, prepared.document.document_id)
        load_ms = (time.perf_counter() - start) * 1000
        retriever.search("marker000000")  # Separate the first search from measured warm queries.
        results = list(retriever.search_many((f"marker{i:06d}" for i in query_indexes), top_k=5))
        timings = [result.elapsed_ms for result in results]
        correct = sum(
            bool(result.hits) and result.hits[0].block.ordinal == index + 1
            for result, index in zip(results, query_indexes, strict=True)
        )
        snapshot_bytes = sum(
            path.stat().st_size for path in (index_root(store) / manifest.index_id).iterdir()
        )
        return {
            "benchmark": "synthetic-bm25-resource-reuse-v1",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {name: version(name) for name in ("bm25s", "jieba", "numpy", "scipy")},
            "paragraphs": paragraphs,
            "queries": queries,
            "corpus_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "index_build_ms": build_ms,
            "retriever_load_ms": load_ms,
            "warm_search_p50_ms": float(np.percentile(timings, 50)),
            "warm_search_p95_ms": float(np.percentile(timings, 95)),
            "warm_search_each_ms": timings,
            "marker_top1_matches": correct,
            "snapshot_bytes": snapshot_bytes,
            "scope": "Generated exact-term English queries; not a RAG quality measurement.",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paragraphs", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=50)
    args = parser.parse_args()
    if args.paragraphs < 1 or args.queries < 1:
        parser.error("paragraphs and queries must be positive")
    print(
        json.dumps(run_benchmark(paragraphs=args.paragraphs, queries=args.queries), sort_keys=True)
    )
