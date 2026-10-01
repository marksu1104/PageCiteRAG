"""Explicit real-model smoke using only synthetic text; not a quality benchmark."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from doc_rag.embedding import DEFAULT_MODEL_DIR, QwenEmbeddingEncoder
from doc_rag.retrieval import RetrievalError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=Path("outputs/embedding-smoke.json"))
    args = parser.parse_args()
    start = time.perf_counter()
    encoder = QwenEmbeddingEncoder(args.model_dir)
    load_ms = (time.perf_counter() - start) * 1000
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
    ]
    queries = ["What is the capital of China?", "Explain gravity", "中國的首都是哪裡？"]
    start = time.perf_counter()
    doc_vectors = encoder.encode_documents(documents)
    query_vectors = encoder.encode_queries(queries)
    encode_ms = (time.perf_counter() - start) * 1000
    scores = query_vectors @ doc_vectors.T
    np.testing.assert_array_equal(scores.argmax(axis=1), [0, 1, 0])
    official = encoder._model.similarity(query_vectors, doc_vectors).numpy()
    np.testing.assert_allclose(scores, official, atol=1e-6)
    long_text = "The result is not significant unless the sample size exceeds 12. " * 300
    spans = encoder.split_document(long_text)
    assert len(spans) > 1 and spans[0][0] == 0 and spans[-1][1] == len(long_text)
    assert all(a[1] >= b[0] for a, b in zip(spans, spans[1:]))
    assert all(encoder._length(long_text[a:b]) <= 512 for a, b in spans)
    try:
        encoder.encode_queries([long_text])
    except RetrievalError:
        oversized_rejected = True
    else:
        raise AssertionError("oversized query was silently truncated")
    report = {
        "identity": encoder.identity,
        "model_load_ms": load_ms,
        "synthetic_encode_ms": encode_ms,
        "scores": scores.tolist(),
        "similarity_max_abs_difference": float(np.max(np.abs(scores - official))),
        "long_passage_segments": len(spans),
        "oversized_query_rejected": oversized_rejected,
        "process_peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scope": "synthetic interface smoke only; not cross-language benchmark quality",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
