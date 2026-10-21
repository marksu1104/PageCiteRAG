"""Compare local scores against a hash-pinned official evaluator, offline.

The caller explicitly obtains the official public script. Only the exact pinned
bytes can execute. No dataset/model downloads or generated semantic judgments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
from pathlib import Path

from page_cite_rag.answer_metrics import (
    OFFICIAL_SHA256,
    AnswerPrediction,
    answer_f1,
    decode_answer,
    evaluate_answers,
    evidence_f1,
)


def compare(official, gold, predictions):
    native = evaluate_answers(gold, predictions)
    expected = official["evaluate"](
        {
            q: [
                {"answer": r.answer, "evidence": list(r.evidence), "type": r.answer_type}
                for r in refs
            ]
            for q, refs in gold.items()
        },
        {q: {"answer": p.answer, "evidence": list(p.evidence)} for q, p in predictions.items()},
    )
    for key in ("Answer F1", "Evidence F1", "Missing predictions"):
        if not math.isclose(native[key], expected[key], abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError("official aggregate scorer mismatch")
    for kind, value in native["Answer F1 by type"].items():
        if not math.isclose(value, expected["Answer F1 by type"][kind], abs_tol=1e-12):
            raise ValueError("official answer-type scorer mismatch")
    return native


def check(evaluator):
    if evaluator.is_symlink() or not evaluator.is_file():
        raise ValueError("an explicitly obtained official evaluator file is required")
    if hashlib.sha256(evaluator.read_bytes()).hexdigest() != OFFICIAL_SHA256:
        raise ValueError("official evaluator checksum mismatch; refusing to execute it")
    official = runpy.run_path(str(evaluator))
    text_pairs = [
        ("", ""),
        ("The cat, cat!", "cat"),
        ("NO", "No"),
        ("Unanswerable", "Unanswerable"),
        ("A precision-recall", "precision recall"),
        ("中文。", "中文"),
        ("77.19", "77.19"),
        ("a an the", ""),
    ]
    for predicted, expected in text_pairs:
        if not math.isclose(
            answer_f1(predicted, expected),
            official["token_f1_score"](predicted, expected),
            abs_tol=1e-12,
        ):
            raise ValueError("official token F1 mismatch")
    evidence_pairs = [
        ((), ()),
        (("x",), ()),
        (("x", "x"), ("x",)),
        (("x",), ("x", "x")),
        (("a b",), ("a  b",)),
        (("x", "y"), ("x",)),
    ]
    for predicted, expected in evidence_pairs:
        if not math.isclose(
            evidence_f1(predicted, expected),
            official["paragraph_f1_score"](predicted, expected),
            abs_tol=1e-12,
        ):
            raise ValueError("official paragraph F1 mismatch")
    base = {
        "unanswerable": False,
        "extractive_spans": [],
        "free_form_answer": "",
        "yes_no": None,
        "evidence": ["source"],
        "annotation_id": "synthetic",
    }
    annotations = [
        {**base, "extractive_spans": ["alpha", "beta"], "free_form_answer": "ignored"},
        {**base, "free_form_answer": "a free answer", "yes_no": True},
        {**base, "yes_no": True},
        {**base, "yes_no": False},
        {**base, "unanswerable": True},
        {**base, "free_form_answer": "figure", "evidence": ["FLOAT SELECTED: table"]},
    ]
    raw = {
        "p": {
            "qas": [
                {"question_id": str(i), "answers": [{"answer": a}]}
                for i, a in enumerate(annotations)
            ]
        }
    }
    decoded = official["get_answers_and_evidence"](raw, False)
    gold = {str(i): (decode_answer(a),) for i, a in enumerate(annotations)}
    for q, refs in gold.items():
        r = refs[0]
        if decoded[q] != [
            {"answer": r.answer, "evidence": list(r.evidence), "type": r.answer_type}
        ]:
            raise ValueError("official answer precedence differs")
    predictions = {
        q: AnswerPrediction(refs[0].answer, refs[0].evidence)
        for q, refs in gold.items()
        if q != "5"
    }
    compare(official, gold, predictions)
    result = {
        "token_edge_cases": len(text_pairs),
        "evidence_edge_cases": len(evidence_pairs),
        "answer_decoding_cases": len(annotations),
        "aggregate_parity": True,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-evaluator", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.official_evaluator), indent=2))


if __name__ == "__main__":
    main()
