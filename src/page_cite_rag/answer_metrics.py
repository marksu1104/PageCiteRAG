"""QASPER answer/evidence scoring, independent of retrieval and generation.

The metric definitions follow AllenAI's evaluator (Apache-2.0), but this
implementation is independently written. A selected text-retrieval subset is
not the official leaderboard. Token overlap is not semantic correctness.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from page_cite_rag import qasper

OFFICIAL_REVISION = "e996b6c7b1b5f95d9308a74e3586416c6e780df1"
OFFICIAL_SHA256 = "781aba7cd8e524bef4f0a1b4bf3504e5b02cb1d8d5bf32a8f0a89dfa83e86bfe"
OFFICIAL_URL = (
    "https://raw.githubusercontent.com/allenai/qasper-led-baseline/"
    f"{OFFICIAL_REVISION}/scripts/evaluator.py"
)
METRIC_VERSION = "qasper-official-compatible/all-references/raw-paragraphs/v1"
ANSWER_TYPES = ("extractive", "abstractive", "boolean", "none")


@dataclass(frozen=True)
class AnswerReference:
    answer: str
    evidence: tuple[str, ...]
    answer_type: str


@dataclass(frozen=True)
class AnswerPrediction:
    answer: str
    evidence: tuple[str, ...]


def decode_answer(answer: dict) -> AnswerReference:
    """Preserve official answer precedence and raw evidence, including figures.

    The separate retrieval loader is deliberately unchanged: no answer text is
    added to passages, indexes, query plans, or generation prompts.
    """
    try:
        unanswerable = answer["unanswerable"]
        spans, free_form, yes_no = (
            answer["extractive_spans"],
            answer["free_form_answer"],
            answer["yes_no"],
        )
        evidence = answer["evidence"]
        if (
            type(unanswerable) is not bool
            or type(spans) is not list
            or any(type(text) is not str for text in spans)
            or type(free_form) is not str
            or (yes_no is not None and type(yes_no) is not bool)
            or type(evidence) is not list
            or any(type(text) is not str for text in evidence)
        ):
            raise ValueError("invalid answer fields")
        if unanswerable:
            return AnswerReference("Unanswerable", (), "none")
        if spans:
            text, kind = ", ".join(spans), "extractive"
        elif free_form:
            text, kind = free_form, "abstractive"
        elif yes_no is not None:
            text, kind = ("Yes" if yes_no else "No"), "boolean"
        else:
            raise ValueError("annotation has no answer")
        if not text.strip():
            raise ValueError("annotation has a blank answer")
        return AnswerReference(text, tuple(evidence), kind)
    except (KeyError, TypeError, ValueError) as error:
        raise qasper.QasperError("unsupported QASPER answer schema") from error


def read_answer_references(
    data_dir: Path, split: str, question_ids: set[str]
) -> dict[str, tuple[AnswerReference, ...]]:
    """Read gold for selected IDs only, from the existing hash-verified snapshot."""
    if split not in qasper.SOURCES or not question_ids:
        raise ValueError("an existing split and nonempty question selection are required")
    path = qasper.snapshot_path(data_dir, split)
    qasper.verify_snapshot(path, split)
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise qasper.QasperError("answer evaluation requires the eval dependency group") from error
    found = {}
    for batch in parquet.ParquetFile(path).iter_batches(batch_size=32, columns=["qas"]):
        for row in batch.to_pylist():
            qas = row["qas"]
            for identifier, references in zip(qas["question_id"], qas["answers"], strict=True):
                if identifier not in question_ids:
                    continue
                if identifier in found or not references["answer"]:
                    raise qasper.QasperError("duplicate or missing selected answer references")
                found[identifier] = tuple(decode_answer(a) for a in references["answer"])
    if set(found) != question_ids:
        raise qasper.QasperError("selected questions are missing from the answer snapshot")
    return found


def normalized_tokens(text: str) -> list[str]:
    translation = str.maketrans("", "", string.punctuation)
    return re.sub(r"\b(a|an|the)\b", " ", text.lower().translate(translation)).split()


def answer_f1(predicted: str, expected: str) -> float:
    left, right = normalized_tokens(predicted), normalized_tokens(expected)
    overlap = sum((Counter(left) & Counter(right)).values())
    # In the official evaluator two empty normalized answers score zero, not one.
    return 2 * overlap / (len(left) + len(right)) if overlap else 0.0


def evidence_f1(predicted: tuple[str, ...], expected: tuple[str, ...]) -> float:
    """Exact raw paragraph matching; preserve official list-length denominators.

    In particular, duplicates in official annotations must not be silently
    removed, nor whitespace normalized to improve a score.
    """
    if not predicted and not expected:
        return 1.0
    overlap = len(set(predicted) & set(expected))
    return 2 * overlap / (len(predicted) + len(expected)) if overlap else 0.0


def evidence_precision_recall(
    predicted: tuple[str, ...], expected: tuple[str, ...]
) -> tuple[float, float]:
    """Expose the exact-match evidence overlap behind Evidence F1."""
    if not predicted and not expected:
        return 1.0, 1.0
    overlap = len(set(predicted) & set(expected))
    precision = overlap / len(predicted) if predicted else 0.0
    recall = overlap / len(expected) if expected else 0.0
    return precision, recall


def score_prediction(
    references: tuple[AnswerReference, ...], prediction: AnswerPrediction | None
) -> dict:
    if not references or any(r.answer_type not in ANSWER_TYPES for r in references):
        raise ValueError("nonempty valid answer references are required")
    if prediction is None:
        return {
            "answer_f1": 0.0,
            "evidence_f1": 0.0,
            "evidence_precision": 0.0,
            "evidence_recall": 0.0,
            "answer_type": None,
        }
    answers = [answer_f1(prediction.answer, reference.answer) for reference in references]
    evidence = [evidence_f1(prediction.evidence, reference.evidence) for reference in references]
    evidence_details = [
        evidence_precision_recall(prediction.evidence, reference.evidence)
        for reference in references
    ]
    answer_index = max(range(len(answers)), key=answers.__getitem__)
    evidence_index = max(range(len(evidence)), key=evidence.__getitem__)
    precision, recall = evidence_details[evidence_index]
    return {
        "answer_f1": answers[answer_index],
        "evidence_f1": evidence[evidence_index],
        "evidence_precision": precision,
        "evidence_recall": recall,
        "answer_type": references[answer_index].answer_type,
        "answer_reference_index": answer_index,
        "evidence_reference_index": evidence_index,
    }


def evaluate_answers(
    gold: dict[str, tuple[AnswerReference, ...]], predictions: dict[str, AnswerPrediction]
) -> dict:
    """Match official aggregate semantics, retaining absent predictions as zero.

    Answer/evidence maxima are independent. Answer type follows the winning
    answer reference (ties follow original annotation order), not a guessed type.
    """
    if not gold or set(predictions) - set(gold):
        raise ValueError("nonempty gold and predictions within its selection are required")
    rows = [
        score_prediction(refs, predictions.get(identifier)) for identifier, refs in gold.items()
    ]
    by_type = {
        kind: [row["answer_f1"] for row in rows if row["answer_type"] == kind]
        for kind in ANSWER_TYPES
    }
    return {
        "Answer F1": sum(row["answer_f1"] for row in rows) / len(rows),
        "Answer F1 by type": {
            kind: sum(values) / len(values) if values else 0.0 for kind, values in by_type.items()
        },
        "Evidence F1": sum(row["evidence_f1"] for row in rows) / len(rows),
        "Evidence precision": sum(row["evidence_precision"] for row in rows) / len(rows),
        "Evidence recall": sum(row["evidence_recall"] for row in rows) / len(rows),
        "Missing predictions": len(set(gold) - set(predictions)),
    }
