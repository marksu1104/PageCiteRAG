"""Metric contracts, not a synthetic substitute for benchmark performance."""

import pytest

from page_cite_rag.answer_metrics import (
    AnswerPrediction,
    AnswerReference,
    answer_f1,
    decode_answer,
    evaluate_answers,
    evidence_f1,
    evidence_precision_recall,
    score_prediction,
)
from page_cite_rag.qasper import QasperError


def annotation(**changes):
    return {
        "unanswerable": False,
        "extractive_spans": [],
        "free_form_answer": "",
        "yes_no": None,
        "evidence": ["raw source"],
        **changes,
    }


def test_official_decoding_precedence_preserves_no_and_raw_figure_evidence():
    assert decode_answer(annotation(yes_no=False)).answer == "No"
    assert decode_answer(annotation(yes_no=True)).answer == "Yes"
    assert decode_answer(annotation(free_form_answer="free", yes_no=False)).answer == "free"
    assert (
        decode_answer(annotation(extractive_spans=["one", "two"], free_form_answer="free")).answer
        == "one, two"
    )
    assert decode_answer(annotation(unanswerable=True)).evidence == ()
    assert decode_answer(
        annotation(free_form_answer="figure", evidence=["FLOAT SELECTED: table"])
    ).evidence == ("FLOAT SELECTED: table",)


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"yes_no": 0},
        {"evidence": "not a list"},
        {"extractive_spans": [1]},
        {"free_form_answer": None},
    ],
)
def test_invalid_annotations_are_not_coerced_or_silently_excluded(change):
    with pytest.raises(QasperError, match="schema"):
        decode_answer(annotation(**change))


def test_official_normalization_empty_and_duplicate_semantics():
    assert answer_f1("The cat, cat!", "cat") == pytest.approx(2 / 3)
    assert answer_f1("a an the", "") == 0
    assert answer_f1("77.19", "77.19") == 1
    assert evidence_f1((), ()) == 1
    assert evidence_f1(("a b",), ("a  b",)) == 0
    assert evidence_f1(("x",), ("x", "x")) == pytest.approx(2 / 3)
    assert evidence_precision_recall(("x", "extra"), ("x",)) == (0.5, 1.0)
    assert evidence_precision_recall((), ()) == (1.0, 1.0)


def test_answer_and_evidence_reference_maxima_are_independent():
    references = (
        AnswerReference("cat", ("x",), "extractive"),
        AnswerReference("dog", ("y",), "abstractive"),
    )
    scores = score_prediction(references, AnswerPrediction("cat", ("y",)))
    assert scores["answer_f1"] == scores["evidence_f1"] == 1
    assert scores["answer_reference_index"] == 0 and scores["evidence_reference_index"] == 1


def test_missing_cases_stay_in_denominator_and_type_ties_use_original_order():
    gold = {
        "q1": (AnswerReference("No", ("x",), "boolean"),),
        "q2": (AnswerReference("cat", ("y",), "extractive"),),
    }
    scores = evaluate_answers(gold, {"q1": AnswerPrediction("no", ("x",))})
    assert scores["Answer F1"] == scores["Evidence F1"] == 0.5
    assert scores["Evidence precision"] == scores["Evidence recall"] == 0.5
    assert scores["Missing predictions"] == 1 and scores["Answer F1 by type"]["boolean"] == 1
    refs = (AnswerReference("cat", (), "extractive"), AnswerReference("dog", (), "abstractive"))
    assert score_prediction(refs, AnswerPrediction("none", ()))["answer_type"] == "extractive"
    with pytest.raises(ValueError):
        evaluate_answers(gold, {"unknown": AnswerPrediction("cat", ())})


def test_selected_answer_reader_uses_verified_parquet_without_loading_other_gold(
    tmp_path, monkeypatch
):
    import hashlib

    from page_cite_rag import qasper
    from page_cite_rag.answer_metrics import read_answer_references

    parquet = pytest.importorskip("pyarrow.parquet")
    arrow = pytest.importorskip("pyarrow")
    path = qasper.snapshot_path(tmp_path, "train")
    path.parent.mkdir(parents=True)
    row = {
        "qas": {
            "question_id": ["selected", "other"],
            "answers": [{"answer": [annotation(yes_no=False)]}, {"answer": [annotation()]}],
        }
    }
    parquet.write_table(arrow.Table.from_pylist([row]), path)
    monkeypatch.setitem(
        qasper.SOURCES,
        "train",
        {
            "filename": path.name,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    references = read_answer_references(tmp_path, "train", {"selected"})
    assert references["selected"] == (AnswerReference("No", ("raw source",), "boolean"),)
    with pytest.raises(QasperError, match="missing"):
        read_answer_references(tmp_path, "train", {"missing"})
    path.write_bytes(b"corrupt")
    with pytest.raises(QasperError, match="checksum"):
        read_answer_references(tmp_path, "train", {"selected"})
