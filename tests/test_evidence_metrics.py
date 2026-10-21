from __future__ import annotations

import pytest

from page_cite_rag.evidence_metrics import (
    EvidenceReference,
    align_references,
    candidate_diagnosis,
    evidence_group,
    neighbor_opportunity,
    reference_metrics,
)
from page_cite_rag.qasper import Annotation, Paper, Passage, Question, align_question


def ref(*ids, index=0):
    return EvidenceReference(index, tuple(ids))


def test_alternative_references_are_not_required_together():
    refs = (ref("a"), ref("b", index=1))
    result = reference_metrics(("a",), refs, k=10)
    assert result["union_recall"] == 0.5
    assert not result["union_complete"]
    assert result["best_reference_recall"] == 1
    assert result["best_reference_paragraph_f1"] == 1
    assert result["any_reference_complete"]
    assert evidence_group(refs)["union_multiple_with_single_reference"]
    assert not evidence_group(refs)["all_references_multiple"]


def test_recall_and_f1_maxima_can_use_different_references():
    # One reference is complete, but the larger reference explains more of the
    # returned paragraphs and therefore has the better precision/F1 tradeoff.
    result = reference_metrics(
        ("a", "b", "c", "noise"), (ref("a"), ref("a", "b", "c", "d", index=1)), k=4
    )
    assert result["best_reference_recall"] == 1
    assert result["best_reference_paragraph_f1"] == 0.75
    assert result["per_reference"][0]["paragraph_f1"] == 0.4
    assert result["per_reference"][1]["recall"] == 0.75


def test_empty_predictions_noise_and_duplicates_use_set_metrics_with_rank_cutoff():
    refs = (ref("a", "b"),)
    empty = reference_metrics((), refs, k=10)
    assert empty["best_reference_recall"] == empty["best_reference_paragraph_f1"] == 0
    assert not empty["any_reference_complete"]
    result = reference_metrics(("a", "a", "b", "noise"), refs, k=2)
    assert result["best_reference_recall"] == 0.5  # Do not deduplicate before cutoff.
    assert result["best_reference_paragraph_f1"] == pytest.approx(2 / 3)
    assert reference_metrics(("a", "b", "noise"), refs, k=3)["best_reference_paragraph_f1"] == 0.8


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_invalid_metric_budget_is_rejected(k):
    with pytest.raises(ValueError, match="positive integer"):
        reference_metrics(("a",), (ref("a"),), k=k)


def test_empty_references_cannot_be_counted_as_success():
    for refs in ((), (ref(),)):
        with pytest.raises(ValueError, match="nonempty"):
            reference_metrics((), refs, k=5)
        with pytest.raises(ValueError, match="nonempty"):
            evidence_group(refs)


def test_alignment_preserves_annotation_identity_and_excludes_entire_mixed_annotations():
    paper = Paper(
        "p", (Passage("a", 0, 0, "Alpha\nsamples."), Passage("b", 0, 1, "Beta controls.")), ()
    )
    question = Question(
        "q",
        "question",
        (
            Annotation(True, ()),
            Annotation(False, ("Alpha samples.", "FLOAT SELECTED table")),
            Annotation(False, ("Alpha  samples.", "Alpha samples.")),
            Annotation(False, ("Beta controls.",)),
            Annotation(False, ("missing", "Beta controls.")),
            Annotation(False, ("Beta controls.",)),
        ),
    )
    refs = align_references(paper, question)
    assert refs == (ref("a", index=2), ref("b", index=3), ref("b", index=5))
    assert set().union(*(set(r.passage_ids) for r in refs)) == set(
        align_question(paper, question).gold_passage_ids
    )
    assert evidence_group(refs)["valid_annotations"] == 3
    assert evidence_group(refs)["distinct_evidence_sets"] == 2


def test_all_references_multiple_is_not_the_same_as_union_multiple():
    group = evidence_group((ref("a", "b"), ref("c", "d", index=1)))
    assert group["all_references_multiple"]
    assert group["minimum_reference_size"] == 2
    assert group["union_evidence_count"] == 4
    assert not group["union_multiple_with_single_reference"]


@pytest.mark.parametrize(
    ("final", "pool", "refs", "expected"),
    [
        (("a",), ("a", "b"), (ref("a"), ref("b", index=1)), "complete_reference"),
        (("a",), ("a", "b"), (ref("a", "b"),), "selection_miss"),
        (("a",), ("a", "noise"), (ref("a", "b"),), "candidate_miss"),
        (("a",), ("a", "b", "c"), (ref("a", "b", "c"),), "reference_exceeds_final_budget"),
    ],
)
def test_candidate_categories_are_gold_aware_and_respect_final_budget(final, pool, refs, expected):
    assert candidate_diagnosis(final, pool, refs, final_k=2) == expected


def test_candidates_must_contain_the_final_results():
    with pytest.raises(ValueError, match="candidate pool"):
        candidate_diagnosis(("a",), ("b",), (ref("b"),))


def test_neighbors_never_cross_sections_and_report_extra_context():
    paper = Paper(
        "p",
        (
            Passage("a", 0, 0, "a"),
            Passage("b", 0, 1, "b"),
            Passage("c", 0, 3, "c"),
            Passage("d", 1, 0, "d"),
        ),
        (),
    )
    result = neighbor_opportunity(paper, ("a",), (ref("a", "b"), ref("d", index=1)))
    assert result["any_reference_complete_with_neighbors"]
    assert result["expanded_paragraph_count"] == 2
    assert result["added_paragraph_count"] == 1
    assert result["reference_gaps"] == [
        {
            "annotation_index": 0,
            "missing": [{"passage_id": "b", "nearest_same_section_distance": 1}],
        },
        {
            "annotation_index": 1,
            "missing": [{"passage_id": "d", "nearest_same_section_distance": None}],
        },
    ]
    assert not neighbor_opportunity(paper, ("a",), (ref("a", "c"),))[
        "any_reference_complete_with_neighbors"
    ]
    with pytest.raises(ValueError, match="outside"):
        neighbor_opportunity(paper, ("unknown",), (ref("a"),))
