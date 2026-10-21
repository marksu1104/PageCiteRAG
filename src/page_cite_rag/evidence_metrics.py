"""Reference-aware text evidence diagnostics, separate from the frozen union baseline.

Best-reference paragraph F1 follows the set-F1/max-reference idea in QASPER,
but this text-only, eligible-question subset is NOT its official leaderboard.
No metric here measures answer accuracy or proves that an annotation is correct.
"""

from __future__ import annotations

from dataclasses import dataclass

from page_cite_rag.qasper import Paper, Question, align_question

METRICS_VERSION = "complete-text-references/independent-max-recall-and-set-f1/v1"


@dataclass(frozen=True)
class EvidenceReference:
    annotation_index: int
    passage_ids: tuple[str, ...]


def align_references(paper: Paper, question: Question) -> tuple[EvidenceReference, ...]:
    """Keep original annotator identity; exclude incomplete annotations in full.

    Reuse the original single-annotation alignment policy rather than introducing
    fuzzy matches or silently scoring the text portion of mixed table evidence.
    """
    references = []
    for index, annotation in enumerate(question.annotations):
        aligned = align_question(
            paper, Question(question.question_id, question.text, (annotation,))
        )
        if aligned.exclusion_reason is None:
            references.append(EvidenceReference(index, aligned.gold_passage_ids))
    return tuple(references)


def reference_metrics(
    retrieved: tuple[str, ...], references: tuple[EvidenceReference, ...], *, k: int
) -> dict:
    """Score each reference separately, then take independent maxima.

    Deduplicate within the supplied top-k, not before cutting the ranked list.
    A reference maximizing F1 need not be the one maximizing recall.
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if not references or any(not reference.passage_ids for reference in references):
        raise ValueError("reference metrics require nonempty aligned evidence sets")
    predicted = set(retrieved[:k])
    per_reference = []
    for reference in references:
        gold = set(reference.passage_ids)
        matched = len(predicted & gold)
        per_reference.append(
            {
                "annotation_index": reference.annotation_index,
                "evidence_count": len(gold),
                "matched": matched,
                "recall": matched / len(gold),
                "paragraph_f1": 2 * matched / (len(predicted) + len(gold)),
                "complete": gold <= predicted,
            }
        )
    union = set().union(*(set(reference.passage_ids) for reference in references))
    return {
        "union_recall": len(predicted & union) / len(union),
        "union_complete": union <= predicted,
        "best_reference_recall": max(item["recall"] for item in per_reference),
        "best_reference_paragraph_f1": max(item["paragraph_f1"] for item in per_reference),
        "any_reference_complete": any(item["complete"] for item in per_reference),
        "per_reference": per_reference,
    }


def evidence_group(references: tuple[EvidenceReference, ...]) -> dict:
    """Distinguish alternative evidence from evidence that always needs multiple paragraphs."""
    if not references or any(not reference.passage_ids for reference in references):
        raise ValueError("evidence grouping requires nonempty references")
    sets = {frozenset(reference.passage_ids) for reference in references}
    union = set().union(*sets)
    return {
        "valid_annotations": len(references),
        "distinct_evidence_sets": len(sets),
        "union_evidence_count": len(union),
        "minimum_reference_size": min(map(len, sets)),
        "all_references_multiple": all(len(ids) >= 2 for ids in sets),
        "union_multiple_with_single_reference": len(union) > 1 and any(len(s) == 1 for s in sets),
    }


def candidate_diagnosis(
    final: tuple[str, ...],
    candidates: tuple[str, ...],
    references: tuple[EvidenceReference, ...],
    *,
    final_k: int = 10,
) -> str:
    """Offline gold-aware categorization, never a live retrieval decision or confidence score."""
    final_score = reference_metrics(final, references, k=final_k)
    if not set(final[:final_k]) <= set(candidates):
        raise ValueError("final results must come from the supplied candidate pool")
    if final_score["any_reference_complete"]:
        return "complete_reference"
    feasible = [r for r in references if len(set(r.passage_ids)) <= final_k]
    if not feasible:
        return "reference_exceeds_final_budget"
    if any(set(reference.passage_ids) <= set(candidates) for reference in feasible):
        return "selection_miss"
    return "candidate_miss"


def neighbor_opportunity(
    paper: Paper, final: tuple[str, ...], references: tuple[EvidenceReference, ...]
) -> dict:
    """Gold-aware upper opportunity for same-section ±1 reading, NOT a budget-fair result.

    Expanded context may contain three times as many paragraphs. Graph adjacency
    is not entailment, and this diagnostic does not implement a reading workflow.
    """
    lookup = {passage.passage_id: passage for passage in paper.passages}
    if len(lookup) != len(paper.passages) or not set(final) <= lookup.keys():
        raise ValueError("neighbor inputs are duplicated or outside the paper")
    anchors = [lookup[identifier] for identifier in final]
    expanded = {
        passage.passage_id
        for passage in paper.passages
        if any(
            passage.section_index == anchor.section_index
            and abs(passage.paragraph_index - anchor.paragraph_index) <= 1
            for anchor in anchors
        )
    }
    missing = []
    for reference in references:
        gaps = []
        for identifier in reference.passage_ids:
            if identifier in final:
                continue
            passage = lookup[identifier]
            distances = [
                abs(passage.paragraph_index - anchor.paragraph_index)
                for anchor in anchors
                if passage.section_index == anchor.section_index
            ]
            gaps.append(
                {
                    "passage_id": identifier,
                    "nearest_same_section_distance": min(distances) if distances else None,
                }
            )
        missing.append({"annotation_index": reference.annotation_index, "missing": gaps})
    return {
        "expanded_paragraph_count": len(expanded),
        "added_paragraph_count": len(expanded - set(final)),
        "any_reference_complete_with_neighbors": any(
            set(reference.passage_ids) <= expanded for reference in references
        ),
        "reference_gaps": missing,
    }
