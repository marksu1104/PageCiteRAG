from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from page_cite_rag.qasper import Annotation, Paper, Passage, Question

_path = Path(__file__).parents[1] / "scripts" / "freeze_qasper_multisection.py"
_spec = importlib.util.spec_from_file_location("freeze_qasper_multisection", _path)
freeze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(freeze)


def paper(paper_id: str, *questions: tuple[str, tuple[str, ...]]) -> Paper:
    passages = (
        Passage(f"{paper_id}-a", 0, 0, "Intro says the model uses attention."),
        Passage(f"{paper_id}-b", 0, 1, "Intro also states the dataset size."),
        Passage(f"{paper_id}-c", 1, 0, "Results report an F1 of 80."),
    )
    return Paper(
        paper_id,
        passages,
        tuple(
            Question(qid, "What?", (Annotation(False, evidence),)) for qid, evidence in questions
        ),
    )


TWO_SECTIONS = ("Intro says the model uses attention.", "Results report an F1 of 80.")
ONE_SECTION = ("Intro says the model uses attention.", "Intro also states the dataset size.")


def test_selection_keeps_multi_section_evidence_once_per_unseen_unreserved_paper():
    papers = [
        paper("p1", ("q1", TWO_SECTIONS), ("q2", TWO_SECTIONS)),
        paper("p2", ("q3", ONE_SECTION)),
        paper("p3", ("q4", TWO_SECTIONS)),
        paper("p4", ("q5", TWO_SECTIONS)),
        paper("p5", ("q6", TWO_SECTIONS)),
    ]
    selected, eligible_questions, eligible_papers = freeze.select_cases(
        papers, exposed={"p3"}, reserved=frozenset({"p4"}), count=2, seed="42"
    )
    assert eligible_questions == 3 and eligible_papers == 2
    assert sorted(item[1].paper_id for item in selected) == ["p1", "p5"]
    assert all(item[4] == 2 for item in selected)


def test_selection_refuses_a_count_the_split_cannot_supply():
    with pytest.raises(ValueError, match="only 1 eligible papers"):
        freeze.select_cases(
            [paper("p1", ("q1", TWO_SECTIONS))],
            exposed=set(),
            reserved=frozenset(),
            count=2,
            seed="42",
        )


def test_no_answer_selection_keeps_only_questions_every_annotator_found_unanswerable():
    def question(qid, *unanswerable):
        return Question(qid, "What?", tuple(Annotation(flag, ()) for flag in unanswerable))

    passages = (Passage("x-a", 0, 0, "Text."),)
    papers = [
        Paper("p1", passages, (question("q1", True, True), question("q2", True, False))),
        Paper("p2", passages, (question("q3", True, False),)),
        Paper("p3", passages, (question("q4", True),)),
    ]
    selected, eligible_questions, _ = freeze.select_cases(
        papers, exposed={"p3"}, reserved=frozenset(), count=1, seed="42", stratum="no-answer"
    )
    assert eligible_questions == 1 and selected[0][2].question_id == "q1"
    # A development set draws only from papers already seen.
    seen, _, _ = freeze.select_cases(
        papers,
        exposed={"p3"},
        reserved=frozenset(),
        count=1,
        seed="42",
        stratum="no-answer",
        exposure="exposed",
    )
    assert seen[0][2].question_id == "q4"
