import importlib.util
import json
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_PREPARE_PATH = Path(__file__).resolve().parents[1] / "scripts/prepare_reading_eval.py"
_PREPARE_SPEC = importlib.util.spec_from_file_location("prepare_reading_eval", _PREPARE_PATH)
prepare_reading_eval = importlib.util.module_from_spec(_PREPARE_SPEC)
sys.modules[_PREPARE_SPEC.name] = prepare_reading_eval
_PREPARE_SPEC.loader.exec_module(prepare_reading_eval)


def _xrag_row(row_id, question_type, *, article_prefix=None):
    prefix = article_prefix or f"row-{row_id}"
    return {
        "id": row_id,
        "language": "zh",
        "task": "mono_retrieval",
        "q_type": question_type,
        "question": f"問題 {row_id}？",
        "answer": f"答案 {row_id}",
        "en_answer": f"Answer {row_id}",
        "crossdoc_type": 0,
        "article_order": [3, 0, 1, 2, 4, 5, 6, 7],
        "articles": [f"{prefix} article {index}." for index in range(8)],
        "is_support": [2, 1, 0, 0, 0, 0, 0, 0],
    }


def test_xrag_bundle_offsets_follow_presentation_order():
    row = _xrag_row(1, "aggregation")

    text, ranges = prepare_reading_eval._xrag_bundle(row)

    assert len(ranges) == 8
    for presented_index, item in enumerate(ranges):
        source_index = row["article_order"][presented_index]
        assert item["source_index"] == source_index
        assert text[item["char_start"] : item["char_end"]].strip() == row["articles"][source_index]
        assert item["article_id"] == f"article-{presented_index + 1}"


def test_xrag_sample_is_stratified_unexposed_and_keeps_gold_separate():
    question_types = ("aggregation", "comparison", "multihop", "set")
    rows = [
        _xrag_row(index, question_type)
        for index, question_type in enumerate(question_types * 4, start=100)
    ]
    exposed = {rows[0]["id"]}

    inputs, gold, selection = prepare_reading_eval._xrag_cases(rows, exposed)

    assert len(inputs) == len(gold) == 12
    assert Counter(row["question_type"] for row in gold) == Counter(
        {question_type: 3 for question_type in question_types}
    )
    assert exposed.isdisjoint(int(row["case_id"].rsplit("-", 1)[1]) for row in inputs)
    assert all("answer" not in row and "is_support" not in row for row in inputs)
    assert all("answer" in row and "article_support_levels" in row for row in gold)
    assert selection["unique_article_contents"] == 96


def test_xrag_sample_rejects_reused_article_content():
    question_types = ("aggregation", "comparison", "multihop", "set")
    rows = [
        _xrag_row(index, question_type, article_prefix="shared")
        for index, question_type in enumerate(question_types * 3, start=100)
    ]

    try:
        prepare_reading_eval._xrag_cases(rows, set())
    except ValueError as error:
        assert "cannot supply" in str(error)
    else:
        raise AssertionError("sample must not reuse article content across selected cases")


def test_qasper_frozen_inputs_keep_original_section_metadata(monkeypatch, tmp_path):
    from page_cite_rag.qasper import Annotation, Paper, Passage, Question

    @dataclass(frozen=True)
    class Reference:
        text: str

    papers = []
    references = {}
    for index in range(8):
        paper_id = f"paper-{index}"
        question_id = f"question-{index}"
        text = f"Evidence {index}."
        passage = Passage(f"{paper_id}/s0002/p0000", 2, 0, text)
        answerable = index < 6
        annotation = Annotation(not answerable, (text,) if answerable else ())
        question = Question(question_id, f"Question {index}?", (annotation,))
        papers.append(
            Paper(
                paper_id,
                (passage,),
                (question,),
                (None, "Introduction", "Methods"),
            )
        )
        references[question_id] = [Reference("Answer")]

    monkeypatch.setattr(prepare_reading_eval, "read_snapshot", lambda *_: iter(papers))
    monkeypatch.setattr(
        prepare_reading_eval,
        "read_answer_references",
        lambda _directory, _split, _question_ids: references,
    )

    inputs, _gold = prepare_reading_eval._qasper_cases(tmp_path, set())

    assert len(inputs) == 8
    assert all(
        row["source"]["section_names"] == [None, "Introduction", "Methods"] for row in inputs
    )
    assert all(row["source"]["paragraphs"][0]["section_index"] == 2 for row in inputs)


def test_qasper_total_sample_prefers_unexposed_papers_and_marks_no_gold_in_inputs(
    monkeypatch, tmp_path
):
    from page_cite_rag.qasper import Annotation, Paper, Passage, Question

    @dataclass(frozen=True)
    class Reference:
        text: str

    papers = []
    references = {}
    for index in range(8):
        paper_id = f"paper-{index}"
        question_id = f"question-{index}"
        text = f"Evidence {index}."
        passage = Passage(f"{paper_id}/s0001/p0000", 1, 0, text)
        annotation = Annotation(False, (text,))
        question = Question(question_id, f"Question {index}?", (annotation,))
        papers.append(Paper(paper_id, (passage,), (question,), (None, "Methods")))
        references[question_id] = [Reference("Answer")]

    monkeypatch.setattr(prepare_reading_eval, "read_snapshot", lambda *_: iter(papers))
    monkeypatch.setattr(
        prepare_reading_eval,
        "read_answer_references",
        lambda _directory, _split, _question_ids: references,
    )

    inputs, gold = prepare_reading_eval._qasper_cases(
        tmp_path, {f"paper-{index}" for index in range(5)}, count=4
    )

    assert len(inputs) == len(gold) == 4
    assert len({row["source_id"] for row in inputs}) == 4
    selected_sources = {row["source_id"] for row in inputs}
    assert {"paper-5", "paper-6", "paper-7"} <= selected_sources
    assert len(selected_sources & {f"paper-{index}" for index in range(5)}) == 1
    assert all("references" not in row for row in inputs)


def test_longbench_v2_preparation_preserves_question_without_repeating_output_contract():
    original_question = "Which option is supported by the document?"
    row = {
        "_id": "longbench-case",
        "domain": "Single-Document QA",
        "sub_domain": "test",
        "difficulty": "easy",
        "length": "long",
        "context": "evidence " * 8_000,
        "question": original_question,
        "choice_A": "first",
        "choice_B": "second",
        "choice_C": "third",
        "choice_D": "fourth",
        "answer": "B",
    }

    inputs, gold, _metadata = prepare_reading_eval._longbench_v2_cases([row], set(), count=1)

    assert len(inputs) == len(gold) == 1
    assert inputs[0]["question"] == (
        f"{original_question}\n\n(A) first\n(B) second\n(C) third\n(D) fourth"
    )
    assert "The correct answer is" not in inputs[0]["question"]
    assert gold[0]["answer"] == "B"


def test_longbench_v2_sample_prefers_unexposed_rows_then_fills_from_exposed():
    rows = []
    for index in range(3):
        rows.append(
            {
                "_id": f"longbench-case-{index}",
                "domain": "Single-Document QA",
                "sub_domain": "test",
                "difficulty": "easy",
                "length": "long",
                "context": f"document {index} " + ("evidence " * 8_000),
                "question": f"Question {index}?",
                "choice_A": "first",
                "choice_B": "second",
                "choice_C": "third",
                "choice_D": "fourth",
                "answer": "B",
            }
        )

    inputs, _gold, metadata = prepare_reading_eval._longbench_v2_cases(
        rows, {"longbench-case-0", "longbench-case-1"}, count=2
    )

    assert len(inputs) == 2
    assert metadata["selected"][0]["previously_exposed"] is False
    assert sum(item["previously_exposed"] for item in metadata["selected"]) == 1


def test_multifieldqa_sample_prefers_unexposed_rows_then_fills_from_exposed(tmp_path):
    path = tmp_path / "data.zip"
    rows = [
        {
            "_id": f"mfqa-{index}",
            "context": f"Context {index}",
            "input": f"Q{index}?",
            "answers": [f"A{index}"],
        }
        for index in range(3)
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "data/multifieldqa_zh.jsonl",
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        )

    inputs, gold, metadata = prepare_reading_eval._multifieldqa_zh_cases(
        path, {"mfqa-0", "mfqa-1"}, count=2
    )

    assert len(inputs) == len(gold) == 2
    assert metadata["selected"][0]["previously_exposed"] is False
    assert sum(item["previously_exposed"] for item in metadata["selected"]) == 1


def test_longbench_v2_full_track_separates_choices_from_gold_and_records_exposure():
    row = {
        "_id": "longbench-full-case",
        "domain": "Single-Document QA",
        "sub_domain": "science",
        "difficulty": "hard",
        "length": "long",
        "context": "The source says $5 is due.",
        "question": "Which statement is supported?",
        "choice_A": "A first claim",
        "choice_B": "A second claim",
        "choice_C": "A third claim",
        "choice_D": "A fourth claim",
        "answer": "C",
    }

    inputs, gold, metadata = prepare_reading_eval._longbench_v2_full_cases(
        [row], {row["_id"]}, expected_count=1
    )

    assert len(inputs) == len(gold) == 1
    assert inputs[0]["benchmark_question"] == row["question"]
    assert inputs[0]["choices"] == {letter: row[f"choice_{letter}"] for letter in "ABCD"}
    assert inputs[0]["question"].startswith("Which statement is supported?\n\n(A)")
    assert "answer" not in inputs[0]
    assert gold[0]["answer"] == "C"
    assert metadata["cases"][0]["previously_exposed"] is True


def test_exposure_scan_ignores_frozen_source_cache_and_binary_indexes(tmp_path):
    identifier = "a" * 40
    frozen = tmp_path / "data" / "frozen-source"
    frozen.mkdir(parents=True)
    (frozen / "manifest.json").write_text('{"state":"frozen_inputs_not_run"}', encoding="utf-8")
    (frozen / "inputs.jsonl").write_text(identifier, encoding="utf-8")
    (frozen / "gold.jsonl").write_text(identifier, encoding="utf-8")
    (frozen / "reading.sqlite3").write_bytes(identifier.encode("ascii"))
    result = tmp_path / "responses" / "answer.json"
    result.parent.mkdir()
    result.write_text(json.dumps({"case_id": identifier}), encoding="utf-8")

    exposed, scanned, skipped = prepare_reading_eval._exposed_ids(tmp_path, {identifier})

    assert exposed == {identifier}
    assert scanned == 1
    assert skipped == 3


def test_exposure_scan_does_not_skip_completed_run_named_reading_eval(tmp_path):
    identifier = "b" * 40
    run = tmp_path / "reading-eval-old-run"
    run.mkdir()
    (run / "manifest.json").write_text('{"state":"completed"}', encoding="utf-8")
    (run / "inputs.jsonl").write_text(identifier, encoding="utf-8")
    (run / "gold.jsonl").write_text(identifier, encoding="utf-8")

    exposed, scanned, skipped = prepare_reading_eval._exposed_ids(tmp_path, {identifier})

    assert exposed == {identifier}
    assert scanned == 3
    assert skipped == 0


def _write_product_freeze_fixture(tmp_path, monkeypatch, *, valid_source_hash=True):
    import hashlib

    monkeypatch.setattr(prepare_reading_eval, "ROOT", tmp_path)
    (tmp_path / "uv.lock").write_text("frozen test lock\n", encoding="utf-8")
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    base_inputs = [{"case_id": "benchmark-1", "dataset": "qasper"}]
    base_gold = [{"case_id": "benchmark-1", "references": []}]
    for name, rows in (("inputs.jsonl", base_inputs), ("gold.jsonl", base_gold)):
        payload = prepare_reading_eval._jsonl_bytes(rows)
        (benchmark / name).write_bytes(payload)
    manifest = {
        "format": "base-eval/v1",
        "files": {
            name: {
                "sha256": prepare_reading_eval._file_sha256(benchmark / name),
                "bytes": (benchmark / name).stat().st_size,
                "rows": 1,
            }
            for name in ("inputs.jsonl", "gold.jsonl")
        },
        "cases": [{"case_id": "benchmark-1", "dataset": "qasper"}],
    }
    (benchmark / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    source = tmp_path / "source.txt"
    source.write_text("A public source statement.", encoding="utf-8")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if not valid_source_hash:
        source_hash = "0" * 64
    cases = []
    gold = []
    for index in range(12):
        case_id = f"product-{index:02d}"
        cases.append(
            {
                "case_id": case_id,
                "dataset": "product-functional",
                "source_id": "one-document",
                "question": f"Question {index}?",
                "question_language": "en",
                "source_language": "en",
                "language": "en",
                "task_category": "fact_detail",
                "source": {
                    "format": "file",
                    "relative_path": "source.txt",
                    "sha256": source_hash,
                },
            }
        )
        gold.append(
            {
                "case_id": case_id,
                "expected_facts": [f"Fact {index}"],
                "source_locations": ["paragraph 1"],
            }
        )
    cases_path = tmp_path / "product-cases.jsonl"
    gold_path = tmp_path / "product-gold.jsonl"
    cases_path.write_bytes(prepare_reading_eval._jsonl_bytes(cases))
    gold_path.write_bytes(prepare_reading_eval._jsonl_bytes(gold))
    (tmp_path / "sources.json").write_text(
        json.dumps(
            {
                "one-document": {
                    "relative_path": "source.txt",
                    "sha256": source_hash,
                    "citation": "Test source",
                    "rights_status": "test only",
                    "sharing": "private",
                }
            }
        ),
        encoding="utf-8",
    )
    return benchmark, cases_path, gold_path


def test_product_freeze_combines_existing_benchmark_and_separate_gold(tmp_path, monkeypatch):
    benchmark, cases_path, gold_path = _write_product_freeze_fixture(tmp_path, monkeypatch)
    output = tmp_path / "combined"

    manifest = prepare_reading_eval.freeze_product_combination(
        benchmark, cases_path, gold_path, output
    )

    inputs = [json.loads(line) for line in (output / "inputs.jsonl").read_text().splitlines()]
    gold = [json.loads(line) for line in (output / "gold.jsonl").read_text().splitlines()]
    assert len(inputs) == len(gold) == 13
    assert "expected_facts" not in inputs[-1]
    assert gold[-1]["expected_facts"] == ["Fact 11"]
    assert manifest["product_case_scope"]["case_count"] == 12
    assert manifest["product_case_scope"]["all_cases_previously_exposed"] is True
    assert (output.stat().st_mode & 0o777) == 0o700
    assert (output / "gold.jsonl").stat().st_mode & 0o777 == 0o600


def test_product_freeze_rejects_source_hash_mismatch_without_writing_output(tmp_path, monkeypatch):
    benchmark, cases_path, gold_path = _write_product_freeze_fixture(
        tmp_path, monkeypatch, valid_source_hash=False
    )
    output = tmp_path / "combined"

    try:
        prepare_reading_eval.freeze_product_combination(benchmark, cases_path, gold_path, output)
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("product source hash mismatch must stop freezing")
    assert not output.exists()
