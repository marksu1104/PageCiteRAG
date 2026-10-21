import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts/run_reading_eval.py"
_RUNNER_SPEC = importlib.util.spec_from_file_location("run_reading_eval", _RUNNER_PATH)
run_reading_eval = importlib.util.module_from_spec(_RUNNER_SPEC)
sys.modules[_RUNNER_SPEC.name] = run_reading_eval
_RUNNER_SPEC.loader.exec_module(run_reading_eval)


def test_missing_gpu_selection_stops_before_loading_inputs(monkeypatch, capsys):
    monkeypatch.setattr(run_reading_eval, "GPU_UUID", "")
    monkeypatch.setattr(sys, "argv", [str(_RUNNER_PATH), "--phase", "prepare"])
    monkeypatch.setattr(
        run_reading_eval, "load_frozen", lambda *_: pytest.fail("loaded inputs without a GPU")
    )

    assert run_reading_eval.main() == 1
    assert "PAGE_CITE_RAG_GPU_UUID" in capsys.readouterr().err


def test_frozen_loader_never_parses_gold(monkeypatch, tmp_path):
    rows = [{"case_id": f"case-{index}"} for index in range(24)]
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    inputs = tmp_path / "inputs.jsonl"
    inputs.write_bytes(payload)
    manifest = {
        "files": {
            "inputs.jsonl": {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        },
        "cases": [{"case_id": row["case_id"]} for row in rows],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    original_read = run_reading_eval.read_jsonl

    def read_inputs_only(path):
        assert path.name == "inputs.jsonl"
        return original_read(path)

    monkeypatch.setattr(run_reading_eval, "read_jsonl", read_inputs_only)
    loaded_manifest, loaded_rows = run_reading_eval.load_frozen(tmp_path)

    assert loaded_manifest == manifest
    assert loaded_rows == rows


def test_qasper_evidence_uses_original_paragraph_not_quote_span():
    passage = "The method improves exact-match by 2.1 points under the stated setting."
    result = {
        "answer_claims": [{"citations": [{"block_id": "block-1", "quote": "2.1 points"}]}],
        "detail_claims": [{"citations": [{"block_id": "block-2", "quote": "additional context"}]}],
    }

    evidence = run_reading_eval._qasper_evidence_texts(
        result,
        {"block-1": "passage-1", "block-2": "passage-2"},
        {"passage-1": passage, "passage-2": "A detail-only paragraph."},
    )

    assert evidence == (passage,)


def test_qasper_evidence_rejects_unmapped_citations():
    result = {"answer_claims": [{"citations": [{"block_id": "unknown"}]}]}

    with pytest.raises(ValueError, match="no original paragraph mapping"):
        run_reading_eval._qasper_evidence_texts(result, {}, {})


def test_search_result_roundtrip_validates_json_arrays_in_strict_mode():
    block_id = "b" * 64
    value = {
        "document_id": "a" * 64,
        "index_id": "index-id",
        "method": "hybrid",
        "query": "question",
        "hits": [
            {
                "rank": 1,
                "score": 0.5,
                "block": {
                    "document_id": "a" * 64,
                    "block_id": block_id,
                    "ordinal": 1,
                    "unit_index": 1,
                    "page_number": None,
                    "line_start": 1,
                    "line_end": 1,
                    "char_start": 0,
                    "char_end": 5,
                    "text": "hello",
                },
            }
        ],
        "elapsed_ms": 1.0,
        "branches": [
            {
                "name": "lexical",
                "index_id": "lexical-index",
                "elapsed_ms": 0.1,
                "block_ids": [block_id],
            }
        ],
    }

    search = run_reading_eval._search_result_from_json(value)

    assert len(search.hits) == 1
    assert search.branches[0].block_ids == (block_id,)


def test_longbench_v2_answer_extraction_requires_official_phrase():
    assert run_reading_eval._longbench_v2_choice("The correct answer is (C).") == "C"
    assert run_reading_eval._longbench_v2_choice("The correct answer is D") == "D"
    assert run_reading_eval._longbench_v2_choice("I think the answer is A") is None


@pytest.mark.parametrize(
    ("dataset", "expected"),
    [
        ("qasper", "short_answer"),
        ("longbench-v1-multifieldqa-zh", "short_answer"),
        ("longbench-v2-single-document-qa", "multiple_choice"),
    ],
)
def test_benchmark_dataset_uses_its_scoring_answer_format(dataset, expected):
    assert run_reading_eval._answer_format_for({"dataset": dataset}) == expected


def test_unknown_benchmark_has_no_implicit_answer_format():
    with pytest.raises(ValueError, match="unknown frozen dataset task"):
        run_reading_eval._answer_format_for({"dataset": "new-task"})


def test_longbench_chinese_f1_matches_official_punctuation_normalization():
    score = run_reading_eval._longbench_zh_f1("逐字引用，保留原文！", ["逐字引用保留原文"])

    assert score == 1.0


def test_longbench_chinese_f1_removes_official_corner_brackets():
    score = run_reading_eval._longbench_zh_f1("【中文】引用", ["中文引用"])

    assert score == 1.0


def test_embedding_budget_is_checked_before_dispatch(monkeypatch):
    class FakeTransport:
        def __init__(self):
            self.dispatches = 0

        def token_ids(self, text):
            return list(range(len(text)))

        def embed(self, texts):
            self.dispatches += 1
            return texts

    transport = FakeTransport()
    counter = run_reading_eval._CountingTransport(transport)
    monkeypatch.setattr(run_reading_eval, "PREP_DOCUMENT_EMBED_TOKEN_LIMIT", 3)

    with pytest.raises(ValueError, match="budget exceeded before dispatch"):
        counter.embed(["four"])

    assert transport.dispatches == 0
    assert counter.document_tokens == 0


def test_score_run_compares_only_selected_arms_and_labels_case_scope(tmp_path):
    case_id = "longbench-v2-case"
    manifest = {"cases": [{"case_id": case_id, "dataset": "longbench-v2-single-document-qa"}]}
    manifest["files"] = {"inputs.jsonl": {"rows": 24}}
    inputs = [{"case_id": case_id}]
    gold = [{"case_id": case_id, "answer": "B"}]
    for arm, answer in (("fixed-rag", "B"), ("single-agent", "A")):
        arm_dir = tmp_path / arm
        arm_dir.mkdir()
        (arm_dir / f"{case_id}.json").write_text(
            json.dumps(
                {
                    "arm": arm,
                    "case_id": case_id,
                    "execution": "ok",
                    "result": {"answer": f"The correct answer is ({answer})."},
                    "gpu_diagnostics": {"observed_gpu_peak": {}},
                }
            ),
            encoding="utf-8",
        )

    result = run_reading_eval.score_run(
        tmp_path,
        manifest,
        inputs,
        gold,
        arms=("fixed-rag", "single-agent"),
    )

    assert result["state"] == "scored_complete"
    assert result["evaluation_scope"] == {
        "kind": "selected_case_and_arm_slice",
        "case_ids": [case_id],
        "arms": ["fixed-rag", "single-agent"],
        "case_count": 1,
        "frozen_dataset_case_count": 24,
    }
    assert set(result["arms"]) == {"fixed-rag", "single-agent"}
    assert (
        result["arms"]["fixed-rag"]["datasets"]["longbench-v2-single-document-qa"]["per_case"][0][
            "case_id"
        ]
        == case_id
    )
    assert (
        result["arms"]["fixed-rag"]["datasets"]["longbench-v2-single-document-qa"]["metrics"][
            "accuracy"
        ]
        == 1.0
    )
    assert (
        result["arms"]["single-agent"]["datasets"]["longbench-v2-single-document-qa"]["metrics"][
            "accuracy"
        ]
        == 0.0
    )


def test_score_run_rejects_raw_results_outside_selected_cases(tmp_path):
    arm_dir = tmp_path / "fixed-rag"
    arm_dir.mkdir()
    (arm_dir / "unexpected.json").write_text(
        json.dumps({"arm": "fixed-rag", "case_id": "unexpected"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="outside the selected evaluation scope"):
        run_reading_eval.score_run(
            tmp_path,
            {"cases": [{"case_id": "selected", "dataset": "longbench-v2-single-document-qa"}]},
            [{"case_id": "selected"}],
            [{"case_id": "selected", "answer": "A"}],
            arms=("fixed-rag",),
        )
