import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

module_path = Path(__file__).parents[1] / "scripts" / "compare_reading_eval.py"
spec = importlib.util.spec_from_file_location("compare_reading_eval", module_path)
assert spec is not None and spec.loader is not None
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_paired_summary_reports_directional_outcomes_and_delta() -> None:
    result = comparison.paired_summary(
        [0.1, 0.2, 0.3],
        [0.2, 0.2, 0.1],
        seed=7,
        replicates=2_000,
    )

    assert result["n"] == 3
    assert result["right_wins"] == 1
    assert result["ties"] == 1
    assert result["right_losses"] == 1
    assert result["mean_delta_right_minus_left"] == pytest.approx(-1 / 30)
    low, high = result["paired_bootstrap_percentile_95_ci"]
    assert low <= result["mean_delta_right_minus_left"] <= high


def test_paired_summary_rejects_unpaired_values() -> None:
    with pytest.raises(ValueError, match="non-empty and have equal length"):
        comparison.paired_summary([1.0], [], seed=7)


def test_paired_cluster_summary_uses_source_cluster_as_resampling_unit() -> None:
    result = comparison.paired_cluster_summary(
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        ["paper-a", "paper-a", "paper-b"],
        seed=7,
        replicates=2_000,
    )

    assert result["case_count"] == 3
    assert result["cluster_count"] == 2
    assert result["left_cluster_mean"] == pytest.approx(0.5)
    assert result["right_cluster_mean"] == pytest.approx(0.5)
    assert result["mean_delta_right_minus_left"] == pytest.approx(0.0)
    assert result["right_cluster_wins"] == 1
    assert result["right_cluster_losses"] == 1
    assert result["tied_clusters"] == 0


def test_paired_cluster_summary_rejects_missing_cluster_ids() -> None:
    with pytest.raises(ValueError, match="source clusters"):
        comparison.paired_cluster_summary([1.0], [1.0], [])


def test_document_cluster_uses_source_hash_when_cluster_is_not_annotated() -> None:
    source_hash = "a" * 64

    assert comparison._document_cluster({"case_id": "case-a", "source_sha256": source_hash}) == (
        f"source-sha256:{source_hash}"
    )
    assert (
        comparison._document_cluster(
            {
                "case_id": "case-b",
                "document_cluster": "explicit-paper-id",
                "source_sha256": source_hash,
            }
        )
        == "explicit-paper-id"
    )


def test_derived_qasper_reports_refusal_metrics_without_lexical_f1() -> None:
    assert comparison._metrics_for_dataset("derived-query-qasper-source-en") == (
        "correct_refusal",
        "false_refusal",
    )


def test_derived_longbench_does_not_claim_an_unavailable_metric() -> None:
    assert (
        comparison._metrics_for_dataset("derived-query-longbench-v1-multifieldqa-zh-source-zh-Hant")
        == ()
    )


def test_product_functional_reports_operational_metrics_and_item_coverage() -> None:
    # item_coverage is present only for rubric rows with anchored items (plan 5.5).
    assert comparison._metrics_for_dataset("product-functional") == (
        "answer_available",
        "refused",
        "item_coverage",
    )


def test_case_id_join_is_order_independent() -> None:
    expected_ids = ["case-a", "case-b"]
    rows = [
        {"case_id": "case-b", "answer_f1": 0.2},
        {"case_id": "case-a", "answer_f1": 0.8},
    ]

    indexed = comparison._rows_by_case_id(rows, expected_ids, "test")

    assert [indexed[case_id]["answer_f1"] for case_id in expected_ids] == [0.8, 0.2]


@pytest.mark.parametrize(
    "rows",
    [
        [{"case_id": "case-a"}, {"case_id": "case-a"}],
        [{"case_id": "case-a"}],
        [{"case_id": "case-a"}, {"case_id": "case-c"}],
        [{"answer_f1": 0.8}, {"case_id": "case-b"}],
    ],
)
def test_case_id_join_rejects_missing_duplicate_and_unexpected_ids(rows) -> None:
    with pytest.raises(ValueError):
        comparison._rows_by_case_id(rows, ["case-a", "case-b"], "test")


def test_paired_run_builder_creates_blinded_source_linked_review(tmp_path, monkeypatch, capsys):
    frozen = tmp_path / "frozen"
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    for root in (frozen, reference_root / "routed-reading", candidate_root / "routed-reading"):
        root.mkdir(parents=True)
    case_id = "paper-case"
    case = {
        "case_id": case_id,
        "dataset": "product-functional",
        "source_id": "paper-1",
        "source_sha256": "a" * 64,
    }
    input_row = {
        **case,
        "question": "What result did the paper report?",
        "source": {"format": "file", "relative_path": "papers/paper.pdf"},
    }
    gold_row = {
        "case_id": case_id,
        "necessary_atoms": [{"atom_id": "a1", "text": "The result is 42."}],
    }
    inputs_payload = (json.dumps(input_row) + "\n").encode()
    gold_payload = (json.dumps(gold_row) + "\n").encode()
    (frozen / "inputs.jsonl").write_bytes(inputs_payload)
    (frozen / "gold.jsonl").write_bytes(gold_payload)
    (frozen / "manifest.json").write_text(
        json.dumps(
            {
                "cases": [case],
                "files": {
                    "inputs.jsonl": {
                        "bytes": len(inputs_payload),
                        "sha256": hashlib.sha256(inputs_payload).hexdigest(),
                    },
                    "gold.jsonl": {
                        "bytes": len(gold_payload),
                        "sha256": hashlib.sha256(gold_payload).hexdigest(),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    shared_static = {
        "models": {"generator": "pinned-model"},
        "retrieval": {"method": "hybrid"},
        "embedding_input": {"max_tokens": 512},
        "gpu": {"uuid": "gpu"},
        "preparation_budget": {"documents": 10_000_000},
        "frozen_inputs_sha256": "b" * 64,
        "dataset_manifest_sha256": "c" * 64,
        "run_selection": {"arms": ["routed-reading"], "case_ids": [case_id]},
    }

    def write_run(root, item_coverage, answer):
        scores = {
            "state": "scored_complete",
            "evaluation_scope": {"arms": ["routed-reading"], "case_ids": [case_id]},
            "arms": {
                "routed-reading": {
                    "datasets": {
                        "product-functional": {
                            "per_case": [{"case_id": case_id, "item_coverage": item_coverage}]
                        }
                    },
                    "generation_calls_attempted": 1,
                    "input_tokens_lower_bound": 20,
                    "output_tokens_lower_bound": 5,
                    "rows_with_unknown_output_usage": 0,
                    "case_elapsed_ms_p50": 100,
                    "case_elapsed_ms_p95": 100,
                }
            },
        }
        (root / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
        (root / "configuration.json").write_text(
            json.dumps({"static": shared_static}), encoding="utf-8"
        )
        raw = {
            "execution": "ok",
            "block_to_passage_id": {"block-1": "passage-1"},
            "result": {
                "status": "answered",
                "answer": answer,
                "answer_claims": [
                    {
                        "text": answer,
                        "citations": [
                            {
                                "block_id": "block-1",
                                "page_number": 2,
                                "quote": "The result is 42.",
                            }
                        ],
                    }
                ],
                "detail_claims": [],
                "limitations": [],
            },
        }
        (root / "routed-reading" / f"{case_id}.json").write_text(json.dumps(raw), encoding="utf-8")

    write_run(reference_root, 0.5, "The result was 40.")
    write_run(candidate_root, 1.0, "The result is 42.")

    report, review_rows, review_key = comparison.build_paired_run_comparison(
        reference_root, candidate_root, frozen, seed=7, replicates=100
    )

    metric = report["datasets"]["product-functional"]["metrics"]["item_coverage"]
    assert metric["mean_delta_right_minus_left"] == pytest.approx(0.5)
    assert report["method"]["independent_human_review"] is False
    assert "review_key" not in report
    assert review_rows[0]["review"]["atom_verdicts"][0]["verdict"] is None
    assert review_rows[0]["source_reference"]["relative_path"] == "papers/paper.pdf"
    assert "passage_id" in review_rows[0]["submission_1"]["claims"][0]["citations"][0]
    assert set(review_key[case_id]) == {"submission_1", "submission_2"}

    output = tmp_path / "paired.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_reading_eval.py",
            str(reference_root),
            "--candidate-run-root",
            str(candidate_root),
            "--frozen-root",
            str(frozen),
            "--output",
            str(output),
        ],
    )
    assert comparison.main() == 0
    assert json.loads(output.read_text(encoding="utf-8"))["case_count"] == 1
    review_path = output.with_name("semantic-review.jsonl")
    key_path = output.with_name("semantic-review-key.json")
    assert len(review_path.read_text(encoding="utf-8").splitlines()) == 1
    assert json.loads(key_path.read_text(encoding="utf-8"))[case_id] == review_key[case_id]
    assert "review_key" not in json.loads(output.read_text(encoding="utf-8"))
    assert capsys.readouterr().out


def test_paired_run_builder_rejects_incomplete_or_mismatched_case_runs(tmp_path):
    frozen = tmp_path / "frozen"
    reference_root = tmp_path / "reference"
    candidate_root = tmp_path / "candidate"
    frozen.mkdir()
    reference_root.mkdir()
    candidate_root.mkdir()
    (frozen / "manifest.json").write_text(
        json.dumps({"cases": [{"case_id": "a"}]}), encoding="utf-8"
    )
    (frozen / "inputs.jsonl").write_text("{}\n", encoding="utf-8")
    (frozen / "gold.jsonl").write_text("{}\n", encoding="utf-8")
    for root in (reference_root, candidate_root):
        shared_static = {
            "models": {"generator": "pinned-model"},
            "retrieval": {"method": "hybrid"},
            "embedding_input": {"max_tokens": 512},
            "gpu": {"uuid": "gpu"},
            "preparation_budget": {"documents": 10_000_000},
            "frozen_inputs_sha256": "b" * 64,
            "dataset_manifest_sha256": "c" * 64,
            "run_selection": {"arms": ["routed-reading"], "case_ids": ["other"]},
        }
        (root / "configuration.json").write_text(
            json.dumps({"static": shared_static}), encoding="utf-8"
        )
        (root / "scores.json").write_text(
            json.dumps(
                {
                    "state": "scored_complete",
                    "evaluation_scope": {"arms": ["routed-reading"], "case_ids": ["other"]},
                    "arms": {"routed-reading": {"datasets": {}}},
                }
            ),
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="same cases"):
        comparison.build_paired_run_comparison(reference_root, candidate_root, frozen)


@pytest.mark.parametrize(
    ("planner_ready", "expected_planner_calls", "expected_planner_input_tokens"),
    [
        (
            {"planner_calls_attempted": 2, "input_tokens": 50, "output_tokens": 10},
            2,
            50,
        ),
        (
            {
                "planner_calls_attempted": 0,
                "new_generation_dispatches": 0,
                "reused_historical_plans": 59,
                "degraded_plans": 13,
                "source_results_sha256": "a" * 64,
            },
            0,
            0,
        ),
    ],
)
def test_quality_v1_builds_three_pairs_and_accepts_reused_planner_costs(
    tmp_path,
    planner_ready,
    expected_planner_calls,
    expected_planner_input_tokens,
):
    run_root = tmp_path / "run"
    frozen_root = tmp_path / "frozen"
    run_root.mkdir()
    frozen_root.mkdir()
    cases = [
        {"case_id": "case-a", "document_cluster": "paper-a", "previously_exposed": False},
        {"case_id": "case-b", "document_cluster": "paper-a", "previously_exposed": True},
    ]
    (frozen_root / "manifest.json").write_text(json.dumps({"cases": cases}), encoding="utf-8")
    arms = comparison.QUALITY_V1_ARMS
    values = {
        arms[0]: [0.1, 0.2],
        arms[1]: [0.2, 0.2],
        arms[2]: [0.3, 0.1],
    }
    scores = {
        "state": "scored_complete",
        "independent_human_review": False,
        "evaluation_scope": {"arms": list(arms)},
        "arms": {},
    }
    for arm, predictions in values.items():
        scores["arms"][arm] = {
            "datasets": {
                "qasper": {
                    "per_case": [
                        {"case_id": case_id, "answer_f1": value}
                        for case_id, value in zip(("case-a", "case-b"), predictions)
                    ]
                }
            },
            "generation_calls_attempted": 2,
            "input_tokens_lower_bound": 100,
            "output_tokens_lower_bound": 10,
            "rows_with_unknown_output_usage": 0,
            "case_elapsed_ms_p50": 1,
            "case_elapsed_ms_p95": 2,
            "model_call_elapsed_ms_p50": 1,
            "model_call_elapsed_ms_p95": 2,
        }
        arm_dir = run_root / arm
        arm_dir.mkdir()
        for case_id in ("case-a", "case-b"):
            (arm_dir / f"{case_id}.json").write_text(
                json.dumps({"execution": "ok", "agent_trace": None}), encoding="utf-8"
            )
    (run_root / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    (run_root / "reading-plans").mkdir()
    (run_root / "planned-retrieval").mkdir()
    (run_root / "reading-plans" / "ready.json").write_text(
        json.dumps(planner_ready), encoding="utf-8"
    )
    (run_root / "planned-retrieval" / "ready.json").write_text(
        json.dumps(
            {
                "embedding_usage": {
                    "requests_attempted": 4,
                    "query_input_tokens_attempted": 200,
                }
            }
        ),
        encoding="utf-8",
    )

    report = comparison.build_comparison(run_root, frozen_root, replicates=200)

    result = report["datasets"]["qasper"]
    assert set(result["pairwise_comparisons"]) == {
        "fixed-rag__vs__decomposition-fixed-rag",
        "decomposition-fixed-rag__vs__decomposition-agent",
        "fixed-rag__vs__decomposition-agent",
    }
    assert result["previously_exposed_cases"] == 1
    assert report["shared_costs"]["planner_calls_attempted"] == expected_planner_calls
    assert report["shared_costs"]["planner_input_tokens"] == expected_planner_input_tokens
    assert (
        report["runtime"]["decomposition-agent"]["deployment_generation_calls_with_shared_planner"]
        == 2 + expected_planner_calls
    )


def test_quality_v2_builds_all_six_pairs_and_allocates_shared_planner_cost(tmp_path):
    run_root = tmp_path / "run"
    frozen_root = tmp_path / "frozen"
    run_root.mkdir()
    frozen_root.mkdir()
    case_id = "case-a"
    (frozen_root / "manifest.json").write_text(
        json.dumps({"cases": [{"case_id": case_id, "document_cluster": "doc-a"}]}),
        encoding="utf-8",
    )
    arms = comparison.QUALITY_V2_ARMS
    scores = {
        "state": "scored_complete",
        "independent_human_review": False,
        "evaluation_scope": {"arms": list(arms)},
        "arms": {},
    }
    for index, arm in enumerate(arms):
        scores["arms"][arm] = {
            "datasets": {"qasper": {"per_case": [{"case_id": case_id, "answer_f1": index / 4}]}},
            "generation_calls_attempted": index + 1,
            "input_tokens_lower_bound": 10 * (index + 1),
            "output_tokens_lower_bound": index + 1,
            "rows_with_unknown_output_usage": 0,
            "case_elapsed_ms_p50": 1,
            "case_elapsed_ms_p95": 2,
            "model_call_elapsed_ms_p50": 1,
            "model_call_elapsed_ms_p95": 2,
        }
        arm_dir = run_root / arm
        arm_dir.mkdir()
        (arm_dir / f"{case_id}.json").write_text(
            json.dumps({"execution": "ok", "agent_trace": None}), encoding="utf-8"
        )
    (run_root / "scores.json").write_text(json.dumps(scores), encoding="utf-8")
    plans = run_root / "reading-plans"
    retrieval = run_root / "planned-retrieval"
    plans.mkdir()
    retrieval.mkdir()
    (plans / "ready.json").write_text(
        json.dumps({"planner_calls_attempted": 1, "input_tokens": 10, "output_tokens": 3}),
        encoding="utf-8",
    )
    (retrieval / "ready.json").write_text(
        json.dumps(
            {
                "embedding_usage": {
                    "requests_attempted": 2,
                    "query_input_tokens_attempted": 10,
                }
            }
        ),
        encoding="utf-8",
    )

    report = comparison.build_comparison(run_root, frozen_root, replicates=100)

    pairs = report["datasets"]["qasper"]["pairwise_comparisons"]
    assert len(pairs) == 6
    assert (
        report["runtime"]["fixed-rag-followup-once"][
            "deployment_generation_calls_with_shared_planner"
        ]
        == 3
    )
