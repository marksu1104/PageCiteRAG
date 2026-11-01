"""Create a source-cluster-paired summary from a completed reading-eval run."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

ARM_PAIR = ("fixed-rag-reranked", "single-agent")
QUALITY_V1_ARMS = (
    "fixed-rag",
    "decomposition-fixed-rag",
    "decomposition-agent",
)
QUALITY_V2_ARMS = (
    "fixed-rag",
    "fixed-rag-followup-once",
    "decomposition-fixed-rag",
    "decomposition-agent",
)
SEED = 20261027
REPLICATES = 5_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_summary(
    left: list[float],
    right: list[float],
    *,
    seed: int,
    replicates: int = REPLICATES,
) -> dict[str, float | int]:
    """Summarize paired values when each row is an independent sampling unit."""
    if not left or len(left) != len(right):
        raise ValueError("paired samples must be non-empty and have equal length")
    if replicates < 1:
        raise ValueError("replicates must be positive")

    differences = [right_value - left_value for left_value, right_value in zip(left, right)]
    rng = random.Random(seed)
    size = len(differences)
    bootstrap_means = [
        sum(differences[rng.randrange(size)] for _ in range(size)) / size for _ in range(replicates)
    ]
    return {
        "n": size,
        "left_mean": sum(left) / size,
        "right_mean": sum(right) / size,
        "mean_delta_right_minus_left": sum(differences) / size,
        "right_wins": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "right_losses": sum(value < 0 for value in differences),
        "paired_bootstrap_percentile_95_ci": [
            _percentile(bootstrap_means, 0.025),
            _percentile(bootstrap_means, 0.975),
        ],
    }


def paired_cluster_summary(
    left: list[float],
    right: list[float],
    clusters: list[str],
    *,
    seed: int = SEED,
    replicates: int = REPLICATES,
) -> dict[str, float | int]:
    """Bootstrap paired cluster means, keeping all rows from a source together."""
    if not left or len(left) != len(right) or len(left) != len(clusters):
        raise ValueError("paired samples and source clusters must have equal non-zero length")
    if replicates < 1:
        raise ValueError("replicates must be positive")

    grouped: dict[str, list[tuple[float, float]]] = {}
    for left_value, right_value, cluster in zip(left, right, clusters):
        if not isinstance(cluster, str) or not cluster:
            raise ValueError("every paired case needs a non-empty source cluster")
        grouped.setdefault(cluster, []).append((left_value, right_value))

    cluster_values = []
    for rows in grouped.values():
        size = len(rows)
        cluster_values.append(
            (
                sum(row[0] for row in rows) / size,
                sum(row[1] for row in rows) / size,
            )
        )
    left_clusters = [row[0] for row in cluster_values]
    right_clusters = [row[1] for row in cluster_values]
    differences = [right_value - left_value for left_value, right_value in cluster_values]

    rng = random.Random(seed)
    cluster_count = len(cluster_values)
    bootstrap_means = [
        sum(differences[rng.randrange(cluster_count)] for _ in range(cluster_count)) / cluster_count
        for _ in range(replicates)
    ]
    return {
        "case_count": len(left),
        "cluster_count": cluster_count,
        "left_cluster_mean": sum(left_clusters) / cluster_count,
        "right_cluster_mean": sum(right_clusters) / cluster_count,
        "mean_delta_right_minus_left": sum(differences) / cluster_count,
        "right_cluster_wins": sum(value > 0 for value in differences),
        "tied_clusters": sum(value == 0 for value in differences),
        "right_cluster_losses": sum(value < 0 for value in differences),
        "paired_cluster_bootstrap_percentile_95_ci": [
            _percentile(bootstrap_means, 0.025),
            _percentile(bootstrap_means, 0.975),
        ],
    }


def _rows_by_case_id(rows: list[dict], expected_ids: list[str], label: str) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"missing case ID in {label}")
        if case_id in indexed:
            raise ValueError(f"duplicate case ID in {label}: {case_id}")
        indexed[case_id] = row
    if indexed.keys() != set(expected_ids):
        missing = set(expected_ids) - indexed.keys()
        unexpected = indexed.keys() - set(expected_ids)
        raise ValueError(
            f"case ID mismatch in {label}: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return indexed


def _numeric_pairs(
    left_rows: dict[str, dict],
    right_rows: dict[str, dict],
    case_ids: list[str],
    clusters_by_id: dict[str, str],
    metric: str,
) -> tuple[list[str], list[float], list[float], list[str]]:
    selected = []
    left_values = []
    right_values = []
    clusters = []
    for case_id in case_ids:
        left_value = left_rows[case_id].get(metric)
        right_value = right_rows[case_id].get(metric)
        if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
            selected.append(case_id)
            left_values.append(float(left_value))
            right_values.append(float(right_value))
            clusters.append(clusters_by_id[case_id])
    return selected, left_values, right_values, clusters


def _metrics_for_dataset(dataset: str) -> tuple[str, ...]:
    if dataset == "product-functional":
        return ("answer_available", "refused")
    if dataset == "qasper":
        return ("answer_f1", "evidence_f1", "correct_refusal", "false_refusal")
    if dataset == "longbench-v1-multifieldqa-zh":
        return ("answer_f1", "longbench_zh_t2s_f1_nonofficial_v1")
    if dataset.startswith("derived-query-qasper-"):
        return ("correct_refusal", "false_refusal")
    if dataset.startswith("derived-query-longbench-"):
        return ()
    if dataset.startswith("derived-query-"):
        return ("answer_available", "refused")
    return ("accuracy", "rouge_l")


def _document_cluster(case: dict) -> str:
    cluster = case.get("document_cluster")
    if isinstance(cluster, str) and cluster.strip():
        return cluster
    source_sha256 = case.get("source_sha256")
    if isinstance(source_sha256, str) and source_sha256.strip():
        return f"source-sha256:{source_sha256}"
    raise ValueError(f"case {case.get('case_id')} has no document cluster identity")


def build_comparison(
    run_root: Path,
    frozen_root: Path,
    *,
    seed: int = SEED,
    replicates: int = REPLICATES,
) -> dict:
    manifest_path = frozen_root / "manifest.json"
    scores_path = run_root / "scores.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    if scores.get("state") != "scored_complete":
        raise ValueError("comparison requires a complete scored run")
    if scores.get("independent_human_review") is not False:
        raise ValueError("unexpected human-review state")
    arms = tuple(scores.get("evaluation_scope", {}).get("arms", ()))
    if arms == ARM_PAIR:
        pairs = (ARM_PAIR,)
    elif arms == QUALITY_V1_ARMS:
        pairs = (
            ("fixed-rag", "decomposition-fixed-rag"),
            ("decomposition-fixed-rag", "decomposition-agent"),
            ("fixed-rag", "decomposition-agent"),
        )
    elif arms == QUALITY_V2_ARMS:
        pairs = (
            ("fixed-rag", "decomposition-fixed-rag"),
            ("fixed-rag", "fixed-rag-followup-once"),
            ("fixed-rag", "decomposition-agent"),
            ("decomposition-fixed-rag", "fixed-rag-followup-once"),
            ("decomposition-fixed-rag", "decomposition-agent"),
            ("fixed-rag-followup-once", "decomposition-agent"),
        )
    else:
        raise ValueError("scored run does not match a supported frozen arm protocol")

    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("frozen manifest has no cases")
    case_by_id = {case["case_id"]: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("frozen manifest contains duplicate case IDs")
    arm_scores = scores["arms"]
    dataset_names = set(arm_scores[arms[0]]["datasets"])
    if any(set(arm_scores[arm]["datasets"]) != dataset_names for arm in arms[1:]):
        raise ValueError("comparison arms have different scored dataset groups")

    comparisons = {}
    for dataset in sorted(dataset_names):
        case_ids = [row["case_id"] for row in arm_scores[arms[0]]["datasets"][dataset]["per_case"]]
        clusters_by_id = {case_id: _document_cluster(case_by_id[case_id]) for case_id in case_ids}
        pair_results = {}
        for left_arm, right_arm in pairs:
            left_dataset = arm_scores[left_arm]["datasets"][dataset]
            right_dataset = arm_scores[right_arm]["datasets"][dataset]
            left_ids = [row["case_id"] for row in left_dataset["per_case"]]
            right_ids = [row["case_id"] for row in right_dataset["per_case"]]
            if left_ids != case_ids or right_ids != case_ids:
                raise ValueError(f"paired case order differs in {dataset}")
            left_rows = _rows_by_case_id(
                left_dataset["per_case"], case_ids, f"{left_arm}/{dataset}"
            )
            right_rows = _rows_by_case_id(
                right_dataset["per_case"], case_ids, f"{right_arm}/{dataset}"
            )
            metrics = {}
            for metric in _metrics_for_dataset(dataset):
                selected, left_values, right_values, clusters = _numeric_pairs(
                    left_rows, right_rows, case_ids, clusters_by_id, metric
                )
                if not selected:
                    continue
                summary = paired_cluster_summary(
                    left_values,
                    right_values,
                    clusters,
                    seed=seed,
                    replicates=replicates,
                )
                summary["paired_case_ids"] = selected
                summary["excluded_unpaired_cases"] = len(case_ids) - len(selected)
                summary["interpretation"] = (
                    "descriptive paired source-cluster bootstrap; the percentile interval is not "
                    "a significance or equivalence test"
                )
                metrics[metric] = summary
            pair_results[f"{left_arm}__vs__{right_arm}"] = {
                "left_arm": left_arm,
                "right_arm": right_arm,
                "metrics": metrics,
            }
        dataset_result = {
            "case_count": len(case_ids),
            "source_cluster_count": len(set(clusters_by_id.values())),
            "previously_exposed_cases": sum(
                case_by_id[case_id].get("previously_exposed") is True for case_id in case_ids
            ),
        }
        if len(pairs) == 1:
            dataset_result["metrics"] = pair_results[f"{pairs[0][0]}__vs__{pairs[0][1]}"]["metrics"]
        else:
            dataset_result["pairwise_comparisons"] = pair_results
        comparisons[dataset] = dataset_result

    runtime = {}
    for arm in arms:
        arm_score = arm_scores[arm]
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((run_root / arm).glob("*.json"))
        ]
        traces = [record.get("agent_trace") for record in records]
        runtime[arm] = {
            "cases": len(records),
            "successful": sum(record.get("execution") == "ok" for record in records),
            "failed": sum(record.get("execution") != "ok" for record in records),
            "generation_calls": arm_score["generation_calls_attempted"],
            "input_tokens_lower_bound": arm_score["input_tokens_lower_bound"],
            "output_tokens_lower_bound": arm_score["output_tokens_lower_bound"],
            "output_usage_unknown_rows": arm_score["rows_with_unknown_output_usage"],
            "case_elapsed_ms_p50": arm_score["case_elapsed_ms_p50"],
            "case_elapsed_ms_p95": arm_score["case_elapsed_ms_p95"],
            "model_call_elapsed_ms_p50": arm_score["model_call_elapsed_ms_p50"],
            "model_call_elapsed_ms_p95": arm_score["model_call_elapsed_ms_p95"],
            "reader_tasks_started": sum(
                trace.get("reader_tasks_started", 0) for trace in traces if isinstance(trace, dict)
            ),
            "data_tool_calls": sum(
                len(trace.get("tool_trace", [])) for trace in traces if isinstance(trace, dict)
            ),
        }

    shared_costs = {}
    if arms in {QUALITY_V1_ARMS, QUALITY_V2_ARMS}:
        planner_path = run_root / "reading-plans" / "ready.json"
        retrieval_path = run_root / "planned-retrieval" / "ready.json"
        if not planner_path.is_file() or not retrieval_path.is_file():
            raise ValueError("reading-plan comparison is missing shared planner/retrieval ledgers")
        planner = json.loads(planner_path.read_text(encoding="utf-8"))
        retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
        planner_calls = planner.get("planner_calls_attempted")
        if not isinstance(planner_calls, int) or planner_calls < 0:
            planner_calls = planner.get("new_generation_dispatches")
        if not isinstance(planner_calls, int) or planner_calls < 0:
            raise ValueError("planner ledger is missing a valid call count")

        planner_input_tokens = planner.get("input_tokens")
        planner_output_tokens = planner.get("output_tokens")
        if planner_input_tokens is None or planner_output_tokens is None:
            if planner_calls != 0 or planner.get("new_generation_dispatches", 0) != 0:
                raise ValueError("planner ledger is missing token usage for dispatched calls")
            planner_input_tokens = 0
            planner_output_tokens = 0

        shared_costs = {
            "planner_calls_attempted": planner_calls,
            "planner_input_tokens": planner_input_tokens,
            "planner_output_tokens": planner_output_tokens,
            "reused_historical_plans": planner.get("reused_historical_plans", 0),
            "degraded_plans": planner.get("degraded_plans", 0),
            "source_results_sha256": planner.get("source_results_sha256"),
            "planner_cost_scope": (
                "current batch incremental cost; reused historical plans are not charged again"
            ),
            "planned_query_embedding_calls": (retrieval.get("embedding_usage") or {}).get(
                "requests_attempted", 0
            ),
            "planned_query_embedding_input_tokens": (retrieval.get("embedding_usage") or {}).get(
                "query_input_tokens_attempted", 0
            ),
            "shared_cost_accounting": (
                "actual shared cost counted once; add planner and planned-query cost to either "
                "decomposition arm for standalone deployment estimates"
            ),
        }
        planned_arms = ("decomposition-fixed-rag", "decomposition-agent")
        if arms == QUALITY_V2_ARMS:
            planned_arms = (
                "decomposition-fixed-rag",
                "fixed-rag-followup-once",
                "decomposition-agent",
            )
        for arm in planned_arms:
            runtime[arm]["deployment_generation_calls_with_shared_planner"] = (
                runtime[arm]["generation_calls"] + shared_costs["planner_calls_attempted"]
            )
            runtime[arm]["deployment_input_tokens_with_shared_planner"] = (
                runtime[arm]["input_tokens_lower_bound"] + shared_costs["planner_input_tokens"]
            )
            runtime[arm]["deployment_output_tokens_with_shared_planner"] = (
                runtime[arm]["output_tokens_lower_bound"] + shared_costs["planner_output_tokens"]
            )

    return {
        "state": "paired_comparison_complete",
        "source_artifacts": {
            "manifest_sha256": _sha256(manifest_path),
            "scores_sha256": _sha256(scores_path),
            "analysis_script_sha256": _sha256(Path(__file__).resolve()),
        },
        "method": {
            "arms": list(arms),
            "pairwise_comparisons": [list(pair) for pair in pairs],
            "resampling": "paired bootstrap over source document clusters",
            "replicates": replicates,
            "seed": seed,
            "interval": "descriptive 95% percentile interval; not a significance test",
            "failed_execution_scores": "preserved by the scorer; missing outputs are not dropped",
            "dataset_metrics_combined": False,
            "human_review": False,
        },
        "datasets": comparisons,
        "runtime": runtime,
        "shared_costs": shared_costs,
        "quality_boundary": (
            "Official lexical metrics do not establish semantic claim support, answer "
            "completeness, or requested-language compliance; those are recorded in the "
            "separate source-review artifact."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_root", type=Path, help="Directory containing scores.json and arm folders"
    )
    parser.add_argument("--frozen-root", type=Path, required=True, help="Frozen manifest directory")
    parser.add_argument("--output", type=Path, help="Output path (must not already exist)")
    args = parser.parse_args()
    report = build_comparison(args.run_root, args.frozen_root)
    destination = args.output or args.run_root / "paired-comparisons.json"
    if destination.exists():
        raise SystemExit(f"refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    destination.chmod(0o600)
    print(
        json.dumps({"output": str(destination), "sha256": _sha256(destination)}, ensure_ascii=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
