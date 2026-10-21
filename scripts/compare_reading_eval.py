"""Create a paired, per-dataset summary from a completed reading-eval run."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

ARM_PAIRS = (
    ("fixed-rag", "single-agent"),
    ("single-agent", "delegated-agent"),
)
METRICS = {
    "qasper": ("answer_f1", "evidence_f1"),
    "longbench-v2-single-document-qa": ("accuracy",),
    "longbench-v1-multifieldqa-zh": ("answer_f1",),
}
SEED = 20261020
REPLICATES = 50_000


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


def build_comparison(run_root: Path, seed: int = SEED, replicates: int = REPLICATES) -> dict:
    manifest_path = run_root.parent.parent / "manifest.json"
    scores_path = run_root / "scores.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    if scores.get("state") != "scored_complete":
        raise ValueError("comparison requires a complete scored run")
    if scores.get("independent_human_review") is not False:
        raise ValueError("unexpected human-review state")

    cases_by_dataset: dict[str, list[dict]] = {}
    for case in manifest["cases"]:
        cases_by_dataset.setdefault(case["dataset"], []).append(case)
    if set(cases_by_dataset) != set(METRICS):
        raise ValueError("frozen manifest has unexpected datasets")

    result = {}
    for dataset, metrics in METRICS.items():
        cases = cases_by_dataset[dataset]
        source_ids = [case["source_id"] for case in cases]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError(f"document-cluster bootstrap required for {dataset}")
        result[dataset] = {}
        expected_ids = [case["case_id"] for case in cases]
        for metric in metrics:
            rows_by_arm = {
                arm: _rows_by_case_id(
                    scores["arms"][arm]["datasets"][dataset]["per_case"],
                    expected_ids,
                    f"{arm}/{dataset}",
                )
                for arm in ("fixed-rag", "single-agent", "delegated-agent")
            }
            selected_ids = list(expected_ids)
            if metric == "evidence_f1":
                selected_ids = [
                    case_id
                    for case_id in expected_ids
                    if all(
                        rows_by_arm[arm][case_id].get("evidence_applicable")
                        and isinstance(rows_by_arm[arm][case_id].get(metric), (float, int))
                        for arm in rows_by_arm
                    )
                ]
            if not selected_ids:
                raise ValueError(f"no paired rows for {dataset}/{metric}")

            result[dataset][metric] = {}
            for pair_index, (left_arm, right_arm) in enumerate(ARM_PAIRS):
                left = [rows_by_arm[left_arm][case_id][metric] for case_id in selected_ids]
                right = [rows_by_arm[right_arm][case_id][metric] for case_id in selected_ids]
                if not all(isinstance(value, (float, int)) for value in (*left, *right)):
                    raise ValueError(f"non-numeric paired score in {dataset}/{metric}")
                pair_seed = seed + sum(map(ord, dataset + metric)) + pair_index
                result[dataset][metric][f"{left_arm}_vs_{right_arm}"] = paired_summary(
                    left,
                    right,
                    seed=pair_seed,
                    replicates=replicates,
                )

    arm_runtime = {}
    for arm in scores["arms"]:
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((run_root / arm).glob("*.json"))
        ]
        traces = [
            record.get("agent_trace")
            for record in records
            if isinstance(record.get("agent_trace"), dict)
        ]
        arm_runtime[arm] = {
            "cases": len(records),
            "successful": sum(record.get("execution") == "ok" for record in records),
            "failed": sum(record.get("execution") != "ok" for record in records),
            "completion_dispatches": sum(
                record.get("completion_calls_attempted", 0) for record in records
            ),
            "input_tokens_lower_bound": sum(
                record.get("input_tokens_lower_bound", 0) for record in records
            ),
            "output_tokens_lower_bound": sum(
                record.get("output_tokens_lower_bound", 0) for record in records
            ),
            "output_usage_unknown_rows": sum(
                bool(record.get("output_usage_unknown")) for record in records
            ),
            "reader_tasks_started": sum(trace.get("reader_tasks_started", 0) for trace in traces),
        }

    return {
        "state": "paired_comparison_complete",
        "source_artifacts": {
            "manifest_sha256": _sha256(manifest_path),
            "scores_sha256": _sha256(scores_path),
            "analysis_script_sha256": _sha256(Path(__file__).resolve()),
        },
        "method": {
            "comparisons": [
                {"left": left, "right": right, "delta": "right minus left"}
                for left, right in ARM_PAIRS
            ],
            "resampling": (
                "paired case bootstrap within each dataset; unique document per sampled case"
            ),
            "replicates": replicates,
            "seed": seed,
            "interval": (
                "descriptive 95% percentile interval; not a significance or equivalence test"
            ),
            "failed_execution_scores": "preserved as zero in the official per-case score",
            "dataset_metrics_combined": False,
            "human_review": False,
        },
        "datasets": result,
        "runtime": arm_runtime,
        "delegated_agent_interpretation": (
            "The arm had delegation available. Its score is not evidence of multi-agent execution "
            "when reader_tasks_started is zero."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_root", type=Path, help="Directory containing scores.json and arm folders"
    )
    parser.add_argument("--output", type=Path, help="Output path (must not already exist)")
    args = parser.parse_args()
    report = build_comparison(args.run_root)
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
