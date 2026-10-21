import importlib.util
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
