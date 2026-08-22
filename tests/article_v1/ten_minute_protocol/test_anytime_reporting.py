import pytest
from pathlib import Path

from reporting.article_v1 import derive_fixed_horizon_anytime_rows
from experiments.article_v1_ten_minute_runner import (
    evaluate_ten_minute,
    report_ten_minute,
    train_ten_minute,
    validate_protocol_sensitivity,
)
from experiments.article_v1_ten_minute_protocol import TenMinuteCheckpoint


ROOT = Path(__file__).resolve().parents[3]


def test_anytime_rows_reuse_one_physical_run():
    rows = derive_fixed_horizon_anytime_rows(
        {
            "budget_mode": "fixed-max-horizon-anytime-v1",
            "target_id": "target",
            "scheduler": "article_sarsa",
            "training_seed": 19,
            "evaluation_seed": 23,
            "executed_max_horizon": 1792,
            "first_certified_hit_expansion": 730,
            "process_cpu_seconds": 12.5,
            "runtime_seconds": 14.0,
        },
        (256, 512, 1024, 1792),
    )
    assert [row["success_by_threshold"] for row in rows] == [False, False, True, True]
    assert all(row["derived_from_raw_run"] for row in rows)
    assert {row["target_id"] for row in rows} == {"target"}


def test_anytime_rows_reject_invalid_thresholds_and_hits():
    raw = {"budget_mode": "fixed-max-horizon-anytime-v1", "executed_max_horizon": 512}
    with pytest.raises(ValueError, match="larger"):
        derive_fixed_horizon_anytime_rows(raw, (1024,))
    with pytest.raises(ValueError, match="outside"):
        derive_fixed_horizon_anytime_rows({**raw, "first_certified_hit_expansion": 513}, (512,))


def test_bounded_evaluator_executes_one_run_and_derives_rows(tmp_path: Path):
    result = evaluate_ten_minute(
        ROOT / "configs/article_v1_10min_pilot.json",
        tmp_path,
        schedulers=("fifo",),
        families=("in_distribution",),
        require_frozen=False,
        maximum_targets_per_family=1,
        horizon_override=1,
    )
    assert result["physical_search_execution_count"] == 1
    assert result["derived_budget_threshold_observation_count"] == 1
    assert len((tmp_path / "raw_runs.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_report_reuses_raw_ledger_and_writes_required_figures(tmp_path: Path):
    run_directory = tmp_path / "run"
    evaluate_ten_minute(
        ROOT / "configs/article_v1_10min_pilot.json",
        run_directory,
        schedulers=("fifo",),
        families=("in_distribution",),
        require_frozen=False,
        maximum_targets_per_family=1,
        horizon_override=1,
    )
    result = report_ten_minute(
        run_directory / "raw_runs.jsonl", tmp_path / "report"
    )
    assert result["raw_run_count"] == 1
    assert result["derived_threshold_row_count"] == 1
    assert result["family_aggregation_separated"] is True
    assert len(list((tmp_path / "report" / "figures").glob("*.svg"))) == 9


def test_protocol_sensitivity_is_validation_only(tmp_path: Path):
    training = train_ten_minute(
        ROOT / "configs/article_v1_10min_pilot.json",
        tmp_path / "training",
        training_seed=19,
        total_expansions_override=2,
    )
    result = validate_protocol_sensitivity(
        ROOT / "configs/article_v1_10min_pilot.json",
        tmp_path / "validation",
        checkpoint=TenMinuteCheckpoint.load(training["checkpoint_path"]),
        maximum_targets=1,
        horizon_override=1,
    )
    assert result["no_test_access"] is True
    assert result["validation_only"] is True
    assert result["fixed_physical_run_count"] == 1
    assert result["independent_physical_run_count"] == 1
