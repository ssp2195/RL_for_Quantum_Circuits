from __future__ import annotations

import json
from pathlib import Path

import pytest

import article_benchmark
from benchmarks.article_v1_calibration import calibrate_certifier
from experiments.article_v1_runner import campaign_plan, main as article_v1_main


def test_calibration_is_deterministic_complete_and_passes_guards(tmp_path: Path):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = calibrate_certifier("pilot", first_path)
    second = calibrate_certifier("pilot", second_path)

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["passed"] is True
    assert first["provisional_tau_cert"] == first["frozen_tau_cert"] == 1e-6
    assert first["tau_at_most_1e_6"] is True
    assert first["separation_at_least_100x"] is True
    assert first["identity_equivalent_floor_covered"] is True
    assert first["identity_separation_at_least_100x"] is True
    assert first["maximum_generator_length"] == 8

    equivalent_names = {row["name"] for row in first["equivalent_pairs"]}
    distinct_names = {row["name"] for row in first["non_equivalent_pairs"]}
    assert {"known-ghz", "known-toffoli", "known-ccz"} <= equivalent_names
    assert any(name.startswith("safe-commuting-reorder-") for name in equivalent_names)
    assert any("length-8" in name for name in equivalent_names)
    assert sum(name.startswith("self-") for name in equivalent_names) == 38
    for fixture in (
        "wrong-relative-phase-",
        "wrong-t-tdg-sign-",
        "omitted-h-",
        "reversed-cnot-",
        "localized-unitary-perturbation-",
        "distinct-short-h-vs-t-",
    ):
        assert any(name.startswith(fixture) for name in distinct_names)


def test_campaign_plan_enumerates_every_required_cardinality_without_execution():
    report = campaign_plan(
        "pilot", worker_count=2, pilot_seconds_per_expansion=1e-6
    )

    assert report["executes_search"] is False
    assert report["target_counts"] == {
        "train": 6,
        "validation": 3,
        "test": 6,
        "ood_test": 4,
    }
    assert report["learner_checkpoint_count"] == 4
    assert report["standard_test_run_count"] == 180
    assert report["ood_run_count"] == 120
    assert report["validation_run_count"] == 10
    assert report["ablation_run_count"] == 33
    assert report["repeated_random_run_count"] == 90
    assert report["expected_raw_ledger_keys"] == 300
    assert report["worst_case_expansion_count"] == sum(
        report["worst_case_expansion_breakdown"].values()
    )
    assert report["estimated_disk_use"]["bytes"] > 0
    assert report["estimated_cpu_time_from_pilot"]["status"] == "estimated"
    assert report["estimated_cpu_time_from_pilot"][
        "wall_seconds_at_selected_workers"
    ] == pytest.approx(
        report["estimated_cpu_time_from_pilot"]["seconds"] / 2
    )
    assert report["selected_worker_count"] == 2
    for split, strata in report["target_counts_by_split_stratum_qubits"].items():
        assert sum(sum(widths.values()) for widths in strata.values()) == report[
            "target_counts"
        ][split]


def test_plan_cli_emits_machine_readable_no_execution_report(capsys):
    assert article_v1_main(["plan", "--config", "configs/article_v1_pilot.json"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["schema_version"] == "article-v1-campaign-plan-v1"
    assert payload["executes_search"] is False
    assert payload["estimated_cpu_time_from_pilot"]["status"] == (
        "awaiting-pilot-measurement"
    )


def test_calibration_cli_writes_a_passing_versioned_artifact(tmp_path, capsys):
    assert article_v1_main([
        "calibrate-certifier",
        "--config",
        "configs/article_v1_pilot.json",
        "--output-root",
        str(tmp_path),
        "--run-id",
        "calibration-test",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    artifact = tmp_path / "calibration-test" / "certifier_calibration.json"

    assert payload["passed"] is True
    assert artifact.is_file()
    assert json.loads(artifact.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize("command", ("mini-ci", "calibrate-certifier", "plan"))
def test_root_cli_dispatches_article_v1_commands(monkeypatch, command):
    captured = []

    def fake_main(arguments):
        captured.append(list(arguments))
        return 17

    monkeypatch.setattr("experiments.article_v1_runner.main", fake_main)

    assert article_benchmark.main([command]) == 17
    assert captured == [[command]]
