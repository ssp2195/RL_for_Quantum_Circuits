from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.article_v1_feature_benchmark import (
    ARTICLE_V1_FEATURE_BASELINE_SCHEMA,
    ARTICLE_V1_FEATURE_PROJECTION_SCHEMA,
    DEFAULT_FRONTIER_SIZES,
    DEFAULT_STAGED_EXPANSION_CAPS,
    CorrectnessGate,
    EndToEndMeasurement,
    MicrobenchmarkMeasurement,
    PilotFeasibilityCriteria,
    PILOT_HARD_3Q_TARGET_ID,
    REQUIRED_CORRECTNESS_CHECKS,
    REQUIRED_PILOT_RELAUNCH_CHECKS,
    baseline_f653193,
    benchmark_feature_evaluator,
    capture_benchmark_environment,
    create_repository_feature_benchmark_adapter,
    write_feature_benchmark_artifacts,
)


def _passing_gate() -> CorrectnessGate:
    return CorrectnessGate(
        checks={name: True for name in REQUIRED_CORRECTNESS_CHECKS},
        command="python -m pytest -q tests/article_v1/test_incremental_feature_index.py",
        evidence=("focused parity suite passed",),
    )


def _passing_relaunch_checks() -> dict[str, bool]:
    return {name: True for name in REQUIRED_PILOT_RELAUNCH_CHECKS}


def _micro_measurements() -> list[MicrobenchmarkMeasurement]:
    values: list[MicrobenchmarkMeasurement] = []
    for size in DEFAULT_FRONTIER_SIZES:
        scale = size / 512.0
        values.append(
            MicrobenchmarkMeasurement(
                frontier_size=size,
                repetitions=7,
                reference_total_seconds=(0.002 * scale**2 if size <= 256 else None),
                optimized_synchronization_seconds=0.002 * scale,
                optimized_compact_batch_seconds=0.006 * scale,
                optimized_scoring_seconds=0.002 * scale,
                optimized_selected_row_seconds=0.001 * scale,
                feature_index_memory_bytes=size * 128,
                process_peak_rss_bytes=10_000_000 + size * 128,
                unique_resource_groups=max(1, size // 4),
            )
        )
    return values


def _end_to_end_measurements() -> list[EndToEndMeasurement]:
    values: list[EndToEndMeasurement] = []
    for cap in DEFAULT_STAGED_EXPANSION_CAPS:
        parity = True if cap in (32, 64) else None
        values.append(
            EndToEndMeasurement(
                expansion_cap=cap,
                expansions_completed=cap,
                runtime_seconds=cap * 0.01,
                feature_time_seconds=cap * 0.002,
                peak_frontier=cap * 8,
                peak_unique_resource_groups=cap * 2,
                peak_feature_index_memory_bytes=cap * 1024,
                terminal_status="truncated",
                reference_runtime_seconds=(cap * 0.1 if cap in (32, 64) else None),
                trace_equivalent=parity,
                final_weights_equivalent=parity,
                terminal_status_equivalent=parity,
                deterministic_counters_equivalent=parity,
            )
        )
    return values


def test_baseline_is_explicitly_diagnostic_and_preserves_supplied_evidence() -> None:
    baseline = baseline_f653193(environment={"machine": "fixed-test-machine"})

    assert baseline["schema_version"] == ARTICLE_V1_FEATURE_BASELINE_SCHEMA
    assert baseline["scientific_scheduler_evidence"] is False
    assert baseline["source"]["commit"].startswith("f653193")
    assert baseline["preflight"]["article_v1_tests_passed"] == 240
    assert baseline["preflight"]["full_repository_tests_passed"] == 408
    assert baseline["hard_three_qubit_episode"][1]["runtime_seconds"] == 61.837
    assert baseline["isolated_reference_feature_batches"][-1] == {
        "frontier_size": 1021,
        "seconds": 2.330,
    }


def test_environment_capture_includes_blas_threads_and_nonscience_label() -> None:
    environment = capture_benchmark_environment()

    assert environment["python_version"]
    assert environment["numpy_version"]
    assert set(environment["thread_environment"]) == {
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    }
    assert set(environment["blas"]) == {"name", "version", "configuration"}
    assert environment["scientific_scheduler_evidence"] is False


def test_repository_adapter_binds_exact_hard_workload_and_cap_one_parity(
    tmp_path: Path,
) -> None:
    adapter = create_repository_feature_benchmark_adapter(
        microbenchmark_repetitions=1,
        profile_caps=(1,),
        profile_frontier_sizes=(32,),
    )

    metadata = adapter.metadata()
    assert metadata["target_id"] == PILOT_HARD_3Q_TARGET_ID
    assert metadata["transfer_target_index"] == 5
    assert metadata["effective_seed"] == 24
    assert metadata["scientific_horizon"] == 8192
    assert metadata["generator_witness_exposed_to_search"] is False

    measurement = adapter.measure_end_to_end(1, include_reference=True)
    assert measurement.expansions_completed == 1
    assert measurement.reference_runtime_seconds is not None
    assert measurement.trace_equivalent is True
    assert measurement.final_weights_equivalent is True
    assert measurement.terminal_status_equivalent is True
    assert measurement.deterministic_counters_equivalent is True

    adapter.prepare_microbenchmarks((32,))
    micro = adapter.measure_microbenchmark(32, include_reference=True)
    assert micro.frontier_size == 32
    assert micro.reference_total_seconds is not None
    assert micro.optimized_total_decision_seconds > 0.0
    assert 0 < micro.unique_resource_groups <= 32
    assert micro.feature_index_memory_bytes > 0

    adapter.write_profiles(tmp_path / "profiles")
    assert (tmp_path / "profiles" / "hard-3q-cap-1-optimized.prof").is_file()
    assert (tmp_path / "profiles" / "hard-3q-cap-1-optimized.txt").is_file()
    assert (tmp_path / "profiles" / "frontier-F32-optimized.prof").is_file()
    assert (tmp_path / "profiles" / "frontier-F32-optimized.txt").is_file()


def test_correctness_gate_fails_closed_on_missing_or_false_required_checks() -> None:
    missing = CorrectnessGate(checks={}, command="pytest parity")
    false = CorrectnessGate(
        checks={name: name != "sarsa_trace_equivalence" for name in REQUIRED_CORRECTNESS_CHECKS},
        command="pytest parity",
    )

    assert missing.passed is False
    assert missing.missing_checks == REQUIRED_CORRECTNESS_CHECKS
    assert false.passed is False
    assert false.failed_checks == ("sarsa_trace_equivalence",)
    with pytest.raises(ValueError, match="must be a bool"):
        CorrectnessGate(
            checks={"snapshot_equivalence": 1},  # type: ignore[dict-item]
            command="pytest parity",
        )


def test_measurement_validation_is_type_strict_and_enforces_physical_bounds() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        MicrobenchmarkMeasurement(
            frontier_size=True,  # type: ignore[arg-type]
            repetitions=1,
            reference_total_seconds=None,
            optimized_synchronization_seconds=0.1,
            optimized_compact_batch_seconds=0.1,
            optimized_scoring_seconds=0.1,
            optimized_selected_row_seconds=0.1,
            feature_index_memory_bytes=1,
            unique_resource_groups=1,
        )
    with pytest.raises(ValueError, match="cannot exceed runtime"):
        EndToEndMeasurement(
            expansion_cap=32,
            expansions_completed=32,
            runtime_seconds=1.0,
            feature_time_seconds=2.0,
            peak_frontier=2,
            peak_unique_resource_groups=1,
            peak_feature_index_memory_bytes=1,
            terminal_status="truncated",
        )


def test_artifact_writer_emits_complete_deterministic_passing_bundle(
    tmp_path: Path,
) -> None:
    output = tmp_path / "article-v1-feature-index-v2"
    profiles = output / "profiles"
    profiles.mkdir(parents=True)
    (profiles / "cap-64.prof").write_bytes(b"profile")
    baseline = baseline_f653193(environment={"machine": "fixed-test-machine"})
    arguments = dict(
        correctness_gate=_passing_gate(),
        microbenchmarks=_micro_measurements(),
        end_to_end=_end_to_end_measurements(),
        implementation_checks={"no_python_nested_frontier_record_loop": True},
        baseline=baseline,
        feasibility_criteria=PilotFeasibilityCriteria(
            maximum_hard_episode_seconds=1000.0,
            maximum_peak_index_memory_bytes=1 << 30,
        ),
        hard_training_episode_count=8,
        pilot_relaunch_checks=_passing_relaunch_checks(),
        benchmark_provenance={
            "code_version": "optimized-test-commit",
            "source_worktree_digest": "sha256:optimized-test-worktree",
            "worktree_clean": True,
        },
    )

    first = write_feature_benchmark_artifacts(output, **arguments)
    first_bytes = {
        path.name: path.read_bytes()
        for path in (
            first.baseline_json,
            first.microbenchmarks_csv,
            first.end_to_end_scaling_csv,
            first.scaling_report_md,
            first.projected_pilot_cost_json,
        )
    }
    second = write_feature_benchmark_artifacts(output, **arguments)

    assert second.qualification["passed"] is True
    assert second.projection["schema_version"] == ARTICLE_V1_FEATURE_PROJECTION_SCHEMA
    assert second.projection["pilot_decision"] == "configured pilot is feasible unchanged"
    assert first_bytes == {
        path.name: path.read_bytes()
        for path in (
            second.baseline_json,
            second.microbenchmarks_csv,
            second.end_to_end_scaling_csv,
            second.scaling_report_md,
            second.projected_pilot_cost_json,
        )
    }
    assert first.profiles_directory.is_dir()
    assert (first.profiles_directory / "cap-64.prof").read_bytes() == b"profile"
    report = first.scaling_report_md.read_text(encoding="utf-8")
    assert "Engineering diagnostic only" in report
    assert "profiles/cap-64.prof" in report
    assert "Overall performance qualification: **PASS**" in report

    baseline_document = json.loads(first.baseline_json.read_text(encoding="utf-8"))
    assert baseline_document["correctness_gate"]["passed"] is True
    assert baseline_document["benchmark_provenance"]["bound"] is True
    assert baseline_document["feature_evaluator_schema_version"] == (
        "article-v1-exact-incremental-v2"
    )
    with first.microbenchmarks_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["frontier_size"]) for row in rows] == list(DEFAULT_FRONTIER_SIZES)
    row_1024 = next(row for row in rows if row["frontier_size"] == "1024")
    assert row_1024["reference_basis"] == "baseline-nearest-F1021"
    assert float(row_1024["speedup"]) >= 10.0


def test_failed_correctness_marks_all_timings_unqualified(tmp_path: Path) -> None:
    failed_gate = CorrectnessGate(
        checks={name: name != "final_weight_equivalence" for name in REQUIRED_CORRECTNESS_CHECKS},
        command="pytest parity",
    )
    artifacts = write_feature_benchmark_artifacts(
        tmp_path / "failed",
        correctness_gate=failed_gate,
        microbenchmarks=_micro_measurements(),
        end_to_end=_end_to_end_measurements(),
        implementation_checks={"no_python_nested_frontier_record_loop": True},
        baseline=baseline_f653193(environment={"machine": "fixed"}),
        feasibility_criteria=PilotFeasibilityCriteria(1000.0, 1 << 30),
    )

    assert artifacts.qualification["passed"] is False
    assert artifacts.projection["pilot_decision"] == (
        "insufficient performance evidence to relaunch"
    )
    assert "**FAIL**" in artifacts.scaling_report_md.read_text(encoding="utf-8")


class _RecordingAdapter:
    def __init__(self) -> None:
        self.micro_calls: list[tuple[int, bool]] = []
        self.end_calls: list[tuple[int, bool]] = []

    def measure_microbenchmark(self, frontier_size: int, *, include_reference: bool):
        self.micro_calls.append((frontier_size, include_reference))
        measurement = next(
            value for value in _micro_measurements() if value.frontier_size == frontier_size
        )
        if include_reference:
            return measurement
        return {
            **{
                name: getattr(measurement, name)
                for name in measurement.__dataclass_fields__
            },
            "reference_total_seconds": None,
        }

    def measure_end_to_end(self, expansion_cap: int, *, include_reference: bool):
        self.end_calls.append((expansion_cap, include_reference))
        return next(
            value
            for value in _end_to_end_measurements()
            if value.expansion_cap == expansion_cap
        )


def test_staged_adapter_api_limits_expensive_reference_work_and_writes_profiles(
    tmp_path: Path,
) -> None:
    adapter = _RecordingAdapter()
    profile_calls: list[Path] = []

    def write_profiles(directory: Path) -> None:
        profile_calls.append(directory)
        (directory / "profile.txt").write_text("diagnostic", encoding="utf-8")

    artifacts = benchmark_feature_evaluator(
        adapter,
        tmp_path / "run",
        correctness_gate=_passing_gate(),
        implementation_checks={"no_python_nested_frontier_record_loop": True},
        baseline=baseline_f653193(environment={"machine": "fixed"}),
        profile_writer=write_profiles,
    )

    assert adapter.micro_calls == [
        (size, size <= 256) for size in DEFAULT_FRONTIER_SIZES
    ]
    assert adapter.end_calls == [
        (cap, cap in (32, 64)) for cap in DEFAULT_STAGED_EXPANSION_CAPS
    ]
    assert profile_calls == [artifacts.profiles_directory]
    assert (artifacts.profiles_directory / "profile.txt").is_file()


def test_staged_adapter_refuses_to_time_before_correctness_passes(tmp_path: Path) -> None:
    adapter = _RecordingAdapter()
    with pytest.raises(ValueError, match="must pass before timing"):
        benchmark_feature_evaluator(
            adapter,
            tmp_path / "never-created",
            correctness_gate=CorrectnessGate(checks={}, command="pytest parity"),
            implementation_checks={"no_python_nested_frontier_record_loop": True},
        )
    assert adapter.micro_calls == []
    assert adapter.end_calls == []
    assert not (tmp_path / "never-created").exists()


def test_invalid_bundle_does_not_overwrite_existing_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "preserve"
    output.mkdir()
    baseline_path = output / "baseline.json"
    baseline_path.write_bytes(b"preserve")
    duplicate = _micro_measurements()
    duplicate.append(duplicate[0])

    with pytest.raises(ValueError, match="must be unique"):
        write_feature_benchmark_artifacts(
            output,
            correctness_gate=_passing_gate(),
            microbenchmarks=duplicate,
            end_to_end=_end_to_end_measurements(),
            implementation_checks={"no_python_nested_frontier_record_loop": True},
        )
    assert baseline_path.read_bytes() == b"preserve"


def test_missing_feasibility_bounds_never_authorizes_pilot_relaunch(
    tmp_path: Path,
) -> None:
    artifacts = write_feature_benchmark_artifacts(
        tmp_path / "review-required",
        correctness_gate=_passing_gate(),
        microbenchmarks=_micro_measurements(),
        end_to_end=_end_to_end_measurements(),
        implementation_checks={"no_python_nested_frontier_record_loop": True},
        baseline=baseline_f653193(environment={"machine": "fixed"}),
    )

    assert artifacts.qualification["passed"] is True
    assert artifacts.projection["within_feasibility_criteria"] is None
    assert artifacts.projection["pilot_decision"] == (
        "insufficient performance evidence to relaunch"
    )


def test_missing_operability_and_clean_source_checks_block_feasible_timing(
    tmp_path: Path,
) -> None:
    artifacts = write_feature_benchmark_artifacts(
        tmp_path / "missing-operability",
        correctness_gate=_passing_gate(),
        microbenchmarks=_micro_measurements(),
        end_to_end=_end_to_end_measurements(),
        implementation_checks={"no_python_nested_frontier_record_loop": True},
        baseline=baseline_f653193(environment={"machine": "fixed"}),
        feasibility_criteria=PilotFeasibilityCriteria(1000.0, 1 << 30),
    )

    assert artifacts.qualification["passed"] is True
    assert artifacts.projection["within_feasibility_criteria"] is True
    assert artifacts.projection["pilot_relaunch_gate"]["passed"] is False
    assert artifacts.projection["pilot_decision"] == (
        "insufficient performance evidence to relaunch"
    )
