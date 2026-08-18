from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess

import pytest

import article_benchmark
import experiments.article_v1_feature_benchmark as feature_benchmark_module
import experiments.article_v1_runner as article_v1_runner_module
from experiments.article_v1_runner import benchmark_article_v1_features
from experiments.article_v1_feature_benchmark import (
    ARTICLE_V1_FEATURE_BASELINE_SCHEMA,
    ARTICLE_V1_FEATURE_PROJECTION_SCHEMA,
    DEFAULT_FRONTIER_SIZES,
    DEFAULT_MICROBENCHMARK_REPETITIONS,
    DEFAULT_MICROBENCHMARK_WARMUPS,
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
    inspect_production_dominance_update,
    run_focused_correctness_gate,
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


def _stable_git_provenance(*, refresh: bool = False) -> dict[str, object]:
    assert isinstance(refresh, bool)
    return {
        "commit_sha": "benchmark-test-commit",
        "branch": "frontier-rl-exper",
        "dirty_worktree": False,
        "source_worktree_digest": "sha256:benchmark-test-source",
        "relevant_untracked_files": [],
    }


def _micro_measurements() -> list[MicrobenchmarkMeasurement]:
    values: list[MicrobenchmarkMeasurement] = []
    for size in DEFAULT_FRONTIER_SIZES:
        scale = size / 512.0
        values.append(
            MicrobenchmarkMeasurement(
                frontier_size=size,
                repetitions=7,
                reference_total_seconds=(0.2 * scale**2 if size <= 1024 else None),
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
                reference_feature_time_seconds=(
                    cap * 0.08 if cap in (32, 64) else None
                ),
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
    unknown = baseline_f653193()["environment"]
    assert unknown["captured"] is False
    assert unknown["status"] == "unknown-not-captured-with-historical-baseline"


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
    assert measurement.reference_feature_time_seconds is not None
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


def test_repository_adapter_defaults_use_stable_microbenchmark_sampling() -> None:
    adapter = create_repository_feature_benchmark_adapter()

    assert adapter.microbenchmark_repetitions == DEFAULT_MICROBENCHMARK_REPETITIONS
    assert adapter.microbenchmark_warmups == DEFAULT_MICROBENCHMARK_WARMUPS
    assert adapter.microbenchmark_repetitions == 31
    assert adapter.microbenchmark_warmups == 5


def test_optimized_frontier_profile_excludes_reference_all_pairs_work(
    tmp_path: Path,
) -> None:
    adapter = object.__new__(
        feature_benchmark_module.RepositoryArticleV1FeatureBenchmarkAdapter
    )
    adapter.profile_caps = ()
    adapter.profile_frontier_sizes = (1024,)
    adapter.reference_safe_frontier_size = 1024
    calls: list[tuple[int, bool]] = []

    def measure_microbenchmark(
        frontier_size: int,
        *,
        include_reference: bool,
    ) -> None:
        calls.append((frontier_size, include_reference))

    adapter.measure_microbenchmark = measure_microbenchmark
    adapter.write_profiles(tmp_path / "profiles")

    assert calls == [(1024, False)]
    assert (tmp_path / "profiles" / "frontier-F1024-optimized.prof").is_file()
    assert (tmp_path / "profiles" / "frontier-F1024-optimized.txt").is_file()


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
    assert row_1024["reference_basis"] == "measured-current-reference"
    assert row_1024["reference_measured"] == "True"
    assert float(row_1024["speedup"]) >= 10.0
    with first.end_to_end_scaling_csv.open(
        newline="", encoding="utf-8"
    ) as handle:
        end_rows = list(csv.DictReader(handle))
    cap_32 = next(row for row in end_rows if row["expansion_cap"] == "32")
    assert float(cap_32["end_to_end_speedup"]) == pytest.approx(10.0)
    assert float(cap_32["feature_time_share"]) == pytest.approx(0.2)
    assert float(cap_32["reference_feature_time_share"]) == pytest.approx(0.8)
    assert cap_32["reference_parity_passed"] == "True"
    cap_128 = next(row for row in end_rows if row["expansion_cap"] == "128")
    assert cap_128["reference_expected"] == "False"
    assert cap_128["reference_parity_passed"] == ""
    staged_report = report.split("## Staged hard-target measurements", 1)[1]
    cap_128_report = next(
        line for line in staged_report.splitlines() if line.startswith("| 128 |")
    )
    assert cap_128_report.endswith("| — |")
    assert second.qualification["checks"][
        "end_to_end_speedup_at_32_and_64_at_least_2x"
    ] is True


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


def test_micro_speedup_cannot_hide_slower_optimized_end_to_end_search(
    tmp_path: Path,
) -> None:
    slower_end_to_end: list[EndToEndMeasurement] = []
    for measurement in _end_to_end_measurements():
        if measurement.expansion_cap not in (32, 64):
            slower_end_to_end.append(measurement)
            continue
        slower_end_to_end.append(
            EndToEndMeasurement(
                expansion_cap=measurement.expansion_cap,
                expansions_completed=measurement.expansions_completed,
                runtime_seconds=measurement.runtime_seconds,
                feature_time_seconds=measurement.feature_time_seconds,
                peak_frontier=measurement.peak_frontier,
                peak_unique_resource_groups=(
                    measurement.peak_unique_resource_groups
                ),
                peak_feature_index_memory_bytes=(
                    measurement.peak_feature_index_memory_bytes
                ),
                terminal_status=measurement.terminal_status,
                reference_runtime_seconds=measurement.runtime_seconds / 2.0,
                reference_feature_time_seconds=(
                    measurement.runtime_seconds / 4.0
                ),
                trace_equivalent=True,
                final_weights_equivalent=True,
                terminal_status_equivalent=True,
                deterministic_counters_equivalent=True,
            )
        )

    artifacts = write_feature_benchmark_artifacts(
        tmp_path / "slower-optimized-search",
        correctness_gate=_passing_gate(),
        microbenchmarks=_micro_measurements(),
        end_to_end=slower_end_to_end,
        implementation_checks={"no_python_nested_frontier_record_loop": True},
        baseline=baseline_f653193(environment={"machine": "fixed"}),
    )

    assert artifacts.qualification["checks"][
        "speedup_at_approximately_1024_at_least_10x"
    ] is True
    assert artifacts.qualification["checks"][
        "end_to_end_speedup_at_32_and_64_at_least_2x"
    ] is False
    assert artifacts.qualification["passed"] is False


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
        (size, size <= 1024) for size in DEFAULT_FRONTIER_SIZES
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


def test_production_dominance_source_check_is_scoped_and_fails_on_nested_loops(
    tmp_path: Path,
) -> None:
    actual = inspect_production_dominance_update()

    assert actual["passed"] is True
    assert actual["debug_or_reference_oracles_excluded"] is True
    assert actual["checks"]["no_python_all_open_record_nested_loop"] is True
    assert not actual["methods"]["_insert_resource"]["python_loop_lines"]
    assert not actual["methods"]["_remove_resource"]["python_loop_lines"]
    assert actual["checks"]["no_reachable_python_nested_loop"] is True
    assert "_grow_groups" in actual["reachable_methods"]

    unsafe = tmp_path / "unsafe_frontier_index.py"
    unsafe.write_text(
        """
class ExactArticleFrontierFeatureIndex:
    def _insert_resource(self, resources):
        active = self._active_group_indices()
        np.all(active)
        for left in self._record_by_id:
            for right in self._record_by_id:
                pass

    def _remove_resource(self, group, resources):
        active = self._active_group_indices()
        np.all(active)
""".lstrip(),
        encoding="utf-8",
    )
    rejected = inspect_production_dominance_update(unsafe)

    assert rejected["passed"] is False
    assert rejected["checks"]["no_python_all_open_record_nested_loop"] is False
    assert rejected["methods"]["_insert_resource"]["nested_python_loop_lines"]

    delegated = tmp_path / "delegated_unsafe_frontier_index.py"
    delegated.write_text(
        """
class ExactArticleFrontierFeatureIndex:
    def _insert_resource(self, resources):
        active = self._active_group_indices()
        np.all(active)
        return self._delegated_all_records(resources)

    def _remove_resource(self, group, resources):
        active = self._active_group_indices()
        np.all(active)

    def _active_group_indices(self):
        return ()

    def _delegated_all_records(self, resources):
        for left in self._record_by_id:
            for right in self._record_by_id:
                pass
""".lstrip(),
        encoding="utf-8",
    )
    delegated_report = inspect_production_dominance_update(delegated)

    assert delegated_report["passed"] is False
    assert "_delegated_all_records" in delegated_report["reachable_methods"]
    assert delegated_report["checks"]["no_reachable_python_nested_loop"] is False


def _successful_gate_junit() -> str:
    return """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="6" failures="0" errors="0" skipped="0">
    <testcase classname="tests.article_v1.test_incremental_feature_index" name="test_incremental_snapshot_is_exactly_equal_to_all_pairs_oracle[32]" time="0.01" />
    <testcase classname="tests.article_v1.test_incremental_feature_index" name="test_incremental_snapshot_is_exactly_equal_to_all_pairs_oracle[128]" time="0.02" />
    <testcase classname="tests.article_v1.test_compact_linear_scoring" name="test_compact_effective_weight_scores_equal_explicit_full_feature_dot_products[provider0]" time="0.01" />
    <testcase classname="tests.article_v1.test_compact_linear_scoring" name="test_selected_row_materialization_matches_reference_31d_exactly" time="0.01" />
    <testcase classname="tests.article_v1.test_search_trace_equivalence" name="test_real_hard_target_reference_trace_is_exact_at_caps_8_16_32_64" time="100.44" />
    <testcase classname="tests.article_v1.test_feature_benchmark_gate" name="test_reference_and_optimized_success_witnesses_certify_identically" time="0.10" />
  </testsuite>
</testsuites>
"""


def test_correctness_gate_is_derived_from_inspectable_junit_cases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command, **kwargs):
        junit_argument = next(
            argument for argument in command if argument.startswith("--junitxml=")
        )
        Path(junit_argument.split("=", 1)[1]).write_text(
            _successful_gate_junit(), encoding="utf-8"
        )
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "tests\\article_v1\\test_search_trace_equivalence.py . [100%]\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(feature_benchmark_module.subprocess, "run", fake_run)
    gate, report = run_focused_correctness_gate(
        tmp_path / "qualification",
        repository_root=tmp_path,
    )

    assert gate.passed is True
    assert gate.checks == {name: True for name in REQUIRED_CORRECTNESS_CHECKS}
    assert report["returncode"] == 0
    assert report["junit"]["case_count"] == 6
    assert report["junit"]["sha256"].startswith("sha256:")
    assert "test_real_hard_target_reference_trace_is_exact_at_caps_8_16_32_64" in (
        gate.command
    )
    for path in gate.evidence:
        assert Path(path).exists()
    persisted = json.loads(
        (tmp_path / "qualification" / "profiles" / "correctness_gate.json").read_text(
            encoding="utf-8"
        )
    )
    assert persisted["correctness_gate"]["passed"] is True


def test_correctness_gate_fails_closed_when_mapped_junit_case_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trace_case = (
        '    <testcase classname="tests.article_v1.test_search_trace_equivalence" '
        'name="test_real_hard_target_reference_trace_is_exact_at_caps_8_16_32_64" '
        'time="100.44" />\n'
    )

    def fake_run(command, **kwargs):
        junit_argument = next(
            argument for argument in command if argument.startswith("--junitxml=")
        )
        Path(junit_argument.split("=", 1)[1]).write_text(
            _successful_gate_junit().replace(trace_case, ""), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="5 passed", stderr="")

    monkeypatch.setattr(feature_benchmark_module.subprocess, "run", fake_run)
    gate, _report = run_focused_correctness_gate(
        tmp_path / "missing-trace",
        repository_root=tmp_path,
    )

    assert gate.passed is False
    assert gate.checks["snapshot_equivalence"] is True
    assert gate.checks["score_equivalence"] is True
    assert gate.checks["selected_record_equivalence"] is True
    assert gate.checks["witness_certification_equivalence"] is True
    assert gate.checks["sarsa_trace_equivalence"] is False
    assert gate.checks["final_weight_equivalence"] is False
    assert gate.checks["terminal_status_equivalence"] is False


def test_staged_adapter_refuses_to_time_before_implementation_check_passes(
    tmp_path: Path,
) -> None:
    adapter = _RecordingAdapter()
    with pytest.raises(ValueError, match="implementation check must pass before timing"):
        benchmark_feature_evaluator(
            adapter,
            tmp_path / "never-created",
            correctness_gate=_passing_gate(),
            implementation_checks={"no_python_nested_frontier_record_loop": False},
        )
    assert adapter.micro_calls == []
    assert adapter.end_calls == []
    assert not (tmp_path / "never-created").exists()


def test_feature_benchmark_command_aborts_before_timing_on_failed_pytest_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    implementation = {
        "schema_version": "article-v1-dominance-source-check-v1",
        "passed": True,
        "checks": {"no_python_all_open_record_nested_loop": True},
    }
    failed_gate = CorrectnessGate(checks={}, command="python -m pytest focused")
    monkeypatch.setattr(
        feature_benchmark_module,
        "inspect_production_dominance_update",
        lambda: implementation,
    )
    monkeypatch.setattr(
        article_v1_runner_module, "git_provenance", _stable_git_provenance
    )
    monkeypatch.setattr(
        feature_benchmark_module,
        "run_focused_correctness_gate",
        lambda *args, **kwargs: (failed_gate, {"returncode": 1}),
    )

    def forbidden_timing(*args, **kwargs):
        raise AssertionError("timing adapter must not run after failed evidence")

    monkeypatch.setattr(
        feature_benchmark_module,
        "run_repository_feature_benchmark",
        forbidden_timing,
    )
    result = benchmark_article_v1_features(
        "pilot",
        output_root=tmp_path,
        run_id="failed-gate",
        write_profiles=False,
    )

    assert result["passed"] is False
    assert result["engineering_qualification_passed"] is False
    assert result["pilot_relaunch_ready"] is False
    assert result["aborted_before_timing"] is True
    assert result["artifacts"] is None
    assert Path(result["implementation_evidence"]).is_file()
    status = json.loads(Path(result["status_manifest"]).read_text(encoding="utf-8"))
    assert status["phase"] == "correctness-gate"
    assert status["written_last"] is True


def test_feature_benchmark_refuses_reused_directory_without_overwrite(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "reused"
    destination.mkdir()
    prior = destination / "baseline.json"
    prior.write_bytes(b"prior-success-must-survive")

    result = benchmark_article_v1_features(
        "pilot", output_root=tmp_path, run_id="reused", write_profiles=False
    )

    assert result["engineering_qualification_passed"] is False
    assert result["status_manifest"] is None
    assert "nonempty" in result["abort_reason"]
    assert prior.read_bytes() == b"prior-success-must-survive"
    assert sorted(path.name for path in destination.iterdir()) == ["baseline.json"]


def test_feature_benchmark_rejects_noncanonical_same_target_config_before_pytest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = json.loads(
        Path("configs/article_v1_pilot.json").read_text(encoding="utf-8")
    )
    payload["experiment"]["learning_rate"] = 0.002
    external = tmp_path / "external-pilot-like.json"
    external.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        article_v1_runner_module, "git_provenance", _stable_git_provenance
    )

    def forbidden_pytest(*args, **kwargs):
        raise AssertionError("pytest must not run for a noncanonical config")

    monkeypatch.setattr(
        feature_benchmark_module,
        "run_focused_correctness_gate",
        forbidden_pytest,
    )
    result = benchmark_article_v1_features(
        external,
        output_root=tmp_path,
        run_id="foreign-config",
        write_profiles=False,
    )

    assert result["engineering_qualification_passed"] is False
    assert "frozen checked-in pilot" in result["abort_reason"]
    status = json.loads(Path(result["status_manifest"]).read_text(encoding="utf-8"))
    assert status["phase"] == "configuration-preflight"
    assert status["config_binding"]["matches_frozen_pilot"] is False
    assert not (tmp_path / "foreign-config" / "microbenchmarks.csv").exists()


def test_feature_benchmark_detects_source_change_before_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def changing_provenance(*, refresh: bool = False):
        nonlocal calls
        calls += 1
        value = _stable_git_provenance(refresh=refresh)
        if calls >= 2:
            value["source_worktree_digest"] = "sha256:changed-before-timing"
        return value

    monkeypatch.setattr(
        article_v1_runner_module, "git_provenance", changing_provenance
    )
    monkeypatch.setattr(
        feature_benchmark_module,
        "inspect_production_dominance_update",
        lambda: {"passed": True, "checks": {}},
    )

    def passing_correctness(output, **kwargs):
        assert kwargs["evidence_binding"]["source"][
            "source_worktree_digest"
        ] == "sha256:benchmark-test-source"
        gate = _passing_gate()
        return gate, {
            "passed": True,
            "junit": {"sha256": "sha256:junit"},
            "evidence_binding": kwargs["evidence_binding"],
        }

    monkeypatch.setattr(
        feature_benchmark_module,
        "run_focused_correctness_gate",
        passing_correctness,
    )

    def forbidden_timing(*args, **kwargs):
        raise AssertionError("source drift must abort before timing")

    monkeypatch.setattr(
        feature_benchmark_module,
        "run_repository_feature_benchmark",
        forbidden_timing,
    )
    result = benchmark_article_v1_features(
        "pilot",
        output_root=tmp_path,
        run_id="source-drift",
        write_profiles=False,
    )

    assert result["engineering_qualification_passed"] is False
    assert result["aborted_before_timing"] is True
    assert "changed before timing" in result["abort_reason"]
    status = json.loads(Path(result["status_manifest"]).read_text(encoding="utf-8"))
    assert status["pre_timing_integrity"]["source_unchanged"] is False


def test_feature_benchmark_command_uses_fixed_axes_and_returns_six_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    implementation = {
        "schema_version": "article-v1-dominance-source-check-v1",
        "passed": True,
        "checks": {"no_python_all_open_record_nested_loop": True},
    }
    monkeypatch.setattr(
        feature_benchmark_module,
        "inspect_production_dominance_update",
        lambda: implementation,
    )
    monkeypatch.setattr(
        article_v1_runner_module, "git_provenance", _stable_git_provenance
    )

    def passing_correctness(output, **kwargs):
        profiles = Path(output) / "profiles"
        profiles.mkdir(parents=True, exist_ok=True)
        evidence = profiles / "correctness_gate.junit.xml"
        evidence.write_text(_successful_gate_junit(), encoding="utf-8")
        gate = CorrectnessGate(
            checks={name: True for name in REQUIRED_CORRECTNESS_CHECKS},
            command="python -m pytest focused",
            evidence=(str(evidence),),
        )
        return gate, {"returncode": 0, "passed": True}

    monkeypatch.setattr(
        feature_benchmark_module,
        "run_focused_correctness_gate",
        passing_correctness,
    )
    captured: dict[str, object] = {}

    def fast_repository_benchmark(output, **kwargs):
        captured.update(kwargs)
        return write_feature_benchmark_artifacts(
            output,
            correctness_gate=kwargs["correctness_gate"],
            microbenchmarks=_micro_measurements(),
            end_to_end=_end_to_end_measurements(),
            implementation_checks=kwargs["implementation_checks"],
            baseline=baseline_f653193(environment={"machine": "fixed"}),
            expected_frontier_sizes=kwargs["frontier_sizes"],
            expected_staged_caps=kwargs["staged_caps"],
            pilot_relaunch_checks=kwargs["pilot_relaunch_checks"],
            benchmark_provenance=kwargs["benchmark_provenance"],
        )

    monkeypatch.setattr(
        feature_benchmark_module,
        "run_repository_feature_benchmark",
        fast_repository_benchmark,
    )
    result = benchmark_article_v1_features(
        "configs/article_v1_pilot.json",
        output_root=tmp_path,
        run_id="fast-fixed-axes",
        write_profiles=False,
    )

    assert result["passed"] is True
    assert result["engineering_qualification_passed"] is True
    assert result["pilot_relaunch_ready"] is False
    assert result["aborted_before_timing"] is False
    assert captured["frontier_sizes"] == DEFAULT_FRONTIER_SIZES
    assert captured["staged_caps"] == DEFAULT_STAGED_EXPANSION_CAPS
    assert len(result["artifacts"]) == 6
    assert all(Path(path).exists() for path in result["artifacts"].values())
    status = json.loads(Path(result["status_manifest"]).read_text(encoding="utf-8"))
    assert status["engineering_qualification_passed"] is True
    assert status["pilot_relaunch_ready"] is False
    assert status["artifact_manifest"]["required_six_artifacts_complete"] is True
    assert status["artifact_manifest"]["files"]["baseline.json"][
        "sha256"
    ].startswith("sha256:")


def test_benchmark_features_cli_dispatches_through_both_entry_points(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: list[tuple[object, dict[str, object]]] = []

    def fake_benchmark(config, **kwargs):
        captured.append((config, kwargs))
        return {
            "schema_version": "article-v1-feature-benchmark-command-v1",
            "passed": True,
            "engineering_qualification_passed": True,
            "pilot_relaunch_ready": False,
            "frontier_sizes": list(DEFAULT_FRONTIER_SIZES),
            "staged_expansion_caps": list(DEFAULT_STAGED_EXPANSION_CAPS),
        }

    monkeypatch.setattr(
        article_v1_runner_module,
        "benchmark_article_v1_features",
        fake_benchmark,
    )
    arguments = [
        "benchmark-features",
        "--config",
        "configs/article_v1_pilot.json",
        "--output-root",
        str(tmp_path),
        "--run-id",
        "cli-test",
        "--no-profiles",
    ]
    assert article_benchmark.main(arguments) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["frontier_sizes"] == list(DEFAULT_FRONTIER_SIZES)
    assert payload["staged_expansion_caps"] == list(DEFAULT_STAGED_EXPANSION_CAPS)
    assert len(captured) == 1
    config, keywords = captured[0]
    assert config == Path("configs/article_v1_pilot.json")
    assert keywords["run_id"] == "cli-test"
    assert keywords["write_profiles"] is False
    assert keywords["microbenchmark_repetitions"] == 31
    assert keywords["microbenchmark_warmups"] == 5
