from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from reporting.article_v1 import (
    ARTICLE_V1_REPORT_SCHEMA,
    AppendOnlyJSONLRunStore,
    aggregate_article_v1_runs,
    bootstrap_mean_ci,
    load_completed_run_keys,
    unique_run_key,
    write_article_v1_report,
)


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _run(
    *,
    target: str,
    scheduler: str,
    seed: int,
    certified: bool,
    expansions: int,
    expansion_budget: int = 16,
    checkpoint: str = "none",
    training_seed: int | None = None,
    runtime: float = 0.01,
) -> dict[str, object]:
    if checkpoint != "none" and training_seed is None:
        training_seed = 0
    return {
        "target_id": target,
        "scheduler": scheduler,
        "resource_budget": {
            "max_t_count": 4,
            "max_two_qubit_count": 4,
            "max_gates": 6,
            "max_depth": 6,
        },
        "expansion_budget": expansion_budget,
        "checkpoint_digest": checkpoint,
        "training_seed": training_seed,
        "evaluation_seed": seed,
        "feature_schema_version": "article-v1-31d",
        "feature_evaluator_schema_version": "article-v1-exact-incremental-v2",
        "reward_schema_version": "article-v1-expansion-potential-amended",
        "target_metric_schema_version": "projective-unitary-metrics-v2",
        "certification_schema_version": "phase-frobenius-raw-v2",
        "code_version": "deadbeef",
        "source_worktree_digest": "sha256:source-a",
        "certified": certified,
        "terminated": certified,
        "truncated": not certified,
        "expansions": expansions,
        "runtime_seconds": runtime,
        "timings": {
            "ranking_time_seconds": runtime * 0.10,
            "feature_time_seconds": runtime * 0.20,
            "target_metric_time_seconds": runtime * 0.15,
            "canonicalization_time_seconds": runtime * 0.10,
            "archive_time_seconds": runtime * 0.10,
            "certification_time_seconds": runtime * 0.05,
        },
        "metrics": {
            "feature_evaluations": expansions * 2,
            "dense_target_evaluations": expansions,
            "target_metric_cache_hits": expansions // 2,
            "peak_frontier": expansions + 2,
            "peak_archive": expansions + 4,
        },
        "solution_resource_vector": (
            [1, 1, 3, 2, 2] if certified else None
        ),
    }


def test_unique_run_key_is_canonical_and_covers_every_identity_coordinate() -> None:
    base = _run(
        target="target-a",
        scheduler="fifo",
        seed=1,
        certified=True,
        expansions=3,
    )
    reordered = dict(reversed(list(base.items())))
    reordered["resource_budget"] = dict(
        reversed(list(base["resource_budget"].items()))
    )
    assert unique_run_key(base) == unique_run_key(reordered)

    mutations = (
        {"target_id": "target-b"},
        {"scheduler": "lifo"},
        {"resource_budget": {**base["resource_budget"], "max_gates": 7}},
        {"expansion_budget": 32},
        {"checkpoint_digest": "sha256:policy", "training_seed": 1},
        {"evaluation_seed": 2},
        {"feature_schema_version": "different-feature"},
        {"feature_evaluator_schema_version": "different-evaluator"},
        {"reward_schema_version": "different-reward"},
        {"reward_parameters": {"beta": 0.0}},
        {"target_metric_schema_version": "different-target-metric"},
        {"certification_schema_version": "different-certifier"},
        {"certification_parameters": {"phase_frobenius_tolerance": 1e-8}},
        {
            "search_reduction": {
                "canonicalization_enabled": True,
                "pareto_dominance_enabled": False,
                "absorb_clifford_angles": True,
            }
        },
        {"code_version": "cafebabe"},
        {"source_worktree_digest": "sha256:source-b"},
    )
    for mutation in mutations:
        assert unique_run_key({**base, **mutation}) != unique_run_key(base)


def test_store_rejects_pre_v2_raw_ledger_schema(tmp_path: Path) -> None:
    store = AppendOnlyJSONLRunStore(tmp_path / "raw_runs.jsonl")
    old = _run(
        target="old", scheduler="fifo", seed=0, certified=False, expansions=1
    )
    old["schema_version"] = "article-v1-raw-run-v1"

    with pytest.raises(ValueError, match="unsupported Article V1 raw-run schema"):
        store.append(old)
    assert not store.path.exists()


def test_jsonl_store_resumes_repairs_partial_tail_and_preserves_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "raw_runs.jsonl"
    store = AppendOnlyJSONLRunStore(path)
    success = _run(
        target="a", scheduler="fifo", seed=1, certified=True, expansions=2
    )
    failure = _run(
        target="b", scheduler="fifo", seed=1, certified=False, expansions=16
    )
    assert store.append(success)
    assert store.append(failure)
    assert not store.append(success)
    assert len(store.completed_keys()) == 2
    assert any(not record["certified"] for record in store.load_records())

    with path.open("ab") as handle:
        handle.write(b'{"target_id":"partial"')
    loaded = store.load_records(repair_partial=True)
    assert len(loaded) == 2
    assert path.read_bytes().endswith(b"\n")
    assert load_completed_run_keys(path) == store.completed_keys()


def test_store_rejects_conflicting_payload_for_one_completed_key(tmp_path: Path) -> None:
    store = AppendOnlyJSONLRunStore(tmp_path / "raw_runs.jsonl")
    run = _run(target="a", scheduler="fifo", seed=1, certified=True, expansions=2)
    assert store.append(run)
    with pytest.raises(ValueError, match="conflicting payloads"):
        store.append({**run, "runtime_seconds": 99.0})


def test_aggregation_uses_targets_not_repeated_eval_seeds_as_sample_units() -> None:
    runs = [
        # Deterministic repeats: target a succeeds twice; target b fails twice.
        _run(target="a", scheduler="fifo", seed=1, certified=True, expansions=2),
        _run(target="a", scheduler="fifo", seed=2, certified=True, expansions=2),
        _run(target="b", scheduler="fifo", seed=1, certified=False, expansions=16),
        _run(target="b", scheduler="fifo", seed=2, certified=False, expansions=16),
        # Random behavior is averaged within target before targets are averaged.
        _run(target="a", scheduler="random", seed=1, certified=True, expansions=4),
        _run(target="a", scheduler="random", seed=2, certified=False, expansions=16),
        _run(target="b", scheduler="random", seed=1, certified=True, expansions=6),
        _run(target="b", scheduler="random", seed=2, certified=True, expansions=8),
    ]
    aggregate = aggregate_article_v1_runs(
        runs,
        stats_seed=7,
        bootstrap_samples=300,
    )
    curves = {row["scheduler"]: row for row in aggregate["success_curves"]}

    assert curves["fifo"]["target_count"] == 2
    assert curves["fifo"]["trajectory_count"] == 4
    assert curves["fifo"]["success_rate"] == 0.5
    assert curves["fifo"]["conditional_successful_expansions_mean"] == 2.0
    assert curves["random"]["target_count"] == 2
    assert curves["random"]["success_rate"] == 0.75
    assert curves["random"]["conditional_successful_expansions_mean"] == 5.5
    fifo_targets = [
        row for row in aggregate["per_target"] if row["scheduler"] == "fifo"
    ]
    assert {row["target_id"] for row in fifo_targets} == {"a", "b"}
    assert sum(row["failures"] for row in fifo_targets) == 2


def test_aggregation_never_synthesizes_a_smaller_budget_from_a_larger_cap() -> None:
    run = _run(
        target="a",
        scheduler="fifo",
        seed=0,
        certified=True,
        expansions=3,
        expansion_budget=10,
    )
    aggregate = aggregate_article_v1_runs(
        [run], budgets=(4, 10), stats_seed=3, bootstrap_samples=50
    )
    assert {row["expansion_budget"] for row in aggregate["per_target"]} == {10}
    assert {row["expansion_budget"] for row in aggregate["success_curves"]} == {10}


def test_bootstrap_and_paired_target_differences_are_deterministic() -> None:
    values = [0.0, 0.5, 1.0]
    assert bootstrap_mean_ci(values, stats_seed=11, samples=400) == bootstrap_mean_ci(
        values, stats_seed=11, samples=400
    )
    runs = [
        _run(target="a", scheduler="fifo", seed=1, certified=True, expansions=5),
        _run(target="b", scheduler="fifo", seed=1, certified=False, expansions=16),
        _run(target="a", scheduler="greedy", seed=1, certified=True, expansions=2),
        _run(target="b", scheduler="greedy", seed=1, certified=True, expansions=3),
    ]
    first = aggregate_article_v1_runs(runs, stats_seed=3, bootstrap_samples=300)
    second = aggregate_article_v1_runs(runs, stats_seed=3, bootstrap_samples=300)
    assert first == second
    pair = first["paired_differences"][0]
    assert pair["paired_target_count"] == 2
    assert pair["success_difference_mean"] == -0.5
    assert pair["paired_successful_expansion_target_count"] == 1
    target_pairs = first["paired_per_target_differences"]
    assert {row["target_id"] for row in target_pairs} == {"a", "b"}
    assert {row["success_difference"] for row in target_pairs} == {0.0, -1.0}


def test_reporting_separates_splits_and_summarizes_independent_learner_seeds():
    runs = []
    outcomes = {
        ("a", "checkpoint-1"): True,
        ("a", "checkpoint-2"): False,
        ("b", "checkpoint-1"): True,
        ("b", "checkpoint-2"): True,
    }
    for (target, checkpoint), certified in outcomes.items():
        runs.append(
            {
                **_run(
                    target=target,
                    scheduler="article_sarsa",
                    seed=0,
                    certified=certified,
                    expansions=3 if certified else 16,
                    checkpoint=checkpoint,
                ),
                "split": "test",
                "difficulty": "easy",
                "training_seed": 1 if checkpoint.endswith("1") else 2,
            }
        )
    runs.append(
        {
            **_run(
                target="ood-a",
                scheduler="fifo",
                seed=0,
                certified=False,
                expansions=16,
            ),
            "split": "ood_test",
            "difficulty": "medium",
        }
    )
    aggregate = aggregate_article_v1_runs(
        runs, stats_seed=31, bootstrap_samples=200
    )

    learner = aggregate["learner_seed_summary"]
    assert len(learner) == 1
    assert learner[0]["split"] == "test"
    assert learner[0]["difficulty"] == "easy"
    assert learner[0]["checkpoint_count"] == 2
    assert learner[0]["learner_seed_count"] == 2
    assert learner[0]["unique_checkpoint_digest_count"] == 2
    assert learner[0]["training_seeds"] == [1, 2]
    assert learner[0]["target_count"] == 2
    assert learner[0]["target_seed_pair_count"] == 4
    assert learner[0]["success_rate_mean"] == pytest.approx(0.75)
    assert learner[0]["success_rate_median"] == pytest.approx(0.75)
    assert learner[0]["success_rate_std"] == pytest.approx(0.25)
    assert learner[0]["learner_success_rate_std"] == pytest.approx(0.25)
    per_learner = aggregate["learner_seed_results"]
    assert [(row["training_seed"], row["success_rate_mean"]) for row in per_learner] == [
        (1, 1.0),
        (2, 0.5),
    ]
    assert {row["split"] for row in aggregate["success_curves"]} == {
        "test",
        "ood_test",
    }


def test_report_artifacts_are_rebuilt_from_raw_jsonl_only(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw_runs.jsonl"
    store = AppendOnlyJSONLRunStore(raw_path)
    for run in (
        _run(target="a", scheduler="fifo", seed=1, certified=True, expansions=3),
        _run(target="b", scheduler="fifo", seed=1, certified=False, expansions=16),
        _run(target="a", scheduler="random", seed=1, certified=True, expansions=2),
        _run(target="b", scheduler="random", seed=1, certified=True, expansions=7),
    ):
        assert store.append(run)
    original_raw = raw_path.read_bytes()

    paths = write_article_v1_report(
        raw_path,
        tmp_path / "publication",
        stats_seed=5,
        bootstrap_samples=200,
    )

    assert raw_path.read_bytes() == original_raw
    assert all(Path(path).is_file() for path in paths.values())
    with Path(paths["success_curves"]).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["scheduler"] for row in rows} == {"fifo", "random"}
    assert "<svg" in Path(paths["success_figure"]).read_text(encoding="utf-8")
    assert "<svg" in Path(paths["expansion_figure"]).read_text(encoding="utf-8")
    summary = Path(paths["completion_summary"]).read_text(encoding="utf-8")
    assert "Failed and truncated runs remain" in summary
    assert "not counted as independent held-out targets" in summary
    assert "do not establish circuit optimality" in summary


def test_report_binds_audited_raw_digest_without_repairing(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw_runs.jsonl"
    store = AppendOnlyJSONLRunStore(raw_path)
    assert store.append(
        _run(target="a", scheduler="fifo", seed=0, certified=True, expansions=2)
    )
    original_raw = raw_path.read_bytes()
    expected_digest = _sha256(raw_path)

    paths = write_article_v1_report(
        raw_path,
        tmp_path / "publication",
        stats_seed=2,
        bootstrap_samples=30,
        expected_raw_sha256=expected_digest,
    )

    metadata = json.loads(
        Path(paths["report_metadata"]).read_text(encoding="utf-8")
    )
    assert raw_path.read_bytes() == original_raw
    assert metadata["schema_version"] == ARTICLE_V1_REPORT_SCHEMA
    assert metadata["raw_ledger_sha256"] == expected_digest
    assert metadata["raw_ledger_digest_bound"] is True


def test_report_rejects_stale_raw_digest_before_writing_artifacts(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw_runs.jsonl"
    store = AppendOnlyJSONLRunStore(raw_path)
    assert store.append(
        _run(target="a", scheduler="fifo", seed=0, certified=True, expansions=2)
    )
    expected_digest = _sha256(raw_path)
    with raw_path.open("ab") as handle:
        handle.write(b"\n")
    destination = tmp_path / "publication"

    with pytest.raises(ValueError, match="differs before report loading"):
        write_article_v1_report(
            raw_path,
            destination,
            expected_raw_sha256=expected_digest,
        )
    assert not destination.exists()


def test_report_rejects_raw_mutation_during_load_before_writing_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_path = tmp_path / "raw_runs.jsonl"
    store = AppendOnlyJSONLRunStore(raw_path)
    assert store.append(
        _run(target="a", scheduler="fifo", seed=0, certified=True, expansions=2)
    )
    expected_digest = _sha256(raw_path)
    original_load = AppendOnlyJSONLRunStore.load_records

    def mutating_load(self, *, repair_partial=True):
        assert repair_partial is False
        records = original_load(self, repair_partial=repair_partial)
        with self.path.open("ab") as handle:
            handle.write(b"\n")
        return records

    monkeypatch.setattr(AppendOnlyJSONLRunStore, "load_records", mutating_load)
    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="changed during report loading"):
        write_article_v1_report(
            raw_path,
            destination,
            expected_raw_sha256=expected_digest,
        )
    assert not destination.exists()


def test_report_without_digest_preserves_partial_line_repair(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw_runs.jsonl"
    store = AppendOnlyJSONLRunStore(raw_path)
    assert store.append(
        _run(target="a", scheduler="fifo", seed=0, certified=True, expansions=2)
    )
    complete_raw = raw_path.read_bytes()
    with raw_path.open("ab") as handle:
        handle.write(b'{"incomplete":')

    write_article_v1_report(raw_path, tmp_path / "publication")

    assert raw_path.read_bytes() == complete_raw


@pytest.mark.parametrize(
    "invalid_digest",
    (
        "sha256:short",
        "SHA256:" + "0" * 64,
        "sha256:" + "A" * 64,
        "sha256:" + "g" * 64,
    ),
)
def test_report_requires_canonical_expected_raw_digest(
    tmp_path: Path, invalid_digest: str
) -> None:
    destination = tmp_path / "publication"
    with pytest.raises(ValueError, match="canonical sha256"):
        write_article_v1_report(
            tmp_path / "not-read.jsonl",
            destination,
            expected_raw_sha256=invalid_digest,
        )
    assert not destination.exists()


def test_report_paths_are_repo_relative_when_output_is_below_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    raw_path = Path("outputs/run/raw_runs.jsonl")
    store = AppendOnlyJSONLRunStore(raw_path)
    assert store.append(
        _run(target="a", scheduler="fifo", seed=0, certified=True, expansions=2)
    )
    paths = write_article_v1_report(
        raw_path,
        Path("outputs/run"),
        stats_seed=2,
        bootstrap_samples=30,
    )
    assert all(not Path(path).is_absolute() for path in paths.values())
    assert all("\\" not in path for path in paths.values())
    assert all(Path(path).is_file() for path in paths.values())


def test_identical_weights_from_independent_training_seeds_do_not_collide() -> None:
    first = _run(
        target="a",
        scheduler="article_sarsa",
        seed=0,
        certified=True,
        expansions=2,
        checkpoint="sha256:identical-weights",
        training_seed=19,
    )
    second = {**first, "training_seed": 23}
    assert unique_run_key(first) != unique_run_key(second)
    aggregate = aggregate_article_v1_runs(
        [first, second], stats_seed=2, bootstrap_samples=50
    )
    assert len(aggregate["per_target"]) == 2
    summary = aggregate["learner_seed_summary"][0]
    assert summary["checkpoint_count"] == 2
    assert summary["unique_checkpoint_digest_count"] == 1
    assert summary["learner_seed_count"] == 2


def test_missing_identity_field_is_rejected() -> None:
    incomplete = _run(
        target="a", scheduler="fifo", seed=1, certified=True, expansions=2
    )
    del incomplete["code_version"]
    with pytest.raises(KeyError, match="code version"):
        unique_run_key(incomplete)


def test_raw_store_is_valid_jsonl_with_explicit_run_keys(tmp_path: Path) -> None:
    path = tmp_path / "raw_runs.jsonl"
    store = AppendOnlyJSONLRunStore(path)
    run = _run(target="a", scheduler="fifo", seed=1, certified=False, expansions=16)
    assert store.append(run)

    decoded = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(decoded) == 1
    assert decoded[0]["run_key"] == unique_run_key(run)
    assert decoded[0]["certified"] is False
