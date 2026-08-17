from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import article_benchmark
from benchmarks.article_native_corpus import (
    ARTICLE_V1_TRAINING_BUDGET_POLICY,
    OOD_LENGTH_CHECKPOINT_FAMILY,
    STANDARD_CHECKPOINT_FAMILY,
    ArticleV1CheckpointScope,
    build_article_v1_corpus,
)
from experiments.article_v1_runner import (
    ARTICLE_V1_CAMPAIGN_AUDIT_SCHEMA,
    ARTICLE_V1_CHECKPOINT_SCHEMA,
    PRIMARY_SCHEDULERS,
    ArticleV1Checkpoint,
    _assert_checked_in_pilot_publication_disjoint,
    _audit_campaign_record,
    _CANONICAL_RAW_RECORD_FIELDS,
    _CANONICAL_SEARCH_METRIC_FIELDS,
    _expected_matrix_runs,
    _independent_witness_certification_diagnostics,
    _independent_witness_resource_vector,
    _REQUIRED_COUNTER_FIELDS,
    _REQUIRED_SUMMARY_COUNTER_FIELDS,
    _REQUIRED_TIMING_FIELDS,
    _SCHEDULER_SEMANTICS,
    audit_article_v1_campaign,
    environment_metadata,
    evaluate_article_v1_run,
    git_provenance,
    main as article_v1_main,
)
from experiments.profiles import ARTICLE_V1_PROFILE
from reporting.article_v1 import ARTICLE_V1_RAW_RUN_SCHEMA, unique_run_key
from rl.article_features import ARTICLE_V1_FEATURE_NAMES, ArticleTargetContext
from search.action_space import generate_actions


def _checkpoint(scope: ArticleV1CheckpointScope, seed: int) -> ArticleV1Checkpoint:
    weights = [0.0] * len(ARTICLE_V1_FEATURE_NAMES)
    weights[0] = 0.125
    return ArticleV1Checkpoint(
        training_seed=seed,
        weights=tuple(weights),
        feature_schema_version=scope.expected_feature_schema_version,
        ordered_feature_names=tuple(ARTICLE_V1_FEATURE_NAMES),
        reward_schema_version=ARTICLE_V1_PROFILE.reward_schema,
        target_metric_schema_version=ARTICLE_V1_PROFILE.target_metric_schema,
        certification_schema_version=ARTICLE_V1_PROFILE.certification_schema,
        learning_rate=scope.expected_learning_rate,
        discount=ARTICLE_V1_PROFILE.gamma,
        epsilon_schedule=scope.expected_epsilon_schedule,
        checkpoint_family=scope.checkpoint_family,
        training_scope_mode=scope.training_scope_mode,
        training_beta=scope.expected_training_beta,
        training_certification_tolerance=(
            scope.expected_certification_tolerance
        ),
        training_episodes_per_target=scope.expected_episodes_per_target,
        training_expansion_cap=scope.expected_expansion_cap,
        training_budget_policy=ARTICLE_V1_TRAINING_BUDGET_POLICY,
        effective_training_expansion_budgets=(
            scope.expected_training_expansion_budgets
        ),
        training_target_ids=scope.allowed_training_target_ids,
        training_histories=(),
        corpus_config_digest=scope.corpus_config_digest,
    )


@pytest.fixture(scope="module")
def campaign_fixture():
    corpus = build_article_v1_corpus("pilot")
    standard_scope = corpus.checkpoint_scope(
        checkpoint_family=STANDARD_CHECKPOINT_FAMILY
    )
    ood_scope = corpus.checkpoint_scope(
        checkpoint_family=OOD_LENGTH_CHECKPOINT_FAMILY
    )
    checkpoints = tuple(
        _checkpoint(standard_scope, seed)
        for seed in standard_scope.allowed_training_seeds
    )
    ood_checkpoints = tuple(
        _checkpoint(ood_scope, seed) for seed in ood_scope.allowed_training_seeds
    )
    provenance = git_provenance()
    specs = (
        _expected_matrix_runs(
            corpus.evaluation_targets(split="test"),
            config=corpus.config,
            checkpoints=checkpoints,
            checkpoint_scope=standard_scope,
            schedulers=PRIMARY_SCHEDULERS,
            provenance=provenance,
        )
        + _expected_matrix_runs(
            corpus.evaluation_targets(split="ood_test"),
            config=corpus.config,
            checkpoints=ood_checkpoints,
            checkpoint_scope=ood_scope,
            schedulers=PRIMARY_SCHEDULERS,
            provenance=provenance,
        )
    )
    records = [_failure_record(spec, corpus.config, provenance) for spec in specs]
    return corpus, checkpoints, ood_checkpoints, specs, records


def _failure_record(spec, config, provenance) -> dict[str, object]:
    case = spec.case
    expansions = spec.expansion_budget
    native_gate_count = len(generate_actions(case.num_qubits))
    generated = expansions * native_gate_count
    certification_nonmatch = generated
    accepted = certification_nonmatch + 1
    frontier_sum = (
        expansions
        + (native_gate_count - 1) * expansions * (expansions - 1) // 2
    )
    final_frontier = 1 + expansions * (native_gate_count - 1)
    search_metrics = {
        name: 0 for name in _REQUIRED_COUNTER_FIELDS
    }
    for name in _REQUIRED_TIMING_FIELDS:
        search_metrics[name.removesuffix("_seconds") + "_ns"] = 0
    search_metrics["wall_time_ns"] = 1_000_000
    search_metrics["environment_step_time_ns"] = 100
    search_metrics["certification_time_ns"] = 100
    search_metrics.update({
        "generated": generated,
        "num_generated": generated,
        "certification_nonmatch": certification_nonmatch,
        "terminal_candidates": generated,
        "certification_count": generated + 1,
        "accepted": accepted,
        "archive_record_count": accepted,
        "archive_size": accepted,
        "active_archive_peak": accepted,
        "peak_active_archive_records": accepted,
        "frontier_peak": final_frontier,
        "peak_frontier": final_frontier,
        "peak_frontier_records": final_frontier,
        "pareto_width_peak": 1,
        "maximum_pareto_antichain_width": 1,
        "frontier_sum": frontier_sum,
        "frontier_observation_count": expansions,
        "frontier_mean": (frontier_sum + final_frontier) / (expansions + 1),
        "frontier_decision_mean": frontier_sum / expansions,
        "num_gate_attempts": expansions * native_gate_count,
        "expanded": expansions,
        "num_expanded": expansions,
    })
    metrics = {
        summary_name: int(search_metrics[search_name])
        for summary_name, search_name in _REQUIRED_SUMMARY_COUNTER_FIELDS.items()
    }
    checkpoint = spec.checkpoint
    scope = spec.checkpoint_scope
    timings = {name: 0.0 for name in _REQUIRED_TIMING_FIELDS}
    timings["wall_time_seconds"] = 0.001
    timings["environment_step_time_seconds"] = 1e-7
    timings["certification_time_seconds"] = 1e-7
    record: dict[str, object] = {
        "schema_version": ARTICLE_V1_RAW_RUN_SCHEMA,
        "target_id": case.target_id,
        "target_fingerprint": ArticleTargetContext(case.target).fingerprint,
        "config_digest": config.digest,
        "split": case.split,
        "difficulty": case.difficulty,
        "num_qubits": case.num_qubits,
        "generator_length": case.generator_length,
        "budget": case.budget.metadata(),
        "resource_budget": dict(spec.identity["resource_budget"]),
        "scheduler": spec.scheduler,
        "scheduler_semantics": _SCHEDULER_SEMANTICS[spec.scheduler],
        "action_semantics": "persistent_frontier_record",
        "expansion_budget": spec.expansion_budget,
        "checkpoint_digest": (
            "none" if checkpoint is None else checkpoint.weight_digest
        ),
        "checkpoint_family": (
            None if checkpoint is None else checkpoint.checkpoint_family
        ),
        "checkpoint_scope_schema": None if scope is None else scope.schema_version,
        "training_seed": None if checkpoint is None else checkpoint.training_seed,
        "evaluation_seed": spec.evaluation_seed,
        "feature_schema_version": spec.identity["feature_schema"],
        "reward_schema_version": spec.identity["reward_schema"],
        "reward_parameters": dict(spec.identity["reward_parameters"]),
        "target_metric_schema_version": spec.identity["target_metric_schema"],
        "certification_schema_version": spec.identity["certifier_schema"],
        "certification_parameters": dict(
            spec.identity["certification_parameters"]
        ),
        "code_version": provenance["commit_sha"],
        "source_worktree_digest": provenance["source_worktree_digest"],
        "dirty_worktree": provenance["dirty_worktree"],
        "certified": False,
        "terminated": False,
        "truncated": True,
        "expansions": expansions,
        "runtime_seconds": 0.001,
        "time_to_solution": None,
        "timings": timings,
        "metrics": metrics,
        "search_metrics": search_metrics,
        "solution_resource_vector": None,
        "witness_operations": [],
        "certification_diagnostics": None,
        "reference_witness_used": False,
        "target_specific_reachability_oracle": False,
        "profile": ARTICLE_V1_PROFILE.metadata(),
        "search_reduction": dict(spec.identity["search_reduction"]),
        "evaluation_weights_frozen": True,
        "evaluation_reward_consumed_by_policy": False,
        "raw_run_schema": ARTICLE_V1_RAW_RUN_SCHEMA,
    }
    record["run_key"] = unique_run_key(record)
    return record


def _write_ledger(
    path: Path,
    records: list[dict[str, object]],
    *,
    terminal_newline: bool = True,
    allow_nan: bool = False,
) -> None:
    lines = [
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=allow_nan,
        )
        for record in records
    ]
    path.write_text(
        "\n".join(lines) + ("\n" if terminal_newline else ""),
        encoding="utf-8",
    )


def _set_native_counter_model(
    record: dict[str, object],
    *,
    expansions: int,
    certified: bool,
    exhausted: bool = False,
) -> None:
    search_metrics = record["search_metrics"]
    native_gate_count = len(generate_actions(int(record["num_qubits"])))
    generated = expansions * native_gate_count
    certification_nonmatch = generated - int(certified)
    duplicate_rejected = certification_nonmatch if exhausted else 0
    accepted = certification_nonmatch + 1 - duplicate_rejected
    frontier_sum = (
        expansions
        + (native_gate_count - 1) * expansions * (expansions - 1) // 2
    )
    final_frontier = (
        0 if exhausted else 1 - expansions + certification_nonmatch
    )
    search_metrics.update({
        "generated": generated,
        "num_generated": generated,
        "certification_nonmatch": certification_nonmatch,
        "terminal_candidates": generated,
        "certification_count": generated + 1,
        "duplicate_rejected": duplicate_rejected,
        "canonical_pruned": duplicate_rejected,
        "num_exact_duplicate_rejections": duplicate_rejected,
        "num_dominance_rejections": 0,
        "accepted": accepted,
        "archive_record_count": accepted,
        "archive_size": accepted,
        "active_archive_peak": accepted,
        "peak_active_archive_records": accepted,
        "frontier_peak": max(1, final_frontier),
        "peak_frontier": max(1, final_frontier),
        "peak_frontier_records": max(1, final_frontier),
        "pareto_width_peak": 1,
        "maximum_pareto_antichain_width": 1,
        "frontier_sum": frontier_sum,
        "frontier_observation_count": expansions,
        "frontier_mean": (frontier_sum + final_frontier) / (expansions + 1),
        "frontier_decision_mean": frontier_sum / expansions,
        "num_gate_attempts": expansions * native_gate_count,
        "expanded": expansions,
        "num_expanded": expansions,
    })
    record["expansions"] = expansions
    record["certified"] = certified
    for summary_name, search_name in _REQUIRED_SUMMARY_COUNTER_FIELDS.items():
        record["metrics"][summary_name] = int(search_metrics[search_name])


def _audit(tmp_path: Path, fixture, records):
    corpus, checkpoints, ood_checkpoints, _specs, _base_records = fixture
    raw_path = tmp_path / "raw_runs.jsonl"
    _write_ledger(raw_path, records)
    return audit_article_v1_campaign(
        corpus,
        checkpoints=checkpoints,
        ood_checkpoints=ood_checkpoints,
        raw_path=raw_path,
        output_path=tmp_path / "campaign_audit.json",
    )


def test_complete_exact_campaign_writes_byte_bound_passing_audit(
    tmp_path: Path, campaign_fixture
) -> None:
    records = deepcopy(campaign_fixture[4])
    result = _audit(tmp_path, campaign_fixture, records)

    assert result["schema_version"] == ARTICLE_V1_CAMPAIGN_AUDIT_SCHEMA
    assert result["passed"] is True
    assert result["expected_run_count"] == result["observed_run_count"] == 300
    assert result["expected_by_split"] == result["observed_by_split"] == {
        "test": 180,
        "ood_test": 120,
    }
    assert result["raw_ledger_path"] == "raw_runs.jsonl"
    assert str(result["raw_ledger_sha256"]).startswith("sha256:")
    assert all(result["integrity_checks"].values())
    assert json.loads(
        (tmp_path / "campaign_audit.json").read_text(encoding="utf-8")
    ) == result


def test_truncation_requires_budget_exhaustion_but_early_terminal_failure_is_valid(
    tmp_path: Path, campaign_fixture
) -> None:
    records = deepcopy(campaign_fixture[4])
    _set_native_counter_model(
        records[0],
        expansions=int(records[0]["expansion_budget"]) - 1,
        certified=False,
    )
    with pytest.raises(ValueError, match="must exhaust its expansion budget"):
        _audit(tmp_path, campaign_fixture, records)

    records = deepcopy(campaign_fixture[4])
    records[0]["terminated"] = True
    records[0]["truncated"] = False
    _set_native_counter_model(
        records[0], expansions=1, certified=False, exhausted=True
    )
    assert _audit(tmp_path, campaign_fixture, records)["passed"] is True

    records[0]["expansions"] = 0
    records[0]["search_metrics"]["expanded"] = 0
    records[0]["search_metrics"]["num_expanded"] = 0
    with pytest.raises(ValueError, match="at least one search expansion"):
        _audit(tmp_path, campaign_fixture, records)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda row: row.__setitem__("target_fingerprint", "sha256:forged"), "fingerprint"),
        (lambda row: row.__setitem__("reference_witness_used", True), "reference_witness"),
        (lambda row: row.__setitem__("scheduler_semantics", "forged"), "scheduler_semantics"),
        (lambda row: row.__setitem__("dirty_worktree", not row["dirty_worktree"]), "dirty_worktree"),
        (lambda row: row["search_metrics"].__setitem__("expanded", -1), "non-negative"),
        (lambda row: row["timings"].__setitem__("wall_time_seconds", 1.0), "disagrees"),
        (lambda row: row.__setitem__("truncated", False), "exactly one"),
        (lambda row: row.__setitem__("unexpected_field", 1), "canonical raw schema"),
        (lambda row: row.pop("certification_parameters"), "invalid run key"),
    ),
)
def test_audit_rejects_incompatible_nonidentity_evidence(
    tmp_path: Path, campaign_fixture, mutation, match
) -> None:
    records = deepcopy(campaign_fixture[4])
    mutation(records[0])
    with pytest.raises(ValueError, match=match):
        _audit(tmp_path, campaign_fixture, records)
    assert not (tmp_path / "campaign_audit.json").exists()


@pytest.mark.parametrize(
    "field",
    (
        "environment_step_time_seconds",
        "ranking_time_seconds",
        "feature_time_seconds",
        "target_metric_time_seconds",
        "symbolic_update_time_seconds",
        "canonicalization_time_seconds",
        "archive_time_seconds",
        "certification_time_seconds",
        "reporting_time_seconds",
    ),
)
def test_every_component_timer_must_fit_inside_wall_time(
    tmp_path: Path, campaign_fixture, field
) -> None:
    records = deepcopy(campaign_fixture[4])
    nanoseconds_field = field.removesuffix("_seconds") + "_ns"
    forged_ns = records[0]["search_metrics"]["wall_time_ns"] + 1
    records[0]["timings"][field] = forged_ns / 1e9
    records[0]["search_metrics"][nanoseconds_field] = forged_ns
    with pytest.raises(ValueError, match="exceeds wall time"):
        _audit(tmp_path, campaign_fixture, records)


@pytest.mark.parametrize(
    ("field", "match"),
    (
        ("generated", "counter aliases"),
        ("duplicate_rejected", "duplicate-rejection counters"),
        ("dominated_retired", "counter aliases"),
        ("pareto_incomparable_accepted", "counter aliases"),
        ("reopened", "counter aliases"),
        ("frontier_peak", "counter aliases"),
        ("active_archive_peak", "counter aliases"),
        ("pareto_width_peak", "counter aliases"),
        ("frontier_sum", "frontier decision mean"),
    ),
)
def test_audit_rejects_every_counter_alias_or_internal_equation_drift(
    tmp_path: Path, campaign_fixture, field, match
) -> None:
    records = deepcopy(campaign_fixture[4])
    records[0]["search_metrics"][field] += 1
    with pytest.raises(ValueError, match=match):
        _audit(tmp_path, campaign_fixture, records)


def _increment_counter(row, name: str) -> None:
    row["search_metrics"][name] += 1


def _drift_frontier_observations(row) -> None:
    metrics = row["search_metrics"]
    metrics["frontier_observation_count"] += 1
    metrics["frontier_decision_mean"] = (
        metrics["frontier_sum"] / metrics["frontier_observation_count"]
    )


def _forge_uncertified_solution_outcome(row) -> None:
    metrics = row["search_metrics"]
    metrics["certification_nonmatch"] = metrics["generated"] - 1
    metrics["accepted"] = metrics["certification_nonmatch"] + 1
    metrics["archive_record_count"] = metrics["accepted"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda row: _increment_counter(row, "terminal_candidates"),
            "terminal_candidates must equal generated",
        ),
        (
            lambda row: _increment_counter(row, "certification_count"),
            "certification_count must equal generated",
        ),
        (
            lambda row: _increment_counter(row, "num_gate_attempts"),
            "native grammar size",
        ),
        (
            lambda row: _increment_counter(row, "archive_record_count"),
            "accepted must equal archive_record_count",
        ),
        (
            lambda row: _increment_counter(row, "certification_nonmatch"),
            "account for every certification nonmatch",
        ),
        (
            lambda row: _increment_counter(
                row, "terminal_certification_failures"
            ),
            "cannot contain terminal certification failures",
        ),
        (_drift_frontier_observations, "must equal completed expansions"),
        (_forge_uncertified_solution_outcome, "certified status disagrees"),
    ),
)
def test_audit_rejects_native_search_counter_impossibilities(
    tmp_path: Path, campaign_fixture, mutation, match
) -> None:
    records = deepcopy(campaign_fixture[4])
    mutation(records[0])
    with pytest.raises(ValueError, match=match):
        _audit(tmp_path, campaign_fixture, records)


def _forge_generated_over_attempts(row) -> None:
    metrics = row["search_metrics"]
    generated = metrics["num_gate_attempts"] + 1
    metrics.update({
        "generated": generated,
        "num_generated": generated,
        "terminal_candidates": generated,
        "certification_count": generated + 1,
        "certification_nonmatch": generated,
        "accepted": generated + 1,
        "archive_record_count": generated + 1,
        "archive_size": generated + 1,
        "active_archive_peak": generated + 1,
        "peak_active_archive_records": generated + 1,
    })


def _forge_nonmatches_over_generated(row) -> None:
    metrics = row["search_metrics"]
    nonmatches = metrics["generated"] + 1
    metrics.update({
        "certification_nonmatch": nonmatches,
        "accepted": nonmatches + 1,
        "archive_record_count": nonmatches + 1,
        "archive_size": nonmatches + 1,
        "active_archive_peak": nonmatches + 1,
        "peak_active_archive_records": nonmatches + 1,
    })


def _forge_frontier_peak(row, value: int) -> None:
    metrics = row["search_metrics"]
    metrics["frontier_peak"] = value
    metrics["peak_frontier"] = value
    metrics["peak_frontier_records"] = value


def _forge_frontier_mean(row, value: float) -> None:
    metrics = row["search_metrics"]
    metrics["frontier_sum"] = int(
        value * metrics["frontier_observation_count"]
    )
    metrics["frontier_decision_mean"] = value


def _forge_zero_accepted(row) -> None:
    metrics = row["search_metrics"]
    metrics["accepted"] = 0
    metrics["archive_record_count"] = 0
    duplicates = metrics["certification_nonmatch"] + 1
    metrics["duplicate_rejected"] = duplicates
    metrics["canonical_pruned"] = duplicates
    metrics["num_exact_duplicate_rejections"] = duplicates


def _forge_frontier_over_active_archive(row) -> None:
    _forge_frontier_peak(
        row, row["search_metrics"]["active_archive_peak"] + 1
    )


def _forge_active_archive_over_records(row) -> None:
    metrics = row["search_metrics"]
    value = metrics["archive_record_count"] + 1
    metrics["active_archive_peak"] = value
    metrics["peak_active_archive_records"] = value


def _forge_pareto_width(row, value: int) -> None:
    metrics = row["search_metrics"]
    metrics["pareto_width_peak"] = value
    metrics["maximum_pareto_antichain_width"] = value


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (_forge_generated_over_attempts, "generated cannot exceed"),
        (_forge_nonmatches_over_generated, "nonmatches cannot exceed"),
        (lambda row: _forge_frontier_peak(row, 0), "frontier peak must be"),
        (lambda row: _forge_frontier_mean(row, 0.0), "decision mean must be"),
        (
            lambda row: _forge_frontier_mean(
                row, row["search_metrics"]["frontier_peak"] + 1.0
            ),
            "decision mean cannot exceed",
        ),
        (_forge_zero_accepted, "accepted/archive record counts"),
        (
            lambda row: row["search_metrics"].__setitem__("archive_size", 0),
            "archive size must lie",
        ),
        (
            lambda row: row["search_metrics"].__setitem__(
                "archive_size",
                row["search_metrics"]["archive_record_count"] + 1,
            ),
            "archive size must lie",
        ),
        (_forge_frontier_over_active_archive, "frontier/archive peaks"),
        (_forge_active_archive_over_records, "frontier/archive peaks"),
        (lambda row: _forge_pareto_width(row, 0), "Pareto width must lie"),
        (
            lambda row: _forge_pareto_width(
                row, row["search_metrics"]["active_archive_peak"] + 1
            ),
            "Pareto width must lie",
        ),
        (
            lambda row: row["search_metrics"].__setitem__(
                "target_metric_evaluation_count",
                row["search_metrics"]["target_metric_cache_misses"] + 1,
            ),
            "evaluations must equal cache misses",
        ),
    ),
)
def test_audit_rejects_impossible_native_search_bounds(
    tmp_path: Path, campaign_fixture, mutation, match
) -> None:
    records = deepcopy(campaign_fixture[4])
    mutation(records[0])
    with pytest.raises(ValueError, match=match):
        _audit(tmp_path, campaign_fixture, records)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda row: (
                row.__setitem__("runtime_seconds", 0.0),
                row["timings"].__setitem__("wall_time_seconds", 0.0),
                row["search_metrics"].__setitem__("wall_time_ns", 0),
            ),
            "positive wall time",
        ),
        (
            lambda row: (
                row["timings"].__setitem__(
                    "environment_step_time_seconds", 0.0
                ),
                row["search_metrics"].__setitem__(
                    "environment_step_time_ns", 0
                ),
            ),
            "require environment step time",
        ),
        (
            lambda row: (
                row["timings"].__setitem__("certification_time_seconds", 0.0),
                row["search_metrics"].__setitem__("certification_time_ns", 0),
            ),
            "require certification time",
        ),
    ),
)
def test_audit_requires_positive_measured_execution_time(
    tmp_path: Path, campaign_fixture, mutation, match
) -> None:
    records = deepcopy(campaign_fixture[4])
    mutation(records[0])
    with pytest.raises(ValueError, match=match):
        _audit(tmp_path, campaign_fixture, records)


@pytest.mark.parametrize(
    "field",
    (
        "dirty_worktree",
        "reference_witness_used",
        "target_specific_reachability_oracle",
        "evaluation_weights_frozen",
        "evaluation_reward_consumed_by_policy",
    ),
)
def test_scientific_boolean_flags_reject_integer_spoofing(
    tmp_path: Path, campaign_fixture, field
) -> None:
    records = deepcopy(campaign_fixture[4])
    records[0][field] = int(bool(records[0][field]))
    with pytest.raises(ValueError, match=field):
        _audit(tmp_path, campaign_fixture, records)


def test_nested_metadata_rejects_boolean_and_numeric_type_spoofing(
    tmp_path: Path, campaign_fixture
) -> None:
    records = deepcopy(campaign_fixture[4])
    records[0]["search_reduction"]["canonicalization_enabled"] = 1
    with pytest.raises(ValueError, match="invalid run key"):
        _audit(tmp_path, campaign_fixture, records)

    records = deepcopy(campaign_fixture[4])
    records[0]["profile"]["gamma"] = 1
    with pytest.raises(ValueError, match="profile"):
        _audit(tmp_path, campaign_fixture, records)


def test_audit_rejects_missing_unexpected_and_duplicate_physical_records(
    tmp_path: Path, campaign_fixture
) -> None:
    records = deepcopy(campaign_fixture[4])
    with pytest.raises(ValueError, match="missing=1"):
        _audit(tmp_path, campaign_fixture, records[:-1])

    unexpected = deepcopy(records[0])
    unexpected["evaluation_seed"] = 999_999
    unexpected["run_key"] = unique_run_key(unexpected)
    with pytest.raises(ValueError, match="unexpected=1"):
        _audit(tmp_path, campaign_fixture, [*records, unexpected])

    raw_path = tmp_path / "raw_runs.jsonl"
    _write_ledger(raw_path, [*records, deepcopy(records[0])])
    corpus, checkpoints, ood_checkpoints, _specs, _records = campaign_fixture
    with pytest.raises(ValueError, match="duplicate run key"):
        audit_article_v1_campaign(
            corpus,
            checkpoints=checkpoints,
            ood_checkpoints=ood_checkpoints,
            raw_path=raw_path,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("target_id", "sha256:foreign-target"),
        ("config_digest", "sha256:foreign-config"),
        ("scheduler", "foreign-scheduler"),
        ("expansion_budget", 999_999),
        ("evaluation_seed", 999_999),
        ("feature_schema_version", "foreign-feature-schema"),
        ("reward_schema_version", "foreign-reward-schema"),
        ("target_metric_schema_version", "foreign-target-metric-schema"),
        ("certification_schema_version", "foreign-certifier-schema"),
        ("source_worktree_digest", "sha256:foreign-source"),
    ),
)
def test_audit_binds_every_primary_identity_coordinate(
    tmp_path: Path, campaign_fixture, field, value
) -> None:
    records = deepcopy(campaign_fixture[4])
    records[0][field] = value
    records[0]["run_key"] = unique_run_key(records[0])
    with pytest.raises(ValueError, match="exact expected matrix"):
        _audit(tmp_path, campaign_fixture, records)


def test_audit_binds_resource_budget_and_learner_checkpoint_identity(
    tmp_path: Path, campaign_fixture
) -> None:
    _corpus, _checkpoints, _ood_checkpoints, specs, base_records = campaign_fixture

    records = deepcopy(base_records)
    records[0]["resource_budget"]["max_gates"] += 1
    records[0]["run_key"] = unique_run_key(records[0])
    with pytest.raises(ValueError, match="exact expected matrix"):
        _audit(tmp_path, campaign_fixture, records)

    for field, value in (
        ("checkpoint_family", STANDARD_CHECKPOINT_FAMILY),
        ("checkpoint_scope_schema", "foreign-checkpoint-scope"),
    ):
        records = deepcopy(base_records)
        records[0][field] = value
        with pytest.raises(ValueError, match="checkpoint"):
            _audit(tmp_path, campaign_fixture, records)

    learner_index = next(
        index for index, spec in enumerate(specs)
        if spec.scheduler == "article_sarsa"
    )
    for field, value in (
        ("checkpoint_digest", "sha256:foreign-checkpoint"),
        ("training_seed", 999_999),
    ):
        records = deepcopy(base_records)
        records[learner_index][field] = value
        records[learner_index]["run_key"] = unique_run_key(
            records[learner_index]
        )
        with pytest.raises(ValueError, match="exact expected matrix"):
            _audit(tmp_path, campaign_fixture, records)

    for field, value in (
        ("checkpoint_family", STANDARD_CHECKPOINT_FAMILY + "-forged"),
        ("checkpoint_scope_schema", "foreign-checkpoint-scope"),
    ):
        records = deepcopy(base_records)
        records[learner_index][field] = value
        with pytest.raises(ValueError, match="checkpoint"):
            _audit(tmp_path, campaign_fixture, records)


def test_audit_rejects_split_drift_and_forged_failure_evidence(
    tmp_path: Path, campaign_fixture
) -> None:
    records = deepcopy(campaign_fixture[4])
    records[0]["split"] = "ood_test"
    with pytest.raises(ValueError, match="split"):
        _audit(tmp_path, campaign_fixture, records)

    records = deepcopy(campaign_fixture[4])
    records[0]["certification_diagnostics"] = {"passed": True}
    with pytest.raises(ValueError, match="forged success evidence"):
        _audit(tmp_path, campaign_fixture, records)


def test_audit_rejects_partial_line_and_nonfinite_json(
    tmp_path: Path, campaign_fixture
) -> None:
    records = deepcopy(campaign_fixture[4])
    raw_path = tmp_path / "raw_runs.jsonl"
    _write_ledger(raw_path, records, terminal_newline=False)
    corpus, checkpoints, ood_checkpoints, _specs, _records = campaign_fixture
    with pytest.raises(ValueError, match="incomplete final line"):
        audit_article_v1_campaign(
            corpus,
            checkpoints=checkpoints,
            ood_checkpoints=ood_checkpoints,
            raw_path=raw_path,
        )


def test_audit_rejects_duplicate_json_members_and_v2_raw_ledgers(
    tmp_path: Path, campaign_fixture
) -> None:
    records = deepcopy(campaign_fixture[4])
    raw_path = tmp_path / "raw_runs.jsonl"
    _write_ledger(raw_path, records)
    lines = raw_path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0][:-1] + ',"certified":true}'
    raw_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    corpus, checkpoints, ood_checkpoints, _specs, _records = campaign_fixture
    with pytest.raises(ValueError, match="duplicate JSON member 'certified'"):
        audit_article_v1_campaign(
            corpus,
            checkpoints=checkpoints,
            ood_checkpoints=ood_checkpoints,
            raw_path=raw_path,
        )

    records[0]["schema_version"] = "article-v1-raw-run-v2"
    records[0]["raw_run_schema"] = "article-v1-raw-run-v2"
    _write_ledger(raw_path, records)
    with pytest.raises(ValueError, match="incompatible raw schema"):
        audit_article_v1_campaign(
            corpus,
            checkpoints=checkpoints,
            ood_checkpoints=ood_checkpoints,
            raw_path=raw_path,
        )

    records[0]["runtime_seconds"] = float("nan")
    _write_ledger(raw_path, records, allow_nan=True)
    with pytest.raises(ValueError, match="nonfinite JSON value"):
        audit_article_v1_campaign(
            corpus,
            checkpoints=checkpoints,
            ood_checkpoints=ood_checkpoints,
            raw_path=raw_path,
        )


def _make_success_record(fixture):
    corpus, _checkpoints, _ood_checkpoints, specs, base_records = fixture
    record = deepcopy(base_records[0])
    spec = specs[0]
    source_case = next(
        case for case in corpus.targets if case.target_id == spec.case.target_id
    )
    witness = [
        {"gate": gate.gate_type.name, "qubits": list(gate.qubits)}
        for gate in source_case.generator_witness
    ]
    expansions = len(witness)
    record.update({
        "certified": True,
        "terminated": True,
        "truncated": False,
        "expansions": expansions,
        "time_to_solution": record["runtime_seconds"],
        "witness_operations": witness,
        "certification_diagnostics": (
            _independent_witness_certification_diagnostics(
                spec.case,
                witness,
                certification_tolerance=float(
                    corpus.config.experiment["certification_tolerance"]
                ),
            )
        ),
        "solution_resource_vector": _independent_witness_resource_vector(
            spec.case, witness
        ),
    })
    _set_native_counter_model(record, expansions=expansions, certified=True)
    return record


def test_success_requires_matching_independent_diagnostics_and_resource_replay(
    tmp_path: Path, campaign_fixture
) -> None:
    records = deepcopy(campaign_fixture[4])
    success = _make_success_record(campaign_fixture)
    records[0] = success
    result = _audit(tmp_path, campaign_fixture, records)
    assert result["independently_certified_success_count"] == 1

    corrupted = deepcopy(records)
    corrupted[0]["certification_diagnostics"]["delta_phi"] = 0.5
    with pytest.raises(ValueError, match="fresh replay"):
        _audit(tmp_path, campaign_fixture, corrupted)

    corrupted = deepcopy(records)
    corrupted[0]["solution_resource_vector"][0] += 1
    with pytest.raises(ValueError, match="resource vector"):
        _audit(tmp_path, campaign_fixture, corrupted)

    corrupted = deepcopy(records)
    corrupted[0]["certification_diagnostics"] = None
    with pytest.raises(ValueError, match="no certification diagnostics"):
        _audit(tmp_path, campaign_fixture, corrupted)

    corrupted = deepcopy(records)
    corrupted[0]["runtime_seconds"] = 1.0
    corrupted[0]["timings"]["wall_time_seconds"] = 1.0
    corrupted[0]["search_metrics"]["wall_time_ns"] = 1_000_000_000
    corrupted[0]["time_to_solution"] = 0.5
    with pytest.raises(ValueError, match="must equal runtime_seconds"):
        _audit(tmp_path, campaign_fixture, corrupted)


def test_evaluator_persists_fresh_success_certification_diagnostics(
    campaign_fixture, monkeypatch
) -> None:
    corpus, _checkpoints, _ood_checkpoints, specs, _records = campaign_fixture
    spec = specs[0]
    source_case = next(
        case for case in corpus.targets if case.target_id == spec.case.target_id
    )
    witness = [
        {"gate": gate.gate_type.name, "qubits": list(gate.qubits)}
        for gate in source_case.generator_witness
    ]

    monkeypatch.setattr(
        "experiments.article_v1_runner.evaluate",
        lambda **_kwargs: {
            "search_metrics": {},
            "scheduler_semantics": "fifo",
            "certified": True,
            "terminated": True,
            "truncated": False,
            "expansions": 1,
            "runtime_seconds": 0.0,
            "time_to_solution": 0.0,
            "solution_resource_vector": _independent_witness_resource_vector(
                spec.case, witness
            ),
            "witness_operations": witness,
        },
    )
    row = evaluate_article_v1_run(
        spec.case,
        scheduler="fifo",
        expansion_budget=spec.expansion_budget,
        evaluation_seed=0,
        config_digest=corpus.config.digest,
        certification_tolerance=float(
            corpus.config.experiment["certification_tolerance"]
        ),
    )

    assert row["certification_diagnostics"]["passed"] is True
    assert row["certification_diagnostics"]["reason"] == (
        "equivalent_phase_frobenius"
    )
    assert row["witness_operations"] == witness


def test_real_evaluator_emits_the_exact_auditable_raw_shape(campaign_fixture) -> None:
    corpus, _checkpoints, _ood_checkpoints, specs, _records = campaign_fixture
    row = evaluate_article_v1_run(
        specs[0].case,
        scheduler="fifo",
        expansion_budget=1,
        evaluation_seed=0,
        config_digest=corpus.config.digest,
        certification_tolerance=float(
            corpus.config.experiment["certification_tolerance"]
        ),
    )

    assert set(row) | {"raw_run_schema", "run_key"} == (
        _CANONICAL_RAW_RECORD_FIELDS
    )
    assert set(row["timings"]) == set(_REQUIRED_TIMING_FIELDS)
    assert set(row["search_metrics"]) == _CANONICAL_SEARCH_METRIC_FIELDS

    spec = _expected_matrix_runs(
        (specs[0].case,),
        config=corpus.config,
        checkpoints=campaign_fixture[1],
        checkpoint_scope=corpus.checkpoint_scope(
            checkpoint_family=STANDARD_CHECKPOINT_FAMILY
        ),
        schedulers=("fifo",),
        provenance=git_provenance(),
        budget_override=1,
    )[0]
    row["raw_run_schema"] = ARTICLE_V1_RAW_RUN_SCHEMA
    row["run_key"] = unique_run_key(row)
    assert spec.checkpoint_scope is None
    assert row["checkpoint_scope_schema"] is None
    assert _audit_campaign_record(
        row,
        spec,
        config=corpus.config,
        provenance=git_provenance(),
    ) is bool(row["certified"])


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("commit_sha", "foreign-commit"),
        ("source_worktree_digest", "sha256:stale"),
        ("dirty_worktree", 1),
    ),
)
def test_checkpoint_loading_rejects_stale_code_provenance(
    tmp_path: Path, campaign_fixture, field, value
) -> None:
    checkpoint = campaign_fixture[1][0]
    path = tmp_path / "checkpoint.json"
    checkpoint.save(path)
    assert ArticleV1Checkpoint.load(path) == checkpoint

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["code"][field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=f"code provenance {field}"):
        ArticleV1Checkpoint.load(path)


def test_aggregate_runs_fail_closed_audit_before_reporting(
    tmp_path: Path, campaign_fixture, monkeypatch
) -> None:
    corpus, checkpoints, ood_checkpoints, _specs, _records = campaign_fixture
    destination = tmp_path / "campaign"
    checkpoint_dir = destination / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_by_name = {}
    for checkpoint in checkpoints:
        path = checkpoint_dir / f"seed-{checkpoint.training_seed}.json"
        path.write_text("{}", encoding="utf-8")
        checkpoint_by_name[path.name] = checkpoint
    for checkpoint in ood_checkpoints:
        path = checkpoint_dir / f"ood-seed-{checkpoint.training_seed}.json"
        path.write_text("{}", encoding="utf-8")
        checkpoint_by_name[path.name] = checkpoint

    monkeypatch.setattr(
        "experiments.article_v1_runner.initialize_run",
        lambda *_args, **_kwargs: (destination, corpus),
    )
    monkeypatch.setattr(
        ArticleV1Checkpoint,
        "load",
        staticmethod(lambda path: checkpoint_by_name[Path(path).name]),
    )
    order = []
    audited_digest = "sha256:" + "0" * 64
    monkeypatch.setattr(
        "experiments.article_v1_runner.audit_article_v1_campaign",
        lambda *_args, **_kwargs: order.append("audit")
        or {"passed": True, "raw_ledger_sha256": audited_digest},
    )

    def fake_report(*_args, **kwargs):
        assert kwargs["expected_raw_sha256"] == audited_digest
        order.append("report")

    monkeypatch.setattr(
        "experiments.article_v1_runner.write_article_v1_report",
        fake_report,
    )

    assert article_v1_main([
        "aggregate",
        "--config",
        "ignored.json",
        "--output-root",
        str(tmp_path),
        "--run-id",
        "campaign",
    ]) == 0
    assert order == ["audit", "report"]

    order.clear()
    monkeypatch.setattr(
        "experiments.article_v1_runner.audit_article_v1_campaign",
        lambda *_args, **_kwargs: order.append("audit") or {"passed": False},
    )
    with pytest.raises(ValueError, match="passed=true"):
        article_v1_main([
            "aggregate",
            "--config",
            "ignored.json",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "campaign",
        ])
    assert order == ["audit"]


def test_root_cli_dispatches_audit(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr(
        "experiments.article_v1_runner.main",
        lambda arguments: captured.append(arguments) or 19,
    )
    assert article_benchmark.main(["audit"]) == 19
    assert captured == [["audit"]]


def test_environment_records_serial_thread_controls(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "2")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    execution = environment_metadata()["execution"]
    assert execution == {
        "concurrency_mode": "serial-single-process",
        "worker_count": 1,
        "thread_environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": None,
        },
    }


def test_checked_in_pilot_guard_fails_on_any_publication_overlap(
    campaign_fixture,
) -> None:
    corpus = campaign_fixture[0]
    with pytest.raises(ValueError, match="pilot and publication corpora overlap"):
        _assert_checked_in_pilot_publication_disjoint(
            corpus.config,
            corpus,
            publication_corpus=corpus,
        )

    source = corpus.targets[0]
    phase_equivalent_with_distinct_digest = SimpleNamespace(
        target_id="sha256:syntactically-distinct",
        unitary=np.exp(0.37j) * source.unitary,
    )
    publication = SimpleNamespace(
        config=SimpleNamespace(tau_identity=corpus.config.tau_identity),
        targets=(phase_equivalent_with_distinct_digest,),
    )
    with pytest.raises(ValueError, match="projective identity rule"):
        _assert_checked_in_pilot_publication_disjoint(
            corpus.config,
            corpus,
            publication_corpus=publication,
        )


def test_config_digest_changes_run_identity(campaign_fixture) -> None:
    record = deepcopy(campaign_fixture[4][0])
    original = unique_run_key(record)
    record["config_digest"] = "sha256:different-config"
    assert unique_run_key(record) != original


def test_checkpoint_schema_constant_remains_explicit() -> None:
    assert ARTICLE_V1_CHECKPOINT_SCHEMA == (
        "article-v1-transferable-linear-checkpoint-v3"
    )
