from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import article_benchmark as root_article_benchmark
import experiments.article_v1_runner as runner_module

from benchmarks.article_native_corpus import (
    ARTICLE_V1_TRAINING_BUDGET_POLICY,
    COMPLETE_TRAINING_SCOPE,
    OOD_LENGTH_CHECKPOINT_FAMILY,
    PARTIAL_SMOKE_TRAINING_SCOPE,
    STANDARD_CHECKPOINT_FAMILY,
    ArticleV1Budget,
    ArticleV1CheckpointScope,
    ArticleV1EvaluationTarget,
)
from certification.article_v1 import ArticleV1CertificationEngine
from certification.simulator import SynthesisTarget, unitary_from_gates
from circuit.gate import Gate
from enums import GateType
from experiments.article_v1_runner import (
    ARTICLE_V1_CHECKPOINT_SCHEMA,
    PRIMARY_SCHEDULERS,
    ArticleV1Checkpoint,
    _EpisodeProgressClock,
    _ProgressFeatureTimingWindow,
    _load_or_train_article_v1_checkpoint,
    _validate_checkpoint_campaign,
    _mini_ci_semantic_checks,
    evaluate_article_v1_run,
    git_provenance,
    initialize_run,
    main,
    mini_ci_benchmark,
)
from experiments.article_v1_progress import (
    ARTICLE_V1_PROGRESS_EVENT_SCHEMA,
    load_progress_events,
    load_progress_status,
)
from experiments.profiles import ARTICLE_V1_PROFILE
from rl.article_features import (
    ARTICLE_V1_FEATURE_NAMES,
    ARTICLE_V1_NO_TARGET_FEATURE_NAMES,
    ARTICLE_V1_NO_TARGET_FEATURE_SCHEMA_VERSION,
    ARTICLE_V1_NO_Z_FEATURE_NAMES,
    ARTICLE_V1_NO_Z_FEATURE_SCHEMA_VERSION,
)


def test_episode_progress_clock_resets_elapsed_time_and_rate() -> None:
    current = [12.0]
    clock = _EpisodeProgressClock(started=10.0, clock=lambda: current[0])
    assert clock.measure(4) == (2.0, 2.0)

    clock.reset(now=20.0)
    current[0] = 21.0
    assert clock.measure(3) == (1.0, 3.0)


def test_replay_capture_and_measure_cli_dispatch_without_starting_workload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_capture(path: Path, **kwargs: object) -> dict[str, object]:
        calls["capture"] = (path, kwargs)
        return {"capture_valid": True}

    evidence = SimpleNamespace(
        engineering_timing_valid=True,
        to_payload=lambda: {"replay_timing_schema": "article-v1-replay-timing-v1"},
    )

    def fake_measure(
        checkpoint: Path,
        output: Path,
        **kwargs: object,
    ) -> object:
        calls["measure"] = (checkpoint, output, kwargs)
        return evidence

    monkeypatch.setattr(
        runner_module, "capture_article_v1_replay_checkpoint", fake_capture
    )
    monkeypatch.setattr(
        runner_module, "measure_article_v1_replay_checkpoint", fake_measure
    )
    assert main(
        (
            "capture-replay-checkpoint",
            "--output-root",
            str(tmp_path),
            "--run-id",
            "capture",
            "--quiet",
        )
    ) == 0
    capture_path, capture_kwargs = calls["capture"]
    assert capture_path == tmp_path / "capture"
    assert capture_kwargs["quiet"] is True
    assert capture_kwargs["checkpoint_cadence"].every_expansions == 64

    checkpoint = tmp_path / "capture" / "training_state" / "latest.json"
    output = tmp_path / "replay_timing.json"
    assert main(
        (
            "measure-replay-timing",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--projected-full-episode-seconds",
            "3515.337979379539",
        )
    ) == 0
    measured_checkpoint, measured_output, measured_kwargs = calls["measure"]
    assert measured_checkpoint == checkpoint
    assert measured_output == output
    assert measured_kwargs["projected_full_episode_seconds"] == pytest.approx(
        3515.337979379539
    )


@pytest.mark.parametrize(
    "command", ("capture-replay-checkpoint", "measure-replay-timing")
)
def test_root_cli_dispatches_replay_operability_commands(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def fake_main(arguments: list[str]) -> int:
        captured.append(list(arguments))
        return 23

    monkeypatch.setattr(runner_module, "main", fake_main)
    assert root_article_benchmark.main([command]) == 23
    assert captured == [[command]]


def test_progress_feature_timing_window_is_exact_and_does_not_double_count() -> None:
    window = _ProgressFeatureTimingWindow()
    all_times: list[int] = []
    last_seconds = rolling_seconds = 0.0
    for count in range(1, 28):
        all_times.append(count * 100)
        recent = tuple(all_times[-25:])
        last_seconds, rolling_seconds = window.observe(
            {
                "compact_batch_count": count,
                "last_compact_batch_time_ns": all_times[-1],
            },
            recent_batch_times_ns=recent,
        )
    assert last_seconds == 2_700 / 1_000_000_000
    assert rolling_seconds == sum(all_times[-25:]) / 25 / 1_000_000_000

    # Episode end observes the same provider count and cannot append twice.
    duplicate = window.observe(
        {
            "compact_batch_count": 27,
            "last_compact_batch_time_ns": 2_700,
        },
        recent_batch_times_ns=tuple(all_times[-25:]),
    )
    assert duplicate == (last_seconds, rolling_seconds)

    # If attachment skips more samples than the provider retains, that exact
    # retained suffix is the complete requested rolling-25 window.
    late = _ProgressFeatureTimingWindow()
    suffix = tuple(range(76, 101))
    assert late.observe(
        {"compact_batch_count": 100, "last_compact_batch_time_ns": 100},
        recent_batch_times_ns=suffix,
    ) == (100 / 1_000_000_000, sum(suffix) / 25 / 1_000_000_000)

    window.reset_episode()
    assert window.observe(
        {"compact_batch_count": 2, "last_compact_batch_time_ns": 150},
        recent_batch_times_ns=(50, 150),
    ) == (150 / 1_000_000_000, 100 / 1_000_000_000)


def _one_h_target() -> ArticleV1EvaluationTarget:
    unitary = unitary_from_gates(2, (Gate(GateType.H, (0,)),))
    return ArticleV1EvaluationTarget(
        target_id="test-h-on-q0",
        split="test",
        difficulty="easy",
        num_qubits=2,
        generator_length=1,
        budget=ArticleV1Budget(
            max_t_count=1,
            max_two_qubit_count=1,
            max_gates=2,
            max_depth=2,
            expansion_budget=4,
        ),
        target=SynthesisTarget(unitary),
    )


def _checkpoint(
    *,
    zero: bool = False,
    feature_schema: str = ARTICLE_V1_PROFILE.feature_schema,
    feature_names: tuple[str, ...] = tuple(ARTICLE_V1_FEATURE_NAMES),
    checkpoint_family: str = STANDARD_CHECKPOINT_FAMILY,
    corpus_config_digest: str = "sha256:test-corpus",
    training_target_ids: tuple[str, ...] = ("train-fixture",),
    training_scope_mode: str = COMPLETE_TRAINING_SCOPE,
    training_beta: float = 1.0,
    training_certification_tolerance: float = 1e-9,
    training_episodes_per_target: int = 2,
    training_expansion_cap: int | None = None,
    training_seed: int = 17,
    learning_rate: float = 1e-3,
    epsilon_schedule: tuple[tuple[str, float], ...] = (
        ("decay", 0.99),
        ("minimum", 0.05),
        ("start", 0.2),
    ),
) -> ArticleV1Checkpoint:
    weights = [0.0] * len(feature_names)
    if not zero:
        weights[0] = 0.125
    return ArticleV1Checkpoint(
        training_seed=training_seed,
        weights=tuple(weights),
        feature_schema_version=feature_schema,
        ordered_feature_names=feature_names,
        reward_schema_version=ARTICLE_V1_PROFILE.reward_schema,
        target_metric_schema_version=ARTICLE_V1_PROFILE.target_metric_schema,
        certification_schema_version=ARTICLE_V1_PROFILE.certification_schema,
        learning_rate=learning_rate,
        discount=ARTICLE_V1_PROFILE.gamma,
        epsilon_schedule=epsilon_schedule,
        checkpoint_family=checkpoint_family,
        training_scope_mode=training_scope_mode,
        training_beta=training_beta,
        training_certification_tolerance=training_certification_tolerance,
        training_episodes_per_target=training_episodes_per_target,
        training_expansion_cap=training_expansion_cap,
        training_budget_policy=ARTICLE_V1_TRAINING_BUDGET_POLICY,
        effective_training_expansion_budgets=tuple(
            (
                target_id,
                4 if training_expansion_cap is None else min(4, training_expansion_cap),
            )
            for target_id in training_target_ids
        ),
        training_target_ids=training_target_ids,
        training_histories=(),
        corpus_config_digest=corpus_config_digest,
    )


def _scope(
    *,
    checkpoint_family: str = STANDARD_CHECKPOINT_FAMILY,
    expected_feature_schema: str = ARTICLE_V1_PROFILE.feature_schema,
    training_scope_mode: str = COMPLETE_TRAINING_SCOPE,
    allowed_training_target_ids: tuple[str, ...] = ("train-fixture",),
    expected_training_beta: float = 1.0,
    expected_certification_tolerance: float = 1e-9,
    expected_episodes_per_target: int = 2,
    expected_expansion_cap: int | None = None,
    expected_learning_rate: float = 1e-3,
    expected_epsilon_schedule: tuple[tuple[str, float], ...] = (
        ("decay", 0.99),
        ("minimum", 0.05),
        ("start", 0.2),
    ),
    allowed_training_seeds: tuple[int, ...] = (17,),
) -> ArticleV1CheckpointScope:
    return ArticleV1CheckpointScope(
        corpus_config_digest="sha256:test-corpus",
        checkpoint_family=checkpoint_family,
        training_scope_mode=training_scope_mode,
        expected_feature_schema_version=expected_feature_schema,
        expected_training_beta=expected_training_beta,
        expected_certification_tolerance=expected_certification_tolerance,
        expected_episodes_per_target=expected_episodes_per_target,
        expected_learning_rate=expected_learning_rate,
        expected_epsilon_schedule=expected_epsilon_schedule,
        allowed_training_seeds=allowed_training_seeds,
        expected_expansion_cap=expected_expansion_cap,
        training_budget_policy=ARTICLE_V1_TRAINING_BUDGET_POLICY,
        allowed_training_target_ids=allowed_training_target_ids,
        expected_training_expansion_budgets=tuple(
            (
                target_id,
                4 if expected_expansion_cap is None else min(4, expected_expansion_cap),
            )
            for target_id in allowed_training_target_ids
        ),
        held_out_target_ids=("test-h-on-q0",),
        permitted_evaluation_target_ids=("test-h-on-q0",),
    )


def test_article_sarsa_rejects_missing_and_zero_checkpoint() -> None:
    case = _one_h_target()
    with pytest.raises(ValueError, match="requires a trained checkpoint"):
        evaluate_article_v1_run(
            case,
            scheduler="article_sarsa",
            expansion_budget=1,
            evaluation_seed=0,
        )

    with pytest.raises(ValueError, match="nonzero trained checkpoint"):
        evaluate_article_v1_run(
            case,
            scheduler="article_sarsa",
            expansion_budget=1,
            evaluation_seed=0,
            checkpoint=_checkpoint(zero=True),
            checkpoint_scope=_scope(),
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    (
        (
            "feature_schema_version",
            "article-v1-no-z-21d",
            "feature names do not match.*schema",
        ),
        ("reward_schema_version", "wrong-reward", "reward schema"),
        (
            "target_metric_schema_version",
            "wrong-target-metric",
            "target-metric schema",
        ),
        ("certification_schema_version", "wrong-certifier", "certification schema"),
    ),
)
def test_article_sarsa_rejects_checkpoint_schema_mismatch(
    field: str,
    bad_value: str,
    message: str,
) -> None:
    checkpoint = replace(_checkpoint(), **{field: bad_value})
    with pytest.raises(ValueError, match=message):
        evaluate_article_v1_run(
            _one_h_target(),
            scheduler="article_sarsa",
            expansion_budget=1,
            evaluation_seed=0,
            checkpoint=checkpoint,
            checkpoint_scope=_scope(),
        )


def test_primary_evaluation_rejects_valid_21d_and_28d_ablation_checkpoints() -> None:
    variants = (
        (
            ARTICLE_V1_NO_Z_FEATURE_SCHEMA_VERSION,
            tuple(ARTICLE_V1_NO_Z_FEATURE_NAMES),
        ),
        (
            ARTICLE_V1_NO_TARGET_FEATURE_SCHEMA_VERSION,
            tuple(ARTICLE_V1_NO_TARGET_FEATURE_NAMES),
        ),
    )
    for feature_schema, feature_names in variants:
        checkpoint = _checkpoint(
            feature_schema=feature_schema,
            feature_names=feature_names,
        )
        checkpoint.validate_contract(require_nonzero=True)
        checkpoint.validate_for_evaluation(
            _scope(expected_feature_schema=feature_schema),
            _one_h_target(),
        )
        with pytest.raises(ValueError, match="expected evaluation schema"):
            evaluate_article_v1_run(
                _one_h_target(),
                scheduler="article_sarsa",
                expansion_budget=1,
                evaluation_seed=0,
                checkpoint=checkpoint,
                checkpoint_scope=_scope(),
            )


def test_checkpoint_evaluation_scope_rejects_digest_training_and_family_leakage() -> None:
    case = _one_h_target()

    with pytest.raises(ValueError, match="corpus config digest"):
        _checkpoint(corpus_config_digest="sha256:foreign").validate_for_evaluation(
            _scope(), case
        )
    with pytest.raises(ValueError, match="outside the permitted training scope"):
        _checkpoint(training_target_ids=("foreign-train",)).validate_for_evaluation(
            _scope(), case
        )
    with pytest.raises(ValueError, match="include held-out evaluation targets"):
        _checkpoint(training_target_ids=(case.target_id,)).validate_for_evaluation(
            _scope(), case
        )
    with pytest.raises(ValueError, match="standard/OOD evaluation scope"):
        _checkpoint(checkpoint_family=OOD_LENGTH_CHECKPOINT_FAMILY).validate_for_evaluation(
            _scope(), case
        )
    with pytest.raises(ValueError, match="standard/OOD evaluation scope"):
        _checkpoint().validate_for_evaluation(
            _scope(checkpoint_family=OOD_LENGTH_CHECKPOINT_FAMILY), case
        )


def test_primary_scope_rejects_beta_zero_checkpoint_and_runtime_mislabel() -> None:
    case = _one_h_target()
    beta_zero = _checkpoint(training_beta=0.0)

    with pytest.raises(ValueError, match="checkpoint training beta"):
        evaluate_article_v1_run(
            case,
            scheduler="article_sarsa",
            expansion_budget=1,
            evaluation_seed=0,
            checkpoint=beta_zero,
            checkpoint_scope=_scope(),
            beta=1.0,
        )

    beta_zero_scope = _scope(expected_training_beta=0.0)
    beta_zero.validate_for_evaluation(beta_zero_scope, case)
    with pytest.raises(ValueError, match="evaluation beta"):
        evaluate_article_v1_run(
            case,
            scheduler="article_sarsa",
            expansion_budget=1,
            evaluation_seed=0,
            checkpoint=beta_zero,
            checkpoint_scope=beta_zero_scope,
            beta=1.0,
        )


@pytest.mark.parametrize(
    ("checkpoint", "message"),
    (
        (
            _checkpoint(training_certification_tolerance=1e-8),
            "certification tolerance",
        ),
        (_checkpoint(training_episodes_per_target=3), "episodes per target"),
        (_checkpoint(training_expansion_cap=2), "expansion cap"),
        (_checkpoint(learning_rate=2e-3), "learning rate"),
        (
            _checkpoint(
                epsilon_schedule=(
                    ("decay", 0.9),
                    ("minimum", 0.05),
                    ("start", 0.2),
                )
            ),
            "epsilon schedule",
        ),
        (_checkpoint(training_seed=99), "training seed"),
    ),
)
def test_primary_scope_rejects_training_protocol_drift(
    checkpoint: ArticleV1Checkpoint,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        checkpoint.validate_for_evaluation(_scope(), _one_h_target())


def test_complete_scope_rejects_partial_training_unless_explicit_smoke_bound() -> None:
    case = _one_h_target()
    primary_scope = _scope(
        allowed_training_target_ids=("train-a", "train-b")
    )
    incomplete_primary = _checkpoint(training_target_ids=("train-a",))
    with pytest.raises(ValueError, match="exactly match the complete"):
        incomplete_primary.validate_for_evaluation(primary_scope, case)

    smoke_checkpoint = _checkpoint(
        training_target_ids=("train-a",),
        training_scope_mode=PARTIAL_SMOKE_TRAINING_SCOPE,
    )
    smoke_scope = _scope(
        training_scope_mode=PARTIAL_SMOKE_TRAINING_SCOPE,
        allowed_training_target_ids=("train-a",),
    )
    smoke_checkpoint.validate_for_evaluation(smoke_scope, case)
    with pytest.raises(ValueError, match="training scope mode"):
        smoke_checkpoint.validate_for_evaluation(primary_scope, case)


def test_checkpoint_campaign_requires_exact_preregistered_seed_set() -> None:
    scope = _scope(allowed_training_seeds=(17, 23))
    first = _checkpoint(training_seed=17)
    second = _checkpoint(training_seed=23)

    with pytest.raises(ValueError, match="exactly match the required training seeds"):
        _validate_checkpoint_campaign((first,), scope)
    with pytest.raises(ValueError, match="duplicate training seeds"):
        _validate_checkpoint_campaign((first, first), scope)
    _validate_checkpoint_campaign((second, first), scope)


def test_article_sarsa_requires_explicit_checkpoint_scope() -> None:
    with pytest.raises(ValueError, match="explicit checkpoint evaluation scope"):
        evaluate_article_v1_run(
            _one_h_target(),
            scheduler="article_sarsa",
            expansion_budget=1,
            evaluation_seed=0,
            checkpoint=_checkpoint(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("checkpoint_family", OOD_LENGTH_CHECKPOINT_FAMILY),
        ("corpus_config_digest", "sha256:foreign"),
        ("training_target_ids", ["foreign-train"]),
        ("training_seed", 99),
        ("learning_rate", 0.002),
        ("epsilon_schedule", {"decay": 0.9, "minimum": 0.05, "start": 0.2}),
    ),
)
def test_checkpoint_digest_binds_evaluation_provenance(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = tmp_path / "checkpoint.json"
    checkpoint = _checkpoint()
    checkpoint.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["checkpoint_schema"] == ARTICLE_V1_CHECKPOINT_SCHEMA
    assert ArticleV1Checkpoint.load(path) == checkpoint

    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="weight digest mismatch"):
        ArticleV1Checkpoint.load(path)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("beta", 0.0),
        ("certification_tolerance", 1e-8),
        ("episodes_per_target", 3),
        ("expansion_cap", 2),
        ("training_scope_mode", PARTIAL_SMOKE_TRAINING_SCOPE),
        (
            "effective_expansion_budgets",
            [{"target_id": "train-fixture", "expansion_budget": 3}],
        ),
    ),
)
def test_checkpoint_digest_binds_training_protocol(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = tmp_path / "checkpoint.json"
    checkpoint = _checkpoint()
    checkpoint.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["training_protocol"][field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="weight digest mismatch"):
        ArticleV1Checkpoint.load(path)


def test_all_seven_schedulers_share_certifier_and_archive_semantics(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    def fake_evaluate(**kwargs):
        captured.append(kwargs)
        return {
            "search_metrics": {},
            "scheduler_semantics": str(kwargs["scheduler"]),
            "certified": False,
            "terminated": False,
            "truncated": True,
            "expansions": int(kwargs["max_steps"]),
            "runtime_seconds": 0.0,
            "time_to_solution": None,
            "solution_resource_vector": None,
            "witness_operations": [],
        }

    monkeypatch.setattr("experiments.article_v1_runner.evaluate", fake_evaluate)
    case = _one_h_target()
    checkpoint = _checkpoint()
    runs = [
        evaluate_article_v1_run(
            case,
            scheduler=scheduler,
            expansion_budget=3,
            evaluation_seed=23,
            checkpoint=checkpoint if scheduler == "article_sarsa" else None,
            checkpoint_scope=_scope() if scheduler == "article_sarsa" else None,
        )
        for scheduler in PRIMARY_SCHEDULERS
    ]

    assert tuple(run["scheduler"] for run in runs) == PRIMARY_SCHEDULERS
    assert len(captured) == len(PRIMARY_SCHEDULERS) == 7
    assert len({id(call["certification_engine"]) for call in captured}) == 7
    for call, run in zip(captured, runs):
        certifier = call["certification_engine"]
        assert isinstance(certifier, ArticleV1CertificationEngine)
        assert np.array_equal(certifier.target.unitary, case.target.unitary)
        assert call["target_gates"] == ()
        assert call["canonicalization_enabled"] is True
        assert call["pareto_dominance_enabled"] is True
        assert call["absorb_clifford_angles"] is True
        assert call["fairness_interval"] == 0
        assert call["reward_mode"] == "article_v1_expansion_potential"
        assert call["article_v1_beta"] == 1.0
        assert call["target_metric"] is not None
        assert call["instrumentation_enabled"] is True
        assert run["action_semantics"] == "persistent_frontier_record"
        assert run["certification_schema_version"] == ARTICLE_V1_PROFILE.certification_schema
        assert run["reference_witness_used"] is False
        assert run["target_specific_reachability_oracle"] is False


@pytest.mark.parametrize(
    "checkpoint_family",
    (STANDARD_CHECKPOINT_FAMILY, OOD_LENGTH_CHECKPOINT_FAMILY),
)
def test_checkpoint_resume_preserves_primary_and_ood_bytes_without_training(
    tmp_path: Path,
    checkpoint_family: str,
) -> None:
    path = tmp_path / f"{checkpoint_family}.json"
    scope = _scope(checkpoint_family=checkpoint_family)
    checkpoint = _checkpoint(checkpoint_family=checkpoint_family)
    checkpoint.save(path)
    original = path.read_bytes()
    trainer_called = False

    def must_not_train() -> ArticleV1Checkpoint:
        nonlocal trainer_called
        trainer_called = True
        raise AssertionError("compatible resume must not retrain")

    resumed, trained = _load_or_train_article_v1_checkpoint(
        path,
        scope=scope,
        expected_training_seed=17,
        train_callback=must_not_train,
    )

    assert resumed == checkpoint
    assert trained is False
    assert trainer_called is False
    assert path.read_bytes() == original


def test_checkpoint_resume_rejects_corruption_and_force_is_explicit(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.json"
    path.write_text("{not-json", encoding="utf-8")
    replacement = _checkpoint()
    trainer_called = False

    def trainer() -> ArticleV1Checkpoint:
        nonlocal trainer_called
        trainer_called = True
        return replacement

    with pytest.raises(ValueError, match="new run ID or explicitly force"):
        _load_or_train_article_v1_checkpoint(
            path,
            scope=_scope(),
            expected_training_seed=17,
            train_callback=trainer,
        )
    assert trainer_called is False
    assert path.read_text(encoding="utf-8") == "{not-json"

    loaded, trained = _load_or_train_article_v1_checkpoint(
        path,
        scope=_scope(),
        expected_training_seed=17,
        train_callback=trainer,
        force_retrain=True,
    )
    assert trainer_called is True
    assert trained is True
    assert loaded == replacement
    assert ArticleV1Checkpoint.load(path) == replacement


def test_checkpoint_resume_rejects_pre_v2_scientific_schema(tmp_path: Path) -> None:
    path = tmp_path / "old-checkpoint.json"
    _checkpoint().save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checkpoint_schema"] = "article-v1-transferable-linear-checkpoint-v2"
    path.write_text(json.dumps(payload), encoding="utf-8")

    old_bytes = path.read_bytes()
    trainer_called = False

    def forbidden_trainer() -> ArticleV1Checkpoint:
        nonlocal trainer_called
        trainer_called = True
        return _checkpoint()

    with pytest.raises(ValueError, match="new run ID or explicitly force"):
        _load_or_train_article_v1_checkpoint(
            path,
            scope=_scope(),
            expected_training_seed=17,
            train_callback=forbidden_trainer,
        )
    assert trainer_called is False
    assert path.read_bytes() == old_bytes


@pytest.mark.parametrize(
    "old_schema",
    ("article-v1-publication-runner-v1", "article-v1-publication-runner-v2"),
)
def test_run_resume_rejects_pre_v3_manifest_without_rewriting(
    tmp_path: Path, old_schema: str
) -> None:
    destination, _ = initialize_run(
        "pilot", output_root=tmp_path, run_id="old-manifest"
    )
    path = destination / "run_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = old_schema
    path.write_text(json.dumps(payload), encoding="utf-8")
    old_bytes = path.read_bytes()

    with pytest.raises(ValueError, match="run manifest conflicts"):
        initialize_run("pilot", output_root=tmp_path, run_id="old-manifest")
    assert path.read_bytes() == old_bytes


def test_mini_ci_writes_artifacts_resumes_and_never_falls_back_to_reference_witness(
    tmp_path: Path,
) -> None:
    first = mini_ci_benchmark(tmp_path, run_id="regression")
    destination = tmp_path / "regression"
    raw_path = destination / "raw_runs.jsonl"
    checkpoint_path = destination / "checkpoints" / "seed-0.json"
    first_raw = raw_path.read_bytes()
    first_checkpoint = checkpoint_path.read_bytes()
    first_checkpoint_sha = sha256(first_checkpoint).hexdigest()
    records = [
        json.loads(line)
        for line in first_raw.decode("utf-8").splitlines()
        if line.strip()
    ]

    assert first["passed"] is True
    assert first["checkpoint_trained_this_run"] is True
    assert all(first["semantic_checks"].values())
    assert first["raw_record_count"] == first["expected_raw_record_count"] == 9
    assert first["no_reference_witness_fallback"] is True
    assert first["matrix"]["appended"] == len(records)
    assert set(PRIMARY_SCHEDULERS) == {row["scheduler"] for row in records}
    assert all(row["reference_witness_used"] is False for row in records)
    assert all(row["target_specific_reachability_oracle"] is False for row in records)
    assert all(row["witness_operations"] == [] for row in records if not row["certified"])
    assert all(row["source_worktree_digest"].startswith("sha256:") for row in records)
    assert all(Path(path).is_file() for path in first["artifacts"].values())
    assert (destination / "mini_ci_summary.json").is_file()
    progress_events = load_progress_events(destination / "progress.jsonl")
    assert progress_events
    assert all(
        event.schema_version == ARTICLE_V1_PROGRESS_EVENT_SCHEMA
        and event.feature_evaluator_schema_version
        == ARTICLE_V1_PROFILE.feature_evaluator_schema
        for event in progress_events
    )
    assert load_progress_status(destination / "status.json") == progress_events[-1]

    second = mini_ci_benchmark(tmp_path, run_id="regression")
    assert raw_path.read_bytes() == first_raw
    assert checkpoint_path.read_bytes() == first_checkpoint
    assert sha256(checkpoint_path.read_bytes()).hexdigest() == first_checkpoint_sha
    assert second["checkpoint_trained_this_run"] is False
    assert second["matrix"] == {
        "appended": 0,
        "skipped": len(records),
        "completed": len(records),
    }
    assert second["checkpoint_digest"] == first["checkpoint_digest"]
    assert second["no_reference_witness_fallback"] is True

    checkpoint = ArticleV1Checkpoint.load(checkpoint_path)
    corrupted = [dict(row) for row in records]
    fifo = next(row for row in corrupted if row["scheduler"] == "fifo")
    fifo["certified"] = False
    experiment = json.loads(
        (destination / "run_manifest.json").read_text(encoding="utf-8")
    )["config"]["experiment"]
    checks = _mini_ci_semantic_checks(
        corrupted,
        target_ids=first["target_ids"],
        experiment=experiment,
        checkpoint=checkpoint,
    )
    assert checks["fifo_independently_certified_known_reachable_target"] is False

    environment = json.loads(
        (destination / "environment.json").read_text(encoding="utf-8")
    )
    assert environment["cwd"] == "."
    assert environment["git"]["source_worktree_digest"].startswith("sha256:")

    manifest_path = destination / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_digest"] = "sha256:stale-config"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    stale_bytes = manifest_path.read_bytes()
    with pytest.raises(ValueError, match="run manifest conflicts"):
        mini_ci_benchmark(tmp_path, run_id="regression")
    assert manifest_path.read_bytes() == stale_bytes


def test_git_provenance_includes_source_worktree_digest() -> None:
    provenance = git_provenance()
    assert {
        "commit_sha",
        "branch",
        "dirty_worktree",
        "source_worktree_digest",
        "relevant_untracked_files",
    } <= set(provenance)
    assert provenance["source_worktree_digest"] == "unknown" or str(
        provenance["source_worktree_digest"]
    ).startswith("sha256:")
