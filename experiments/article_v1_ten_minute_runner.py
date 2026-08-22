"""Executable orchestration for the Article V1 ten-minute protocol."""

from __future__ import annotations

from contextlib import redirect_stdout
import csv
from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from benchmarks.article_native_corpus import build_article_v1_corpus
from certification.article_v1 import ArticleV1CertificationEngine
from config import Config
from env.rl_env import CircuitSynthesisEnv
from experiments.article_v1_runner import (
    PRIMARY_SCHEDULERS,
    _feature_provider,
    _replay_training_checkpoint,
    _restore_rng_state,
    _target,
    _training_provenance,
    _training_state_digests,
    evaluate_article_v1_run,
)
from experiments.article_v1_training_checkpoint import (
    ArticleV1EventJournal,
    ArticleV1JournalEntry,
    ArticleV1TrainingCheckpointStore,
    MidEpisodeCheckpoint,
    ResumeExpectation,
    feature_row_digest,
    policy_weight_digest,
    validate_resume_compatibility,
)
from experiments.article_v1_ten_minute_protocol import (
    CurriculumAccounting,
    RUNTIME_PROTOCOL_SCHEMA,
    TEN_MINUTE_CHECKPOINT_SCHEMA,
    TRAINING_PROTOCOL_SCHEMA,
    TEN_MINUTE_FRONTIER_ENUMERATION_SCHEMA,
    TenMinuteCheckpoint,
    audit_ten_minute_runs,
    load_ten_minute_config,
    load_ten_minute_corpus_config,
)
from rl.article_features import ARTICLE_V1_FEATURE_SCHEMA_VERSION
from rl.policy import LinearQPolicy
from train import Trainer, TrainerBoundaryEvent, TrainerEpisodeResume
from reporting.article_v1 import derive_fixed_horizon_anytime_rows


TEN_MINUTE_TRAINING_RUN_SCHEMA = "article-v1-10min-training-run-v1"
TEN_MINUTE_STATUS_SCHEMA = "article-v1-10min-status-v1"
TEN_MINUTE_EVALUATION_SCHEMA = "article-v1-10min-evaluation-v1"
TEN_MINUTE_CALIBRATION_SCHEMA = "article-v1-10min-horizon-calibration-v1"
TEN_MINUTE_PLAN_SCHEMA = "article-v1-10min-campaign-plan-v1"
TEN_MINUTE_CPU_EVALUATION_SCHEMA = "article-v1-10min-equal-cpu-evaluation-v1"
TEN_MINUTE_SENSITIVITY_SCHEMA = "article-v1-10min-protocol-sensitivity-v1"
TEN_MINUTE_REPORT_SCHEMA = "article-v1-10min-report-v1"
TEN_MINUTE_TRAINING_SELECTION_SCHEMA = "article-v1-10min-training-selection-v1"


class TenMinuteOperabilityTimeout(RuntimeError):
    def __init__(self, event: TrainerBoundaryEvent, cpu_seconds: float):
        super().__init__("Article V1 episode exceeded its process-CPU limit")
        self.event = event
        self.cpu_seconds = float(cpu_seconds)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def train_ten_minute(
    config_path: str | Path,
    output_directory: str | Path,
    *,
    training_seed: int,
    total_expansions_override: int | None = None,
    process_cpu_clock_ns: Callable[[], int] = time.process_time_ns,
    resume_training: bool = True,
    checkpoint_every_expansions: int = 64,
    interrupt_after_expansions: int | None = None,
) -> dict[str, Any]:
    """Train one persistent Article V1 policy under a fixed interaction total."""

    protocol = load_ten_minute_config(config_path)
    corpus = build_article_v1_corpus(load_ten_minute_corpus_config(config_path))
    eligible = tuple(
        case
        for case in corpus.cases(split="train")
        if case.difficulty in protocol.training.eligible_difficulties
    )
    if not eligible or any(case.difficulty == "hard" for case in eligible):
        raise ValueError("primary ten-minute training requires easy/medium train cases")
    total_budget = (
        protocol.training.total_expansions_per_seed
        if total_expansions_override is None
        else int(total_expansions_override)
    )
    if total_budget is None or total_budget < 1:
        raise ValueError("training total must be resolved and positive")

    if checkpoint_every_expansions < 1:
        raise ValueError("checkpoint cadence must be positive")
    experiment = protocol.payload["experiment"]
    epsilon = experiment["epsilon"]
    accounting = CurriculumAccounting(
        tuple(case.target_id for case in eligible),
        {case.target_id: case.difficulty for case in eligible},
        int(total_budget),
        protocol.training.episode_caps_by_difficulty,
        int(training_seed),
        protocol.training.allow_partial_final_episode,
    )
    cases_by_id = {case.target_id: case for case in eligible}
    destination = Path(output_directory)
    checkpoint_path = destination / "checkpoints" / f"seed-{training_seed}.json"
    recovery_store = ArticleV1TrainingCheckpointStore(
        destination / "training_state" / f"seed-{training_seed}"
    )
    loaded_checkpoint: TenMinuteCheckpoint | None = None
    if resume_training and checkpoint_path.exists():
        loaded_checkpoint = TenMinuteCheckpoint.load(checkpoint_path)
        if (
            loaded_checkpoint.training_seed != int(training_seed)
            or loaded_checkpoint.corpus_config_digest != protocol.digest
            or loaded_checkpoint.total_expansion_budget != int(total_budget)
            or loaded_checkpoint.ordered_target_ids
            != tuple(case.target_id for case in eligible)
        ):
            raise ValueError("existing V5 checkpoint is incompatible with this curriculum")

    loaded_mid: MidEpisodeCheckpoint | None = None
    if resume_training and (
        recovery_store.checkpoint_path("latest").exists()
        or recovery_store.manifest_path("latest").exists()
    ):
        candidate = recovery_store.load_latest_or_previous()
        if isinstance(candidate, MidEpisodeCheckpoint):
            loaded_mid = candidate

    policy: LinearQPolicy | None = None
    current_epsilon = float(epsilon["start"])
    histories: list[dict[str, Any]] = []
    effective_caps: list[int] = []
    episode_expansions: list[int] = []
    status = "COMPLETE"
    progress_path = destination / "progress.jsonl"
    progress_reporting_time_ns = 0

    recovery_schedule: tuple[str, ...] = ()
    recovery_caps: tuple[int, ...] = ()
    recovery_expansions: tuple[int, ...] = ()
    if loaded_checkpoint is not None:
        recovery_expansions = loaded_checkpoint.episode_expansions
        completed_count = len(recovery_expansions)
        recovery_schedule = loaded_checkpoint.executed_target_schedule[:completed_count]
        recovery_caps = loaded_checkpoint.effective_episode_caps[:completed_count]
        current_epsilon = loaded_checkpoint.current_epsilon
    if loaded_mid is not None:
        aggregates = dict(loaded_mid.training_aggregates)
        mid_expansions = tuple(
            int(value) for value in aggregates.get("completed_episode_expansions", ())
        )
        if len(mid_expansions) >= len(recovery_expansions):
            recovery_expansions = mid_expansions
            recovery_schedule = tuple(
                str(value) for value in aggregates.get("completed_target_schedule", ())
            )
            recovery_caps = tuple(
                int(value) for value in aggregates.get("completed_effective_caps", ())
            )
    if recovery_expansions:
        accounting.restore_completed_episodes(
            target_schedule=recovery_schedule,
            effective_caps=recovery_caps,
            episode_expansions=recovery_expansions,
        )
        effective_caps.extend(recovery_caps)
        episode_expansions.extend(recovery_expansions)
        summary_path = destination / "training_summary.json"
        if summary_path.exists():
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            histories = [
                dict(value)
                for value in previous.get("training_histories", ())
                if value.get("complete") is True
            ][: len(recovery_expansions)]

    while accounting.remaining:
        episode = accounting.next_episode()
        assert episode is not None
        target_id, effective_cap = episode
        effective_caps.append(effective_cap)
        case = cases_by_id[target_id]
        provider, context = _feature_provider(
            case,
            expansion_budget=effective_cap,
            feature_schema=ARTICLE_V1_FEATURE_SCHEMA_VERSION,
        )
        previous_weights = (
            None
            if policy is None
            else np.asarray(policy.theta, dtype=np.float64).copy()
        )
        previous_rng_state = (
            None
            if policy is None
            else json.loads(json.dumps(policy.rng.bit_generator.state))
        )
        policy = LinearQPolicy(
            feature_provider=provider,
            lr=float(experiment["learning_rate"]),
            gamma=1.0,
            seed=int(training_seed),
        )
        if previous_weights is not None:
            policy.theta[:] = previous_weights
            assert previous_rng_state is not None
            _restore_rng_state(policy.rng, previous_rng_state)
        environment = CircuitSynthesisEnv(
            Config(
                num_qubits=case.num_qubits,
                budget=case.budget.resource_budget(),
                max_steps=effective_cap,
                max_frontier=64,
                discount=1.0,
                seed=int(training_seed) + len(episode_expansions),
                fairness_interval=0,
                reward_mode="article_v1_expansion_potential",
                article_v1_beta=float(experiment["beta"]),
            ),
            ArticleV1CertificationEngine(
                _target(case),
                tau_cert=float(experiment["certification_tolerance"]),
            ),
            feature_provider=provider,
            target_metric=context,
            observation_features=False,
        )
        trainer = Trainer(environment, policy=policy)
        if loaded_checkpoint is not None and len(histories) == len(recovery_expansions):
            policy.theta[:] = np.asarray(loaded_checkpoint.weights, dtype=np.float64)
            if loaded_checkpoint.policy_rng_state:
                _restore_rng_state(policy.rng, loaded_checkpoint.policy_rng_state)
            loaded_checkpoint = None
        trainer.epsilon = current_epsilon
        trainer.min_epsilon = float(epsilon["minimum"])
        trainer.epsilon_decay = float(epsilon["decay"])
        episode_cpu_started = int(process_cpu_clock_ns())
        episode_initial_theta = tuple(float(value) for value in policy.theta)
        journal = ArticleV1EventJournal()
        td_errors: list[float] = []
        resume_episode: TrainerEpisodeResume | None = None
        provenance = _training_provenance(
            case,
            corpus_config_digest=protocol.digest,
            target_fingerprint=context.fingerprint,
            feature_schema=ARTICLE_V1_FEATURE_SCHEMA_VERSION,
            feature_evaluator_schema=str(provider.evaluator_schema_version),
        )
        if loaded_mid is not None:
            aggregates = dict(loaded_mid.training_aggregates)
            if (
                len(aggregates.get("completed_episode_expansions", ()))
                == len(episode_expansions)
                and loaded_mid.provenance.target_id == target_id
            ):
                validate_resume_compatibility(
                    loaded_mid,
                    ResumeExpectation(
                        provenance=provenance,
                        training_seed=int(training_seed),
                        episode_index=0,
                        episode_count=1,
                        expansion_cap=effective_cap,
                        feature_dimension=len(policy.theta),
                    ),
                )
                journal = ArticleV1EventJournal(loaded_mid.journal.entries)
                td_errors = [
                    float(value)
                    for value in aggregates.get("td_errors", ())
                ]
                episode_initial_theta = loaded_mid.episode_initial_theta
                resume_episode = _replay_training_checkpoint(
                    loaded_mid,
                    environment=environment,
                    policy=policy,
                    trainer=trainer,
                )
                current_epsilon = float(trainer.epsilon)
                loaded_mid = None

        def save_mid(event: TrainerBoundaryEvent) -> None:
            if event.next_record_id is None or event.next_features is None:
                return
            frontier_digest, archive_digest, generation_digest = (
                _training_state_digests(environment)
            )
            journal.bind_latest_state_digests(
                frontier_active_ids_digest=frontier_digest,
                archive_digest=archive_digest,
                generation_count_digest=generation_digest,
            )
            recovery_store.save_latest(
                MidEpisodeCheckpoint(
                    provenance=provenance,
                    training_seed=int(training_seed),
                    episode_index=0,
                    episode_count=1,
                    expansion_count=event.expansion,
                    expansion_cap=effective_cap,
                    journal=journal,
                    episode_initial_theta=episode_initial_theta,
                    theta=event.policy_weights_after_update,
                    epsilon=event.epsilon,
                    policy_rng_state=event.policy_rng_state,
                    environment_rng_state=event.environment_rng_state,
                    pending_next_record_id=event.next_record_id,
                    pending_next_feature_row=event.next_features,
                    total_reward=event.total_reward,
                    training_aggregates={
                        "td_errors": list(td_errors),
                        "completed_episode_expansions": list(episode_expansions),
                        "completed_target_schedule": list(accounting.executed_target_schedule[:-1]),
                        "completed_effective_caps": list(effective_caps[:-1]),
                    },
                    search_metrics={
                        name: value
                        for name, value in event.search_metrics.items()
                        if not name.endswith("_time_ns")
                    },
                    frontier_revision=event.frontier_revision,
                    frontier_active_ids_digest=frontier_digest,
                    archive_digest=archive_digest,
                    generation_count_digest=generation_digest,
                )
            )

        def boundary_callback(event: TrainerBoundaryEvent) -> None:
            nonlocal progress_reporting_time_ns
            if event.boundary != "expansion":
                return
            assert event.selected_record_id is not None
            assert event.selected_features is not None
            assert event.reward is not None and event.td_error is not None
            journal.append(
                ArticleV1JournalEntry(
                    expansion_index=event.expansion,
                    selected_record_id=event.selected_record_id,
                    selected_feature_digest=feature_row_digest(event.selected_features),
                    reward=event.reward,
                    terminated=event.terminated,
                    truncated=event.truncated,
                    frontier_revision=event.frontier_revision,
                    state_digest_verified=False,
                    frontier_active_ids_digest=None,
                    archive_digest=None,
                    generation_count_digest=None,
                    policy_weight_digest_after_update=policy_weight_digest(
                        event.policy_weights_after_update
                    ),
                    pending_next_record_id=event.next_record_id,
                )
            )
            td_errors.append(float(event.td_error))
            progress_started = time.perf_counter_ns()
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "schema_version": "article-v1-10min-progress-v1",
                            "training_seed": int(training_seed),
                            "episode_index": len(episode_expansions),
                            "target_id": target_id,
                            "current_target_difficulty": case.difficulty,
                            "effective_episode_cap": effective_cap,
                            "episode_expansion": event.expansion,
                            "selected_record_id": event.selected_record_id,
                            "policy_weight_digest": event.policy_weight_digest_after_update,
                            "total_training_expansions_completed": accounting.completed + event.expansion,
                            "total_training_expansions_remaining": total_budget - accounting.completed - event.expansion,
                            "curriculum_cycle": len(accounting.executed_target_schedule) // len(eligible),
                            "episode_cpu_seconds": (
                                int(process_cpu_clock_ns()) - episode_cpu_started
                            ) / 1e9,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            progress_reporting_time_ns += time.perf_counter_ns() - progress_started
            if event.terminated or event.truncated:
                return
            elapsed = (int(process_cpu_clock_ns()) - episode_cpu_started) / 1e9
            due = event.expansion % checkpoint_every_expansions == 0
            interrupted = (
                interrupt_after_expansions is not None
                and event.expansion == int(interrupt_after_expansions)
            )
            timed_out = elapsed >= protocol.runtime.hard_episode_cpu_limit_seconds
            if due or interrupted or timed_out:
                save_mid(event)
            if interrupted:
                raise KeyboardInterrupt
            if timed_out:
                raise TenMinuteOperabilityTimeout(event, elapsed)

        trainer.checkpoint_callback = boundary_callback
        wall_started = time.perf_counter_ns()
        try:
            with redirect_stdout(StringIO()):
                result = trainer.train(1, resume_episode=resume_episode)[0]
            expansions = int(result["steps"])
            episode_cpu_seconds = (
                int(process_cpu_clock_ns()) - episode_cpu_started
            ) / 1e9
            history = dict(result)
            complete = True
        except TenMinuteOperabilityTimeout as timeout:
            expansions = int(timeout.event.expansion)
            episode_cpu_seconds = timeout.cpu_seconds
            history = {
                "steps": expansions,
                "certified": False,
                "truncated": False,
                "reward": float(timeout.event.total_reward),
                "policy_weight_digest": timeout.event.policy_weight_digest_after_update,
            }
            complete = False
            status = "OPERABILITY_TIMEOUT"
        if complete:
            accounting.record_expansions(target_id, expansions)
            episode_expansions.append(expansions)
        current_epsilon = float(trainer.epsilon)
        history.update(
            {
                "target_id": target_id,
                "difficulty": case.difficulty,
                "effective_episode_cap": effective_cap,
                "episode_index": len(histories),
                "episode_cpu_seconds": float(episode_cpu_seconds),
                "episode_wall_seconds": (time.perf_counter_ns() - wall_started) / 1e9,
                "complete": complete,
                "terminal_reason": status if not complete else (
                    "CERTIFIED" if history.get("certified") else "UNSOLVED_WITHIN_EXPANSION_BUDGET"
                ),
            }
        )
        histories.append(history)

        checkpoint = TenMinuteCheckpoint(
            training_seed=int(training_seed),
            weights=tuple(float(value) for value in policy.theta),
            feature_schema_version=ARTICLE_V1_FEATURE_SCHEMA_VERSION,
            feature_evaluator_schema_version=str(provider.evaluator_schema_version),
            frontier_enumeration_schema_version=TEN_MINUTE_FRONTIER_ENUMERATION_SCHEMA,
            ordered_feature_names=tuple(str(name) for name in provider.names),
            learning_rate=float(experiment["learning_rate"]),
            corpus_config_digest=protocol.digest,
            runtime_protocol_schema_version=RUNTIME_PROTOCOL_SCHEMA,
            training_protocol_schema_version=TRAINING_PROTOCOL_SCHEMA,
            total_expansion_budget=int(total_budget),
            total_completed_expansions=(
                accounting.completed if complete else accounting.completed + expansions
            ),
            eligible_splits=protocol.training.eligible_splits,
            eligible_difficulties=protocol.training.eligible_difficulties,
            ordered_target_ids=tuple(case.target_id for case in eligible),
            executed_target_schedule=tuple(accounting.executed_target_schedule),
            target_schedule_seed=int(training_seed),
            episode_caps_by_difficulty=tuple(sorted(protocol.training.episode_caps_by_difficulty.items())),
            effective_episode_caps=tuple(effective_caps),
            episode_expansions=tuple(episode_expansions),
            current_epsilon=current_epsilon,
            policy_rng_state=json.loads(json.dumps(policy.rng.bit_generator.state)),
        )
        checkpoint.save(checkpoint_path)
        if not complete:
            break

    assert policy is not None
    reported_completed = int(checkpoint.total_completed_expansions)
    training_payload = {
        "schema_version": TEN_MINUTE_TRAINING_RUN_SCHEMA,
        "status": status,
        "complete": status == "COMPLETE" and accounting.remaining == 0,
        "config_digest": protocol.digest,
        "checkpoint_schema": TEN_MINUTE_CHECKPOINT_SCHEMA,
        "checkpoint_digest": checkpoint.digest,
        "checkpoint_path": str(checkpoint_path),
        "training_seed": int(training_seed),
        "training_histories": histories,
        "progress_reporting_time_ns": int(progress_reporting_time_ns),
        "checkpoint_io_time_ns": int(recovery_store.checkpoint_io_time_ns),
        **accounting.metadata(),
        "total_training_expansions_completed": reported_completed,
        "total_training_expansions_remaining": total_budget - reported_completed,
    }
    _atomic_json(destination / "training_summary.json", training_payload)
    _atomic_json(
        destination / "status.json",
        {
            "schema_version": TEN_MINUTE_STATUS_SCHEMA,
            "phase": "training",
            "status": status,
            "complete": training_payload["complete"],
            "checkpoint_path": str(checkpoint_path),
            "total_training_expansions_completed": reported_completed,
            "total_training_expansions_remaining": total_budget - reported_completed,
        },
    )
    return training_payload


def evaluate_ten_minute(
    config_path: str | Path,
    output_directory: str | Path,
    *,
    checkpoints: tuple[TenMinuteCheckpoint, ...] = (),
    schedulers: tuple[str, ...] = PRIMARY_SCHEDULERS,
    families: tuple[str, ...] = (
        "in_distribution",
        "hard_generalization",
        "length_ood",
    ),
    require_frozen: bool = True,
    maximum_targets_per_family: int | None = None,
    horizon_override: int | None = None,
    budget_mode: str = "fixed-max-horizon-anytime-v1",
    cpu_limit_override: float | None = None,
) -> dict[str, Any]:
    """Execute one maximum-horizon trajectory per physical run identity."""

    protocol = load_ten_minute_config(config_path, require_frozen=require_frozen)
    if budget_mode not in {
        "fixed-max-horizon-anytime-v1",
        "equal-cpu-budget-secondary-v1",
    }:
        raise ValueError("unsupported ten-minute evaluation budget mode")
    corpus = build_article_v1_corpus(load_ten_minute_corpus_config(config_path))
    family_cases = {
        "in_distribution": tuple(
            case for case in corpus.cases(split="test")
            if case.difficulty in ("easy", "medium")
        ),
        "hard_generalization": corpus.cases(split="test", difficulty="hard"),
        "length_ood": corpus.cases(split="ood_test"),
    }
    if any(family not in family_cases for family in families):
        raise ValueError("unknown ten-minute evaluation family")
    if tuple(schedulers) != tuple(dict.fromkeys(schedulers)) or any(
        scheduler not in PRIMARY_SCHEDULERS for scheduler in schedulers
    ):
        raise ValueError("invalid or duplicate scheduler selection")
    if "article_sarsa" in schedulers and not checkpoints:
        raise ValueError("article_sarsa evaluation requires V5 checkpoints")
    if any(cp.corpus_config_digest != protocol.digest for cp in checkpoints):
        raise ValueError("evaluation checkpoint/config digest mismatch")

    raw_rows: list[dict[str, Any]] = []
    derived_rows: list[dict[str, Any]] = []
    experiment = protocol.payload["experiment"]
    for family in families:
        cases = family_cases[family]
        if maximum_targets_per_family is not None:
            cases = cases[: int(maximum_targets_per_family)]
        for case in cases:
            horizon = int(
                horizon_override
                if horizon_override is not None
                else protocol.budget.maximum_horizon_by_difficulty[case.difficulty]
            )
            thresholds = tuple(
                value
                for value in protocol.budget.thresholds_by_difficulty[case.difficulty]
                if value <= horizon
            )
            if not thresholds:
                thresholds = (horizon,)
            for scheduler in schedulers:
                run_checkpoints = checkpoints if scheduler == "article_sarsa" else (None,)
                seeds = (
                    tuple(int(v) for v in experiment["random_scheduler_seeds"])
                    if scheduler == "seeded_random"
                    else (0,)
                )
                for checkpoint in run_checkpoints:
                    for evaluation_seed in seeds:
                        row = evaluate_article_v1_run(
                            case,
                            scheduler=scheduler,
                            expansion_budget=horizon,
                            evaluation_seed=evaluation_seed,
                            checkpoint=checkpoint,
                            beta=float(experiment["beta"]),
                            certification_tolerance=float(experiment["certification_tolerance"]),
                            config_digest=protocol.digest,
                            budget_mode=budget_mode,
                            budget_thresholds=thresholds,
                            process_cpu_limit_seconds=(
                                protocol.runtime.hard_episode_cpu_limit_seconds
                                if cpu_limit_override is None
                                else float(cpu_limit_override)
                            ),
                        )
                        row["evaluation_family"] = family
                        raw_rows.append(row)
                        if budget_mode == "fixed-max-horizon-anytime-v1":
                            derived_rows.extend(
                                {
                                    **derived,
                                    "evaluation_family": family,
                                    "curve_label": "fixed-horizon anytime budget-success curve",
                                }
                                for derived in derive_fixed_horizon_anytime_rows(row, thresholds)
                            )

    audit = audit_ten_minute_runs(raw_rows)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    raw_path = destination / "raw_runs.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_rows),
        encoding="utf-8",
    )
    table_name = (
        "fixed_horizon_anytime_rows.json"
        if budget_mode == "fixed-max-horizon-anytime-v1"
        else "equal_cpu_rows.json"
    )
    _atomic_json(destination / "tables" / table_name, derived_rows if derived_rows else raw_rows)
    _atomic_json(destination / "audits" / "campaign_audit.json", audit)
    result = {
        "schema_version": (
            TEN_MINUTE_EVALUATION_SCHEMA
            if budget_mode == "fixed-max-horizon-anytime-v1"
            else TEN_MINUTE_CPU_EVALUATION_SCHEMA
        ),
        "budget_mode": budget_mode,
        "passed": audit["passed"],
        "physical_search_execution_count": len(raw_rows),
        "derived_budget_threshold_observation_count": len(derived_rows),
        "raw_runs_path": str(raw_path),
        "audit": audit,
    }
    _atomic_json(destination / "evaluation_summary.json", result)
    return result


def evaluate_cpu_budget(
    config_path: str | Path,
    output_directory: str | Path,
    *,
    checkpoints: tuple[TenMinuteCheckpoint, ...] = (),
    cpu_seconds: float | None = None,
) -> dict[str, Any]:
    protocol = load_ten_minute_config(config_path, require_frozen=True)
    if protocol.secondary_cpu.enabled is not True or protocol.secondary_cpu.report_separately is not True:
        raise ValueError("secondary CPU experiment must be enabled and separate")
    return evaluate_ten_minute(
        config_path,
        output_directory,
        checkpoints=checkpoints,
        schedulers=protocol.secondary_cpu.schedulers,
        require_frozen=True,
        budget_mode="equal-cpu-budget-secondary-v1",
        cpu_limit_override=(
            protocol.secondary_cpu.cpu_budget_seconds
            if cpu_seconds is None
            else float(cpu_seconds)
        ),
    )


def select_feasible_hard_cap(
    rows: list[dict[str, Any]],
    *,
    candidate_caps: tuple[int, ...],
    target_cpu_seconds: float,
    hard_cpu_limit_seconds: float,
    maximum_feature_index_memory_bytes: int,
    selection_quantile: float = 0.95,
    correctness_parity_passed: bool,
) -> dict[str, Any]:
    """Select the largest validation-only cap satisfying every operability gate."""

    decisions = []
    selected = None
    for cap in candidate_caps:
        samples = [row for row in rows if int(row["executed_max_horizon"]) == cap]
        cpu = np.asarray([float(row["process_cpu_seconds"]) for row in samples])
        memory = [int(row["search_metrics"].get("feature_index_memory_bytes", 0)) for row in samples]
        q95 = None if not len(cpu) else float(np.quantile(cpu, selection_quantile))
        maximum = None if not len(cpu) else float(np.max(cpu))
        peak_memory = None if not memory else max(memory)
        checks = {
            "samples_present": bool(samples),
            "cpu_quantile_within_target": q95 is not None and q95 <= target_cpu_seconds,
            "maximum_cpu_within_hard_limit": maximum is not None and maximum <= hard_cpu_limit_seconds,
            "zero_operability_timeouts": all(row.get("terminal_reason") != "OPERABILITY_TIMEOUT" for row in samples),
            "feature_index_memory_within_limit": peak_memory is not None and peak_memory <= maximum_feature_index_memory_bytes,
            "correctness_parity_passed": correctness_parity_passed is True,
        }
        passed = all(checks.values())
        decisions.append({"candidate_cap": cap, "passed": passed, "cpu_quantile": q95, "maximum_cpu_seconds": maximum, "peak_feature_index_memory_bytes": peak_memory, "checks": checks})
        if passed: selected = cap
    return {"selected_hard_expansion_cap": selected, "candidate_decisions": decisions, "selection_quantile": selection_quantile, "target_cpu_seconds": target_cpu_seconds, "hard_cpu_limit_seconds": hard_cpu_limit_seconds}


def calibrate_ten_minute_horizon(
    config_path: str | Path,
    output_directory: str | Path,
    *,
    correctness_parity_passed: bool,
    checkpoint: TenMinuteCheckpoint | None = None,
    schedulers: tuple[str, ...] = (
        "zero_weight_linear",
        "article_target_distance",
        "fifo",
    ),
) -> dict[str, Any]:
    """Measure candidate caps on train/validation hard targets only."""

    protocol = load_ten_minute_config(config_path)
    corpus = build_article_v1_corpus(load_ten_minute_corpus_config(config_path))
    calibration_cases = tuple(
        case for case in corpus.targets
        if case.split in ("train", "validation") and case.difficulty == "hard"
    )
    if not calibration_cases or any(case.split in ("test", "ood_test") for case in calibration_cases):
        raise ValueError("calibration target leakage detected")
    selected_schedulers = tuple(schedulers)
    if checkpoint is not None and "article_sarsa" not in selected_schedulers:
        selected_schedulers = (*selected_schedulers, "article_sarsa")
    rows: list[dict[str, Any]] = []
    experiment = protocol.payload["experiment"]
    for cap in protocol.runtime.candidate_hard_expansion_caps:
        for case in calibration_cases:
            for scheduler in selected_schedulers:
                rows.append(evaluate_article_v1_run(
                    case,
                    scheduler=scheduler,
                    expansion_budget=cap,
                    evaluation_seed=int(experiment["validation_seeds"][0]),
                    checkpoint=checkpoint if scheduler == "article_sarsa" else None,
                    beta=float(experiment["beta"]),
                    certification_tolerance=float(experiment["certification_tolerance"]),
                    config_digest=protocol.digest,
                    budget_mode="fixed-max-horizon-anytime-v1",
                    budget_thresholds=(cap,),
                    process_cpu_limit_seconds=protocol.runtime.hard_episode_cpu_limit_seconds,
                ))
    selection = select_feasible_hard_cap(
        rows,
        candidate_caps=protocol.runtime.candidate_hard_expansion_caps,
        target_cpu_seconds=protocol.runtime.target_episode_cpu_seconds,
        hard_cpu_limit_seconds=protocol.runtime.hard_episode_cpu_limit_seconds,
        maximum_feature_index_memory_bytes=int(protocol.runtime.maximum_feature_index_memory_mb * 1024 * 1024),
        selection_quantile=protocol.runtime.selection_quantile,
        correctness_parity_passed=correctness_parity_passed,
    )
    result = {"schema_version": TEN_MINUTE_CALIBRATION_SCHEMA, "config_digest": protocol.digest, "no_test_access": True, "calibration_splits": ["train", "validation"], "calibration_target_ids": [case.target_id for case in calibration_cases], "run_count": len(rows), "rows": rows, **selection}
    destination = Path(output_directory)
    _atomic_json(destination / "calibration.json", result)
    return result


def select_training_interaction_budget(
    rows: list[dict[str, Any]],
    *,
    candidate_totals: tuple[int, ...],
) -> dict[str, Any]:
    """Select by validation success, then efficiency, then smaller compute."""

    if not rows:
        raise ValueError("training-budget selection requires validation rows")
    if any(row.get("split") != "validation" for row in rows):
        raise ValueError("training-budget selection cannot read non-validation rows")
    decisions = []
    for total in candidate_totals:
        samples = [
            row
            for row in rows
            if int(row.get("total_training_expansions_per_seed", -1)) == total
        ]
        if not samples:
            raise ValueError(f"missing validation evidence for candidate total {total}")
        successes = [bool(row.get("certified", False)) for row in samples]
        successful_expansions = [
            int(row.get("first_certified_hit_expansion", row.get("expansions", 0)))
            for row, success in zip(samples, successes)
            if success
        ]
        decisions.append(
            {
                "candidate_total_expansions_per_seed": int(total),
                "validation_run_count": len(samples),
                "validation_success_rate": sum(successes) / len(successes),
                "mean_successful_expansions": (
                    None
                    if not successful_expansions
                    else float(np.mean(successful_expansions))
                ),
            }
        )
    selected = min(
        decisions,
        key=lambda row: (
            -float(row["validation_success_rate"]),
            float("inf")
            if row["mean_successful_expansions"] is None
            else float(row["mean_successful_expansions"]),
            int(row["candidate_total_expansions_per_seed"]),
        ),
    )
    return {
        "schema_version": TEN_MINUTE_TRAINING_SELECTION_SCHEMA,
        "no_test_access": True,
        "selection_split": "validation",
        "selection_rule": (
            "maximum validation success rate; then minimum mean successful "
            "expansions; then minimum total training expansions"
        ),
        "candidate_decisions": decisions,
        "selected_total_expansions_per_seed": selected[
            "candidate_total_expansions_per_seed"
        ],
    }


def plan_ten_minute_campaign(
    config_path: str | Path,
    *,
    training_cpu_seconds_per_expansion: float | None = None,
) -> dict[str, Any]:
    """Return physical and derived counts without executing search."""

    protocol = load_ten_minute_config(config_path)
    corpus = build_article_v1_corpus(load_ten_minute_corpus_config(config_path))
    experiment = protocol.payload["experiment"]
    learner_count = len(experiment["training_seeds"])
    random_count = len(experiment["random_scheduler_seeds"])
    physical = 0
    derived = 0
    by_family: dict[str, dict[str, int]] = {}
    family_cases = {
        "in_distribution": tuple(case for case in corpus.cases(split="test") if case.difficulty in ("easy", "medium")),
        "hard_generalization": corpus.cases(split="test", difficulty="hard"),
        "length_ood": corpus.cases(split="ood_test"),
    }
    for family, cases in family_cases.items():
        family_physical = 0
        family_derived = 0
        for case in cases:
            thresholds = protocol.budget.thresholds_by_difficulty[case.difficulty]
            for scheduler in PRIMARY_SCHEDULERS:
                repetitions = learner_count if scheduler == "article_sarsa" else random_count if scheduler == "seeded_random" else 1
                family_physical += repetitions
                family_derived += repetitions * len(thresholds)
        by_family[family] = {"target_count": len(cases), "physical_search_executions": family_physical, "derived_budget_threshold_observations": family_derived}
        physical += family_physical
        derived += family_derived
    training_total = protocol.training.total_expansions_per_seed
    total_interactions = (
        None if training_total is None else int(training_total) * learner_count
    )
    if training_cpu_seconds_per_expansion is not None:
        if not np.isfinite(training_cpu_seconds_per_expansion) or training_cpu_seconds_per_expansion <= 0:
            raise ValueError("training CPU seconds per expansion must be positive")
    return {"schema_version": TEN_MINUTE_PLAN_SCHEMA, "executes_search": False, "config_digest": protocol.digest, "training_total_expansions_per_seed": training_total, "candidate_training_totals_per_seed": list(protocol.training.candidate_total_expansions_per_seed), "learner_seed_count": learner_count, "random_scheduler_seed_count": random_count, "total_training_interactions_all_seeds": total_interactions, "training_cpu_seconds_per_expansion_assumption": training_cpu_seconds_per_expansion, "projected_training_cpu_seconds_all_seeds": (None if total_interactions is None or training_cpu_seconds_per_expansion is None else total_interactions * training_cpu_seconds_per_expansion), "physical_search_execution_count": physical, "derived_budget_threshold_observation_count": derived, "by_evaluation_family": by_family}


def validate_protocol_sensitivity(
    config_path: str | Path,
    output_directory: str | Path,
    *,
    checkpoint: TenMinuteCheckpoint,
    maximum_targets: int = 1,
    horizon_override: int | None = None,
) -> dict[str, Any]:
    """Compare anytime and independent horizons on validation targets only."""

    protocol = load_ten_minute_config(config_path)
    if checkpoint.corpus_config_digest != protocol.digest:
        raise ValueError("protocol-sensitivity checkpoint/config mismatch")
    corpus = build_article_v1_corpus(load_ten_minute_corpus_config(config_path))
    cases = tuple(
        case
        for case in corpus.cases(split="validation")
        if case.difficulty in ("easy", "medium")
    )[: int(maximum_targets)]
    if not cases or any(case.split != "validation" for case in cases):
        raise ValueError("protocol sensitivity requires a validation-only subset")
    experiment = protocol.payload["experiment"]
    raw_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for case in cases:
        maximum = int(
            horizon_override
            if horizon_override is not None
            else protocol.budget.maximum_horizon_by_difficulty[case.difficulty]
        )
        thresholds = tuple(
            value
            for value in protocol.budget.thresholds_by_difficulty[case.difficulty]
            if value <= maximum
        ) or (maximum,)
        fixed = evaluate_article_v1_run(
            case,
            scheduler="article_sarsa",
            expansion_budget=maximum,
            evaluation_seed=int(experiment["validation_seeds"][0]),
            checkpoint=checkpoint,
            beta=float(experiment["beta"]),
            certification_tolerance=float(experiment["certification_tolerance"]),
            config_digest=protocol.digest,
            budget_mode="fixed-max-horizon-anytime-v1",
            budget_thresholds=thresholds,
            process_cpu_limit_seconds=protocol.runtime.hard_episode_cpu_limit_seconds,
        )
        fixed["validation_protocol_mode"] = "fixed-max-horizon-anytime-v1"
        raw_rows.append(fixed)
        fixed_rows = {
            int(row["threshold"]): row
            for row in derive_fixed_horizon_anytime_rows(fixed, thresholds)
        }
        for threshold in thresholds:
            independent = evaluate_article_v1_run(
                case,
                scheduler="article_sarsa",
                expansion_budget=int(threshold),
                evaluation_seed=int(experiment["validation_seeds"][0]),
                checkpoint=checkpoint,
                beta=float(experiment["beta"]),
                certification_tolerance=float(experiment["certification_tolerance"]),
                config_digest=protocol.digest,
                budget_mode="fixed-max-horizon-anytime-v1",
                budget_thresholds=(int(threshold),),
                process_cpu_limit_seconds=protocol.runtime.hard_episode_cpu_limit_seconds,
            )
            independent["validation_protocol_mode"] = "independent-horizons-v1"
            raw_rows.append(independent)
            fixed_success = bool(fixed_rows[int(threshold)]["success_by_threshold"])
            independent_success = bool(independent["certified"])
            comparisons.append(
                {
                    "target_id": case.target_id,
                    "threshold": int(threshold),
                    "fixed_max_horizon_success": fixed_success,
                    "independent_horizon_success": independent_success,
                    "success_difference": int(fixed_success) - int(independent_success),
                }
            )
    audit = audit_ten_minute_runs(raw_rows)
    result = {
        "schema_version": TEN_MINUTE_SENSITIVITY_SCHEMA,
        "no_test_access": True,
        "validation_only": True,
        "target_ids": [case.target_id for case in cases],
        "fixed_physical_run_count": len(cases),
        "independent_physical_run_count": len(comparisons),
        "comparisons": comparisons,
        "audit": audit,
        "passed": audit["passed"],
    }
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "raw_runs.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_rows),
        encoding="utf-8",
    )
    _atomic_json(destination / "validation.json", result)
    return result


def _write_ten_minute_svg(
    path: Path,
    *,
    title: str,
    points: list[tuple[str, float]],
    y_label: str,
) -> None:
    """Write a small dependency-free SVG suitable for publication QA."""

    width, height = 760, 420
    left, top, plot_width, plot_height = 80, 55, 630, 290
    maximum = max((value for _, value in points), default=1.0)
    maximum = max(maximum, 1e-12)
    escaped = lambda value: str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{escaped(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827"/>',
        f'<text x="18" y="{top + plot_height / 2}" transform="rotate(-90 18 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="12">{escaped(y_label)}</text>',
    ]
    if not points:
        elements.append(f'<text x="{left + plot_width / 2}" y="{top + plot_height / 2}" text-anchor="middle" font-family="sans-serif" fill="#6b7280">No applicable rows supplied</text>')
    else:
        slot = plot_width / len(points)
        for index, (label, value) in enumerate(points):
            bar_height = plot_height * max(0.0, float(value)) / maximum
            x = left + index * slot + slot * 0.18
            y = top + plot_height - bar_height
            elements.extend(
                [
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{slot * 0.64:.2f}" height="{bar_height:.2f}" fill="#2563eb"/>',
                    f'<text x="{x + slot * 0.32:.2f}" y="{y - 5:.2f}" text-anchor="middle" font-family="sans-serif" font-size="10">{value:.3g}</text>',
                    f'<text x="{x + slot * 0.32:.2f}" y="{top + plot_height + 18}" text-anchor="middle" font-family="sans-serif" font-size="9">{escaped(label[:18])}</text>',
                ]
            )
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def report_ten_minute(
    raw_runs_path: str | Path,
    output_directory: str | Path,
    *,
    training_summaries: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    """Aggregate audited V3 runs without re-executing scientific searches."""

    rows = [
        json.loads(line)
        for line in Path(raw_runs_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    audit = audit_ten_minute_runs(rows)
    if not audit["passed"]:
        raise ValueError("cannot report an incomplete or timed-out ten-minute campaign")
    derived: list[dict[str, Any]] = []
    for row in rows:
        if row.get("budget_mode") == "fixed-max-horizon-anytime-v1":
            derived.extend(
                {
                    **item,
                    "evaluation_family": row.get("evaluation_family", "unspecified"),
                }
                for item in derive_fixed_horizon_anytime_rows(
                    row, tuple(int(value) for value in row["budget_thresholds"])
                )
            )
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in derived:
        key = (
            str(row["evaluation_family"]),
            str(row["scheduler"]),
            int(row["threshold"]),
        )
        groups.setdefault(key, []).append(row)
    aggregate_rows = []
    for (family, scheduler, threshold), values in sorted(groups.items()):
        successes = [bool(value["success_by_threshold"]) for value in values]
        aggregate_rows.append(
            {
                "evaluation_family": family,
                "scheduler": scheduler,
                "threshold": threshold,
                "observation_count": len(values),
                "success_rate": sum(successes) / len(successes),
                "mean_process_cpu_seconds": float(np.mean([value["process_cpu_seconds"] for value in values])),
                "mean_wall_time_seconds": float(np.mean([value["wall_time_seconds"] for value in values])),
            }
        )
    destination = Path(output_directory)
    tables = destination / "tables"
    figures = destination / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    with (tables / "fixed_horizon_success.csv").open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(aggregate_rows[0]) if aggregate_rows else ["evaluation_family", "scheduler", "threshold", "observation_count", "success_rate", "mean_process_cpu_seconds", "mean_wall_time_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    _atomic_json(tables / "fixed_horizon_success.json", aggregate_rows)

    def means(field: str, *, family: str | None = None, schedulers: tuple[str, ...] | None = None) -> list[tuple[str, float]]:
        selected = [
            row for row in aggregate_rows
            if (family is None or row["evaluation_family"] == family)
            and (schedulers is None or row["scheduler"] in schedulers)
        ]
        by_scheduler: dict[str, list[float]] = {}
        for row in selected:
            by_scheduler.setdefault(str(row["scheduler"]), []).append(float(row[field]))
        return [(name, float(np.mean(values))) for name, values in sorted(by_scheduler.items())]

    success_points = [
        (f'{row["scheduler"]}@{row["threshold"]}', float(row["success_rate"]))
        for row in aggregate_rows
    ]
    _write_ten_minute_svg(figures / "fixed_horizon_success_curves.svg", title="Fixed-horizon budget-success", points=success_points, y_label="Success rate")
    _write_ten_minute_svg(figures / "conditional_expansions.svg", title="Conditional expansions", points=[(label, float(next((r["threshold"] for r in aggregate_rows if f'{r["scheduler"]}@{r["threshold"]}' == label), 0))) for label, _ in success_points], y_label="Expansions")
    _write_ten_minute_svg(figures / "process_cpu_time.svg", title="Process CPU time", points=means("mean_process_cpu_seconds"), y_label="Seconds")
    _write_ten_minute_svg(figures / "wall_time.svg", title="Wall time", points=means("mean_wall_time_seconds"), y_label="Seconds")
    _write_ten_minute_svg(figures / "sarsa_vs_target_distance.svg", title="SARSA vs target distance", points=means("success_rate", schedulers=("article_sarsa", "article_target_distance")), y_label="Success rate")
    _write_ten_minute_svg(figures / "hard_generalization_success.svg", title="Hard generalization", points=means("success_rate", family="hard_generalization"), y_label="Success rate")
    _write_ten_minute_svg(figures / "length_ood_success.svg", title="Length OOD", points=means("success_rate", family="length_ood"), y_label="Success rate")
    cpu_rows = [row for row in rows if row.get("budget_mode") == "equal-cpu-budget-secondary-v1"]
    _write_ten_minute_svg(figures / "cpu_budget_success.svg", title="Equal-CPU secondary evidence", points=[(str(row["scheduler"]), float(bool(row["certified"]))) for row in cpu_rows], y_label="Success")
    training_points = []
    for source in training_summaries:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        training_points.append((str(payload["training_seed"]), float(payload["total_training_expansions_completed"])))
    _write_ten_minute_svg(figures / "training_interaction_curve.svg", title="Training interactions", points=training_points, y_label="Expansions")
    result = {
        "schema_version": TEN_MINUTE_REPORT_SCHEMA,
        "passed": True,
        "raw_run_count": len(rows),
        "derived_threshold_row_count": len(derived),
        "evaluation_families": sorted({str(row.get("evaluation_family", "unspecified")) for row in rows}),
        "family_aggregation_separated": True,
        "audit": audit,
        "table": str(tables / "fixed_horizon_success.csv"),
        "figure_count": 9,
    }
    _atomic_json(destination / "report_summary.json", result)
    (destination / "completion_summary.md").write_text(
        "# Article V1 ten-minute report\n\n"
        f"Audited {len(rows)} physical runs and derived {len(derived)} threshold rows.\n\n"
        "Validation, hard-generalization, and length-OOD families remain separately labelled.\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "TEN_MINUTE_STATUS_SCHEMA",
    "TEN_MINUTE_TRAINING_RUN_SCHEMA",
    "TEN_MINUTE_EVALUATION_SCHEMA",
    "TEN_MINUTE_CALIBRATION_SCHEMA",
    "TEN_MINUTE_PLAN_SCHEMA",
    "TEN_MINUTE_CPU_EVALUATION_SCHEMA",
    "TEN_MINUTE_SENSITIVITY_SCHEMA",
    "TEN_MINUTE_REPORT_SCHEMA",
    "TEN_MINUTE_TRAINING_SELECTION_SCHEMA",
    "TenMinuteOperabilityTimeout",
    "train_ten_minute",
    "evaluate_ten_minute",
    "evaluate_cpu_budget",
    "select_feasible_hard_cap",
    "calibrate_ten_minute_horizon",
    "plan_ten_minute_campaign",
    "select_training_interaction_budget",
    "validate_protocol_sensitivity",
    "report_ten_minute",
]
