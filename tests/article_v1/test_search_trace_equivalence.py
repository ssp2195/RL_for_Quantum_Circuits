from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import dataclass
import hashlib
import io
import json
from typing import Mapping

import numpy as np

from certification.article_v1 import ArticleV1CertificationEngine
from config import Config
from env.rl_env import CircuitSynthesisEnv
from experiments.article_v1_feature_benchmark import (
    create_repository_feature_benchmark_adapter,
)
from experiments.article_v1_training_checkpoint import policy_weight_digest
from rl.policy import LinearQPolicy
from train import Trainer, TrainerBoundaryEvent


_STAGED_CAPS = (8, 16, 32, 64)
_EXPECTED_SELECTED_RECORD_IDS = (
    0,
    1,
    40,
    45,
    10,
    8,
    6,
    95,
    77,
    167,
    13,
    2,
    67,
    214,
    20,
    61,
    91,
    220,
    300,
    343,
    43,
    280,
    44,
    403,
    335,
    265,
    469,
    247,
    436,
    534,
    9,
    14,
    83,
    108,
    48,
    448,
    386,
    538,
    166,
    5,
    126,
    31,
    15,
    370,
    667,
    611,
    173,
    588,
    248,
    23,
    121,
    354,
    902,
    553,
    276,
    7,
    11,
    462,
    177,
    1010,
    968,
    17,
    974,
    853,
)


def _json_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise TypeError(f"trace payload contains unsupported {type(value).__name__}")


def _digest(value: object, *, domain: str) -> str:
    encoded = json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\0")
    digest.update(encoded)
    return f"sha256:{digest.hexdigest()}"


def _action_payload(action: object) -> tuple[str, tuple[int, ...]]:
    gate_type = getattr(action, "gate_type", None)
    name = getattr(gate_type, "name", str(gate_type))
    return str(name), tuple(int(qubit) for qubit in getattr(action, "qubits", ()))


def _witness_payload(node: object | None) -> tuple[tuple[str, tuple[int, ...]], ...] | None:
    if node is None:
        return None
    actions = getattr(node, "reconstruct_actions")()
    return tuple(_action_payload(action) for action in actions)


def _record_payload(environment: CircuitSynthesisEnv, record: object) -> dict[str, object]:
    node = getattr(record, "node")
    state = getattr(node, "state")
    resources = getattr(record, "resources")
    return {
        "record_id": int(getattr(record, "record_id")),
        "identity_hash": str(environment.canonicalizer.identity_hash(state)),
        "resources": [int(value) for value in resources.as_tuple()],
        "expanded": bool(getattr(record, "expanded")),
        "active": bool(getattr(record, "active")),
        "queued": bool(getattr(record, "queued")),
        "tombstoned": bool(getattr(record, "tombstoned")),
        "witness": _witness_payload(node),
    }


def _frontier_digest(environment: CircuitSynthesisEnv) -> str:
    payload = [
        _record_payload(environment, record)
        for record in environment.current_records()
    ]
    return _digest(payload, domain="article-v1-trace-frontier-v1")


def _archive_digest(environment: CircuitSynthesisEnv) -> str:
    assert environment.frontier is not None
    payload = [
        _record_payload(environment, record)
        for record in environment.frontier.archive.all_records()
    ]
    return _digest(payload, domain="article-v1-trace-archive-v1")


def _generation_count_digest(environment: CircuitSynthesisEnv) -> str:
    # Canonical keys are immutable tuples.  Their repr is the repository's
    # canonicalizer serialization, so this binds both identity and count while
    # keeping the test payload strict JSON.
    payload = sorted(
        (
            hashlib.sha256(repr(key).encode("utf-8")).hexdigest(),
            int(count),
        )
        for key, count in environment.generation_counts.items()
    )
    return _digest(payload, domain="article-v1-trace-generation-counts-v1")


def _deterministic_metrics(values: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    result: list[tuple[str, object]] = []
    for name, value in sorted(values.items()):
        if name.endswith("_time_ns"):
            continue
        if name.startswith("feature_") or name.startswith("target_metric_"):
            continue
        if name == "ranking_time_ns":
            continue
        result.append((str(name), _json_value(value)))
    return tuple(result)


def _certification_payload(environment: CircuitSynthesisEnv) -> object:
    solution = environment.solution_node
    if solution is None:
        return None
    result = environment.cert_engine.certify(solution.state)
    return {
        "status": result.status.name,
        "score": float(result.score),
        "info": _json_value(result.info or {}),
    }


@dataclass(frozen=True, slots=True)
class _UpdateTrace:
    expansion: int
    selected_record_id: int
    reward: float
    td_error: float
    actual_next_record_id: int | None
    selected_features: tuple[float, ...]
    next_features: tuple[float, ...] | None
    theta_after_update: tuple[float, ...]
    theta_digest_after_update: str
    terminated: bool
    truncated: bool
    frontier_revision: int


@dataclass(frozen=True, slots=True)
class _SearchStateTrace:
    expansion: int
    frontier_digest: str
    archive_digest: str
    generation_count_digest: str
    deterministic_search_counters: tuple[tuple[str, object], ...]
    certified: bool
    certification_result: object
    witness: tuple[tuple[str, tuple[int, ...]], ...] | None
    status_if_stopped_here: str


@dataclass(frozen=True, slots=True)
class _EpisodeTrace:
    cap: int
    evaluator_schema: str
    updates: tuple[_UpdateTrace, ...]
    staged_state: _SearchStateTrace
    final_theta: tuple[float, ...]
    final_theta_digest: str
    terminal_status: str
    certified: bool
    certification_result: object
    witness: tuple[tuple[str, tuple[int, ...]], ...] | None
    deterministic_search_counters: tuple[tuple[str, object], ...]


def _run_trace(evaluator: str, cap: int) -> _EpisodeTrace:
    adapter = create_repository_feature_benchmark_adapter(
        microbenchmark_repetitions=1,
        profile_caps=(1,),
        profile_frontier_sizes=(32,),
    )
    provider, context = adapter._provider(evaluator)  # type: ignore[attr-defined]
    experiment = adapter.experiment
    policy = LinearQPolicy(
        feature_provider=provider,
        lr=float(experiment["learning_rate"]),
        gamma=1.0,
        seed=adapter.effective_seed,
    )
    environment = CircuitSynthesisEnv(
        Config(
            num_qubits=adapter.case.num_qubits,
            budget=adapter.case.budget.resource_budget(),
            max_steps=adapter.scientific_horizon,
            max_frontier=64,
            discount=1.0,
            seed=adapter.effective_seed,
            fairness_interval=0,
            canonicalization_enabled=bool(experiment["canonicalization_enabled"]),
            pareto_dominance_enabled=bool(experiment["pareto_dominance_enabled"]),
            absorb_clifford_angles=bool(experiment["absorb_clifford_angles"]),
            canonicalization_mode=str(experiment["canonicalization_mode"]),
            reward_mode="article_v1_expansion_potential",
            article_v1_beta=float(experiment["beta"]),
        ),
        ArticleV1CertificationEngine(
            adapter.case.target,
            tau_cert=float(experiment["certification_tolerance"]),
        ),
        feature_provider=provider,
        target_metric=context,
        instrumentation_enabled=True,
        observation_features=False,
    )

    updates: list[_UpdateTrace] = []
    staged_state: _SearchStateTrace | None = None

    def capture_boundary(event: TrainerBoundaryEvent) -> None:
        nonlocal staged_state
        if event.boundary != "expansion":
            return
        assert event.selected_record_id is not None
        assert event.selected_features is not None
        assert event.reward is not None
        assert event.td_error is not None
        theta = tuple(float(value) for value in event.policy_weights_after_update)
        updates.append(
            _UpdateTrace(
                expansion=int(event.expansion),
                selected_record_id=int(event.selected_record_id),
                reward=float(event.reward),
                td_error=float(event.td_error),
                actual_next_record_id=(
                    None if event.next_record_id is None else int(event.next_record_id)
                ),
                selected_features=tuple(float(value) for value in event.selected_features),
                next_features=(
                    None
                    if event.next_features is None
                    else tuple(float(value) for value in event.next_features)
                ),
                theta_after_update=theta,
                # The policy's own digest intentionally binds evaluator schema,
                # so compare the portable raw-weight digest across evaluators.
                theta_digest_after_update=policy_weight_digest(theta),
                terminated=bool(event.terminated),
                truncated=bool(event.truncated),
                frontier_revision=int(event.frontier_revision),
            )
        )
        if event.expansion == cap:
            certified = environment.solution_node is not None
            staged_state = _SearchStateTrace(
                expansion=int(event.expansion),
                frontier_digest=_frontier_digest(environment),
                archive_digest=_archive_digest(environment),
                generation_count_digest=_generation_count_digest(environment),
                deterministic_search_counters=_deterministic_metrics(
                    event.search_metrics
                ),
                certified=certified,
                certification_result=_certification_payload(environment),
                witness=_witness_payload(environment.solution_node),
                status_if_stopped_here=(
                    "certified"
                    if certified
                    else "frontier_exhausted"
                    if bool(event.terminated)
                    else "truncated"
                ),
            )

    trainer = Trainer(environment, policy=policy, checkpoint_callback=capture_boundary)
    epsilon = experiment["epsilon"]
    trainer.epsilon = float(epsilon["start"])
    trainer.min_epsilon = float(epsilon["minimum"])
    trainer.epsilon_decay = float(epsilon["decay"])

    original_select_record = environment.select_record

    def bounded_select_record(record_id: int):
        observation, reward, terminated, truncated, info = original_select_record(
            record_id
        )
        if environment.steps >= cap and not terminated:
            truncated = True
        return observation, reward, terminated, truncated, info

    environment.select_record = bounded_select_record  # type: ignore[method-assign]
    with redirect_stdout(io.StringIO()):
        history = trainer.train(1)
    assert len(history) == 1
    episode = history[0]
    assert staged_state is not None
    assert len(updates) == cap
    certified = environment.solution_node is not None
    terminal_status = (
        "certified"
        if certified
        else "truncated"
        if bool(episode["truncated"])
        else "frontier_exhausted"
    )
    theta = tuple(float(value) for value in policy.theta)
    return _EpisodeTrace(
        cap=cap,
        evaluator_schema=str(provider.evaluator_schema_version),
        updates=tuple(updates),
        staged_state=staged_state,
        final_theta=theta,
        final_theta_digest=policy_weight_digest(theta),
        terminal_status=terminal_status,
        certified=certified,
        certification_result=_certification_payload(environment),
        witness=_witness_payload(environment.solution_node),
        deterministic_search_counters=_deterministic_metrics(
            dict(episode["search_metrics"])
        ),
    )


def _assert_exact_trace_equivalence(
    optimized: _EpisodeTrace,
    reference: _EpisodeTrace,
) -> None:
    assert optimized.cap == reference.cap
    assert optimized.evaluator_schema == "article-v1-exact-incremental-v2"
    assert reference.evaluator_schema == "article-v1-reference-all-pairs-v1"

    # These are the actual SARSA decisions and updates, not only final-state
    # checks.  Exact tuple equality catches even one-bit float drift.
    expected_ids = _EXPECTED_SELECTED_RECORD_IDS[: optimized.cap]
    optimized_ids = tuple(item.selected_record_id for item in optimized.updates)
    reference_ids = tuple(item.selected_record_id for item in reference.updates)
    assert optimized_ids == expected_ids
    assert reference_ids == expected_ids
    assert [item.reward for item in optimized.updates] == [
        item.reward for item in reference.updates
    ]
    assert [item.actual_next_record_id for item in optimized.updates] == [
        item.actual_next_record_id for item in reference.updates
    ]
    assert tuple(item.actual_next_record_id for item in optimized.updates) == (
        *expected_ids[1:],
        None,
    )
    assert [item.selected_features for item in optimized.updates] == [
        item.selected_features for item in reference.updates
    ]
    assert [item.next_features for item in optimized.updates] == [
        item.next_features for item in reference.updates
    ]
    assert [item.td_error for item in optimized.updates] == [
        item.td_error for item in reference.updates
    ]
    assert [item.theta_after_update for item in optimized.updates] == [
        item.theta_after_update for item in reference.updates
    ]
    assert [item.theta_digest_after_update for item in optimized.updates] == [
        item.theta_digest_after_update for item in reference.updates
    ]
    assert [
        (item.terminated, item.truncated, item.frontier_revision)
        for item in optimized.updates
    ] == [
        (item.terminated, item.truncated, item.frontier_revision)
        for item in reference.updates
    ]

    assert optimized.staged_state == reference.staged_state
    assert optimized.final_theta == reference.final_theta
    assert optimized.final_theta_digest == reference.final_theta_digest
    assert optimized.terminal_status == reference.terminal_status
    assert optimized.certified == reference.certified
    assert optimized.certification_result == reference.certification_result
    assert optimized.witness == reference.witness
    assert (
        optimized.deterministic_search_counters
        == reference.deterministic_search_counters
    )


def test_real_hard_target_reference_trace_is_exact_at_caps_8_16_32_64() -> None:
    # Keep the intentionally expensive reference qualification consolidated in
    # one deterministic test.  The four runs are independent because the
    # engineering truncation changes the final SARSA target at each cap.
    for cap in _STAGED_CAPS:
        _assert_exact_trace_equivalence(
            _run_trace("optimized", cap),
            _run_trace("reference", cap),
        )
