"""Deterministic evaluator for small dense-certified synthesis instances.

Example:

``python evaluate.py --qubits 2 --target H:0,T:1,CNOT:0-1 --max-steps 100``
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Iterable, Sequence
import warnings

import numpy as np

from certification.simulator import (
    SimulatorCertificationEngine,
    SynthesisTarget,
    unitary_from_gates,
)
from circuit.gate import Gate
from ckt_types import ResourceBudget
from config import Config, TargetProgressRewardConfig
from enums import GateType
from env.rl_env import CircuitSynthesisEnv
from rl.policy import LinearQPolicy


def select_article_target_distance(nodes: Sequence[object], target_metric) -> object | None:
    """Select minimum Article Eq. (86) distance with stable-ID tie-breaking."""

    candidates = tuple(nodes)
    if not candidates:
        return None
    distance = getattr(target_metric, "distance", None)
    if not callable(distance):
        raise TypeError("article target-distance scheduler requires distance(state)")
    return min(
        candidates,
        key=lambda candidate: (
            float(distance(candidate.state)),
            int(candidate.record_id or 0),
        ),
    )


def parse_target(specification: str, num_qubits: int) -> list[Gate]:
    """Parse a compact ``GATE:q[,q]`` target witness specification."""
    if not specification.strip():
        return []
    gates: list[Gate] = []
    for token in specification.split(","):
        try:
            name, operands = token.strip().upper().split(":", 1)
            gate_type = GateType[name]
            qubits = tuple(int(value) for value in operands.split("-"))
        except (KeyError, ValueError) as error:
            raise ValueError(
                f"invalid target token {token!r}; use e.g. H:0 or CNOT:0-1"
            ) from error
        expected_arity = 2 if gate_type is GateType.CNOT else 1
        if len(qubits) != expected_arity or len(set(qubits)) != len(qubits):
            raise ValueError(f"invalid operands for {gate_type.name}: {qubits!r}")
        if any(qubit < 0 or qubit >= num_qubits for qubit in qubits):
            raise ValueError(f"target operands out of range: {qubits!r}")
        gates.append(Gate(gate_type, qubits))
    return gates


def evaluate(
    *,
    num_qubits: int,
    target_gates: Sequence[Gate],
    target_unitary: np.ndarray | None = None,
    budget: ResourceBudget,
    max_steps: int = 100,
    seed: int | None = 0,
    scheduler: str = "fifo",
    collect_trace: bool = False,
    policy: LinearQPolicy | None = None,
    target_aware_features: bool = False,
    reward_mode: str = "legacy_archive_shaping",
    target_progress_reward: TargetProgressRewardConfig | None = None,
    fairness_interval: int = 0,
    feature_provider=None,
    canonicalization_enabled: bool = True,
    pareto_dominance_enabled: bool = True,
    absorb_clifford_angles: bool = True,
    canonicalization_mode: str = "enhanced",
    certification_engine=None,
    target_metric=None,
    article_v1_beta: float = 1.0,
    instrumentation_enabled: bool = True,
    observation_features: bool = True,
) -> dict:
    """Run a baseline or frozen learned frontier-record scheduler.

    ``collect_trace`` records compact per-expansion metrics for optional
    post-processing.  ``scheduler='learned'`` requires an explicitly supplied
    nonzero policy and freezes both epsilon and fairness: it cannot silently
    fall back to a fresh zero-weight ranker or a FIFO interleave.
    """
    supplied_policy = policy is not None
    dense_target = (
        unitary_from_gates(num_qubits, target_gates)
        if target_unitary is None
        else np.asarray(target_unitary, dtype=np.complex128)
    )
    target = SynthesisTarget(dense_target)
    if certification_engine is None:
        certification_engine = SimulatorCertificationEngine(target)

    scheduler_aliases = {
        "seeded_random": "random",
        "zero_weight_linear": "zero_policy",
        "article_sarsa": "learned",
        "composite_target_progress": "composite_target_progress",
    }
    requested_scheduler = scheduler
    scheduler = scheduler_aliases.get(scheduler, scheduler)
    if requested_scheduler == "target_potential":
        warnings.warn(
            "target_potential is the composite process/support/entanglement "
            "scheduler; use composite_target_progress in new reports",
            DeprecationWarning,
            stacklevel=2,
        )

    needs_article_metric = (
        scheduler == "article_target_distance"
        or reward_mode == "article_v1_expansion_potential"
    )
    if needs_article_metric and target_metric is None:
        from rl.article_features import ArticleTargetContext

        target_metric = ArticleTargetContext(target)
    config = Config(
        num_qubits=num_qubits,
        budget=budget,
        max_steps=max_steps,
        # This is only the Gym adapter cap; the core archive stays dynamic.
        max_frontier=max(1, 64),
        seed=seed,
        fairness_interval=fairness_interval,
        canonicalization_enabled=canonicalization_enabled,
        pareto_dominance_enabled=pareto_dominance_enabled,
        absorb_clifford_angles=absorb_clifford_angles,
        canonicalization_mode=canonicalization_mode,
        target_aware_features=target_aware_features,
        reward_mode=reward_mode,
        article_v1_beta=article_v1_beta,
        target_progress_reward=(
            TargetProgressRewardConfig()
            if target_progress_reward is None
            else target_progress_reward
        ),
    )
    env = CircuitSynthesisEnv(
        config,
        certification_engine,
        feature_provider=feature_provider,
        target_metric=target_metric,
        instrumentation_enabled=instrumentation_enabled,
        observation_features=observation_features,
    )
    feature_context = getattr(env, "_feature_target_context", None)
    if policy is None:
        if feature_provider is None:
            policy = LinearQPolicy(
                env.feature_dim,
                seed=seed,
                target_context=feature_context,
            )
        else:
            policy = LinearQPolicy(feature_provider=feature_provider, seed=seed)
    elif feature_context is not None:
        policy.bind_target_context(feature_context)
    bind_policy = getattr(env, "_bind_policy_target_context", None)
    if callable(bind_policy):
        bind_policy(policy)

    valid_schedulers = {
        "fifo",
        "lifo",
        "uniform_cost",
        "greedy",
        "zero_policy",
        "random",
        "target_potential",
        "composite_target_progress",
        "article_target_distance",
        "learned",
    }
    if scheduler not in valid_schedulers:
        raise ValueError(
            "unsupported scheduler; choose fifo, lifo, uniform_cost, random, "
            "composite_target_progress, article_target_distance, zero_policy, "
            "greedy, or learned"
        )
    if scheduler in {"target_potential", "composite_target_progress"} and env.target_context is None:
        raise ValueError("target_potential scheduling requires a dense target context")
    if scheduler == "learned":
        if not supplied_policy:
            raise ValueError("learned scheduling requires an explicitly supplied trained policy")
        if not np.any(np.abs(policy.theta) > 0.0):
            raise ValueError("learned scheduling requires nonzero trained policy weights")
        if fairness_interval:
            raise ValueError("learned scheduling requires fairness_interval=0")
    if scheduler == "zero_policy" and np.any(np.abs(policy.theta) > 0.0):
        raise ValueError("zero_policy scheduling requires exactly zero weights")

    started_ns = time.perf_counter_ns()
    env.reset(seed=seed)
    terminated = env.solution_node is not None
    truncated = False
    trace: list[dict] = []
    external_ranking_time_ns = 0

    while not (terminated or truncated):
        nodes = env.current_nodes()
        if not nodes:
            break
        selection_started = time.perf_counter_ns()
        metric_cache = getattr(target_metric, "cache_metrics", None)
        target_time_before = (
            int(metric_cache().get("target_metric_time_ns", 0))
            if callable(metric_cache)
            else 0
        )
        if scheduler == "fifo":
            node = min(nodes, key=lambda candidate: int(candidate.record_id or 0))
        elif scheduler == "lifo":
            node = max(nodes, key=lambda candidate: int(candidate.record_id or 0))
        elif scheduler == "uniform_cost":
            node = min(
                nodes,
                key=lambda candidate: (
                    int(candidate.state.t_count),
                    int(candidate.state.two_qubit_count),
                    int(candidate.state.num_gates),
                    int(candidate.state.depth),
                    int(candidate.record_id or 0),
                ),
            )
        elif scheduler in {"greedy", "zero_policy", "learned"}:
            node = policy.select_node(nodes, epsilon=0.0)
        elif scheduler == "random":
            node = policy.select_node(nodes, epsilon=1.0)
        elif scheduler in {"target_potential", "composite_target_progress"}:
            assert env.target_context is not None
            node = max(
                nodes,
                key=lambda candidate: (
                    env.target_context.potential(candidate.state),
                    -int(candidate.record_id or 0),
                ),
            )
        else:  # article_target_distance: direct Eq. (86), no learned weights.
            assert target_metric is not None
            node = select_article_target_distance(nodes, target_metric)
        if scheduler not in {"greedy", "zero_policy", "learned", "random"}:
            elapsed = time.perf_counter_ns() - selection_started
            target_time_after = target_time_before
            if callable(metric_cache):
                target_time_after = int(
                    metric_cache().get("target_metric_time_ns", target_time_before)
                )
            external_ranking_time_ns += max(
                0, elapsed - max(0, target_time_after - target_time_before)
            )
        assert node is not None
        prefix = [repr(action) for action in node.reconstruct_actions()]
        selected_q_value = (
            float(policy.node_value(node, nodes))
            if collect_trace
            and scheduler in {"greedy", "zero_policy", "learned", "random"}
            else None
        )
        _, reward, terminated, truncated, info = env.select_record(int(node.record_id))
        if collect_trace:
            row: dict[str, Any] = {
                "expansion": env.steps,
                "selected_record_id": info.get("selected_record_id"),
                "selected_prefix": prefix,
                "selected_by_fairness": bool(info.get("selected_by_fairness", False)),
                "selected_q_value": selected_q_value,
                "frontier_size": int(info.get("frontier_size", 0)),
                "num_children": int(info.get("num_children", 0)),
                "num_accepted": int(info.get("num_accepted", 0)),
                "num_pruned": int(info.get("num_pruned", 0)),
                "reward": float(reward),
                "article_target_distance": (
                    float(target_metric.distance(node.state))
                    if target_metric is not None
                    else None
                ),
            }
            for name in (
                "potential_before",
                "potential_after",
                "potential_delta",
                "selected_node_potential",
                "best_generated_child_potential",
                "terminal_bonus",
                "step_cost",
                "dead_end_cost",
                "raw_reward",
                "clipped_reward",
            ):
                if name in info:
                    row[name] = float(info[name])
            trace.append(row)

    witness_actions = []
    if env.solution_node is not None:
        witness_actions = env.solution_node.reconstruct_actions()
    solution_state = None if env.solution_node is None else env.solution_node.state
    total_wall_time_ns = time.perf_counter_ns() - started_ns
    runtime_seconds = float(total_wall_time_ns / 1e9)
    search_metrics = dict(getattr(env, "search_metrics", {}))
    # The environment-level timer intentionally covers only executed step()
    # bodies.  Publication "wall time" must include reset, frontier ranking,
    # feature construction, and stopping logic as well.  Preserve the narrower
    # value under an explicit name and expose the end-to-end evaluation timer
    # under the protocol's wall_time_ns field.  Disabled instrumentation stays
    # zero so toggling it cannot masquerade as measured timing.
    environment_step_time_ns = int(search_metrics.get("wall_time_ns", 0))
    search_metrics["environment_step_time_ns"] = environment_step_time_ns
    search_metrics["wall_time_ns"] = (
        int(total_wall_time_ns) if instrumentation_enabled else 0
    )
    policy_metrics = policy.instrumentation()
    search_metrics["ranking_time_ns"] = int(
        policy_metrics["ranking_time_ns"] + external_ranking_time_ns
    )
    search_metrics["feature_time_ns"] = int(
        search_metrics.get("feature_time_ns", 0)
        + policy_metrics["feature_time_ns"]
    )
    search_metrics["feature_evaluation_count"] = int(
        search_metrics.get("feature_evaluation_count", 0)
        + policy_metrics["feature_evaluation_count"]
    )
    if target_metric is not None:
        cache_metrics = getattr(target_metric, "cache_metrics", None)
        if callable(cache_metrics):
            metric_values = dict(cache_metrics())
            search_metrics["target_metric_evaluation_count"] = int(
                metric_values.get(
                    "target_metric_evaluation_count",
                    metric_values.get("evaluations", metric_values.get("evaluation_count", 0)),
                )
            )
            search_metrics["target_metric_cache_hits"] = int(
                metric_values.get(
                    "target_metric_cache_hits",
                    metric_values.get("hits", metric_values.get("cache_hits", 0)),
                )
            )
            search_metrics["target_metric_cache_misses"] = int(
                metric_values.get(
                    "target_metric_cache_misses",
                    metric_values.get("misses", metric_values.get("cache_misses", 0)),
                )
            )
            search_metrics["target_metric_time_ns"] = int(
                metric_values.get(
                    "target_metric_time_ns",
                    metric_values.get("time_ns", metric_values.get("evaluation_time_ns", 0)),
                )
            )
    mean_frontier = (
        float(search_metrics.get("frontier_sum", 0))
        / max(1, int(search_metrics.get("frontier_observation_count", 0)))
    )
    search_metrics["frontier_decision_mean"] = mean_frontier
    report = {
        "certified": env.solution_node is not None,
        "terminated": terminated,
        "truncated": truncated,
        "expansions": env.steps,
        "frontier_size": len(env.current_nodes()),
        "witness": [repr(action) for action in witness_actions],
        "witness_operations": [
            {
                "gate": action.gate_type.name,
                "qubits": list(action.qubits),
            }
            for action in witness_actions
        ],
        "scheduler": requested_scheduler,
        "scheduler_semantics": (
            "composite_target_progress"
            if scheduler == "target_potential"
            else scheduler
        ),
        "action_semantics": "persistent_frontier_record",
        "fairness_interval": fairness_interval,
        "target_aware_features": target_aware_features,
        "reward_mode": config.reward_mode,
        "reward_coefficients": env.reward_spec(),
        "feature_schema": policy.metadata(),
        "policy_weight_norm": float(np.linalg.norm(policy.theta)),
        "canonicalization_enabled": canonicalization_enabled,
        "pareto_dominance_enabled": pareto_dominance_enabled,
        "absorb_clifford_angles": absorb_clifford_angles,
        "canonicalization_mode": canonicalization_mode,
        "instrumentation_enabled": instrumentation_enabled,
        "runtime_seconds": runtime_seconds,
        "time_to_solution": runtime_seconds if env.solution_node is not None else None,
        "search_metrics": search_metrics,
        "archive_size_final": int(search_metrics.get("archive_size", 0)),
        "certification_schema": str(
            getattr(certification_engine, "schema_version", "legacy-dense-phase-quotient-v1")
        ),
        "solution_resource_vector": (
            None
            if solution_state is None
            else [
                int(solution_state.t_count),
                int(solution_state.two_qubit_count),
                int(solution_state.num_gates),
                *(int(value) for value in solution_state.wire_depths),
            ]
        ),
    }
    if env.target_context is not None:
        report["target"] = {
            "fingerprint": env.target_context.fingerprint,
            "context_schema_version": env.target_context.schema_version,
            "phase_mode": env.target_context.phase_mode,
        }
    if target_metric is not None:
        report["article_target_metric"] = {
            "schema_version": str(getattr(target_metric, "schema_version", "unknown")),
            "target_fingerprint": str(getattr(target_metric, "fingerprint", "unknown")),
            "cache_metrics": (
                dict(target_metric.cache_metrics())
                if callable(getattr(target_metric, "cache_metrics", None))
                else {}
            ),
        }
    if collect_trace:
        report["trace"] = trace
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qubits", type=int, required=True)
    parser.add_argument("--target", default="", help="comma-separated GATE:operands witness")
    parser.add_argument("--max-t", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--max-gates", type=int, default=12)
    parser.add_argument("--max-two-qubit", type=int)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scheduler",
        choices=("fifo", "lifo", "uniform_cost", "zero_policy", "greedy", "random"),
        default="fifo",
    )
    args = parser.parse_args(argv)

    target_gates = parse_target(args.target, args.qubits)
    report = evaluate(
        num_qubits=args.qubits,
        target_gates=target_gates,
        budget=ResourceBudget(
            args.max_t,
            args.max_depth,
            args.max_gates,
            args.max_two_qubit,
        ),
        max_steps=args.max_steps,
        seed=args.seed,
        scheduler=args.scheduler,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["certified"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
