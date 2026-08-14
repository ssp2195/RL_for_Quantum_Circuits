"""Train and evaluate a target-aware linear frontier policy on GHZ-3.

This is intentionally separate from :mod:`ghz3_smoke`: the smoke runner is a
deterministic FIFO reference, whereas this module trains a linear SARSA policy
whose only action is choosing a persistent *frontier record*.  Once selected,
the existing search engine still expands that record through every legal
native one-gate continuation.

The runner makes no fallback substitution.  A circuit diagram is emitted only
when a fresh, frozen learned-policy evaluation has a certified ``solution_node``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from certification.base import CertStatus
from certification.simulator import (
    SimulatorCertificationEngine,
    SynthesisTarget,
    state_fidelity,
    unitary_from_gates,
)
from circuit.gate import Gate
from config import Config
from env.rl_env import CircuitSynthesisEnv
from enums import GateType
from ghz3_smoke import (
    GHZ3_BUDGET,
    GHZ3_GATES,
    expected_ghz3_state,
    validate_ghz3_state_preparation,
)
from reporting import save_ghz3_rl_artifacts
from rl.policy import LinearQPolicy
from train import Trainer


DEFAULT_SEED = 19
DEFAULT_EPISODES = 50
# A solution needs exactly three selected frontier records (root, H(0), and a
# correct Bell prefix).  Four steps gives direct SARSA one spare selection for
# exploration while keeping the dense small-instance training run practical.
DEFAULT_TRAINING_MAX_STEPS = 4
# Evaluation has a conventional generous cap.  The zero-weight baseline thus
# has the same target, resource budget, and evaluation horizon as the learned
# policy; it is not artificially made to fail by the compact training horizon.
DEFAULT_EVALUATION_MAX_STEPS = 32
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_EPSILON_START = 0.25
DEFAULT_EPSILON_MINIMUM = 0.03
DEFAULT_EPSILON_DECAY = 0.97


def _target() -> SynthesisTarget:
    """Construct the exact, qubit-labelled canonical GHZ synthesis target."""

    return SynthesisTarget(unitary_from_gates(3, GHZ3_GATES))


def _make_environment(*, seed: int, max_steps: int) -> CircuitSynthesisEnv:
    """Return a fresh direct-GHZ target-aware environment.

    A new environment is used for every baseline, training, and frozen
    evaluation.  In particular, learned evaluation cannot reuse a completed
    training frontier or its solution node.
    """

    config = Config(
        num_qubits=3,
        budget=GHZ3_BUDGET,
        max_steps=max_steps,
        max_frontier=256,
        discount=1.0,
        fairness_interval=0,
        seed=seed,
        target_aware_features=True,
        reward_mode="target_progress",
    )
    return CircuitSynthesisEnv(
        config,
        SimulatorCertificationEngine(_target()),
    )


def _new_policy(
    environment: CircuitSynthesisEnv,
    *,
    seed: int,
    learning_rate: float,
) -> LinearQPolicy:
    """Create a zero-weight policy bound to this environment's target context."""

    if environment.target_context is None:  # pragma: no cover - config contract
        raise RuntimeError("the learned GHZ environment must expose a target context")
    return LinearQPolicy(
        feature_dim=environment.feature_dim,
        lr=learning_rate,
        gamma=environment.config.discount,
        seed=seed,
        target_context=environment.target_context,
    )


def _operation(gate: Gate) -> dict[str, Any]:
    return {
        "gate": gate.gate_type.name,
        "qubits": [int(qubit) for qubit in gate.qubits],
    }


def _as_gate(action: object) -> Gate:
    """Convert a reconstructed action to the public gate witness type."""

    try:
        gate_type = getattr(action, "gate_type")
        qubits = tuple(getattr(action, "qubits"))
    except AttributeError as exc:  # pragma: no cover - search-node contract
        raise TypeError(f"reconstructed action is not a circuit gate: {action!r}") from exc
    return Gate(gate_type, qubits)


def _display_operations(gates: Sequence[Gate]) -> list[str]:
    return [repr(gate) for gate in gates]


def _witness_structure_matches_ghz3(gates: Sequence[Gate]) -> bool:
    """Validate the allowed commuting-CNOT GHZ witness structure.

    This validates a certified output; it is not consulted by policy scoring,
    deterministic child generation, or certification, so it cannot act as a
    gate-generation shortcut.
    """

    if len(gates) != 3:
        return False
    h_gates = [gate for gate in gates if gate.gate_type is GateType.H]
    cnot_pairs = {
        gate.qubits for gate in gates if gate.gate_type is GateType.CNOT
    }
    return bool(
        gates[0] == Gate(GateType.H, (0,))
        and h_gates == [Gate(GateType.H, (0,))]
        and cnot_pairs == {(0, 1), (0, 2)}
        and sum(gate.gate_type is GateType.CNOT for gate in gates) == 2
        and all(gate.gate_type not in {GateType.T, GateType.TDG} for gate in gates)
    )


def _rollout_frozen_policy(
    environment: CircuitSynthesisEnv,
    policy: LinearQPolicy,
    *,
    seed: int,
) -> tuple[dict[str, Any], tuple[Gate, ...]]:
    """Evaluate a frozen policy on a fresh frontier without any exploration."""

    if environment.config.fairness_interval != 0:
        raise ValueError("learned GHZ evaluation requires fairness_interval=0")
    policy.bind_target_context(environment.target_context)
    environment.policy = policy
    _, reset_info = environment.reset(seed=seed)
    terminated = bool(reset_info.get("initial_certified", False))
    truncated = False
    trace: list[dict[str, Any]] = []

    while not (terminated or truncated):
        nodes_before = environment.current_nodes()
        selected = policy.select_node(nodes_before, epsilon=0.0)
        if selected is None:
            break
        selected_index = next(
            index for index, candidate in enumerate(nodes_before) if candidate is selected
        )
        selected_value = policy.node_value(selected, nodes_before)
        _, reward, terminated, truncated, info = environment.step(selected_index)
        selected_record_id = info.get("selected_record_id")
        actual = next(
            (
                node
                for node in nodes_before
                if node.record_id == selected_record_id
            ),
            selected,
        )
        prefix = tuple(_as_gate(action) for action in actual.reconstruct_actions())
        trace.append(
            {
                "expansion": int(environment.steps),
                "selected_record_id": selected_record_id,
                "selected_by_fairness": bool(info.get("selected_by_fairness", False)),
                "selected_q_value": float(selected_value),
                "selected_prefix": [_operation(gate) for gate in prefix],
                "frontier_size": int(info.get("frontier_size", 0)),
                "num_children": int(info.get("num_children", 0)),
                "num_accepted": int(info.get("num_accepted", 0)),
                "num_pruned": int(info.get("num_pruned", 0)),
                "reward": float(reward),
                "potential_before": float(info.get("potential_before", 0.0)),
                "potential_after": float(info.get("potential_after", 0.0)),
                "potential_delta": float(info.get("potential_delta", 0.0)),
                "selected_node_potential": float(
                    info.get("selected_node_potential", 0.0)
                ),
                "best_generated_child_potential": float(
                    info.get("best_generated_child_potential", 0.0)
                ),
            }
        )

    solution = environment.solution_node
    gates = (
        tuple(_as_gate(action) for action in solution.reconstruct_actions())
        if solution is not None
        else ()
    )
    exact_certified = bool(
        solution is not None
        and environment.cert_engine.certify(solution.state).status is CertStatus.SUCCESS
    )
    return (
        {
            "certified": exact_certified,
            "search_solution_present": solution is not None,
            "truncated": bool(truncated),
            "expansions": int(environment.steps),
            "fairness_override_observed": any(
                row["selected_by_fairness"] for row in trace
            ),
            "trace": trace,
            "witness": _display_operations(gates),
            "witness_operations": [_operation(gate) for gate in gates],
            "exact_unitary_certification": exact_certified,
        },
        gates,
    )


def run_ghz3_rl(
    output_dir: str | Path,
    *,
    episodes: int = DEFAULT_EPISODES,
    seed: int = DEFAULT_SEED,
    training_max_steps: int = DEFAULT_TRAINING_MAX_STEPS,
    evaluation_max_steps: int = DEFAULT_EVALUATION_MAX_STEPS,
    max_steps: int | None = None,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    epsilon_start: float = DEFAULT_EPSILON_START,
    epsilon_minimum: float = DEFAULT_EPSILON_MINIMUM,
    epsilon_decay: float = DEFAULT_EPSILON_DECAY,
) -> dict[str, Any]:
    """Train direct GHZ-3 SARSA from zeros, then run frozen learned search.

    The return report is JSON-ready.  On learned-search failure it contains no
    reference witness and the artifacts render an explicit failed-circuit
    diagram; callers can use ``report['correct']`` as a process-success flag.
    """

    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    # ``max_steps`` is retained as a convenient uniform override for tiny
    # failure-path tests and callers of the initial runner draft.  Normal
    # direct training uses the short training horizon above and evaluates both
    # zero and learned policies under the same longer horizon.
    if max_steps is not None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        training_max_steps = max_steps
        evaluation_max_steps = max_steps
    for name, value in (
        ("training_max_steps", training_max_steps),
        ("evaluation_max_steps", evaluation_max_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not 0.0 <= epsilon_minimum <= epsilon_start <= 1.0:
        raise ValueError("epsilon values must satisfy 0 <= minimum <= start <= 1")
    if not 0.0 < epsilon_decay <= 1.0:
        raise ValueError("epsilon_decay must be in (0, 1]")

    # The baseline is intentionally the same target-aware *zero-weight*
    # frontier scheduler.  It receives no learned weights and no exploration.
    baseline_environment = _make_environment(
        seed=seed + 1,
        max_steps=evaluation_max_steps,
    )
    zero_policy = _new_policy(
        baseline_environment,
        seed=seed + 1,
        learning_rate=learning_rate,
    )
    zero_baseline, _ = _rollout_frozen_policy(
        baseline_environment,
        zero_policy,
        seed=seed + 1,
    )

    training_environment = _make_environment(
        seed=seed,
        max_steps=training_max_steps,
    )
    learned_policy = _new_policy(
        training_environment,
        seed=seed,
        learning_rate=learning_rate,
    )
    trainer = Trainer(training_environment, policy=learned_policy)
    trainer.epsilon = float(epsilon_start)
    trainer.min_epsilon = float(epsilon_minimum)
    trainer.epsilon_decay = float(epsilon_decay)
    history = trainer.train(episodes)

    # Evaluation uses a new environment and no exploration/fairness.  Rebind
    # to that environment's immutable equivalent context so every evaluation
    # feature is constructed from its own per-search dense-metric cache.
    evaluation_environment = _make_environment(
        seed=seed + 2,
        max_steps=evaluation_max_steps,
    )
    evaluation, witness_gates = _rollout_frozen_policy(
        evaluation_environment,
        learned_policy,
        seed=seed + 2,
    )

    statevector: np.ndarray
    state_preparation: dict[str, Any]
    if evaluation["certified"]:
        # The witness comes exclusively from ``solution_node`` above.  This
        # independent state check is intentionally post-search validation.
        state = evaluation_environment.solution_node.state
        statevector, state_preparation = validate_ghz3_state_preparation(state)
    else:
        statevector = np.zeros(1 << 3, dtype=np.complex128)
        state_preparation = {
            "passed": False,
            "reason": "frozen learned search did not return a certified solution_node",
            "resources": {},
        }

    expected_resources = {
        "num_gates": 3,
        "two_qubit_count": 2,
        "t_count": 0,
        "depth": 3,
        "ancilla_count": 0,
    }
    # CNOT(0, 1) and CNOT(0, 2) commute, but their sequential DAG insertion
    # order changes the per-wire depth tuple.  Both exact witnesses have the
    # same gate/CNOT/T/depth optimum and must be accepted.
    accepted_wire_depths = ([3, 2, 3], [3, 3, 2])
    structure_match = _witness_structure_matches_ghz3(witness_gates)
    actual_resources = state_preparation.get("resources", {})
    resource_match = bool(
        all(actual_resources.get(name) == value for name, value in expected_resources.items())
        and list(actual_resources.get("wire_depths", ())) in accepted_wire_depths
    )
    expected_positive_ghz = expected_ghz3_state()
    negative_relative_phase_ghz = expected_positive_ghz.copy()
    negative_relative_phase_ghz[-1] *= -1.0
    negative_relative_phase_fidelity = state_fidelity(
        expected_positive_ghz,
        negative_relative_phase_ghz,
    )
    negative_relative_phase_rejected = bool(negative_relative_phase_fidelity < 1e-12)
    learned_improves_baseline = bool(
        evaluation["certified"]
        and evaluation["expansions"] < zero_baseline["expansions"]
    )
    correct = bool(
        evaluation["certified"]
        and not evaluation["truncated"]
        and not evaluation["fairness_override_observed"]
        and state_preparation.get("passed", False)
        and state_preparation.get("symbolic_agrees_with_dense_witness", False)
        and structure_match
        and resource_match
        and learned_improves_baseline
        and negative_relative_phase_rejected
    )

    policy_metadata = learned_policy.metadata()
    policy_metadata["weights"] = [float(value) for value in learned_policy.theta]
    report: dict[str, Any] = {
        "correct": correct,
        "scope": (
            "Direct GHZ-3 target-aware linear SARSA frontier-record benchmark; "
            "not evidence of general circuit-synthesis performance."
        ),
        "learning": {
            "algorithm": "linear semi-gradient SARSA(0)",
            "episodes": episodes,
            "seed": seed,
            "learning_rate": float(learning_rate),
            "discount": 1.0,
            "epsilon_start": float(epsilon_start),
            "epsilon_minimum": float(epsilon_minimum),
            "epsilon_decay": float(epsilon_decay),
            "training_max_steps": training_max_steps,
            "evaluation_max_steps": evaluation_max_steps,
            "uniform_max_steps_override": max_steps,
            "fairness_interval": 0,
            "curriculum_used": False,
            "target_fingerprint": learned_policy.target_fingerprint,
            "feature_schema_version": learned_policy.feature_schema_version,
            "target_context_schema_version": learned_policy.target_context_schema_version,
            "reward": asdict(training_environment.config.target_progress_reward),
        },
        "policy": policy_metadata,
        "training_history": history,
        "zero_policy": zero_baseline,
        "zero_policy_expansions": zero_baseline["expansions"],
        "evaluation": evaluation,
        "state_preparation": state_preparation,
        "validation": {
            "witness_structure_matches_canonical_ghz3": structure_match,
            "resources_match_optimal_native_contract": resource_match,
            "expected_resources": expected_resources,
            "accepted_wire_depths_for_commuting_cnot_order": accepted_wire_depths,
            "learned_improves_zero_policy": learned_improves_baseline,
            "negative_relative_phase_fidelity": negative_relative_phase_fidelity,
            "negative_relative_phase_rejected": negative_relative_phase_rejected,
        },
    }
    report["artifacts"] = save_ghz3_rl_artifacts(
        output_dir,
        report=report,
        gates=witness_gates,
        statevector=statevector,
    )
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("outputs") / "ghz3-rl",
        help="directory for deterministic learned-GHZ JSON, CSV, and SVG artifacts",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--training-max-steps",
        type=int,
        default=DEFAULT_TRAINING_MAX_STEPS,
        help="per-episode direct-training horizon (default: 4)",
    )
    parser.add_argument(
        "--evaluation-max-steps",
        type=int,
        default=DEFAULT_EVALUATION_MAX_STEPS,
        help="common zero/learned frozen-evaluation horizon (default: 32)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override both horizons, useful for a deliberate failure-path check",
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    args = parser.parse_args(argv)
    report = run_ghz3_rl(
        args.artifacts_dir,
        episodes=args.episodes,
        seed=args.seed,
        training_max_steps=args.training_max_steps,
        evaluation_max_steps=args.evaluation_max_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["correct"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
