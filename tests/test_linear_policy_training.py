"""Deterministic end-to-end checks for linear SARSA record scheduling.

Training plan exercised here:

1. Use a small, target-specific curriculum instance with a fixed seed and a
   tight resource budget.  This keeps the regression fast while requiring the
   scheduler to prefer a non-first frontier record.
2. Measure a greedy rollout of the fresh zero-weight policy.
3. Train the real :class:`train.Trainer` for a few exploratory SARSA episodes.
4. Freeze exploration and require the learned policy to recover the certified
   witness in fewer frontier expansions.

The policy selects persistent frontier *records*, never gates.  The
environment continues to enumerate every legal gate for each selected record.
The current feature vector is intentionally target-agnostic, so this is a
regression for learning resource-class scheduling rather than a claim that the
linear policy has learned arbitrary target-specific qubit or gate directions.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from certification.simulator import (
    SimulatorCertificationEngine,
    SynthesisTarget,
    unitary_from_gates,
)
from circuit.gate import Gate
from ckt_types import ResourceBudget
from config import Config
from enums import GateType
from env.rl_env import CircuitSynthesisEnv
from rl.policy import LinearQPolicy
from train import Trainer


_SEED = 7
_TARGET_GATES = (
    Gate(GateType.CNOT, (0, 1)),
    Gate(GateType.H, (0,)),
)
_BUDGET = ResourceBudget(
    max_t_count=0,
    max_depth=2,
    max_gates=2,
    max_two_qubit_count=1,
)


def _make_env(*, seed: int = _SEED, max_steps: int = 24) -> CircuitSynthesisEnv:
    target = SynthesisTarget(unitary_from_gates(2, _TARGET_GATES))
    return CircuitSynthesisEnv(
        Config(
            num_qubits=2,
            budget=_BUDGET,
            max_steps=max_steps,
            max_frontier=16,
            seed=seed,
        ),
        SimulatorCertificationEngine(target),
    )


def _greedy_rollout(policy: LinearQPolicy) -> dict[str, object]:
    """Evaluate a frozen node policy without changing its weights."""

    env = _make_env(seed=99)
    env.reset(seed=99)
    terminated = env.solution_node is not None
    truncated = False
    selected = []

    while not (terminated or truncated):
        nodes = env.current_nodes()
        node = policy.select_node(nodes, epsilon=0.0)
        assert node is not None
        selected.append((node.record_id, repr(node.action) if node.action else "root"))
        # SearchNode equality intentionally compares only queue priority, so
        # adapt this concrete chosen object to the Gym index by identity.
        index = next(i for i, candidate in enumerate(nodes) if candidate is node)
        _, _, terminated, truncated, _ = env.step(index)

    assert env.solution_node is not None
    return {
        "steps": env.steps,
        "selected": selected,
        "witness": [repr(action) for action in env.solution_node.reconstruct_actions()],
    }


def _train_curriculum_once() -> tuple[LinearQPolicy, list[dict[str, object]]]:
    """Train one independently seeded curriculum run for reproducibility checks."""

    training_env = _make_env()
    policy = LinearQPolicy(
        training_env.feature_dim,
        lr=0.05,
        gamma=1.0,
        seed=_SEED,
    )
    trainer = Trainer(training_env, policy=policy)
    # Fixed, decaying exploration is part of the deterministic curriculum.
    trainer.epsilon = 0.2
    trainer.min_epsilon = 0.05
    trainer.epsilon_decay = 0.995
    with redirect_stdout(StringIO()):
        history = trainer.train(5)
    return policy, history


class _RecordingEnv(CircuitSynthesisEnv):
    """Expose selected record IDs to regress Trainer's Gym-index adapter."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.expanded_record_ids: list[int | None] = []

    def step(self, action):
        result = super().step(action)
        self.expanded_record_ids.append(result[-1].get("selected_record_id"))
        return result


class _ChooseCnotPolicy(LinearQPolicy):
    """Select a tied-priority CNOT record after the root expansion."""

    def select_node(self, nodes, epsilon=0.1):
        if len(nodes) == 1:
            return nodes[0]
        return next(
            node
            for node in nodes
            if node.action is not None
            and node.action.gate_type is GateType.CNOT
            and node.action.qubits == (0, 1)
        )


def test_trainer_expands_the_policy_selected_record_when_priorities_tie():
    """A policy-selected CNOT must not be replaced by the first equal node."""

    # X is not in the expansion grammar, so two steps cannot terminate before
    # the selected second record is observed.
    env = _RecordingEnv(
        Config(
            num_qubits=2,
            budget=_BUDGET,
            max_steps=2,
            max_frontier=16,
            seed=_SEED,
        ),
        SimulatorCertificationEngine(
            SynthesisTarget(unitary_from_gates(2, (Gate(GateType.X, (0,)),)))
        ),
    )
    policy = _ChooseCnotPolicy(env.feature_dim, seed=_SEED)
    trainer = Trainer(env, policy=policy)
    trainer.epsilon = 0.0
    trainer.min_epsilon = 0.0

    with redirect_stdout(StringIO()):
        history = trainer.train(1)

    assert history[0]["truncated"]
    # Record 1 is H(0); record 7 is CNOT(0, 1).  Both have queue priority 2,
    # so this specifically proves identity-based selection rather than
    # value-based list.index selection.
    assert env.expanded_record_ids == [0, 7]


def test_linear_sarsa_training_improves_resource_class_scheduling():
    """Training learns a two-step record schedule without gate injection."""

    fresh_env = _make_env()
    policy = LinearQPolicy(fresh_env.feature_dim, lr=0.05, gamma=1.0, seed=_SEED)
    baseline = _greedy_rollout(policy)

    policy, history = _train_curriculum_once()
    repeated_policy, repeated_history = _train_curriculum_once()

    learned = _greedy_rollout(policy)

    assert all(episode["certified"] for episode in history)
    assert history == repeated_history
    np.testing.assert_allclose(policy.theta, repeated_policy.theta)
    assert np.linalg.norm(policy.theta) > 0.0
    assert baseline["witness"] == ["CNOT(0, 1)", "H(0,)"]
    assert learned["witness"] == ["CNOT(0, 1)", "H(0,)"]
    assert baseline["steps"] == 8
    assert learned["steps"] == 2
    assert learned["steps"] < baseline["steps"]
    # The learned policy prioritizes an entangling record after root; it does
    # not emit a gate itself.  Stable record order resolves the equal-featured
    # directed CNOT alternatives, then exhaustive expansion certifies H next.
    assert learned["selected"] == [(0, "root"), (7, "CNOT(0, 1)")]
