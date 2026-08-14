"""Focused checks for the opt-in potential-shaped reward model."""

from __future__ import annotations

import pytest

from certification.simulator import (
    SimulatorCertificationEngine,
    SynthesisTarget,
    unitary_from_gates,
)
from circuit.gate import Gate
from ckt_types import ResourceBudget
from config import TargetProgressRewardConfig
from config import Config
from enums import GateType
from env.rl_env import CircuitSynthesisEnv
from rl.reward import TargetProgressRewardModel


def test_target_progress_reward_uses_only_terminal_step_dead_end_and_potential_terms():
    """The shaped formula is explicit and has no archive-pruning bonus."""

    model = TargetProgressRewardModel(
        TargetProgressRewardConfig(
            terminal_bonus=10.0,
            step_cost=0.1,
            potential_scale=4.0,
            dead_end_cost=1.0,
            reward_clip=None,
        )
    )
    result = model.reward(
        potential_before=0.20,
        potential_after=0.65,
        selected_node_potential=0.20,
        best_generated_child_potential=0.95,
        certified=False,
        dead_end=False,
    )

    assert result.potential_delta == pytest.approx(0.45)
    assert result.raw_reward == pytest.approx(-0.1 + 4.0 * 0.45)
    assert result.reward == pytest.approx(result.raw_reward)
    assert result.terminal_bonus == 0.0
    assert result.dead_end_cost == 0.0
    # ``best_generated_child_potential`` is diagnostic only.  It is not an
    # implicit pruning/child-count reward term.
    assert result.best_generated_child_potential == pytest.approx(0.95)


def test_target_progress_terminal_child_is_rewarded_and_clipped_with_diagnostics():
    model = TargetProgressRewardModel(
        TargetProgressRewardConfig(
            terminal_bonus=10.0,
            step_cost=0.1,
            potential_scale=4.0,
            dead_end_cost=1.0,
            reward_clip=10.0,
        )
    )
    result = model.reward(
        potential_before=0.50,
        potential_after=1.0,
        selected_node_potential=0.50,
        best_generated_child_potential=1.0,
        certified=True,
        dead_end=False,
    )

    assert result.raw_reward == pytest.approx(11.9)
    assert result.clipped_reward == pytest.approx(10.0)
    assert result.reward == pytest.approx(10.0)
    assert result.info() == {
        "potential_before": pytest.approx(0.50),
        "potential_after": pytest.approx(1.0),
        "potential_delta": pytest.approx(0.50),
        "selected_node_potential": pytest.approx(0.50),
        "best_generated_child_potential": pytest.approx(1.0),
        "terminal_bonus": pytest.approx(10.0),
        "step_cost": pytest.approx(0.1),
        "dead_end_cost": pytest.approx(0.0),
        "raw_reward": pytest.approx(11.9),
        "clipped_reward": pytest.approx(10.0),
    }


def test_target_progress_dead_end_cost_is_applied_once():
    model = TargetProgressRewardModel(
        TargetProgressRewardConfig(reward_clip=None)
    )
    result = model.reward(
        potential_before=0.25,
        potential_after=0.0,
        selected_node_potential=0.25,
        best_generated_child_potential=0.0,
        certified=False,
        dead_end=True,
    )

    assert result.raw_reward == pytest.approx(-0.1 - 4.0 * 0.25 - 1.0)
    assert result.dead_end_cost == pytest.approx(1.0)


def test_target_progress_environment_includes_terminal_child_in_post_potential():
    """A certified H child is not enqueued but must still shape the transition."""

    target = SynthesisTarget(unitary_from_gates(1, (Gate(GateType.H, (0,)),)))
    config = Config(
        num_qubits=1,
        budget=ResourceBudget(max_t_count=0, max_depth=1, max_gates=1),
        max_steps=2,
        target_aware_features=True,
        reward_mode="target_progress",
    )
    env = CircuitSynthesisEnv(config, SimulatorCertificationEngine(target))
    _, _ = env.reset(seed=4)
    root = env.current_nodes()[0]
    expected_before = env.target_context.potential(root.state)

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated
    assert not truncated
    assert env.solution_node is not None
    assert info["num_certified"] == 1
    assert info["potential_before"] == pytest.approx(expected_before)
    assert info["potential_after"] == pytest.approx(1.0)
    assert info["best_generated_child_potential"] == pytest.approx(1.0)
    assert info["terminal_bonus"] == pytest.approx(10.0)
    assert info["step_cost"] == pytest.approx(0.1)
    assert info["dead_end_cost"] == pytest.approx(0.0)
    assert info["raw_reward"] > 10.0
    assert info["clipped_reward"] == pytest.approx(10.0)
    assert reward == pytest.approx(info["clipped_reward"])


def test_legacy_reward_and_features_remain_opt_in_defaults():
    """Adding target shaping must not change a legacy environment's behavior."""

    target = SynthesisTarget(unitary_from_gates(1, (Gate(GateType.H, (0,)),)))
    env = CircuitSynthesisEnv(
        Config(
            num_qubits=1,
            budget=ResourceBudget(max_t_count=1, max_depth=1, max_gates=1),
            max_steps=2,
        ),
        SimulatorCertificationEngine(target),
    )
    observation, _ = env.reset(seed=4)
    _, reward, terminated, _, info = env.step(0)

    assert env.target_context is None
    assert observation.shape == (16,)
    assert terminated
    # One terminal H child and four accepted nonterminal children yield the
    # unchanged legacy resource term: 10 + 0.1 * (-(2 + 2 + 3 + 3)).
    assert reward == pytest.approx(9.0)
    assert "potential_before" not in info


def test_target_progress_environment_does_not_add_an_archive_pruning_bonus():
    """A dominated identity child remains logged but cannot alter shaping."""

    # X is intentionally outside the ordinary action grammar, which gives us
    # a nonterminal two-step expansion.  Expanding H after the root generates
    # H;H = I, already dominated by the root archive record.
    target = SynthesisTarget(unitary_from_gates(1, (Gate(GateType.X, (0,)),)))
    env = CircuitSynthesisEnv(
        Config(
            num_qubits=1,
            budget=ResourceBudget(max_t_count=0, max_depth=2, max_gates=2),
            max_steps=4,
            target_aware_features=True,
            reward_mode="target_progress",
        ),
        SimulatorCertificationEngine(target),
    )
    env.reset(seed=1)
    env.step(0)  # Expand root and expose H, S, and SDG records.
    h_index = next(
        index
        for index, node in enumerate(env.current_nodes())
        if node.action is not None and node.action.gate_type is GateType.H
    )

    _, reward, terminated, truncated, info = env.step(h_index)

    assert not terminated
    assert not truncated
    assert info["num_pruned"] == 1
    assert info["terminal_bonus"] == pytest.approx(0.0)
    assert info["dead_end_cost"] == pytest.approx(0.0)
    assert info["raw_reward"] == pytest.approx(
        -info["step_cost"] + 4.0 * info["potential_delta"]
    )
    assert reward == pytest.approx(info["raw_reward"])
