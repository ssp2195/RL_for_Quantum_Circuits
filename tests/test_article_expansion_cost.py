from __future__ import annotations

import numpy as np

from certification.simulator import SimulatorCertificationEngine, SynthesisTarget, unitary_from_gates
from circuit.gate import Gate
from ckt_types import ResourceBudget
from config import Config
from enums import GateType
from env.rl_env import CircuitSynthesisEnv
from train import Trainer


def _environment(target_gates, *, max_steps: int) -> CircuitSynthesisEnv:
    config = Config(
        num_qubits=1,
        budget=ResourceBudget(3, max_steps + 1, max_steps + 1),
        max_steps=max_steps,
        max_frontier=1,
        discount=1.0,
        seed=5,
        reward_mode="expansion_cost",
    )
    target = SynthesisTarget(unitary_from_gates(1, target_gates))
    return CircuitSynthesisEnv(config, SimulatorCertificationEngine(target))


def test_historical_reward_aliases_normalize_to_reportable_names():
    budget = ResourceBudget(0, 0, 0)

    assert Config(1, budget, reward_mode="legacy").reward_mode == "legacy_archive_shaping"
    assert (
        Config(1, budget, reward_mode="target_progress").reward_mode
        == "target_progress_shaping"
    )


def test_expansion_cost_success_step_has_the_article_terminal_correction():
    environment = _environment([Gate(GateType.H, (0,))], max_steps=2)
    environment.reset(seed=5)

    _, reward, terminated, truncated, info = environment.select_record(0)

    assert terminated and not truncated
    assert reward == 0.0  # -1 expansion cost +1 terminal success correction
    assert info["reward_mode"] == "expansion_cost"
    assert info["expansion_cost"] == 1.0
    assert info["terminal_success_correction"] == 1.0
    assert info["terminal_failure_correction"] == 0.0


def test_expansion_cost_failure_return_is_negative_budget_without_visit_bonus():
    # X is supported by the semantic engine but deliberately absent from the
    # native expansion grammar, so this two-expansion episode cannot certify.
    environment = _environment([Gate(GateType.X, (0,))], max_steps=2)
    trainer = Trainer(environment)

    result = trainer.train(1)[0]

    assert result["truncated"]
    assert not result["certified"]
    assert result["steps"] == 2
    assert np.isclose(result["reward"], -2.0)
    assert trainer.exploration_beta == 0.0
    assert result["reward_mode"] == "expansion_cost"
    assert result["mean_absolute_td_error"] >= 0.0
    assert result["weight_norm"] >= 0.0


def test_expansion_cost_early_frontier_exhaustion_still_returns_negative_budget():
    config = Config(
        num_qubits=1,
        budget=ResourceBudget(0, 0, 0),
        max_steps=25,
        max_frontier=1,
        discount=1.0,
        seed=7,
        reward_mode="expansion_cost",
    )
    target = SynthesisTarget(
        unitary_from_gates(1, [Gate(GateType.X, (0,))])
    )
    trainer = Trainer(
        CircuitSynthesisEnv(config, SimulatorCertificationEngine(target))
    )

    result = trainer.train(1)[0]

    assert not result["certified"]
    assert not result["truncated"]
    assert result["steps"] == 1
    assert result["reward"] == -25.0
