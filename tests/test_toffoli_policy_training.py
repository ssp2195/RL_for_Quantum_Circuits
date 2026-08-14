"""Reproducible SARSA mechanics for the real Toffoli parity provider.

The complete fresh learned-synthesis experiment is exercised by
``toffoli_search.py --train``.  This compact regression isolates its learning
boundary: actual selected frontier records, the 66-D provider, transition
reward, and semi-gradient SARSA updates must be deterministic before a broad
normal-form rollout is attempted.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from benchmarks.toffoli import TOFFOLI_NUM_QUBITS, toffoli_reference_unitary
from certification.simulator import SimulatorCertificationEngine, SynthesisTarget
from ckt_types import ResourceBudget
from config import Config
from env.rl_env import CircuitSynthesisEnv
from rl.policy import LinearQPolicy
from rl.toffoli_parity import (
    TOFFOLI_PARITY_FEATURE_SCHEMA_VERSION,
    ToffoliParityFeatureProvider,
    ToffoliParityRewardModel,
)
from search.problems.toffoli_parity import ToffoliParityNetworkProblem
from train import Trainer


def _train_short_seeded_run() -> tuple[list[dict[str, object]], np.ndarray, str]:
    """Run genuine SARSA updates from the identity root on the real problem."""

    problem = ToffoliParityNetworkProblem()
    provider = ToffoliParityFeatureProvider(
        problem,
        target_fingerprint="toffoli-policy-training-regression",
    )
    environment = CircuitSynthesisEnv(
        Config(
            num_qubits=TOFFOLI_NUM_QUBITS,
            budget=ResourceBudget(
                max_t_count=7,
                max_two_qubit_count=6,
                max_gates=15,
                max_depth=12,
            ),
            max_steps=3,
            max_frontier=64,
            fairness_interval=0,
            discount=1.0,
            seed=29,
        ),
        SimulatorCertificationEngine(SynthesisTarget(toffoli_reference_unitary())),
        problem=problem,
        feature_provider=provider,
        reward_model=ToffoliParityRewardModel(provider),
        observation_features=False,
    )
    policy = LinearQPolicy(feature_provider=provider, lr=0.01, gamma=1.0, seed=29)
    trainer = Trainer(environment, policy=policy)
    trainer.epsilon = 0.25
    trainer.min_epsilon = 0.05
    trainer.epsilon_decay = 0.97

    with redirect_stdout(StringIO()):
        history = trainer.train(2)
    return history, policy.theta.copy(), policy.weight_digest()


def test_toffoli_sarsa_updates_are_seeded_and_provider_bound() -> None:
    """Two short real runs have identical SARSA histories and learned weights."""

    first_history, first_weights, first_digest = _train_short_seeded_run()
    second_history, second_weights, second_digest = _train_short_seeded_run()

    assert len(first_history) == len(second_history) == 2
    assert first_history == second_history
    np.testing.assert_array_equal(first_weights, second_weights)
    assert first_digest == second_digest
    assert np.any(first_weights != 0.0)

    # The full frozen exact-Toffoli success assertion belongs to the public
    # calibrated runner; this unit regression establishes that its 66-D
    # parity weights genuinely came from reproducible SARSA transitions.
    assert TOFFOLI_PARITY_FEATURE_SCHEMA_VERSION == "frontier-toffoli-parity-v1"
