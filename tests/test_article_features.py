from __future__ import annotations

import numpy as np

from certification.simulator import (
    SimulatorCertificationEngine,
    SynthesisTarget,
    unitary_from_gates,
)
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from config import Config
from env.rl_env import CircuitSynthesisEnv
from enums import GateType
from rl.article_features import (
    ARTICLE_CANDIDATE_FEATURE_NAMES,
    ARTICLE_FEATURE_SCHEMA_VERSION,
    ArticleFeatureProvider,
)
from rl.policy import LinearQPolicy
from search.node import SearchNode


_BUDGET = ResourceBudget(
    max_t_count=4,
    max_two_qubit_count=2,
    max_gates=4,
    max_depth=4,
)


def _state(*gates: Gate) -> CircuitState:
    state = CircuitState(CircuitDAG(1), _BUDGET)
    for gate in gates:
        assert state.apply_gate(gate)
    return state


def test_article_eq19_schema_has_bias_candidate_z_and_budget_interactions():
    provider = ArticleFeatureProvider()
    root = _state()

    features = provider.extract(root, [root])

    assert provider.schema_version == ARTICLE_FEATURE_SCHEMA_VERSION
    assert provider.dimension == 1 + 12 + 12 + 12
    assert features.shape == (provider.dimension,)
    assert features[0] == 1.0
    assert len(provider.names) == provider.dimension
    assert provider.metadata()["article_equation"] == 19


def test_article_budget_interaction_uses_remaining_expansion_horizon():
    provider = ArticleFeatureProvider(search_horizon=4)
    root = _state()

    provider.set_search_step(1)
    features = provider.extract(root, [root])
    candidate = features[1:13]
    interactions = features[25:]

    assert provider.remaining_search_budget_fraction == 0.75
    assert np.allclose(interactions, 0.75 * candidate)
    provider.set_search_step(4)
    assert np.array_equal(provider.extract(root, [root])[25:], np.zeros(12))


def test_environment_binds_and_advances_article_search_horizon():
    provider = ArticleFeatureProvider(search_horizon=99)
    config = Config(
        num_qubits=1,
        budget=ResourceBudget(0, 2, 2),
        max_steps=4,
        reward_mode="expansion_cost",
    )
    target = SynthesisTarget(
        unitary_from_gates(1, [Gate(GateType.X, (0,))])
    )
    environment = CircuitSynthesisEnv(
        config,
        SimulatorCertificationEngine(target),
        feature_provider=provider,
    )

    environment.reset(seed=3)
    assert provider.search_horizon == 4
    assert provider.remaining_search_budget_fraction == 1.0
    environment.select_record(0)
    assert provider.remaining_search_budget_fraction == 0.75


def test_article_features_are_bitwise_invariant_to_frontier_permutation():
    provider = ArticleFeatureProvider()
    root = _state()
    h = _state(Gate(GateType.H, (0,)))
    t = _state(Gate(GateType.T, (0,)))

    forward = provider.extract(h, [root, h, t])
    reverse = provider.extract(h, [t, h, root])

    assert np.array_equal(forward, reverse)


def test_article_features_expose_pareto_rank_and_semantic_novelty():
    provider = ArticleFeatureProvider()
    root = _state()
    identity_with_cost = _state(Gate(GateType.H, (0,)), Gate(GateType.H, (0,)))
    frontier = [root, identity_with_cost]
    candidate_offset = 1
    rank_index = candidate_offset + ARTICLE_CANDIDATE_FEATURE_NAMES.index(
        "pareto_rank_fraction"
    )
    novelty_index = candidate_offset + ARTICLE_CANDIDATE_FEATURE_NAMES.index(
        "semantic_novelty"
    )

    root_features = provider.extract(root, frontier)
    costly_features = provider.extract(identity_with_cost, frontier)

    assert root_features[rank_index] == 0.0
    assert costly_features[rank_index] == 1.0
    assert root_features[novelty_index] == 0.5
    assert costly_features[novelty_index] == 0.5


def test_article_provider_binds_policy_and_scores_records_without_positions():
    provider = ArticleFeatureProvider()
    policy = LinearQPolicy(feature_provider=provider, seed=11)
    root = SearchNode(priority=0.0, state=_state(), record_id=9)
    t = SearchNode(
        priority=0.0,
        state=_state(Gate(GateType.T, (0,))),
        record_id=3,
    )
    policy.theta[provider.names.index("gate_count_fraction")] = 1.0

    selected_forward = policy.select_node([root, t], epsilon=0.0)
    selected_reverse = policy.select_node([t, root], epsilon=0.0)

    assert selected_forward is t
    assert selected_reverse is t
    assert policy.metadata()["feature_schema_version"] == ARTICLE_FEATURE_SCHEMA_VERSION
