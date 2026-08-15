from __future__ import annotations

import numpy as np

from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType
from rl.baselines import LinearContextualBanditPolicy, LinearExpectedSarsaPolicy
from search.node import SearchNode


class _GateCountProvider:
    schema_version = "test-gate-count-v1"
    dimension = 1
    names = ("gate_count_plus_bias",)

    @staticmethod
    def extract(state, frontier=None):
        return np.asarray([state.num_gates + 1.0], dtype=np.float32)

    @staticmethod
    def metadata():
        return {
            "feature_schema_version": "test-gate-count-v1",
            "feature_dim": 1,
            "feature_names": ("gate_count_plus_bias",),
        }


def _state(count: int) -> CircuitState:
    state = CircuitState(CircuitDAG(1), ResourceBudget(0, 4, 4))
    for _ in range(count):
        assert state.apply_gate(Gate(GateType.H, (0,)))
    return state


def test_expected_sarsa_uses_the_full_epsilon_greedy_frontier_expectation():
    policy = LinearExpectedSarsaPolicy(
        feature_provider=_GateCountProvider(), lr=0.1, gamma=1.0, seed=3
    )
    policy.theta[:] = 2.0
    current = _state(0)
    next_nodes = [
        SearchNode(0.0, _state(1), record_id=4),
        SearchNode(0.0, _state(2), record_id=7),
    ]
    policy.select_node(next_nodes, epsilon=0.2)

    assert np.isclose(policy.expected_value(next_nodes), 5.8)
    td_error = policy.update(
        current,
        reward=-1.0,
        next_frontier=next_nodes,
        next_node=next_nodes[1],
        frontier=[SearchNode(0.0, current, record_id=1)],
    )

    assert np.isclose(td_error, 2.8)
    assert np.isclose(policy.theta[0], 2.28)
    assert policy.metadata()["algorithm"] == "linear-semi-gradient-expected-sarsa(0)"


def test_contextual_bandit_has_no_next_frontier_bootstrap():
    policy = LinearContextualBanditPolicy(
        feature_provider=_GateCountProvider(), lr=0.1, gamma=1.0, seed=3
    )
    policy.theta[:] = 2.0
    current = _state(0)
    next_node = SearchNode(0.0, _state(2), record_id=7)

    td_error = policy.update(
        current,
        reward=-1.0,
        next_frontier=[next_node],
        next_node=next_node,
        frontier=[SearchNode(0.0, current, record_id=1)],
    )

    assert np.isclose(td_error, -3.0)
    assert np.isclose(policy.theta[0], 1.7)
    assert policy.metadata()["algorithm"] == "linear-contextual-bandit"
