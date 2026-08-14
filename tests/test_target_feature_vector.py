"""Feature-vector and policy-binding coverage for target-aware GHZ ranking."""

from __future__ import annotations

import numpy as np
import pytest

from certification.simulator import SynthesisTarget, unitary_from_gates
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType
from rl.features import (
    DIRECTED_CNOT_PAIRS,
    LEGACY_FEATURE_DIMENSION,
    TARGET_AWARE_FEATURE_DIMENSION,
    TARGET_AWARE_FEATURE_SCHEMA_VERSION,
    extract_features,
    feature_dimension,
    feature_names,
)
from rl.policy import LinearQPolicy
from rl.target_context import DenseTargetContext
from search.node import SearchNode


_BUDGET = ResourceBudget(
    max_t_count=0,
    max_depth=3,
    max_gates=3,
    max_two_qubit_count=2,
)
_GHZ3_GATES = (
    Gate(GateType.H, (0,)),
    Gate(GateType.CNOT, (0, 1)),
    Gate(GateType.CNOT, (0, 2)),
)


def _state(*gates: Gate) -> CircuitState:
    state = CircuitState(CircuitDAG(3), _BUDGET)
    for gate in gates:
        assert state.apply_gate(gate)
    state.dag.validate()
    return state


def _context(*gates: Gate) -> DenseTargetContext:
    target_gates = gates or _GHZ3_GATES
    return DenseTargetContext.from_synthesis_target(
        SynthesisTarget(unitary_from_gates(3, target_gates))
    )


def _feature(feature_vector: np.ndarray, context: DenseTargetContext, name: str) -> float:
    return float(feature_vector[feature_names(context).index(name)])


def test_target_free_feature_schema_is_still_the_legacy_16_coordinates():
    root = _state()
    h0 = _state(Gate(GateType.H, (0,)))

    features = extract_features(h0, [root, h0])

    assert features.shape == (LEGACY_FEATURE_DIMENSION,)
    assert feature_dimension(h0) == LEGACY_FEATURE_DIMENSION
    assert feature_dimension() == LEGACY_FEATURE_DIMENSION
    assert "bias" not in feature_names()


def test_target_aware_features_distinguish_labelled_hadamard_qubits():
    context = _context()
    h0 = _state(Gate(GateType.H, (0,)))
    h1 = _state(Gate(GateType.H, (1,)))
    h2 = _state(Gate(GateType.H, (2,)))
    frontier = [h0, h1, h2]

    h0_features = extract_features(h0, frontier, context)
    h1_features = extract_features(h1, frontier, context)
    h2_features = extract_features(h2, frontier, context)

    assert h0_features.shape == (TARGET_AWARE_FEATURE_DIMENSION,)
    assert feature_dimension(target_context=context) == TARGET_AWARE_FEATURE_DIMENSION
    assert not np.array_equal(h0_features, h1_features)
    assert not np.array_equal(h0_features, h2_features)
    assert _feature(h0_features, context, "target_process_fidelity") > _feature(
        h1_features, context, "target_process_fidelity"
    )
    assert _feature(h0_features, context, "target_progress_potential") > _feature(
        h1_features, context, "target_progress_potential"
    )
    assert _feature(h0_features, context, "target_progress_potential") > _feature(
        h2_features, context, "target_progress_potential"
    )
    assert _feature(h0_features, context, "qubit_0_hadamard_fraction") > 0.0
    assert _feature(h1_features, context, "qubit_1_hadamard_fraction") > 0.0
    assert _feature(h2_features, context, "qubit_2_hadamard_fraction") > 0.0


def test_target_aware_features_encode_directed_cnot_and_last_operation():
    context = _context()
    h0 = Gate(GateType.H, (0,))
    correct = _state(h0, Gate(GateType.CNOT, (0, 1)))
    reversed_direction = _state(h0, Gate(GateType.CNOT, (1, 0)))
    other_correct = _state(h0, Gate(GateType.CNOT, (0, 2)))
    other_reversed = _state(h0, Gate(GateType.CNOT, (2, 0)))
    frontier = [correct, reversed_direction, other_correct, other_reversed]

    correct_features = extract_features(correct, frontier, context)
    reversed_features = extract_features(reversed_direction, frontier, context)
    other_correct_features = extract_features(other_correct, frontier, context)
    other_reversed_features = extract_features(other_reversed, frontier, context)

    assert DIRECTED_CNOT_PAIRS == ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
    assert _feature(correct_features, context, "cnot_0_to_1_fraction") == pytest.approx(0.5)
    assert _feature(correct_features, context, "cnot_1_to_0_fraction") == 0.0
    assert _feature(reversed_features, context, "cnot_1_to_0_fraction") == pytest.approx(0.5)
    assert _feature(other_correct_features, context, "cnot_0_to_2_fraction") == pytest.approx(0.5)
    assert _feature(other_reversed_features, context, "cnot_2_to_0_fraction") == pytest.approx(0.5)
    assert _feature(correct_features, context, "last_gate_cnot") == 1.0
    assert _feature(correct_features, context, "last_first_operand_0") == 1.0
    assert _feature(correct_features, context, "last_second_operand_1") == 1.0
    assert _feature(reversed_features, context, "last_first_operand_1") == 1.0
    assert _feature(reversed_features, context, "last_second_operand_0") == 1.0
    assert _feature(correct_features, context, "target_progress_potential") > _feature(
        reversed_features, context, "target_progress_potential"
    )
    assert _feature(other_correct_features, context, "target_progress_potential") > _feature(
        other_reversed_features, context, "target_progress_potential"
    )


def test_target_relative_block_changes_without_changing_legacy_features():
    candidate = _state(Gate(GateType.H, (0,)))
    frontier = [candidate, _state(Gate(GateType.H, (1,)))]
    ghz_context = _context()
    h1_context = _context(Gate(GateType.H, (1,)))

    ghz_features = extract_features(candidate, frontier, ghz_context)
    h1_features = extract_features(candidate, frontier, h1_context)

    np.testing.assert_array_equal(
        ghz_features[:LEGACY_FEATURE_DIMENSION],
        h1_features[:LEGACY_FEATURE_DIMENSION],
    )
    assert not np.allclose(
        ghz_features[LEGACY_FEATURE_DIMENSION:],
        h1_features[LEGACY_FEATURE_DIMENSION:],
    )


def test_target_context_features_and_policy_ranking_are_frontier_order_invariant():
    context = _context()
    root = SearchNode(priority=0.0, state=_state(), record_id=11)
    h0 = SearchNode(priority=0.0, state=_state(Gate(GateType.H, (0,))), record_id=12)
    h1 = SearchNode(priority=0.0, state=_state(Gate(GateType.H, (1,))), record_id=13)
    nodes = [root, h0, h1]
    reversed_nodes = list(reversed(nodes))

    left = extract_features(h0.state, nodes, context)
    right = extract_features(h0.state, reversed_nodes, context)
    np.testing.assert_array_equal(left, right)

    policy = LinearQPolicy(target_context=context, seed=8)
    policy.theta = np.linspace(-0.25, 0.25, policy.feature_dim)
    assert policy.node_value(h0, nodes) == pytest.approx(
        policy.node_value(h0, reversed_nodes), abs=0.0
    )
    # Use the semantic potential alone to make the intended record the unique
    # greedy choice.  Record IDs only resolve equal learned values.
    policy.theta.fill(0.0)
    policy.theta[feature_names(context).index("target_progress_potential")] = 1.0
    assert policy.select_node(nodes, epsilon=0.0) is h0
    assert policy.select_node(reversed_nodes, epsilon=0.0) is h0


def test_policy_binds_target_schema_and_reports_a_weight_digest():
    context = _context()
    policy = LinearQPolicy(target_context=context, lr=0.02, gamma=1.0, seed=4)

    assert policy.feature_dim == TARGET_AWARE_FEATURE_DIMENSION
    assert policy.feature_schema_version == TARGET_AWARE_FEATURE_SCHEMA_VERSION
    assert policy.target_fingerprint == context.fingerprint
    metadata = policy.metadata()
    assert metadata["target_fingerprint"] == context.fingerprint
    assert metadata["feature_dim"] == TARGET_AWARE_FEATURE_DIMENSION
    assert metadata["target_context_binding_digest"] == policy.target_context_binding_digest
    assert policy.target_context_binding_digest.startswith("sha256:")
    before = policy.weight_digest()
    policy.theta[0] = 0.125
    assert policy.weight_digest() != before

    with pytest.raises(ValueError, match="does not match"):
        LinearQPolicy(feature_dim=LEGACY_FEATURE_DIMENSION, target_context=context)


def test_target_aware_sarsa_update_uses_the_policy_bound_context():
    context = _context()
    current = SearchNode(priority=0.0, state=_state(), record_id=1)
    next_node = SearchNode(
        priority=0.0,
        state=_state(Gate(GateType.H, (0,))),
        record_id=2,
    )
    policy = LinearQPolicy(target_context=context, lr=0.125, gamma=0.5, seed=3)
    before = np.linspace(-0.15, 0.15, policy.feature_dim)
    policy.theta = before.copy()
    frontier = [current]
    next_frontier = [next_node]
    phi = extract_features(current.state, frontier, context).astype(np.float64)
    expected_error = (
        1.25
        + 0.5 * policy.node_value(next_node, next_frontier)
        - float(np.dot(before, phi))
    )

    returned_error = policy.update(
        current.state,
        1.25,
        next_frontier=next_frontier,
        done=False,
        next_node=next_node,
        frontier=frontier,
    )

    assert returned_error == pytest.approx(expected_error)
    np.testing.assert_allclose(policy.theta, before + 0.125 * expected_error * phi)
