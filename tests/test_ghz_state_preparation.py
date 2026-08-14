"""Deterministic GHZ-3 state-preparation smoke coverage.

This module intentionally imports neither the Gymnasium environment nor the
RL trainer.  It validates the public circuit/DAG and dense statevector paths.
"""

import numpy as np
import pytest

from certification.simulator import (
    equivalent_up_to_global_phase,
    state_fidelity,
    statevector_from_gates,
    unitary_from_gates,
)
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType


MAX_STATE_INFIDELITY = 1e-12
GHZ3_GATES = (
    Gate(GateType.H, (0,)),
    Gate(GateType.CNOT, (0, 1)),
    Gate(GateType.CNOT, (0, 2)),
)


def _ghz3_state() -> CircuitState:
    state = CircuitState(
        CircuitDAG(3),
        ResourceBudget(max_t_count=0, max_depth=3, max_gates=3, max_two_qubit_count=2),
    )
    for gate in GHZ3_GATES:
        assert state.apply_gate(gate)
        state.dag.validate()
    return state


def _expected_ghz_plus() -> np.ndarray:
    expected = np.zeros(8, dtype=np.complex128)
    expected[0] = 1.0 / np.sqrt(2.0)
    expected[7] = 1.0 / np.sqrt(2.0)
    return expected


def test_ghz3_public_witness_has_directed_structure_and_resources():
    state = _ghz3_state()
    gates = state.dag.gates

    assert state.dag.size() == 3
    assert gates[0] == Gate(GateType.H, (0,))
    assert {gate.qubits for gate in gates if gate.gate_type is GateType.CNOT} == {
        (0, 1),
        (0, 2),
    }
    assert sum(gate.gate_type is GateType.CNOT for gate in gates) == 2
    assert all(gate.gate_type not in {GateType.T, GateType.TDG} for gate in gates)
    assert state.num_gates == 3
    assert state.two_qubit_count == 2
    assert state.t_count == 0
    assert state.depth == 3
    assert state.wire_depths == (3, 2, 3)
    assert "no-ancilla" in state.continuation_interface

    dense_witness = unitary_from_gates(3, gates)
    assert equivalent_up_to_global_phase(state.symbolic_unitary(), dense_witness)


def test_ghz3_statevector_has_positive_ghz_fidelity_and_ideal_probabilities():
    actual = statevector_from_gates(3, _ghz3_state().dag.gates)
    expected = _expected_ghz_plus()
    fidelity = state_fidelity(expected, actual, atol=MAX_STATE_INFIDELITY)
    probabilities = np.abs(actual) ** 2

    assert actual.shape == (8,)
    assert np.isfinite(actual).all()
    assert np.isclose(np.linalg.norm(actual), 1.0, atol=MAX_STATE_INFIDELITY)
    assert fidelity >= 1.0 - MAX_STATE_INFIDELITY
    np.testing.assert_allclose(probabilities[[0, 7]], (0.5, 0.5), atol=MAX_STATE_INFIDELITY)
    np.testing.assert_allclose(
        probabilities[[1, 2, 3, 4, 5, 6]],
        0.0,
        atol=MAX_STATE_INFIDELITY,
    )
    assert np.isclose(probabilities.sum(), 1.0, atol=MAX_STATE_INFIDELITY)


def test_ghz3_fidelity_rejects_the_negative_relative_phase_state():
    expected = _expected_ghz_plus()
    negative_ghz = expected.copy()
    negative_ghz[7] *= -1.0

    assert np.isclose(state_fidelity(expected, negative_ghz), 0.0, atol=MAX_STATE_INFIDELITY)


def test_statevector_helper_validates_custom_input_shape_values_and_norm():
    with pytest.raises(ValueError, match="shape"):
        statevector_from_gates(3, (), initial_state=np.zeros(7, dtype=np.complex128))
    with pytest.raises(ValueError, match="finite"):
        statevector_from_gates(
            3,
            (),
            initial_state=np.array([np.nan] + [0.0] * 7, dtype=np.complex128),
        )
    with pytest.raises(ValueError, match="normalized"):
        statevector_from_gates(3, (), initial_state=np.ones(8, dtype=np.complex128))
