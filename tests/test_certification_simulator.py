from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from certification.algebraic import AlgebraicCertificationEngine
from certification.base import CertStatus
from certification.simulator import (
    CNOT_MATRIX,
    H_MATRIX,
    SDG_MATRIX,
    S_MATRIX,
    TDG_MATRIX,
    T_MATRIX,
    SynthesisTarget,
    SimulatorCertificationEngine,
    equivalent_up_to_global_phase,
    gate_matrix,
    unitary_from_gates,
)
from circuit.dag import CircuitDAG
from enums import GateType


@dataclass(frozen=True)
class WitnessGate:
    """Small gate fixture that also works before the enum gains dagger gates."""

    gate_type: object
    qubits: tuple[int, ...]


def _gate(name: str, *qubits: int) -> WitnessGate:
    return WitnessGate(getattr(GateType, name, name), qubits)


def _state_from_gates(num_qubits: int, gates: list[WitnessGate]):
    dag = CircuitDAG.from_gates(num_qubits, gates)
    return SimpleNamespace(dag=dag)


@pytest.mark.parametrize(
    ("gate_name", "expected"),
    [
        ("H", H_MATRIX),
        ("S", S_MATRIX),
        ("SDG", SDG_MATRIX),
        ("T", T_MATRIX),
        ("TDG", TDG_MATRIX),
        ("CNOT", CNOT_MATRIX),
    ],
)
def test_gate_matrices_are_exact(gate_name, expected):
    np.testing.assert_allclose(gate_matrix(getattr(GateType, gate_name, gate_name)), expected)


def test_cnot_uses_control_then_target_operands():
    # Qubit 0 is the least-significant register bit.  CNOT(0, 1) maps
    # |01> -> |11> and |11> -> |01> in the global vector indexing.
    cnot = unitary_from_gates(2, [_gate("CNOT", 0, 1)])
    np.testing.assert_allclose(cnot[:, 1], np.eye(4)[:, 3])
    np.testing.assert_allclose(cnot[:, 3], np.eye(4)[:, 1])


def test_simulator_certifies_a_witness_with_every_required_gate():
    gates = [
        _gate("H", 0),
        _gate("S", 1),
        _gate("SDG", 0),
        _gate("T", 1),
        _gate("TDG", 0),
        _gate("CNOT", 0, 1),
    ]
    state = _state_from_gates(2, gates)
    target = SynthesisTarget(unitary_from_gates(2, gates))

    result = SimulatorCertificationEngine(target).certify(state)

    assert result.status is CertStatus.SUCCESS
    assert result.score == 1.0


def test_equivalence_quotients_global_phase():
    target = unitary_from_gates(1, [_gate("H", 0), _gate("T", 0)])
    candidate = np.exp(0.371j) * target

    assert equivalent_up_to_global_phase(candidate, target)

    state = _state_from_gates(1, [])
    phased_identity = np.exp(0.371j) * np.eye(2, dtype=np.complex128)
    assert (
        SimulatorCertificationEngine(SynthesisTarget(phased_identity)).certify(state).status
        is CertStatus.SUCCESS
    )
    assert (
        SimulatorCertificationEngine(
            SynthesisTarget(phased_identity, quotient_global_phase=False)
        ).certify(state).status
        is CertStatus.INCONCLUSIVE
    )


def test_non_target_prefix_is_inconclusive_and_ht_does_not_match_t():
    target = SynthesisTarget(unitary_from_gates(1, [_gate("T", 0)]))
    ht_state = _state_from_gates(1, [_gate("H", 0), _gate("T", 0)])

    assert (
        SimulatorCertificationEngine(target).certify(ht_state).status
        is CertStatus.INCONCLUSIVE
    )


def test_algebraic_certifier_never_false_succeeds_on_matching_phase_terms():
    # This matches the legacy phase-polynomial payload that used to certify
    # H;T as T.  A general Clifford+T certifier must leave it inconclusive.
    matching_phase_poly = SimpleNamespace(terms={1: 1})
    state = SimpleNamespace(phase_poly=matching_phase_poly)

    result = AlgebraicCertificationEngine(((1, 1),)).certify(state)

    assert result.status is CertStatus.INCONCLUSIVE


def test_equivalence_rejects_shape_mismatch_and_bad_phase_anchor():
    assert not equivalent_up_to_global_phase(np.eye(2), np.eye(4))
    assert not equivalent_up_to_global_phase(np.zeros((2, 2)), np.zeros((2, 2)))


def test_synthesis_target_rejects_nonunitary_inputs():
    with pytest.raises(ValueError, match="unitary"):
        SynthesisTarget(np.array([[1.0, 0.0], [0.0, 2.0]]))
