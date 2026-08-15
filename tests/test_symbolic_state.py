import random

import numpy as np
import pytest

from algebra.pauli import PauliAxis
from algebra.tableau import CliffordFrame
from canonical.canonicalizer import Canonicalizer
from certification.simulator import equivalent_up_to_global_phase, unitary_from_gates
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType


def _state(num_qubits=1, *, depth=20, gates=30, t_count=20):
    return CircuitState(
        CircuitDAG(num_qubits),
        ResourceBudget(t_count, depth, gates),
    )


def _append(state, gate_type, qubits):
    assert state.apply_gate(Gate(gate_type, qubits))
    return state


def test_per_wire_depth_allows_disjoint_parallel_gates():
    state = _state(2, depth=1)

    _append(state, GateType.H, (0,))
    _append(state, GateType.H, (1,))

    assert state.wire_depths == (1, 1)
    assert state.depth == 1
    assert state.dag.depth() == 1


def test_tdg_and_two_qubit_resources_are_tracked_independently():
    state = _state(2, t_count=2, gates=3)
    _append(state, GateType.T, (0,))
    _append(state, GateType.TDG, (1,))
    _append(state, GateType.CNOT, (0, 1))

    assert state.t_count == 2
    assert state.two_qubit_count == 1
    assert state.resource_vector() == (2, 1, 3, 2, 2)


def test_t_axes_are_transported_by_the_current_clifford_frame():
    hth = _state()
    _append(hth, GateType.H, (0,))
    _append(hth, GateType.T, (0,))
    _append(hth, GateType.H, (0,))
    assert hth.rotations[0].axis == PauliAxis.x_axis(1, 0)

    cnot_t_cnot = _state(2)
    _append(cnot_t_cnot, GateType.CNOT, (0, 1))
    _append(cnot_t_cnot, GateType.T, (1,))
    _append(cnot_t_cnot, GateType.CNOT, (0, 1))
    assert cnot_t_cnot.rotations[0].axis == PauliAxis(2, 0, 0b11)


def test_symbolic_invariant_matches_independent_witness_simulation():
    rng = random.Random(12)
    choices = [
        (GateType.H, (0,)),
        (GateType.S, (0,)),
        (GateType.SDG, (0,)),
        (GateType.T, (0,)),
        (GateType.TDG, (0,)),
        (GateType.H, (1,)),
        (GateType.S, (1,)),
        (GateType.T, (1,)),
        (GateType.CNOT, (0, 1)),
        (GateType.CNOT, (1, 0)),
    ]
    for _ in range(25):
        state = _state(2)
        for _ in range(8):
            gate_type, qubits = rng.choice(choices)
            _append(state, gate_type, qubits)
        witness = unitary_from_gates(2, state.dag.gates)
        assert equivalent_up_to_global_phase(state.symbolic_unitary(), witness)


def test_bare_prepopulated_dag_is_replayed_not_mistaken_for_identity():
    dag = CircuitDAG.from_gates(
        1,
        (Gate(GateType.H, (0,)), Gate(GateType.T, (0,))),
    )

    state = CircuitState(dag, ResourceBudget(2, 2, 2))
    empty = _state()

    assert state.num_gates == 2
    assert state.t_count == 1
    assert equivalent_up_to_global_phase(
        state.symbolic_unitary(), unitary_from_gates(1, state.dag.gates)
    )
    assert Canonicalizer().semantic_key(state) != Canonicalizer().semantic_key(empty)


def test_forged_cached_snapshot_is_ignored_in_favor_of_the_dag_witness():
    dag = CircuitDAG.from_gates(
        1,
        (Gate(GateType.H, (0,)), Gate(GateType.T, (0,))),
    )

    forged = CircuitState(
        dag,
        ResourceBudget(2, 2, 2),
        t_count=0,
        num_gates=2,
        wire_depths=(2,),
        frame=None,
        rotations=(),
    )

    assert forged.t_count == 1
    assert forged.rotations
    assert Canonicalizer().semantic_key(forged) != Canonicalizer().semantic_key(_state())


def test_empty_dag_ignores_a_forged_literal_phase_frame():
    frame = CliffordFrame(1)
    for gate_type in (GateType.H, GateType.S) * 3:
        if gate_type is GateType.H:
            frame.apply_H(0)
        else:
            frame.apply_S(0)
    fresh_frame = CliffordFrame(1)
    assert frame.canonical_payload() == fresh_frame.canonical_payload()
    assert frame.phase_sensitive_payload() != fresh_frame.phase_sensitive_payload()

    state = CircuitState(
        CircuitDAG(1),
        ResourceBudget(0, 0, 0),
        frame=frame,
        global_phase_eighths=7,
    )
    identity = CircuitState(CircuitDAG(1), ResourceBudget(0, 0, 0))

    # Public construction derives semantics entirely from the empty witness,
    # rather than retaining its forged literal phase or frame history.
    np.testing.assert_allclose(state.symbolic_unitary(), np.eye(2))
    literal = Canonicalizer(phase_sensitive=True)
    assert literal.semantic_key(state) == literal.semantic_key(identity)


def test_continuation_key_includes_budget_legality_contract():
    no_gates = CircuitState(CircuitDAG(1), ResourceBudget(0, 0, 0))
    one_gate = CircuitState(CircuitDAG(1), ResourceBudget(0, 1, 1))
    assert Canonicalizer().semantic_key(no_gates) != Canonicalizer().semantic_key(one_gate)


def test_invalid_boolean_qubits_are_rejected_without_mutating_the_witness():
    state = _state(2)
    assert not state.apply_gate(Gate(GateType.T, (True,)))
    assert state.dag.size() == 0
    assert state.resource_vector() == (0, 0, 0, 0, 0)

    with pytest.raises(ValueError):
        CircuitDAG.from_gates(2, (Gate(GateType.H, (True,)),))


def test_public_dag_mutation_is_a_hard_failure():
    dag = CircuitDAG(1)

    with pytest.raises(RuntimeError, match="CircuitState.apply_gate"):
        dag.add_gate(Gate(GateType.H, (0,)))

    assert dag.gates == []
    dag.validate()

    with pytest.raises(AttributeError):
        dag.num_qubits = 2


def test_supported_mutation_and_copy_preserve_independent_consistent_snapshots():
    original = _append(_state(), GateType.H, (0,))
    original.validate_consistency()

    copied = original.copy()
    _append(copied, GateType.T, (0,))

    assert original.dag.gates == [Gate(GateType.H, (0,))]
    assert copied.dag.gates == [Gate(GateType.H, (0,)), Gate(GateType.T, (0,))]
    assert original.frame is not copied.frame
    assert original.dag is not copied.dag
    original.validate_consistency()
    copied.validate_consistency()


def test_consistency_validation_detects_unauthorized_raw_dag_mutation():
    state = _append(_state(), GateType.H, (0,))
    state.dag._append_gate_unchecked(Gate(GateType.T, (0,)))

    with pytest.raises(AssertionError, match="resource vector"):
        state.validate_consistency()


def test_replaying_serialized_dag_reproduces_dense_and_symbolic_unitaries():
    state = _state(2)
    for gate_type, qubits in (
        (GateType.H, (0,)),
        (GateType.CNOT, (0, 1)),
        (GateType.TDG, (1,)),
        (GateType.S, (0,)),
    ):
        _append(state, gate_type, qubits)

    serialized_gates = tuple(state.dag.gates)
    replayed = CircuitState(
        CircuitDAG.from_gates(2, serialized_gates),
        state.budget,
    )

    state.validate_consistency()
    replayed.validate_consistency()
    np.testing.assert_allclose(state.symbolic_unitary(), replayed.symbolic_unitary())
    np.testing.assert_allclose(
        replayed.symbolic_unitary(),
        unitary_from_gates(2, serialized_gates),
    )


def test_dag_validation_requires_the_latest_gate_cache():
    dag = CircuitDAG.from_gates(
        1,
        (Gate(GateType.H, (0,)), Gate(GateType.T, (0,))),
    )
    dag._last_gate_on_qubit[0] = 0

    with pytest.raises(AssertionError, match="latest"):
        dag.validate()


def test_canonicalisation_is_sound_for_key_regressions_and_global_phase_modes():
    ht = _append(_append(_state(), GateType.H, (0,)), GateType.T, (0,))
    th = _append(_append(_state(), GateType.T, (0,)), GateType.H, (0,))
    canonicalizer = Canonicalizer()

    assert canonicalizer.semantic_key(ht) != canonicalizer.semantic_key(th)
    assert canonicalizer.identity_hash(ht) != canonicalizer.identity_hash(th)

    phase_shifted = ht.copy()
    phase_shifted.global_phase_eighths += 1
    assert canonicalizer.semantic_key(ht) == canonicalizer.semantic_key(phase_shifted)
    literal = Canonicalizer(phase_sensitive=True)
    assert literal.semantic_key(ht) != literal.semantic_key(phase_shifted)

    # (H,S)^3 is projectively identity but carries a Clifford global phase.
    clifford_phase = _state()
    for gate_type in (GateType.H, GateType.S) * 3:
        _append(clifford_phase, gate_type, (0,))
    identity = _state()
    assert canonicalizer.semantic_key(clifford_phase) == canonicalizer.semantic_key(identity)
    assert literal.semantic_key(clifford_phase) != literal.semantic_key(identity)


def test_commuting_rotation_order_merges_but_anticommuting_order_does_not():
    t0_t1 = _append(_append(_state(2), GateType.T, (0,)), GateType.T, (1,))
    t1_t0 = _append(_append(_state(2), GateType.T, (1,)), GateType.T, (0,))
    assert Canonicalizer().semantic_key(t0_t1) == Canonicalizer().semantic_key(t1_t0)

    x_then_z = _append(_append(_state(), GateType.H, (0,)), GateType.T, (0,))
    z_then_x = _append(_append(_state(), GateType.T, (0,)), GateType.H, (0,))
    assert Canonicalizer().semantic_key(x_then_z) != Canonicalizer().semantic_key(z_then_x)
