"""Public-witness and negative-control coverage for exact Toffoli validation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, is_dataclass

import numpy as np
import pytest

from benchmarks.toffoli import (
    KNOWN_TOFFOLI_BUDGET,
    KNOWN_TOFFOLI_GATES,
    TOFFOLI_CONTROLS,
    TOFFOLI_NUM_QUBITS,
    ToffoliValidation,
    build_known_toffoli_state,
    toffoli_reference_unitary,
    toffoli_resource_summary,
    validate_exact_toffoli_state,
    validate_toffoli_unitary,
)
from certification.simulator import equivalent_up_to_global_phase, unitary_from_gates
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget


def _replay_public_witness() -> CircuitState:
    state = CircuitState(CircuitDAG(TOFFOLI_NUM_QUBITS), KNOWN_TOFFOLI_BUDGET)
    for gate in KNOWN_TOFFOLI_GATES:
        assert state.apply_gate(gate), f"public witness gate was rejected: {gate!r}"
    state.dag.validate()
    return state


def _state_from_public_gates(gates) -> CircuitState:
    # A malformed witness can have a different dependency depth.  Validation
    # negatives must reach the semantic oracle rather than being rejected by
    # the *reference* witness's tight resource profile first.
    state = CircuitState(
        CircuitDAG(TOFFOLI_NUM_QUBITS),
        ResourceBudget(
            max_t_count=32,
            max_depth=32,
            max_gates=32,
            max_two_qubit_count=32,
        ),
    )
    for gate in gates:
        assert state.apply_gate(gate), f"negative-control gate was rejected: {gate!r}"
    state.dag.validate()
    return state


def _state_snapshot(state: CircuitState) -> tuple[object, ...]:
    """Capture every observable mutation target of ``CircuitState.apply_gate``."""

    assert state.frame is not None
    return (
        tuple(state.dag.gates),
        state.resource_vector(),
        state.depth,
        state.frame.canonical_payload(),
        tuple(state.rotations),
        state.global_phase_eighths,
    )


def _assert_budget_rejection_is_atomic(gate, budget: ResourceBudget) -> None:
    state = CircuitState(CircuitDAG(TOFFOLI_NUM_QUBITS), budget)
    before = _state_snapshot(state)

    assert not state.can_apply(gate)
    assert not state.apply_gate(gate)
    assert _state_snapshot(state) == before


def test_public_known_witness_replays_through_the_dag_and_exactly_validates():
    built = build_known_toffoli_state()
    replayed = _replay_public_witness()
    replayed_from_authoritative_dag = CircuitState(
        built.dag.copy(),
        built.budget,
    )

    assert isinstance(KNOWN_TOFFOLI_GATES, tuple)
    assert tuple((gate.gate_type.name, gate.qubits) for gate in KNOWN_TOFFOLI_GATES) == (
        ("H", (2,)),
        ("T", (0,)),
        ("T", (1,)),
        ("T", (2,)),
        ("CNOT", (0, 1)),
        ("TDG", (1,)),
        ("CNOT", (0, 1)),
        ("CNOT", (0, 2)),
        ("TDG", (2,)),
        ("CNOT", (1, 2)),
        ("T", (2,)),
        ("CNOT", (0, 2)),
        ("TDG", (2,)),
        ("CNOT", (1, 2)),
        ("H", (2,)),
    )
    assert len(KNOWN_TOFFOLI_GATES) == 15
    names = [gate.gate_type.name for gate in KNOWN_TOFFOLI_GATES]
    assert names.count("H") == 2
    assert names.count("T") == 4
    assert names.count("TDG") == 3
    assert names.count("CNOT") == 6
    assert (
        KNOWN_TOFFOLI_BUDGET.max_t_count,
        KNOWN_TOFFOLI_BUDGET.max_two_qubit_count,
        KNOWN_TOFFOLI_BUDGET.max_gates,
        KNOWN_TOFFOLI_BUDGET.max_depth,
    ) == (7, 6, 15, 12)

    assert tuple(built.dag.gates) == tuple(KNOWN_TOFFOLI_GATES)
    assert tuple(replayed.dag.gates) == tuple(KNOWN_TOFFOLI_GATES)
    assert tuple(replayed_from_authoritative_dag.dag.gates) == tuple(
        KNOWN_TOFFOLI_GATES
    )
    assert toffoli_resource_summary(built) == toffoli_resource_summary(replayed)
    assert toffoli_resource_summary(built) == toffoli_resource_summary(
        replayed_from_authoritative_dag
    )
    assert built.resource_vector() == replayed.resource_vector()
    assert built.resource_vector() == replayed_from_authoritative_dag.resource_vector()
    assert (built.t_count, built.two_qubit_count, built.num_gates, built.depth) == (
        7,
        6,
        15,
        12,
    )
    assert built.wire_depths == (9, 11, 12)

    dense_witness = unitary_from_gates(TOFFOLI_NUM_QUBITS, built.dag.gates)
    reference = toffoli_reference_unitary()
    assert equivalent_up_to_global_phase(dense_witness, reference)
    assert equivalent_up_to_global_phase(built.symbolic_unitary(), dense_witness)
    assert equivalent_up_to_global_phase(
        replayed_from_authoritative_dag.symbolic_unitary(),
        dense_witness,
    )

    validation = validate_exact_toffoli_state(built)
    assert isinstance(validation, ToffoliValidation)
    assert validation.exact_certified
    assert validation.global_phase_equivalent
    assert validation.truth_table_correct
    assert validation.column_phase_consistent
    assert validation.symbolic_agrees_with_dense
    assert validation.max_phase_aligned_matrix_error <= 1e-9
    assert validation.process_fidelity >= 1.0 - 1e-12
    resource_summary = toffoli_resource_summary(built)
    for key in (
        "num_gates",
        "t_count",
        "two_qubit_count",
        "depth",
        "wire_depths",
        "resource_vector",
        "resource_accounting_correct",
    ):
        assert validation.resources[key] == resource_summary[key]
    assert validation.resources["resource_accounting_correct"]
    assert validation.resources["semantic_correct"]


def test_phase_aware_oracle_accepts_one_global_phase_but_rejects_columnwise_phases():
    reference = toffoli_reference_unitary()
    global_phase = np.exp(0.317j)
    globally_phased = global_phase * reference

    accepted = validate_toffoli_unitary(globally_phased)
    assert accepted.global_phase_equivalent
    assert accepted.truth_table_correct
    assert accepted.column_phase_consistent
    assert accepted.max_phase_aligned_matrix_error <= 1e-9
    assert accepted.process_fidelity >= 1.0 - 1e-12

    # Right multiplication changes individual input columns while retaining
    # the same classical output permutation.  A truth-table-only oracle would
    # incorrectly accept it; exact unitary validation must not.
    relative_phase = np.eye(reference.shape[0], dtype=np.complex128)
    relative_phase[0, 0] = -1.0
    impostor = reference @ relative_phase
    rejected = validate_toffoli_unitary(impostor)

    assert not rejected.global_phase_equivalent
    assert rejected.truth_table_correct
    assert not rejected.column_phase_consistent
    assert rejected.max_phase_aligned_matrix_error > 1e-6
    assert rejected.process_fidelity < 1.0 - 1e-6


def test_unitary_diagnostics_expose_immutable_per_column_truth_table_evidence():
    diagnostics = validate_toffoli_unitary(toffoli_reference_unitary())

    assert is_dataclass(diagnostics)
    rows = diagnostics.truth_table
    assert len(rows) == 1 << TOFFOLI_NUM_QUBITS
    assert all(is_dataclass(row) for row in rows)
    assert tuple(rows) == rows

    with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
        rows[0].expected_output_index = -1


def test_known_witness_mutations_fail_exact_certification_not_just_resource_checks():
    gates = list(KNOWN_TOFFOLI_GATES)
    first_cnot = next(
        index for index, gate in enumerate(gates) if gate.gate_type.name == "CNOT"
    )
    first_t_like = next(
        index
        for index, gate in enumerate(gates)
        if gate.gate_type.name in {"T", "TDG"}
    )
    final_h = max(
        index for index, gate in enumerate(gates) if gate.gate_type.name == "H"
    )
    outer_hs = [
        index for index, gate in enumerate(gates) if gate.gate_type.name == "H"
    ]

    wrong_direction = gates.copy()
    cnot = wrong_direction[first_cnot]
    wrong_direction[first_cnot] = Gate(cnot.gate_type, tuple(reversed(cnot.qubits)))

    removed_t_like = [
        gate for index, gate in enumerate(gates) if index != first_t_like
    ]
    removed_final_h = [gate for index, gate in enumerate(gates) if index != final_h]

    wrong_outer_h_target = gates.copy()
    for index in outer_hs:
        h_gate = wrong_outer_h_target[index]
        wrong_outer_h_target[index] = Gate(h_gate.gate_type, (TOFFOLI_CONTROLS[0],))

    for mutation_name, candidate_gates in (
        ("wrong CNOT direction", wrong_direction),
        ("removed T/TDG", removed_t_like),
        ("removed final H", removed_final_h),
        ("wrong outer-H targets", wrong_outer_h_target),
    ):
        validation = validate_exact_toffoli_state(_state_from_public_gates(candidate_gates))
        assert not validation.exact_certified, mutation_name
        assert not validation.global_phase_equivalent, mutation_name


def test_each_resource_budget_rejection_is_atomic_for_a_mandatory_witness_gate():
    single_gate = next(gate for gate in KNOWN_TOFFOLI_GATES if len(gate.qubits) == 1)
    t_gate = next(
        gate for gate in KNOWN_TOFFOLI_GATES if gate.gate_type.name in {"T", "TDG"}
    )
    cnot_gate = next(
        gate for gate in KNOWN_TOFFOLI_GATES if gate.gate_type.name == "CNOT"
    )

    # Gate-count and depth checks must reject before appending to the DAG.
    _assert_budget_rejection_is_atomic(
        single_gate,
        ResourceBudget(max_t_count=99, max_depth=99, max_gates=0, max_two_qubit_count=99),
    )
    _assert_budget_rejection_is_atomic(
        single_gate,
        ResourceBudget(max_t_count=99, max_depth=0, max_gates=99, max_two_qubit_count=99),
    )
    _assert_budget_rejection_is_atomic(
        t_gate,
        ResourceBudget(max_t_count=0, max_depth=99, max_gates=99, max_two_qubit_count=99),
    )
    _assert_budget_rejection_is_atomic(
        cnot_gate,
        ResourceBudget(max_t_count=99, max_depth=99, max_gates=99, max_two_qubit_count=0),
    )


@pytest.mark.parametrize(
    ("causal_resource", "budget"),
    [
        (
            "t_count",
            ResourceBudget(
                max_t_count=6,
                max_depth=99,
                max_gates=99,
                max_two_qubit_count=99,
            ),
        ),
        (
            "two_qubit_count",
            ResourceBudget(
                max_t_count=99,
                max_depth=99,
                max_gates=99,
                max_two_qubit_count=5,
            ),
        ),
        (
            "num_gates",
            ResourceBudget(
                max_t_count=99,
                max_depth=99,
                max_gates=14,
                max_two_qubit_count=99,
            ),
        ),
        (
            "depth",
            ResourceBudget(
                max_t_count=99,
                max_depth=11,
                max_gates=99,
                max_two_qubit_count=99,
            ),
        ),
    ],
)
def test_full_known_witness_insufficient_budgets_reject_atomically_at_the_causal_gate(
    causal_resource: str,
    budget: ResourceBudget,
):
    """A near-complete witness must leave no partial failed continuation."""

    state = CircuitState(CircuitDAG(TOFFOLI_NUM_QUBITS), budget)
    for gate in KNOWN_TOFFOLI_GATES:
        before = _state_snapshot(state)
        if state.apply_gate(gate):
            continue

        # The failed public transition cannot leak a DAG node, resource count,
        # frame update, Pauli word, or literal global phase update.
        assert _state_snapshot(state) == before
        if causal_resource == "t_count":
            assert gate.gate_type.name in {"T", "TDG"}
            assert state.t_count == budget.max_t_count
        elif causal_resource == "two_qubit_count":
            assert gate.gate_type.name == "CNOT"
            assert state.two_qubit_count == budget.max_two_qubit_count
        elif causal_resource == "num_gates":
            assert state.num_gates == budget.max_gates
        else:
            projected_layer = 1 + max(state.wire_depths[q] for q in gate.qubits)
            assert projected_layer > budget.max_depth
            assert state.depth == budget.max_depth
        break
    else:  # pragma: no cover - every parameterisation must reject one gate
        pytest.fail(f"{causal_resource} budget unexpectedly accepted full witness")
