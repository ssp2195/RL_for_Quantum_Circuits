from __future__ import annotations

import numpy as np
import pytest

from hybrid_qcs.ancilla_benchmarks import (
    clean_contract,
    controlled_t_with_clean_ancilla,
    qft3_clean_ancilla_target,
    qft3_clean_ancilla_witness,
)
from hybrid_qcs.ancilla_certify import certify_ancilla_state
from hybrid_qcs.ancilla_contract import target_from_hidden_ancilla_gates
from hybrid_qcs.model import Gate, HybridState
from hybrid_qcs.qft_guided import (
    controlled_t_relative_phase_compute,
    exact_qft_macros,
    qft_matrix,
    synthesize_qft_decomposition_guided,
)


def _state(target, gates):
    state = HybridState.identity(target.num_qubits, target.budget)
    for gate in gates:
        child = state.apply(gate, partial_order_reduction=False)
        assert child is not None
        state = child
    return state


def test_relative_phase_compute_uncompute_is_exact_controlled_t() -> None:
    legacy = controlled_t_with_clean_ancilla(0, 1, 2)
    target = target_from_hidden_ancilla_gates(
        "controlled-t-clean",
        "test",
        clean_contract(2, 1),
        legacy,
        gate_slack=0,
        depth_slack=0,
    )
    optimized = controlled_t_relative_phase_compute(0, 1, 2)
    result = certify_ancilla_state(target, _state(target, optimized))
    assert result.success
    assert result.ancilla_leakage < 1e-12
    assert len(optimized) == 19
    assert sum(gate.is_non_clifford for gate in optimized) == 9
    assert sum(gate.is_two_qubit for gate in optimized) == 6
    assert len(optimized) < len(legacy)


def test_qft3_verified_macro_plan_generates_shorter_certified_circuit() -> None:
    target = qft3_clean_ancilla_target()
    result = synthesize_qft_decomposition_guided(target)
    assert result.success
    assert result.certified
    assert result.stop_reason == "certified"
    assert result.macro_count == 7
    assert result.native_gate_count == 35
    assert result.t_count == 15
    assert result.cnot_count == 13
    assert result.depth == 26
    assert result.ancilla_leakage is not None
    assert result.ancilla_leakage < 1e-12
    assert result.projective_isometry_error is not None
    assert result.projective_isometry_error < 1e-12
    assert result.native_gate_count < len(qft3_clean_ancilla_witness())
    assert tuple(result.witness) != tuple(
        gate.label() for gate in qft3_clean_ancilla_witness()
    )


def test_qft_plan_is_derived_from_contract_wire_order() -> None:
    macros = exact_qft_macros(clean_contract(3, 1))
    assert tuple(macro.name for macro in macros) == (
        "H(q2)",
        "CS(q1->q2)",
        "CT(q0->q2;a3)",
        "H(q1)",
        "CS(q0->q1)",
        "H(q0)",
        "SWAP(q0,q2)",
    )
    assert np.allclose(qft_matrix(3), qft3_clean_ancilla_target().logical_unitary)


def test_qft3_macro_plan_requires_clean_workspace() -> None:
    with pytest.raises(ValueError, match="requires one clean ancilla"):
        exact_qft_macros(clean_contract(3, 0))


def test_non_qft_target_is_not_silently_rewritten() -> None:
    target = target_from_hidden_ancilla_gates(
        "not-qft",
        "test",
        clean_contract(1, 1),
        (Gate("S", (0,)),),
    )
    result = synthesize_qft_decomposition_guided(target)
    assert not result.success
    assert result.stop_reason == "not_qft"
    assert result.native_gate_count == 0
