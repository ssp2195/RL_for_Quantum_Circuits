"""Independent certification for clean and borrowed ancilla contracts."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ancilla_contract import (
    AncillaSynthesisTarget,
    PhaseMode,
    contract_metrics,
)
from .certify import unitary_from_gates
from .model import HybridState


@dataclass(frozen=True)
class AncillaCertificationResult:
    success: bool
    replay_match: bool
    contract_match: bool
    clean_return: bool
    borrowed_identity: bool
    phase_mode: str
    projective_isometry_error: float
    exact_isometry_error: float
    ancilla_leakage: float
    gate_count: int
    t_count: int
    cnot_count: int
    depth: int
    witness: tuple[str, ...]


def certify_ancilla_state(
    target: AncillaSynthesisTarget,
    state: HybridState,
    *,
    tolerance: float = 1e-9,
) -> AncillaCertificationResult:
    """Reconstruct a DAG witness and verify the declared ancilla isometry.

    The candidate need not equal the hidden reference unitary on physical input
    states that violate the clean-ancilla promise.  Correctness is equality of
    the induced isometry on the logical-plus-borrowed input domain, together
    with return of every clean workspace qubit to |0>.
    """

    if state.num_qubits != target.contract.total_qubits:
        raise ValueError("candidate width does not match the ancilla contract")
    dag = state.materialize_dag()
    replay = HybridState.identity(target.num_qubits, target.budget)
    for gate in dag.gates:
        child = replay.apply(gate, partial_order_reduction=False)
        if child is None:
            raise AssertionError("terminal witness failed symbolic replay")
        replay = child
    replay_match = (
        replay.canonical_key == state.canonical_key
        and replay.resource_vector() == state.resource_vector()
    )

    full_unitary = unitary_from_gates(target.num_qubits, dag.gates)
    candidate_isometry = full_unitary @ target.contract.input_embedding
    metrics = contract_metrics(candidate_isometry, target, tolerance=tolerance)
    selected_error = (
        metrics.projective_error
        if target.contract.phase_mode is PhaseMode.PROJECTIVE
        else metrics.exact_error
    )
    contract_match = selected_error <= tolerance
    clean_return = metrics.leakage <= tolerance

    # Borrowed-ancilla identity is part of target.target_isometry.  Keeping a
    # separate diagnostic makes the contract explicit without performing an
    # additional weaker reduced-state test.
    borrowed_identity = contract_match if target.contract.borrowed_ancillas else True
    success = bool(replay_match and contract_match and clean_return and borrowed_identity)
    return AncillaCertificationResult(
        success=success,
        replay_match=bool(replay_match),
        contract_match=bool(contract_match),
        clean_return=bool(clean_return),
        borrowed_identity=bool(borrowed_identity),
        phase_mode=target.contract.phase_mode.value,
        projective_isometry_error=metrics.projective_error,
        exact_isometry_error=metrics.exact_error,
        ancilla_leakage=metrics.leakage,
        gate_count=state.gate_count,
        t_count=state.t_count,
        cnot_count=state.cnot_count,
        depth=state.depth,
        witness=tuple(gate.label() for gate in dag.gates),
    )


__all__ = ["AncillaCertificationResult", "certify_ancilla_state"]
