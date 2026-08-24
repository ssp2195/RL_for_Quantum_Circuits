"""Independent dense certification of terminal persistent-DAG witnesses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, TYPE_CHECKING

import numpy as np

from .model import Gate, HybridState

if TYPE_CHECKING:
    from .benchmarks import SynthesisTarget


@dataclass(frozen=True, slots=True)
class CertificationResult:
    success: bool
    symbolic_match: bool
    replay_match: bool
    dense_match: bool
    maximum_matrix_error: float
    gate_count: int
    t_count: int
    cnot_count: int
    depth: int
    witness: tuple[str, ...]


def _single_qubit_matrix(n: int, matrix: np.ndarray, qubit: int) -> np.ndarray:
    dimension = 1 << n
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    for column in range(dimension):
        source = (column >> qubit) & 1
        for target_bit in (0, 1):
            row = (column & ~(1 << qubit)) | (target_bit << qubit)
            result[row, column] = matrix[target_bit, source]
    return result


def gate_matrix(n: int, gate: Gate) -> np.ndarray:
    if gate.name == "CNOT":
        control, target = gate.qubits
        dimension = 1 << n
        result = np.zeros((dimension, dimension), dtype=np.complex128)
        for column in range(dimension):
            row = column ^ ((1 << target) if ((column >> control) & 1) else 0)
            result[row, column] = 1.0
        return result
    if gate.name == "H":
        local = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
    elif gate.name == "S":
        local = np.diag([1.0, 1j]).astype(np.complex128)
    elif gate.name == "SDG":
        local = np.diag([1.0, -1j]).astype(np.complex128)
    elif gate.name == "T":
        local = np.diag([1.0, np.exp(1j * np.pi / 4)]).astype(np.complex128)
    elif gate.name == "TDG":
        local = np.diag([1.0, np.exp(-1j * np.pi / 4)]).astype(np.complex128)
    else:
        raise ValueError(f"unsupported gate {gate.name!r}")
    return _single_qubit_matrix(n, local, gate.qubits[0])


def unitary_from_gates(n: int, gates: Iterable[Gate]) -> np.ndarray:
    result = np.eye(1 << n, dtype=np.complex128)
    for gate in gates:
        result = gate_matrix(n, gate) @ result
    return result


def equal_up_to_global_phase(
    candidate: np.ndarray, target: np.ndarray, tolerance: float = 1e-9
) -> tuple[bool, float]:
    overlap = np.vdot(target.ravel(), candidate.ravel())
    phase = 1.0 + 0j if abs(overlap) < tolerance else overlap / abs(overlap)
    error = float(np.max(np.abs(candidate - phase * target)))
    return error <= tolerance, error


def certify_state(
    target: "SynthesisTarget",
    state: HybridState,
    *,
    tolerance: float = 1e-9,
) -> CertificationResult:
    """Reconstruct from the persistent DAG and verify independently."""
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
    symbolic_match = state.canonical_key == target.canonical_key
    candidate = unitary_from_gates(target.num_qubits, dag.gates)
    dense_match, error = equal_up_to_global_phase(candidate, target.unitary, tolerance)
    return CertificationResult(
        success=bool(symbolic_match and replay_match and dense_match),
        symbolic_match=bool(symbolic_match),
        replay_match=bool(replay_match),
        dense_match=bool(dense_match),
        maximum_matrix_error=error,
        gate_count=state.gate_count,
        t_count=state.t_count,
        cnot_count=state.cnot_count,
        depth=state.depth,
        witness=tuple(gate.label() for gate in dag.gates),
    )


__all__ = [
    "CertificationResult",
    "certify_state",
    "equal_up_to_global_phase",
    "gate_matrix",
    "unitary_from_gates",
]
