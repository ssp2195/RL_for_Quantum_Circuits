"""Exact clean/borrowed-ancilla contracts for unitary Clifford+T synthesis.

The physical circuit remains a unitary on every wire.  Correctness is evaluated
on the permitted input subspace: clean ancillas are initialized in |0>, while
borrowed ancillas range over an arbitrary input state and must be returned
unchanged.  This is the isometry semantics used throughout the ancilla-aware
search and independent certifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from hashlib import sha256
from typing import Iterable

import numpy as np

from .certify import equal_up_to_global_phase, unitary_from_gates
from .model import Budget, Gate, HybridState


class PhaseMode(str, Enum):
    """Whether a synthesized block is compared projectively or exactly."""

    PROJECTIVE = "projective"
    EXACT = "exact"


@dataclass(frozen=True)
class AncillaContract:
    """Logical interface and workspace promise for a fixed physical register.

    Domain-basis bits are ordered as ``logical_qubits`` followed by
    ``borrowed_ancillas``.  Clean ancillas are inserted in |0>.  All physical
    wires must have exactly one role; output qubits that are part of the desired
    transformation belong in ``logical_qubits`` rather than ``clean_ancillas``.
    """

    total_qubits: int
    logical_qubits: tuple[int, ...]
    clean_ancillas: tuple[int, ...] = ()
    borrowed_ancillas: tuple[int, ...] = ()
    phase_mode: PhaseMode = PhaseMode.PROJECTIVE

    def __post_init__(self) -> None:
        if isinstance(self.total_qubits, bool) or not isinstance(self.total_qubits, int):
            raise TypeError("total_qubits must be an integer")
        if self.total_qubits <= 0:
            raise ValueError("total_qubits must be positive")
        object.__setattr__(self, "logical_qubits", tuple(self.logical_qubits))
        object.__setattr__(self, "clean_ancillas", tuple(self.clean_ancillas))
        object.__setattr__(self, "borrowed_ancillas", tuple(self.borrowed_ancillas))
        object.__setattr__(self, "phase_mode", PhaseMode(self.phase_mode))

        groups = (self.logical_qubits, self.clean_ancillas, self.borrowed_ancillas)
        flat = tuple(q for group in groups for q in group)
        if not self.logical_qubits:
            raise ValueError("at least one logical qubit is required")
        if any(isinstance(q, bool) or not isinstance(q, int) for q in flat):
            raise TypeError("qubit indices must be integers")
        if any(q < 0 or q >= self.total_qubits for q in flat):
            raise ValueError("a qubit role lies outside the physical register")
        if len(flat) != len(set(flat)):
            raise ValueError("logical, clean, and borrowed qubit roles must be disjoint")
        if set(flat) != set(range(self.total_qubits)):
            raise ValueError("every physical qubit must have exactly one declared role")

    @property
    def domain_qubits(self) -> tuple[int, ...]:
        """Free input wires: logical data followed by borrowed workspace."""

        return (*self.logical_qubits, *self.borrowed_ancillas)

    @property
    def num_logical_qubits(self) -> int:
        return len(self.logical_qubits)

    @property
    def num_clean_ancillas(self) -> int:
        return len(self.clean_ancillas)

    @property
    def num_borrowed_ancillas(self) -> int:
        return len(self.borrowed_ancillas)

    @property
    def physical_dimension(self) -> int:
        return 1 << self.total_qubits

    @property
    def logical_dimension(self) -> int:
        return 1 << self.num_logical_qubits

    @property
    def domain_dimension(self) -> int:
        return 1 << len(self.domain_qubits)

    @cached_property
    def input_embedding(self) -> np.ndarray:
        """Return J mapping free inputs to physical states with clean |0>s."""

        matrix = np.zeros(
            (self.physical_dimension, self.domain_dimension),
            dtype=np.complex128,
        )
        for domain_basis in range(self.domain_dimension):
            physical_basis = 0
            for position, physical_qubit in enumerate(self.domain_qubits):
                physical_basis |= ((domain_basis >> position) & 1) << physical_qubit
            matrix[physical_basis, domain_basis] = 1.0
        matrix.setflags(write=False)
        return matrix

    @cached_property
    def clean_subspace_projector(self) -> np.ndarray:
        embedding = self.input_embedding
        projector = embedding @ embedding.conj().T
        projector.setflags(write=False)
        return projector


    @cached_property
    def invalid_clean_output_rows(self) -> np.ndarray:
        """Rows in which at least one clean ancilla is |1>."""

        if not self.clean_ancillas:
            rows = np.empty(0, dtype=np.int64)
        else:
            clean_mask = sum(1 << q for q in self.clean_ancillas)
            rows = np.asarray(
                [basis for basis in range(self.physical_dimension) if basis & clean_mask],
                dtype=np.int64,
            )
        rows.setflags(write=False)
        return rows

    def domain_target_unitary(self, logical_unitary: np.ndarray) -> np.ndarray:
        """Extend U_logical by identity on every borrowed ancilla."""

        logical = np.asarray(logical_unitary, dtype=np.complex128)
        expected_shape = (self.logical_dimension, self.logical_dimension)
        if logical.shape != expected_shape:
            raise ValueError(
                f"logical target has shape {logical.shape}, expected {expected_shape}"
            )
        borrowed_dimension = 1 << self.num_borrowed_ancillas
        domain = np.zeros(
            (self.domain_dimension, self.domain_dimension), dtype=np.complex128
        )
        logical_mask = self.logical_dimension - 1
        for column in range(self.domain_dimension):
            logical_in = column & logical_mask
            borrowed = column >> self.num_logical_qubits
            for logical_out in range(self.logical_dimension):
                row = logical_out | (borrowed << self.num_logical_qubits)
                domain[row, column] = logical[logical_out, logical_in]
        if borrowed_dimension * self.logical_dimension != self.domain_dimension:
            raise AssertionError("domain-dimension factorization failed")
        return domain

    def target_isometry(self, logical_unitary: np.ndarray) -> np.ndarray:
        target = self.input_embedding @ self.domain_target_unitary(logical_unitary)
        target.setflags(write=False)
        return target

    def operand_role(self, qubit: int) -> str:
        if qubit in self.logical_qubits:
            return "logical"
        if qubit in self.clean_ancillas:
            return "clean"
        return "borrowed"

    def gate_role_class(self, gate: Gate) -> str:
        roles = tuple(self.operand_role(q) for q in gate.qubits)
        return f"{gate.name}:{'-'.join(roles)}"

    def canonical_payload(self) -> tuple[object, ...]:
        return (
            "ancilla-contract-v1",
            self.total_qubits,
            self.logical_qubits,
            self.clean_ancillas,
            self.borrowed_ancillas,
            self.phase_mode.value,
        )


@dataclass(frozen=True)
class AncillaSynthesisTarget:
    """Search-facing target with a logical-unitary/ancilla-return contract."""

    name: str
    split: str
    contract: AncillaContract
    budget: Budget
    logical_unitary: np.ndarray
    target_isometry: np.ndarray
    canonical_key: tuple[object, ...]
    tableau_payload: tuple[tuple[int, int, int], ...]
    rotation_payloads: tuple[tuple[int, int, int], ...]
    reference_unitary: np.ndarray
    generator_length: int
    target_digest: str
    family: str = "ancilla-clean-clifford-t"
    convention: str = "q0 is the least-significant basis bit"

    @property
    def num_qubits(self) -> int:
        return self.contract.total_qubits


@dataclass(frozen=True)
class ContractMetrics:
    projective_error: float
    exact_error: float
    leakage: float
    overlap: complex
    success: bool


def contract_metrics(
    candidate_isometry: np.ndarray,
    target: AncillaSynthesisTarget,
    *,
    tolerance: float = 1e-9,
) -> ContractMetrics:
    candidate = np.asarray(candidate_isometry, dtype=np.complex128)
    expected = target.target_isometry
    if candidate.shape != expected.shape:
        raise ValueError("candidate isometry has the wrong shape")
    overlap = np.vdot(expected.ravel(), candidate.ravel())
    phase = 1.0 + 0.0j if abs(overlap) < tolerance else overlap / abs(overlap)
    projective_error = float(np.max(np.abs(candidate - phase * expected)))
    exact_error = float(np.max(np.abs(candidate - expected)))
    invalid_rows = target.contract.invalid_clean_output_rows
    leakage = float(
        0.0
        if invalid_rows.size == 0
        else np.sum(np.abs(candidate[invalid_rows, :]) ** 2)
        / target.contract.domain_dimension
    )
    selected_error = (
        projective_error
        if target.contract.phase_mode is PhaseMode.PROJECTIVE
        else exact_error
    )
    return ContractMetrics(
        projective_error=projective_error,
        exact_error=exact_error,
        leakage=leakage,
        overlap=complex(overlap),
        success=bool(selected_error <= tolerance and leakage <= tolerance),
    )


def contract_distance_and_leakage(
    candidate_isometry: np.ndarray,
    target: AncillaSynthesisTarget,
) -> tuple[float, float]:
    """Cheap phase-invariant search distance plus clean-workspace leakage."""

    candidate = np.asarray(candidate_isometry, dtype=np.complex128)
    expected = target.target_isometry
    overlap = np.vdot(expected.ravel(), candidate.ravel())
    dimension = target.contract.domain_dimension
    distance = max(
        0.0,
        1.0 - float(abs(overlap) ** 2 / (dimension * dimension)),
    )
    invalid_rows = target.contract.invalid_clean_output_rows
    leakage = float(
        0.0
        if invalid_rows.size == 0
        else np.sum(np.abs(candidate[invalid_rows, :]) ** 2) / dimension
    )
    return distance, leakage


def _unitary_matches_contract(
    full_unitary: np.ndarray,
    contract: AncillaContract,
    logical_unitary: np.ndarray,
    *,
    tolerance: float = 1e-9,
) -> tuple[bool, float, float]:
    candidate = np.asarray(full_unitary, dtype=np.complex128) @ contract.input_embedding
    expected = contract.target_isometry(logical_unitary)
    if contract.phase_mode is PhaseMode.PROJECTIVE:
        matches, error = equal_up_to_global_phase(candidate, expected, tolerance)
    else:
        error = float(np.max(np.abs(candidate - expected)))
        matches = error <= tolerance
    invalid_rows = contract.invalid_clean_output_rows
    leakage = float(
        0.0
        if invalid_rows.size == 0
        else np.sum(np.abs(candidate[invalid_rows, :]) ** 2)
        / contract.domain_dimension
    )
    return bool(matches and leakage <= tolerance), error, leakage


def target_from_hidden_ancilla_gates(
    name: str,
    split: str,
    contract: AncillaContract,
    gates: Iterable[Gate],
    *,
    logical_unitary: np.ndarray | None = None,
    gate_slack: int = 0,
    depth_slack: int = 0,
    t_slack: int = 0,
    cnot_slack: int = 0,
    family: str = "ancilla-clean-clifford-t",
    convention: str = "q0 is the least-significant basis bit",
) -> AncillaSynthesisTarget:
    """Construct a contract target from a hidden exact native witness."""

    hidden = tuple(gates)
    t_count = sum(gate.is_non_clifford for gate in hidden)
    cnot_count = sum(gate.is_two_qubit for gate in hidden)
    provisional = Budget(
        t_count + t_slack,
        cnot_count + cnot_slack,
        len(hidden) + gate_slack,
        max(1, len(hidden) + depth_slack),
    )
    state = HybridState.identity(contract.total_qubits, provisional)
    for gate in hidden:
        child = state.apply(gate, partial_order_reduction=False)
        if child is None:
            raise ValueError(f"hidden target {name!r} violates its provisional budget")
        state = child

    budget = Budget(
        t_count + t_slack,
        cnot_count + cnot_slack,
        len(hidden) + gate_slack,
        max(1, state.depth + depth_slack),
    )
    final_state = HybridState.identity(contract.total_qubits, budget)
    for gate in hidden:
        child = final_state.apply(gate, partial_order_reduction=False)
        if child is None:
            raise AssertionError("final target budget cannot replay its hidden witness")
        final_state = child

    reference_unitary = unitary_from_gates(contract.total_qubits, hidden)
    candidate_isometry = reference_unitary @ contract.input_embedding
    clean_block = contract.input_embedding.conj().T @ candidate_isometry
    if logical_unitary is None:
        if contract.num_borrowed_ancillas:
            raise ValueError(
                "logical_unitary must be supplied when borrowed ancillas are present"
            )
        logical = np.asarray(clean_block, dtype=np.complex128)
    else:
        logical = np.asarray(logical_unitary, dtype=np.complex128)

    matches, error, leakage = _unitary_matches_contract(
        reference_unitary, contract, logical
    )
    if not matches:
        raise ValueError(
            f"hidden witness for {name!r} violates the ancilla contract; "
            f"error={error:.3e}, leakage={leakage:.3e}"
        )

    logical = np.array(logical, dtype=np.complex128, copy=True)
    target_isometry = contract.target_isometry(logical)
    reference_unitary = np.array(reference_unitary, dtype=np.complex128, copy=True)
    logical.setflags(write=False)
    target_isometry.setflags(write=False)
    reference_unitary.setflags(write=False)

    digest = sha256(
        repr(
            (
                name,
                split,
                contract.canonical_payload(),
                final_state.canonical_key,
                budget,
                family,
                logical.tobytes(),
                target_isometry.tobytes(),
            )
        ).encode("utf-8")
    ).hexdigest()
    return AncillaSynthesisTarget(
        name=name,
        split=split,
        contract=contract,
        budget=budget,
        logical_unitary=logical,
        target_isometry=target_isometry,
        canonical_key=final_state.canonical_key,
        tableau_payload=final_state.tableau.canonical_payload(),
        rotation_payloads=tuple(
            rotation.canonical_payload() for rotation in final_state.rotations
        ),
        reference_unitary=reference_unitary,
        generator_length=len(hidden),
        target_digest=f"sha256:{digest}",
        family=family,
        convention=convention,
    )


__all__ = [
    "AncillaContract",
    "AncillaSynthesisTarget",
    "ContractMetrics",
    "PhaseMode",
    "contract_distance_and_leakage",
    "contract_metrics",
    "target_from_hidden_ancilla_gates",
]
