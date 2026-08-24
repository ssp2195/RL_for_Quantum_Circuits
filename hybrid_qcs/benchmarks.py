"""Small exact Clifford+T targets generated from hidden reachable witnesses."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

import numpy as np

from .certify import equal_up_to_global_phase, unitary_from_gates
from .model import Budget, Gate, HybridState


@dataclass(frozen=True, slots=True)
class SynthesisTarget:
    """Search-facing target data with no exposed generator gate sequence."""

    name: str
    split: str
    num_qubits: int
    budget: Budget
    canonical_key: tuple[object, ...]
    tableau_payload: tuple[tuple[int, int, int], ...]
    rotation_payloads: tuple[tuple[int, int, int], ...]
    unitary: np.ndarray
    generator_length: int
    target_digest: str
    family: str = "unrestricted-native"
    convention: str = "q0 is the least-significant basis bit"


def _target_from_hidden_gates(
    name: str,
    split: str,
    num_qubits: int,
    gates: Iterable[Gate],
    *,
    gate_slack: int = 0,
    depth_slack: int = 0,
    expected_unitary: np.ndarray | None = None,
    family: str = "unrestricted-native",
    convention: str = "q0 is the least-significant basis bit",
) -> SynthesisTarget:
    """Build an exact target while keeping the reachability witness private.

    ``expected_unitary`` is constructed independently for named analytical
    benchmarks such as QFT-2 and Toffoli.  The hidden native witness must agree
    with it up to global phase before the target is admitted.
    """

    hidden = tuple(gates)
    t_count = sum(gate.is_non_clifford for gate in hidden)
    cnot_count = sum(gate.is_two_qubit for gate in hidden)
    provisional = Budget(t_count, cnot_count, len(hidden), max(1, len(hidden)))
    state = HybridState.identity(num_qubits, provisional)
    for gate in hidden:
        child = state.apply(gate, partial_order_reduction=False)
        if child is None:
            raise ValueError(f"hidden target {name!r} violates its provisional budget")
        state = child
    budget = Budget(
        t_count,
        cnot_count,
        len(hidden) + gate_slack,
        max(1, state.depth + depth_slack),
    )

    # Replay under the final budget so all target metadata comes from the same
    # exact transition boundary used by synthesis.
    final_state = HybridState.identity(num_qubits, budget)
    for gate in hidden:
        child = final_state.apply(gate, partial_order_reduction=False)
        if child is None:
            raise AssertionError("final target budget cannot replay its hidden witness")
        final_state = child

    witness_unitary = unitary_from_gates(num_qubits, hidden)
    if expected_unitary is None:
        unitary = witness_unitary
    else:
        expected = np.asarray(expected_unitary, dtype=np.complex128)
        if expected.shape != witness_unitary.shape:
            raise ValueError("analytical target has the wrong matrix dimension")
        matches, error = equal_up_to_global_phase(witness_unitary, expected, 1e-9)
        if not matches:
            raise ValueError(
                f"hidden witness for {name!r} disagrees with the analytical target; "
                f"maximum phase-aligned error={error:.3e}"
            )
        unitary = np.array(expected, dtype=np.complex128, copy=True)

    unitary.setflags(write=False)
    digest = sha256(
        repr(
            (
                name,
                split,
                num_qubits,
                final_state.canonical_key,
                budget,
                family,
                convention,
                unitary.tobytes(),
            )
        ).encode("utf-8")
    ).hexdigest()
    return SynthesisTarget(
        name=name,
        split=split,
        num_qubits=num_qubits,
        budget=budget,
        canonical_key=final_state.canonical_key,
        tableau_payload=final_state.tableau.canonical_payload(),
        rotation_payloads=tuple(
            rotation.canonical_payload() for rotation in final_state.rotations
        ),
        unitary=unitary,
        generator_length=len(hidden),
        target_digest=f"sha256:{digest}",
        family=family,
        convention=convention,
    )


def _controlled_s(control: int, target: int) -> tuple[Gate, ...]:
    """Exact native decomposition of diag(1,1,1,i)."""

    return (
        Gate("T", (control,)),
        Gate("T", (target,)),
        Gate("CNOT", (control, target)),
        Gate("TDG", (target,)),
        Gate("CNOT", (control, target)),
    )


def _swap_network(left: int, right: int) -> tuple[Gate, ...]:
    """Exact native three-CNOT SWAP decomposition."""

    return (
        Gate("CNOT", (left, right)),
        Gate("CNOT", (right, left)),
        Gate("CNOT", (left, right)),
    )


def qft2_matrix(*, include_output_swap: bool = False) -> np.ndarray:
    """Return the exact forward two-qubit QFT under the repository convention.

    With ``include_output_swap=True`` this function returns the conventional
    matrix ``F[j,k]=exp(+2*pi*i*j*k/4)/2`` used by the held-out benchmark.
    The false branch remains available as the exact no-final-SWAP,
    bit-reversed-output convention for diagnostics.
    """

    dimension = 4
    row = np.arange(dimension, dtype=np.int64)[:, None]
    column = np.arange(dimension, dtype=np.int64)[None, :]
    standard = np.exp(2j * np.pi * row * column / dimension) / 2.0
    if include_output_swap:
        return np.asarray(standard, dtype=np.complex128)
    swap = np.zeros((dimension, dimension), dtype=np.complex128)
    for column_index in range(dimension):
        reversed_index = ((column_index & 1) << 1) | ((column_index >> 1) & 1)
        swap[reversed_index, column_index] = 1.0
    return swap @ standard


def qft2_target() -> SynthesisTarget:
    """Conventional exact QFT-2 target over the unrestricted native grammar."""

    hidden = (
        Gate("H", (1,)),
        *_controlled_s(0, 1),
        Gate("H", (0,)),
        *_swap_network(0, 1),
    )
    return _target_from_hidden_gates(
        "heldout-qft2-exact",
        "test",
        2,
        hidden,
        expected_unitary=qft2_matrix(include_output_swap=True),
        family="unrestricted-native-qft2",
        convention=(
            "conventional forward exact QFT-2 including a final native SWAP; "
            "q0 is the least-significant basis bit"
        ),
    )


def toffoli_matrix() -> np.ndarray:
    """Analytical CCX matrix with controls q0,q1 and target q2."""

    result = np.zeros((8, 8), dtype=np.complex128)
    for column in range(8):
        controls_active = ((column >> 0) & 1) and ((column >> 1) & 1)
        row = column ^ ((1 << 2) if controls_active else 0)
        result[row, column] = 1.0
    return result


def structured_toffoli_target() -> SynthesisTarget:
    """Analytical Toffoli target for the separate parity-network stress test."""

    hidden = (
        Gate("H", (2,)),
        Gate("T", (1,)),
        Gate("T", (2,)),
        Gate("T", (0,)),
        Gate("CNOT", (2, 0)),
        Gate("TDG", (0,)),
        Gate("CNOT", (2, 1)),
        Gate("TDG", (1,)),
        Gate("CNOT", (1, 0)),
        Gate("TDG", (0,)),
        Gate("CNOT", (2, 0)),
        Gate("T", (0,)),
        Gate("CNOT", (1, 0)),
        Gate("CNOT", (2, 1)),
        Gate("H", (2,)),
    )
    return _target_from_hidden_gates(
        "stress-toffoli3-structured-parity-network",
        "stress",
        3,
        hidden,
        depth_slack=3,
        expected_unitary=toffoli_matrix(),
        family="structured-toffoli-parity-network",
        convention="CCX controls q0,q1; target q2; q0 is LSB",
    )


def training_targets() -> tuple[SynthesisTarget, ...]:
    """Curriculum with genuine Hadamard/non-Clifford axis transport."""

    return (
        _target_from_hidden_gates(
            "train-h-t-x-axis",
            "train",
            1,
            (Gate("H", (0,)), Gate("T", (0,))),
        ),
        _target_from_hidden_gates(
            "train-t-h-t",
            "train",
            1,
            (Gate("T", (0,)), Gate("H", (0,)), Gate("T", (0,))),
        ),
        _target_from_hidden_gates(
            "train-entangle-phase",
            "train",
            2,
            (Gate("H", (0,)), Gate("CNOT", (0, 1)), Gate("T", (1,))),
        ),
        _target_from_hidden_gates(
            "train-two-axes",
            "train",
            2,
            (
                Gate("H", (1,)),
                Gate("T", (1,)),
                Gate("CNOT", (1, 0)),
                Gate("TDG", (0,)),
            ),
        ),
        _target_from_hidden_gates(
            "train-three-qubit-short",
            "train",
            3,
            (
                Gate("H", (0,)),
                Gate("CNOT", (0, 1)),
                Gate("T", (1,)),
                Gate("CNOT", (1, 2)),
            ),
        ),
        _target_from_hidden_gates(
            "train-three-qubit-mixed",
            "train",
            3,
            (
                Gate("H", (2,)),
                Gate("T", (2,)),
                Gate("CNOT", (2, 1)),
                Gate("S", (0,)),
            ),
        ),
        _target_from_hidden_gates(
            "train-controlled-s-core",
            "train",
            2,
            _controlled_s(0, 1),
        ),
        _target_from_hidden_gates(
            "train-h-controlled-s-prefix",
            "train",
            2,
            (Gate("H", (1,)), *_controlled_s(0, 1)),
        ),
        _target_from_hidden_gates(
            "train-swap-network",
            "train",
            2,
            _swap_network(0, 1),
        ),
        _target_from_hidden_gates(
            "train-qft2-substructure-without-final-h",
            "train",
            2,
            (Gate("H", (1,)), *_controlled_s(0, 1), *_swap_network(0, 1)),
        ),
    )


def validation_targets() -> tuple[SynthesisTarget, ...]:
    return (
        _target_from_hidden_gates(
            "validation-h-t-h",
            "validation",
            1,
            (Gate("H", (0,)), Gate("T", (0,)), Gate("H", (0,))),
        ),
        _target_from_hidden_gates(
            "validation-directed",
            "validation",
            2,
            (
                Gate("T", (1,)),
                Gate("H", (0,)),
                Gate("CNOT", (0, 1)),
                Gate("T", (0,)),
            ),
        ),
    )


def held_out_targets() -> tuple[SynthesisTarget, ...]:
    return (
        _target_from_hidden_gates(
            "heldout-anticommuting-word",
            "test",
            1,
            (
                Gate("H", (0,)),
                Gate("T", (0,)),
                Gate("H", (0,)),
                Gate("T", (0,)),
            ),
        ),
        _target_from_hidden_gates(
            "heldout-entangled-two-rotation",
            "test",
            2,
            (
                Gate("H", (0,)),
                Gate("T", (0,)),
                Gate("CNOT", (0, 1)),
                Gate("TDG", (1,)),
                Gate("H", (1,)),
            ),
        ),
        _target_from_hidden_gates(
            "heldout-three-qubit",
            "test",
            3,
            (
                Gate("H", (0,)),
                Gate("CNOT", (0, 1)),
                Gate("T", (1,)),
                Gate("CNOT", (1, 2)),
                Gate("H", (2,)),
            ),
        ),
        _target_from_hidden_gates(
            "heldout-s-phase-interaction",
            "test",
            2,
            (
                Gate("H", (1,)),
                Gate("S", (1,)),
                Gate("CNOT", (1, 0)),
                Gate("T", (0,)),
            ),
        ),
        qft2_target(),
    )


def all_targets() -> tuple[SynthesisTarget, ...]:
    """Return identity-disjoint unrestricted train/validation/test targets."""

    targets = (*training_targets(), *validation_targets(), *held_out_targets())
    keys = [target.canonical_key for target in targets]
    if len(keys) != len(set(keys)):
        raise AssertionError("target partitions contain a canonical duplicate")
    for index, left in enumerate(targets):
        for right in targets[index + 1 :]:
            if left.num_qubits != right.num_qubits:
                continue
            equivalent, _ = equal_up_to_global_phase(
                left.unitary, right.unitary, tolerance=1e-9
            )
            if equivalent:
                raise AssertionError(
                    "target partitions contain a projective-unitary duplicate: "
                    f"{left.name!r} and {right.name!r}"
                )
    return targets


def structured_stress_targets() -> tuple[SynthesisTarget, ...]:
    return (structured_toffoli_target(),)


__all__ = [
    "SynthesisTarget",
    "all_targets",
    "held_out_targets",
    "qft2_matrix",
    "qft2_target",
    "structured_stress_targets",
    "structured_toffoli_target",
    "toffoli_matrix",
    "training_targets",
    "validation_targets",
]
