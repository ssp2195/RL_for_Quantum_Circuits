"""Exact ancilla-aware Clifford+T targets and a certified QFT-3 witness."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .ancilla_contract import (
    AncillaContract,
    AncillaSynthesisTarget,
    PhaseMode,
    target_from_hidden_ancilla_gates,
)
from .model import Gate

ANCILLA_FAMILY = "clean-ancilla-clifford-t"


def clean_contract(num_logical: int, num_clean: int = 1, *, exact_phase: bool = False) -> AncillaContract:
    total = num_logical + num_clean
    return AncillaContract(
        total_qubits=total,
        logical_qubits=tuple(range(num_logical)),
        clean_ancillas=tuple(range(num_logical, total)),
        phase_mode=PhaseMode.EXACT if exact_phase else PhaseMode.PROJECTIVE,
    )


def parity_phase_witness(
    logical_qubits: Iterable[int],
    ancilla: int,
    phase_gate: str = "T",
) -> tuple[Gate, ...]:
    """Compute a parity into a clean ancilla, phase it, and uncompute."""

    logical = tuple(logical_qubits)
    compute = tuple(Gate("CNOT", (q, ancilla)) for q in logical)
    return (*compute, Gate(phase_gate, (ancilla,)), *reversed(compute))


def toffoli_decomposition(control0: int, control1: int, target: int) -> tuple[Gate, ...]:
    """Exact 15-gate Clifford+T decomposition of CCX."""

    if len({control0, control1, target}) != 3:
        raise ValueError("Toffoli operands must be distinct")
    return (
        Gate("H", (target,)),
        Gate("T", (control1,)),
        Gate("T", (target,)),
        Gate("T", (control0,)),
        Gate("CNOT", (target, control0)),
        Gate("TDG", (control0,)),
        Gate("CNOT", (target, control1)),
        Gate("TDG", (control1,)),
        Gate("CNOT", (control1, control0)),
        Gate("TDG", (control0,)),
        Gate("CNOT", (target, control0)),
        Gate("T", (control0,)),
        Gate("CNOT", (control1, control0)),
        Gate("CNOT", (target, control1)),
        Gate("H", (target,)),
    )


def controlled_s_decomposition(control: int, target: int) -> tuple[Gate, ...]:
    """Exact native decomposition of diag(1,1,1,i)."""

    return (
        Gate("T", (control,)),
        Gate("T", (target,)),
        Gate("CNOT", (control, target)),
        Gate("TDG", (target,)),
        Gate("CNOT", (control, target)),
    )


def controlled_t_with_clean_ancilla(
    control: int,
    target: int,
    ancilla: int,
) -> tuple[Gate, ...]:
    """Compute control AND target, apply T by phase kickback, and uncompute."""

    compute = toffoli_decomposition(control, target, ancilla)
    return (*compute, Gate("T", (ancilla,)), *compute)


def swap_decomposition(left: int, right: int) -> tuple[Gate, ...]:
    return (
        Gate("CNOT", (left, right)),
        Gate("CNOT", (right, left)),
        Gate("CNOT", (left, right)),
    )


def qft3_matrix() -> np.ndarray:
    dimension = 8
    rows = np.arange(dimension, dtype=np.int64)[:, None]
    columns = np.arange(dimension, dtype=np.int64)[None, :]
    return np.asarray(
        np.exp(2j * np.pi * rows * columns / dimension) / np.sqrt(dimension),
        dtype=np.complex128,
    )


def qft3_clean_ancilla_witness() -> tuple[Gate, ...]:
    """Exact QFT-3 with controlled-T realized through one clean ancilla."""

    ancilla = 3
    return (
        Gate("H", (2,)),
        *controlled_s_decomposition(1, 2),
        *controlled_t_with_clean_ancilla(0, 2, ancilla),
        Gate("H", (1,)),
        *controlled_s_decomposition(0, 1),
        Gate("H", (0,)),
        *swap_decomposition(0, 2),
    )


def qft3_clean_ancilla_target() -> AncillaSynthesisTarget:
    return target_from_hidden_ancilla_gates(
        "heldout-qft3-clean-ancilla",
        "stress",
        clean_contract(3, 1),
        qft3_clean_ancilla_witness(),
        logical_unitary=qft3_matrix(),
        family="clean-ancilla-qft3",
        convention=(
            "conventional forward exact QFT-3 including final q0<->q2 SWAP; "
            "q3 is a clean workspace qubit returned to |0>; q0 is LSB"
        ),
    )


def ancilla_training_targets() -> tuple[AncillaSynthesisTarget, ...]:
    """Short compute-phase-uncompute curriculum with one clean workspace."""

    specifications: tuple[tuple[str, AncillaContract, tuple[Gate, ...]], ...] = (
        (
            "train-clean-t-echo",
            clean_contract(1, 1),
            parity_phase_witness((0,), 1, "T"),
        ),
        (
            "train-clean-tdg-echo",
            clean_contract(1, 1),
            parity_phase_witness((0,), 1, "TDG"),
        ),
        (
            "train-clean-s-echo",
            clean_contract(1, 1),
            parity_phase_witness((0,), 1, "S"),
        ),
        (
            "train-parity-t-2q",
            clean_contract(2, 1),
            parity_phase_witness((0, 1), 2, "T"),
        ),
        (
            "train-parity-tdg-2q",
            clean_contract(2, 1),
            parity_phase_witness((0, 1), 2, "TDG"),
        ),
        (
            "train-h-parity-t-2q",
            clean_contract(2, 1),
            (
                Gate("H", (0,)),
                *parity_phase_witness((0, 1), 2, "T"),
                Gate("H", (0,)),
            ),
        ),
        (
            "train-s-parity-tdg-2q",
            clean_contract(2, 1),
            (
                Gate("S", (1,)),
                *parity_phase_witness((0, 1), 2, "TDG"),
            ),
        ),
        (
            "train-parity-t-3q",
            clean_contract(3, 1),
            parity_phase_witness((0, 1, 2), 3, "T"),
        ),
        (
            "train-mixed-clean-3q",
            clean_contract(3, 1),
            (
                Gate("H", (2,)),
                Gate("S", (0,)),
                *parity_phase_witness((0, 2), 3, "TDG"),
                Gate("CNOT", (1, 2)),
            ),
        ),
    )
    return tuple(
        target_from_hidden_ancilla_gates(
            name,
            "train",
            contract,
            gates,
            gate_slack=1,
            depth_slack=1,
            family=ANCILLA_FAMILY,
        )
        for name, contract, gates in specifications
    )


def ancilla_evaluation_targets() -> tuple[AncillaSynthesisTarget, ...]:
    """Held-out contract targets whose hidden witnesses use clean workspace.

    The synthesizer is free to return an ancilla-free realization when one is
    cheaper.  This is intentional: workspace is an available resource, not a
    mandatory syntactic decoration of the returned circuit.
    """

    return (
        target_from_hidden_ancilla_gates(
            "heldout-clean-h-t-echo-1q",
            "test",
            clean_contract(1, 1),
            (
                Gate("H", (0,)),
                *parity_phase_witness((0,), 1, "T"),
            ),
            gate_slack=1,
            depth_slack=1,
            family=ANCILLA_FAMILY,
        ),
        target_from_hidden_ancilla_gates(
            "heldout-clean-s-t-product-2q",
            "test",
            clean_contract(2, 1),
            (
                Gate("S", (1,)),
                *parity_phase_witness((0,), 2, "T"),
            ),
            gate_slack=1,
            depth_slack=1,
            family=ANCILLA_FAMILY,
        ),
        target_from_hidden_ancilla_gates(
            "heldout-clean-mixed-product-3q",
            "ood",
            clean_contract(3, 1),
            (
                Gate("H", (2,)),
                *parity_phase_witness((1,), 3, "TDG"),
                Gate("S", (0,)),
            ),
            gate_slack=1,
            depth_slack=1,
            family=ANCILLA_FAMILY,
        ),
    )



__all__ = [
    "ANCILLA_FAMILY",
    "ancilla_evaluation_targets",
    "ancilla_training_targets",
    "clean_contract",
    "controlled_s_decomposition",
    "controlled_t_with_clean_ancilla",
    "parity_phase_witness",
    "qft3_clean_ancilla_target",
    "qft3_clean_ancilla_witness",
    "qft3_matrix",
    "swap_decomposition",
    "toffoli_decomposition",
]
