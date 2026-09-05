"""Decomposition-guided exact QFT synthesis with clean ancillas.

The unrestricted native search remains available as an honest baseline.  This
module provides a separate structured path for targets whose algebraic QFT
factorization is known: it constructs Hadamard, controlled-phase, and final
bit-reversal blocks, lowers every block to the native Clifford+T grammar, and
independently certifies the resulting ancilla isometry.

For the controlled-T block required by QFT-3, a relative-phase AND compute
circuit is used together with its exact inverse.  The relative phases cancel in
``R; T(ancilla); R^dagger``, so the logical action is an exact controlled-T on
the clean-input subspace while requiring fewer native gates than two full
Toffoli decompositions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Iterable

import numpy as np

from .ancilla_certify import AncillaCertificationResult, certify_ancilla_state
from .ancilla_contract import AncillaContract, AncillaSynthesisTarget, PhaseMode
from .model import Gate, HybridState, INVERSE_GATE


@dataclass(frozen=True, slots=True)
class NativeMacro:
    """A named exact high-level operation lowered to native gates."""

    name: str
    gates: tuple[Gate, ...]

    @property
    def gate_count(self) -> int:
        return len(self.gates)

    @property
    def t_count(self) -> int:
        return sum(gate.is_non_clifford for gate in self.gates)

    @property
    def cnot_count(self) -> int:
        return sum(gate.is_two_qubit for gate in self.gates)


@dataclass(frozen=True, slots=True)
class QFTGuidedResult:
    """Result of exact decomposition-guided QFT generation."""

    method: str
    target: str
    success: bool
    certified: bool
    stop_reason: str
    wall_seconds: float
    cpu_seconds: float
    macro_count: int
    native_gate_count: int
    t_count: int
    cnot_count: int
    depth: int
    projective_isometry_error: float | None
    exact_isometry_error: float | None
    ancilla_leakage: float | None
    macros: tuple[str, ...]
    witness: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "target": self.target,
            "success": self.success,
            "certified": self.certified,
            "stop_reason": self.stop_reason,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "macro_count": self.macro_count,
            "native_gate_count": self.native_gate_count,
            "t_count": self.t_count,
            "cnot_count": self.cnot_count,
            "depth": self.depth,
            "projective_isometry_error": self.projective_isometry_error,
            "exact_isometry_error": self.exact_isometry_error,
            "ancilla_leakage": self.ancilla_leakage,
            "macros": list(self.macros),
            "witness": "; ".join(self.witness),
        }


def inverse_gate_sequence(gates: Iterable[Gate]) -> tuple[Gate, ...]:
    """Return the exact native inverse of a chronological gate sequence."""

    sequence = tuple(gates)
    return tuple(
        Gate(INVERSE_GATE[gate.name], gate.qubits) for gate in reversed(sequence)
    )


def relative_phase_and_compute(
    control0: int,
    control1: int,
    target: int,
) -> tuple[Gate, ...]:
    """Compute ``control0 AND control1`` into ``target`` up to relative phase.

    On computational basis inputs with a clean target this circuit maps
    ``|x,y,0>`` to ``exp(i phi(x,y)) |x,y,x AND y>``.  It is deliberately not
    advertised as a full Toffoli replacement.  Its input-dependent phase is
    harmless only when the circuit is paired with its exact inverse around a
    diagonal operation on the computed target.
    """

    if len({control0, control1, target}) != 3:
        raise ValueError("relative-phase AND operands must be distinct")
    return (
        Gate("H", (target,)),
        Gate("T", (target,)),
        Gate("CNOT", (control1, target)),
        Gate("TDG", (target,)),
        Gate("CNOT", (control0, target)),
        Gate("T", (target,)),
        Gate("CNOT", (control1, target)),
        Gate("TDG", (target,)),
        Gate("H", (target,)),
    )


def controlled_t_relative_phase_compute(
    control: int,
    target: int,
    clean_ancilla: int,
) -> tuple[Gate, ...]:
    """Exact controlled-T using one clean ancilla and phase-cancelled AND.

    If ``R`` is :func:`relative_phase_and_compute`, the sequence
    ``R; T(clean_ancilla); R^dagger`` applies ``exp(i*pi/4)`` exactly when the
    logical control and target are both one and returns the clean ancilla to
    ``|0>``.  Relative phases introduced by ``R`` cancel against ``R^dagger``.
    """

    compute = relative_phase_and_compute(control, target, clean_ancilla)
    return (
        *compute,
        Gate("T", (clean_ancilla,)),
        *inverse_gate_sequence(compute),
    )


def controlled_s_native(control: int, target: int) -> tuple[Gate, ...]:
    """Exact native decomposition of controlled-S."""

    if control == target:
        raise ValueError("controlled-S operands must be distinct")
    return (
        Gate("T", (control,)),
        Gate("T", (target,)),
        Gate("CNOT", (control, target)),
        Gate("TDG", (target,)),
        Gate("CNOT", (control, target)),
    )


def swap_native(left: int, right: int) -> tuple[Gate, ...]:
    """Exact CNOT-only SWAP decomposition."""

    if left == right:
        raise ValueError("SWAP operands must be distinct")
    return (
        Gate("CNOT", (left, right)),
        Gate("CNOT", (right, left)),
        Gate("CNOT", (left, right)),
    )


def qft_matrix(num_qubits: int) -> np.ndarray:
    """Return the conventional forward QFT matrix in little-endian ordering."""

    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
        raise TypeError("num_qubits must be an integer")
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    dimension = 1 << num_qubits
    rows = np.arange(dimension, dtype=np.int64)[:, None]
    columns = np.arange(dimension, dtype=np.int64)[None, :]
    return np.asarray(
        np.exp(2j * np.pi * rows * columns / dimension) / np.sqrt(dimension),
        dtype=np.complex128,
    )


def _matrix_error(
    candidate: np.ndarray,
    expected: np.ndarray,
    phase_mode: PhaseMode,
) -> float:
    actual = np.asarray(candidate, dtype=np.complex128)
    target = np.asarray(expected, dtype=np.complex128)
    if actual.shape != target.shape:
        return float("inf")
    if phase_mode is PhaseMode.EXACT:
        aligned = target
    else:
        overlap = np.vdot(target.ravel(), actual.ravel())
        phase = 1.0 + 0.0j if abs(overlap) == 0.0 else overlap / abs(overlap)
        aligned = phase * target
    return float(np.max(np.abs(actual - aligned)))


def exact_qft_macros(contract: AncillaContract) -> tuple[NativeMacro, ...]:
    """Build an exact QFT plan for one, two, or three logical qubits.

    QFT-3 contains controlled-S and controlled-T phases.  The latter requires
    one declared clean ancilla in this native construction.  Wider exact QFTs
    would require controlled phases below ``pi/4`` and are intentionally not
    represented as exact Clifford+T macros here.
    """

    logical = contract.logical_qubits
    width = len(logical)
    if width > 3:
        raise ValueError(
            "exact Clifford+T QFT macro generation currently supports at most "
            "three logical qubits"
        )
    clean_ancilla = contract.clean_ancillas[0] if contract.clean_ancillas else None
    if width == 3 and clean_ancilla is None:
        raise ValueError("QFT-3 controlled-T lowering requires one clean ancilla")

    macros: list[NativeMacro] = []
    for target_position in range(width - 1, -1, -1):
        target = logical[target_position]
        macros.append(NativeMacro(f"H(q{target})", (Gate("H", (target,)),)))
        for control_position in range(target_position - 1, -1, -1):
            control = logical[control_position]
            separation = target_position - control_position
            if separation == 1:
                macros.append(
                    NativeMacro(
                        f"CS(q{control}->q{target})",
                        controlled_s_native(control, target),
                    )
                )
            elif separation == 2:
                assert clean_ancilla is not None
                macros.append(
                    NativeMacro(
                        f"CT(q{control}->q{target};a{clean_ancilla})",
                        controlled_t_relative_phase_compute(
                            control, target, clean_ancilla
                        ),
                    )
                )
            else:
                raise AssertionError("unsupported QFT phase separation")

    for left_position in range(width // 2):
        right_position = width - 1 - left_position
        left = logical[left_position]
        right = logical[right_position]
        macros.append(
            NativeMacro(
                f"SWAP(q{left},q{right})",
                swap_native(left, right),
            )
        )
    return tuple(macros)


def synthesize_qft_decomposition_guided(
    target: AncillaSynthesisTarget,
    *,
    tolerance: float = 1e-9,
) -> QFTGuidedResult:
    """Generate and certify an exact QFT circuit from verified native macros.

    This function does not inspect or replay the target's hidden generator
    witness.  It recognizes the analytical logical QFT matrix, constructs the
    standard QFT block factorization from the declared logical-wire ordering,
    lowers those blocks to native gates, and certifies the resulting DAG against
    the target ancilla contract.
    """

    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    macros: tuple[NativeMacro, ...] = ()
    state = HybridState.identity(target.num_qubits, target.budget)
    certification: AncillaCertificationResult | None = None
    stop_reason = "not_qft"

    expected = qft_matrix(target.contract.num_logical_qubits)
    target_error = _matrix_error(
        target.logical_unitary,
        expected,
        target.contract.phase_mode,
    )
    if target_error > tolerance:
        return QFTGuidedResult(
            method="qft_exact_verified_macro_plan",
            target=target.name,
            success=False,
            certified=False,
            stop_reason=stop_reason,
            wall_seconds=time.perf_counter() - wall_start,
            cpu_seconds=time.process_time() - cpu_start,
            macro_count=0,
            native_gate_count=0,
            t_count=0,
            cnot_count=0,
            depth=0,
            projective_isometry_error=None,
            exact_isometry_error=None,
            ancilla_leakage=None,
            macros=(),
            witness=(),
        )

    try:
        macros = exact_qft_macros(target.contract)
    except ValueError:
        stop_reason = "unsupported_contract"
    else:
        stop_reason = "budget_infeasible"
        for macro in macros:
            for gate in macro.gates:
                child = state.apply(gate, partial_order_reduction=False)
                if child is None:
                    break
                state = child
            else:
                continue
            break
        else:
            certification = certify_ancilla_state(
                target,
                state,
                tolerance=tolerance,
            )
            stop_reason = "certified" if certification.success else "certification_failed"

    success = bool(certification and certification.success)
    return QFTGuidedResult(
        method="qft_exact_verified_macro_plan",
        target=target.name,
        success=success,
        certified=success,
        stop_reason=stop_reason,
        wall_seconds=time.perf_counter() - wall_start,
        cpu_seconds=time.process_time() - cpu_start,
        macro_count=len(macros),
        native_gate_count=state.gate_count,
        t_count=state.t_count,
        cnot_count=state.cnot_count,
        depth=state.depth,
        projective_isometry_error=(
            None if certification is None else certification.projective_isometry_error
        ),
        exact_isometry_error=(
            None if certification is None else certification.exact_isometry_error
        ),
        ancilla_leakage=(
            None if certification is None else certification.ancilla_leakage
        ),
        macros=tuple(macro.name for macro in macros),
        witness=() if certification is None else certification.witness,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/qft3-guided/result.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Imported lazily so this module's synthesis primitives remain independent
    # of benchmark witness construction.
    from .ancilla_benchmarks import qft3_clean_ancilla_target

    args = build_parser().parse_args(argv)
    result = synthesize_qft_decomposition_guided(qft3_clean_ancilla_target())
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.success else 1


__all__ = [
    "NativeMacro",
    "QFTGuidedResult",
    "controlled_s_native",
    "controlled_t_relative_phase_compute",
    "exact_qft_macros",
    "inverse_gate_sequence",
    "qft_matrix",
    "relative_phase_and_compute",
    "swap_native",
    "synthesize_qft_decomposition_guided",
]


if __name__ == "__main__":
    raise SystemExit(main())
