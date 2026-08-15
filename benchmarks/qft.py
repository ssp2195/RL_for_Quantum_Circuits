"""SDK-neutral QFT references and an explicit native capability boundary.

This module does not extend :mod:`enums`, :mod:`search.action_space`, or the
native Clifford+T grammar.  ``ControlledPhase`` and ``SWAP`` are reference-only
operations used to define and score exact/approximate QFT targets.

Conventions
-----------
* q0 is the least-significant bit of a computational-basis integer.
* ``analytical_qft_matrix`` uses ``F[j,k] = exp(+2*pi*i*j*k/N)/sqrt(N)``.
* The forward circuit with swaps implements that matrix exactly.
* Without swaps, forward outputs are bit-reversed.  The inverse of that
  convention has bit-reversed inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import math
from typing import TypeAlias

import numpy as np

from certification.simulator import SynthesisTarget


QFT_REFERENCE_SCHEMA = "sdk-neutral-qft-reference-v1"
QUBIT_ORDER = "little-endian; q0 is the least-significant basis bit"


def _validate_qubit(qubit: int, *, name: str = "qubit") -> int:
    if isinstance(qubit, bool) or not isinstance(qubit, int) or qubit < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return qubit


@dataclass(frozen=True, slots=True)
class H:
    """Reference-only Hadamard operation."""

    qubit: int

    def __post_init__(self) -> None:
        _validate_qubit(self.qubit)


@dataclass(frozen=True, slots=True)
class ControlledPhase:
    """Reference-only controlled phase with an exact rational multiple of pi."""

    angle_pi: Fraction
    control: int
    target: int

    def __post_init__(self) -> None:
        try:
            angle_pi = Fraction(self.angle_pi)
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise TypeError("angle_pi must be an exact rational number") from exc
        control = _validate_qubit(self.control, name="control")
        target = _validate_qubit(self.target, name="target")
        if control == target:
            raise ValueError("controlled phase requires distinct qubits")
        object.__setattr__(self, "angle_pi", angle_pi)

    @property
    def angle(self) -> float:
        """The angle in radians, for numerical reference simulation only."""

        return float(self.angle_pi) * math.pi


@dataclass(frozen=True, slots=True)
class SWAP:
    """Reference-only swap operation."""

    qubit_a: int
    qubit_b: int

    def __post_init__(self) -> None:
        qubit_a = _validate_qubit(self.qubit_a, name="qubit_a")
        qubit_b = _validate_qubit(self.qubit_b, name="qubit_b")
        if qubit_a == qubit_b:
            raise ValueError("SWAP requires distinct qubits")


ReferenceOperation: TypeAlias = H | ControlledPhase | SWAP


def _operation_payload(operation: ReferenceOperation) -> dict[str, object]:
    if isinstance(operation, H):
        return {"operation": "H", "qubits": [operation.qubit]}
    if isinstance(operation, ControlledPhase):
        return {
            "operation": "ControlledPhase",
            "qubits": [operation.control, operation.target],
            "angle_pi": str(operation.angle_pi),
            "angle_radians": operation.angle,
        }
    if isinstance(operation, SWAP):
        return {
            "operation": "SWAP",
            "qubits": [operation.qubit_a, operation.qubit_b],
        }
    raise TypeError(f"unsupported reference operation {operation!r}")


@dataclass(frozen=True, slots=True)
class QFTReference:
    """A declared exact or approximate QFT operation sequence."""

    num_qubits: int
    operations: tuple[ReferenceOperation, ...]
    direction: str
    include_final_swaps: bool
    mode: str
    omitted_operations: tuple[ControlledPhase, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.num_qubits, bool)
            or not isinstance(self.num_qubits, int)
            or self.num_qubits < 1
        ):
            raise ValueError("num_qubits must be a positive integer")
        if self.direction not in {"forward", "inverse"}:
            raise ValueError("direction must be 'forward' or 'inverse'")
        if self.mode not in {"exact", "approximate"}:
            raise ValueError("mode must be 'exact' or 'approximate'")
        if not isinstance(self.include_final_swaps, bool):
            raise TypeError("include_final_swaps must be a bool")
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "omitted_operations", tuple(self.omitted_operations))

        for operation in (*self.operations, *self.omitted_operations):
            qubits = _operation_qubits(operation)
            if any(qubit >= self.num_qubits for qubit in qubits):
                raise ValueError("reference operation exceeds the declared register")
        if self.mode == "exact" and self.omitted_operations:
            raise ValueError("an exact QFT reference cannot omit operations")
        if self.mode == "approximate" and not self.omitted_operations:
            raise ValueError("an approximate QFT reference must declare omissions")

    @property
    def inverse(self) -> bool:
        return self.direction == "inverse"

    @property
    def permutation_convention(self) -> str:
        if self.include_final_swaps:
            return "identity"
        return "bit-reversed-input" if self.inverse else "bit-reversed-output"

    @property
    def output_qubit_order(self) -> tuple[int, ...]:
        if self.include_final_swaps or self.inverse:
            return tuple(range(self.num_qubits))
        return tuple(reversed(range(self.num_qubits)))

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": QFT_REFERENCE_SCHEMA,
            "num_qubits": self.num_qubits,
            "direction": self.direction,
            "mode": self.mode,
            "include_final_swaps": self.include_final_swaps,
            "qubit_order": QUBIT_ORDER,
            "matrix_convention": "F[j,k] = exp(+2*pi*i*j*k/N)/sqrt(N)",
            "permutation_convention": self.permutation_convention,
            "output_qubit_order": list(self.output_qubit_order),
            "operations": [_operation_payload(op) for op in self.operations],
            "omitted_operations": [
                _operation_payload(op) for op in self.omitted_operations
            ],
        }


def _operation_qubits(operation: ReferenceOperation) -> tuple[int, ...]:
    if isinstance(operation, H):
        return (operation.qubit,)
    if isinstance(operation, ControlledPhase):
        return (operation.control, operation.target)
    if isinstance(operation, SWAP):
        return (operation.qubit_a, operation.qubit_b)
    raise TypeError(f"unsupported reference operation {operation!r}")


def _inverse_operation(operation: ReferenceOperation) -> ReferenceOperation:
    if isinstance(operation, ControlledPhase):
        return ControlledPhase(
            angle_pi=-operation.angle_pi,
            control=operation.control,
            target=operation.target,
        )
    return operation


def _forward_qft_operations(
    num_qubits: int,
    *,
    include_final_swaps: bool,
    omit_angle_magnitudes: frozenset[Fraction] = frozenset(),
) -> tuple[tuple[ReferenceOperation, ...], tuple[ControlledPhase, ...]]:
    operations: list[ReferenceOperation] = []
    omitted: list[ControlledPhase] = []
    for target in reversed(range(num_qubits)):
        operations.append(H(target))
        for control in reversed(range(target)):
            phase = ControlledPhase(
                angle_pi=Fraction(1, 2 ** (target - control)),
                control=target,
                target=control,
            )
            if abs(phase.angle_pi) in omit_angle_magnitudes:
                omitted.append(phase)
            else:
                operations.append(phase)
    if include_final_swaps:
        operations.extend(
            SWAP(qubit, num_qubits - qubit - 1)
            for qubit in range(num_qubits // 2)
        )
    return tuple(operations), tuple(omitted)


def qft_reference(
    num_qubits: int,
    *,
    inverse: bool = False,
    include_final_swaps: bool = True,
) -> QFTReference:
    """Build the exact high-level QFT operation list."""

    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer")
    operations, _ = _forward_qft_operations(
        num_qubits, include_final_swaps=include_final_swaps
    )
    if inverse:
        operations = tuple(_inverse_operation(op) for op in reversed(operations))
    return QFTReference(
        num_qubits=num_qubits,
        operations=operations,
        direction="inverse" if inverse else "forward",
        include_final_swaps=include_final_swaps,
        mode="exact",
    )


def aqft3_reference(
    *,
    inverse: bool = False,
    include_final_swaps: bool = True,
) -> QFTReference:
    """Build AQFT-3 by omitting only the smallest controlled phase (pi/4)."""

    operations, omitted = _forward_qft_operations(
        3,
        include_final_swaps=include_final_swaps,
        omit_angle_magnitudes=frozenset({Fraction(1, 4)}),
    )
    if inverse:
        operations = tuple(_inverse_operation(op) for op in reversed(operations))
        omitted = tuple(
            _inverse_operation(op) for op in omitted
        )  # type: ignore[assignment]
    return QFTReference(
        num_qubits=3,
        operations=operations,
        direction="inverse" if inverse else "forward",
        include_final_swaps=include_final_swaps,
        mode="approximate",
        omitted_operations=omitted,
    )


def analytical_qft_matrix(num_qubits: int, *, inverse: bool = False) -> np.ndarray:
    """Construct the independent analytical Fourier matrix for ``2**n``."""

    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer")
    dimension = 1 << num_qubits
    indices = np.arange(dimension, dtype=np.float64)
    sign = -1.0 if inverse else 1.0
    matrix = np.exp(
        sign * 2.0j * np.pi * np.outer(indices, indices) / dimension
    ) / np.sqrt(float(dimension))
    matrix.setflags(write=False)
    return matrix


def bit_reversal_permutation(num_qubits: int) -> tuple[int, ...]:
    """Map each basis index to the index obtained by reversing its n bits."""

    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer")
    return tuple(
        int(f"{index:0{num_qubits}b}"[::-1], 2)
        for index in range(1 << num_qubits)
    )


def bit_reversal_matrix(num_qubits: int) -> np.ndarray:
    permutation = bit_reversal_permutation(num_qubits)
    matrix = np.zeros((len(permutation), len(permutation)), dtype=np.complex128)
    for input_index, output_index in enumerate(permutation):
        matrix[output_index, input_index] = 1.0
    matrix.setflags(write=False)
    return matrix


def declared_qft_target_matrix(reference: QFTReference) -> np.ndarray:
    """Return the exact target implied by the declared swap convention."""

    fourier = analytical_qft_matrix(reference.num_qubits, inverse=reference.inverse)
    if reference.include_final_swaps:
        return fourier
    reversal = bit_reversal_matrix(reference.num_qubits)
    matrix = fourier @ reversal if reference.inverse else reversal @ fourier
    matrix.setflags(write=False)
    return matrix


def _embedded_operation_matrix(
    operation: ReferenceOperation, num_qubits: int
) -> np.ndarray:
    dimension = 1 << num_qubits
    matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    if isinstance(operation, H):
        qubit = operation.qubit
        inverse_sqrt_two = 1.0 / np.sqrt(2.0)
        for input_index in range(dimension):
            input_bit = (input_index >> qubit) & 1
            zero_index = input_index & ~(1 << qubit)
            one_index = zero_index | (1 << qubit)
            matrix[zero_index, input_index] = inverse_sqrt_two
            matrix[one_index, input_index] = (
                inverse_sqrt_two if input_bit == 0 else -inverse_sqrt_two
            )
        return matrix
    if isinstance(operation, ControlledPhase):
        np.fill_diagonal(matrix, 1.0)
        phase = np.exp(1.0j * operation.angle)
        for basis_index in range(dimension):
            if (
                ((basis_index >> operation.control) & 1)
                and ((basis_index >> operation.target) & 1)
            ):
                matrix[basis_index, basis_index] = phase
        return matrix
    if isinstance(operation, SWAP):
        for input_index in range(dimension):
            bit_a = (input_index >> operation.qubit_a) & 1
            bit_b = (input_index >> operation.qubit_b) & 1
            output_index = input_index
            if bit_a != bit_b:
                output_index ^= (1 << operation.qubit_a) | (1 << operation.qubit_b)
            matrix[output_index, input_index] = 1.0
        return matrix
    raise TypeError(f"unsupported reference operation {operation!r}")


def reference_unitary(reference: QFTReference) -> np.ndarray:
    """Simulate the high-level reference independently of the native engine."""

    dimension = 1 << reference.num_qubits
    unitary = np.eye(dimension, dtype=np.complex128)
    for operation in reference.operations:
        unitary = _embedded_operation_matrix(operation, reference.num_qubits) @ unitary
    unitary.setflags(write=False)
    return unitary


class ExactCapability(str, Enum):
    EXACT_NATIVE = "EXACT_NATIVE"
    EXACT_DECOMPOSABLE = "EXACT_DECOMPOSABLE"
    APPROXIMATION_REQUIRED = "APPROXIMATION_REQUIRED"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True, slots=True)
class CapabilityIssue:
    operation_index: int
    operation: ControlledPhase
    reason_code: str


@dataclass(frozen=True, slots=True)
class NativeExactCapability:
    classification: ExactCapability
    reason_code: str
    model: str
    issues: tuple[CapabilityIssue, ...] = ()

    @property
    def exact_search_allowed(self) -> bool:
        return self.classification in {
            ExactCapability.EXACT_NATIVE,
            ExactCapability.EXACT_DECOMPOSABLE,
        }

    def metadata(self) -> dict[str, object]:
        return {
            "classification": self.classification.value,
            "reason_code": self.reason_code,
            "model": self.model,
            "exact_search_allowed": self.exact_search_allowed,
            "issues": [
                {
                    "operation_index": issue.operation_index,
                    "reason_code": issue.reason_code,
                    **_operation_payload(issue.operation),
                }
                for issue in self.issues
            ],
        }


NATIVE_EXACT_MODEL = (
    "no-ancilla all-to-all {H,S,SDG,T,TDG,directed-CNOT}; "
    "no parameterized reference operations"
)


def _controlled_phase_has_registered_exact_lowering(
    operation: ControlledPhase,
) -> bool:
    # Multiples of pi/2 have exact Clifford+T phase-polynomial lowerings.  The
    # present project registers no exact no-ancilla lowering below that lattice.
    return (operation.angle_pi * 2).denominator == 1


def assess_native_exact_capability(reference: QFTReference) -> NativeExactCapability:
    """Classify a reference before the current exact native search is invoked."""

    if reference.mode != "exact":
        return NativeExactCapability(
            classification=ExactCapability.UNSUPPORTED,
            reason_code="approximate_reference_requires_approximate_acceptance_api",
            model=NATIVE_EXACT_MODEL,
        )

    issues = tuple(
        CapabilityIssue(
            operation_index=index,
            operation=operation,
            reason_code="no_registered_exact_no_ancilla_lowering_below_pi_over_2",
        )
        for index, operation in enumerate(reference.operations)
        if isinstance(operation, ControlledPhase)
        and not _controlled_phase_has_registered_exact_lowering(operation)
    )
    if issues:
        return NativeExactCapability(
            classification=ExactCapability.APPROXIMATION_REQUIRED,
            reason_code="reference_contains_unlowered_controlled_phase_angles",
            model=NATIVE_EXACT_MODEL,
            issues=issues,
        )

    if any(isinstance(operation, (ControlledPhase, SWAP)) for operation in reference.operations):
        return NativeExactCapability(
            classification=ExactCapability.EXACT_DECOMPOSABLE,
            reason_code="all_reference_operations_have_registered_exact_native_identities",
            model=NATIVE_EXACT_MODEL,
        )
    return NativeExactCapability(
        classification=ExactCapability.EXACT_NATIVE,
        reason_code="reference_uses_only_native_operations",
        model=NATIVE_EXACT_MODEL,
    )


@dataclass(frozen=True, slots=True)
class NativeExactSearchRequest:
    """Guarded target handoff to the exact search layer."""

    capability: NativeExactCapability
    target: SynthesisTarget | None

    @property
    def accepted(self) -> bool:
        return self.target is not None and self.capability.exact_search_allowed


def prepare_native_exact_search(reference: QFTReference) -> NativeExactSearchRequest:
    """Return no target when exact native support has not been established."""

    capability = assess_native_exact_capability(reference)
    target = (
        SynthesisTarget(declared_qft_target_matrix(reference))
        if capability.exact_search_allowed
        else None
    )
    return NativeExactSearchRequest(capability=capability, target=target)


def process_fidelity(candidate: np.ndarray, target: np.ndarray) -> float:
    """Return the phase-insensitive unitary process fidelity."""

    candidate_array = np.asarray(candidate, dtype=np.complex128)
    target_array = np.asarray(target, dtype=np.complex128)
    if candidate_array.shape != target_array.shape or candidate_array.ndim != 2:
        raise ValueError("candidate and target must be same-shaped matrices")
    dimension = candidate_array.shape[0]
    value = abs(np.trace(target_array.conj().T @ candidate_array)) ** 2 / (
        dimension * dimension
    )
    return min(1.0, max(0.0, float(value.real)))


@dataclass(frozen=True, slots=True)
class StateFidelityMetric:
    label: str
    fidelity: float


@dataclass(frozen=True, slots=True)
class AQFTMetrics:
    reference: QFTReference
    process_fidelity: float
    maximum_matrix_error: float
    state_fidelities: tuple[StateFidelityMetric, ...]

    def metadata(self) -> dict[str, object]:
        return {
            "benchmark": "AQFT-3",
            "acceptance_mode": "approximate-metrics-only",
            "reference": self.reference.metadata(),
            "process_fidelity": self.process_fidelity,
            "maximum_matrix_error": self.maximum_matrix_error,
            "state_fidelities": {
                metric.label: metric.fidelity for metric in self.state_fidelities
            },
        }


def _selected_probe_states(num_qubits: int) -> tuple[tuple[str, np.ndarray], ...]:
    dimension = 1 << num_qubits
    basis_indices = tuple(dict.fromkeys((0, 1, dimension // 2 - 1, dimension - 1)))
    probes: list[tuple[str, np.ndarray]] = []
    for index in basis_indices:
        state = np.zeros(dimension, dtype=np.complex128)
        state[index] = 1.0
        probes.append((f"basis_{index:0{num_qubits}b}", state))
    uniform = np.ones(dimension, dtype=np.complex128) / np.sqrt(float(dimension))
    probes.append(("uniform_plus", uniform))
    phase_ramp = np.exp(2.0j * np.pi * np.arange(dimension) / dimension) / np.sqrt(
        float(dimension)
    )
    probes.append(("fourier_phase_ramp", phase_ramp))
    return tuple(probes)


def aqft3_metrics(
    *,
    inverse: bool = False,
    include_final_swaps: bool = True,
) -> AQFTMetrics:
    """Score the separately-labelled pi/4-omitting AQFT-3 target."""

    reference = aqft3_reference(
        inverse=inverse, include_final_swaps=include_final_swaps
    )
    approximate = reference_unitary(reference)
    exact = declared_qft_target_matrix(reference)
    state_metrics = []
    for label, initial_state in _selected_probe_states(3):
        expected = exact @ initial_state
        observed = approximate @ initial_state
        fidelity = abs(np.vdot(expected, observed)) ** 2
        state_metrics.append(
            StateFidelityMetric(
                label=label,
                fidelity=min(1.0, max(0.0, float(fidelity.real))),
            )
        )
    return AQFTMetrics(
        reference=reference,
        process_fidelity=process_fidelity(approximate, exact),
        maximum_matrix_error=float(np.max(np.abs(approximate - exact))),
        state_fidelities=tuple(state_metrics),
    )


__all__ = [
    "AQFTMetrics",
    "CapabilityIssue",
    "ControlledPhase",
    "ExactCapability",
    "H",
    "NATIVE_EXACT_MODEL",
    "NativeExactCapability",
    "NativeExactSearchRequest",
    "QFTReference",
    "QFT_REFERENCE_SCHEMA",
    "QUBIT_ORDER",
    "ReferenceOperation",
    "SWAP",
    "StateFidelityMetric",
    "analytical_qft_matrix",
    "aqft3_metrics",
    "aqft3_reference",
    "assess_native_exact_capability",
    "bit_reversal_matrix",
    "bit_reversal_permutation",
    "declared_qft_target_matrix",
    "prepare_native_exact_search",
    "process_fidelity",
    "qft_reference",
    "reference_unitary",
]
