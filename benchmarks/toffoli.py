"""Independent, deterministic certification helpers for a known Toffoli-3 witness.

The analytical CCX oracle in this module is deliberately separate from the
native gate witness.  The oracle is constructed directly from the Boolean
mapping on computational-basis indices, while the candidate is always rebuilt
from the authoritative :class:`~circuit.dag.CircuitDAG` by the dense simulator.
This is a certification reference, not a search or RL benchmark.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType


TOFFOLI_NUM_QUBITS = 3
"""The fixed width of the Stage 2 benchmark."""

TOFFOLI_CONTROLS = (0, 1)
"""Controls of the benchmark CCX, under the repository LSB convention."""

TOFFOLI_TARGET_QUBIT = 2
"""Target of the benchmark CCX, under the repository LSB convention."""


# This is the ordered native Clifford+T witness specified for the benchmark.
# ``Gate`` is frozen and the outer tuple is immutable, so callers cannot
# accidentally alter the witness that the public builder replays.
KNOWN_TOFFOLI_GATES: tuple[Gate, ...] = (
    Gate(GateType.H, (2,)),
    Gate(GateType.T, (0,)),
    Gate(GateType.T, (1,)),
    Gate(GateType.T, (2,)),
    Gate(GateType.CNOT, (0, 1)),
    Gate(GateType.TDG, (1,)),
    Gate(GateType.CNOT, (0, 1)),
    Gate(GateType.CNOT, (0, 2)),
    Gate(GateType.TDG, (2,)),
    Gate(GateType.CNOT, (1, 2)),
    Gate(GateType.T, (2,)),
    Gate(GateType.CNOT, (0, 2)),
    Gate(GateType.TDG, (2,)),
    Gate(GateType.CNOT, (1, 2)),
    Gate(GateType.H, (2,)),
)

# ``ResourceBudget`` is intentionally the existing mutable project type.  The
# public builder below creates a fresh equivalent value, so it never exposes
# this module-level reference through a returned CircuitState.
KNOWN_TOFFOLI_BUDGET = ResourceBudget(
    max_t_count=7,
    max_two_qubit_count=6,
    max_gates=15,
    max_depth=12,
)


@dataclass(frozen=True, slots=True)
class ToffoliTruthTableRow:
    """One exhaustive computational-basis diagnostic row.

    ``expected_output_index`` follows the repository's convention that q0 is
    the least-significant bit of a basis integer.  The complex ``amplitude``
    is retained so callers can distinguish a correct classical truth table
    from a truth-table-preserving relative-phase impostor.
    """

    input_index: int
    expected_output_index: int
    # A compact alias retained in the serialisable row shape.  It is a real
    # frozen field (rather than a mutable convenience attribute), so consumers
    # get the same immutability guarantee whichever descriptive name they use.
    expected_output: int
    observed_output_index: int
    expected_output_probability: float
    maximum_off_target_probability: float
    amplitude: complex
    correct: bool


@dataclass(frozen=True, slots=True)
class ToffoliUnitaryDiagnostics:
    """Dense diagnostics reusable for a state or an analytical impostor."""

    global_phase_equivalent: bool
    max_phase_aligned_matrix_error: float
    process_fidelity: float
    truth_table_correct: bool
    column_phase_consistent: bool
    truth_table: tuple[ToffoliTruthTableRow, ...]


@dataclass(frozen=True, slots=True)
class ToffoliValidation:
    """Independent semantic and resource diagnostics for a candidate state."""

    exact_certified: bool
    global_phase_equivalent: bool
    max_phase_aligned_matrix_error: float
    process_fidelity: float
    truth_table_correct: bool
    column_phase_consistent: bool
    symbolic_agrees_with_dense: bool
    resources: Mapping[str, object]


def _validate_benchmark_layout(
    *,
    num_qubits: int,
    controls: tuple[int, int] | Sequence[int],
    target: int,
) -> tuple[tuple[int, int], int]:
    """Validate the fixed-width Toffoli labels and return normalized values."""

    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
        raise TypeError("num_qubits must be an integer")
    if num_qubits != TOFFOLI_NUM_QUBITS:
        raise ValueError(
            "the Toffoli reference benchmark supports exactly "
            f"{TOFFOLI_NUM_QUBITS} qubits"
        )
    if isinstance(controls, (str, bytes)):
        raise TypeError("controls must be a two-qubit sequence")
    try:
        normalized_controls = tuple(controls)
    except TypeError as exc:
        raise TypeError("controls must be a two-qubit sequence") from exc
    if len(normalized_controls) != 2:
        raise ValueError("Toffoli requires exactly two controls")

    labels = (*normalized_controls, target)
    for label in labels:
        if isinstance(label, bool) or not isinstance(label, int):
            raise TypeError(f"qubit labels must be integers, got {label!r}")
        if label < 0 or label >= num_qubits:
            raise ValueError(
                f"qubit label {label!r} is outside a {num_qubits}-qubit register"
            )
    if len(set(labels)) != len(labels):
        raise ValueError("Toffoli controls and target must be distinct")
    return (normalized_controls[0], normalized_controls[1]), target


def _expected_index_unchecked(
    index: int,
    *,
    controls: tuple[int, int],
    target: int,
) -> int:
    """Return the CCX output index after already validating all parameters."""

    control_mask = (1 << controls[0]) | (1 << controls[1])
    if (index & control_mask) == control_mask:
        return index ^ (1 << target)
    return index


def expected_toffoli_basis_index(
    index: int,
    *,
    num_qubits: int = TOFFOLI_NUM_QUBITS,
    controls: tuple[int, int] = TOFFOLI_CONTROLS,
    target: int = TOFFOLI_TARGET_QUBIT,
) -> int:
    """Return the analytical CCX output index for one computational input.

    Qubit 0 is the least-significant bit, hence with the default labels the
    only exchanged integer basis indices are 3 and 7.  This helper has no
    dependence on the native witness, dense simulator, or symbolic engine.
    """

    normalized_controls, normalized_target = _validate_benchmark_layout(
        num_qubits=num_qubits,
        controls=controls,
        target=target,
    )
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError("basis index must be an integer")
    dimension = 1 << num_qubits
    if index < 0 or index >= dimension:
        raise ValueError(
            f"basis index {index!r} is outside the [0, {dimension}) register range"
        )
    return _expected_index_unchecked(
        index,
        controls=normalized_controls,
        target=normalized_target,
    )


def toffoli_reference_unitary(
    *,
    num_qubits: int = TOFFOLI_NUM_QUBITS,
    controls: tuple[int, int] = TOFFOLI_CONTROLS,
    target: int = TOFFOLI_TARGET_QUBIT,
) -> np.ndarray:
    """Construct the CCX target analytically from its Boolean action.

    The implementation intentionally does *not* replay ``KNOWN_TOFFOLI_GATES``
    or call :func:`certification.simulator.unitary_from_gates`; doing either
    would make the certification oracle share a failure mode with its witness.
    """

    normalized_controls, normalized_target = _validate_benchmark_layout(
        num_qubits=num_qubits,
        controls=controls,
        target=target,
    )
    dimension = 1 << num_qubits
    unitary = np.zeros((dimension, dimension), dtype=np.complex128)
    for input_index in range(dimension):
        output_index = _expected_index_unchecked(
            input_index,
            controls=normalized_controls,
            target=normalized_target,
        )
        unitary[output_index, input_index] = 1.0

    identity = np.eye(dimension, dtype=np.complex128)
    is_unitary = np.allclose(unitary.conj().T @ unitary, identity, atol=0.0, rtol=0.0)
    is_permutation = bool(
        np.all((unitary == 0.0) | (unitary == 1.0))
        and np.all(np.count_nonzero(unitary, axis=0) == 1)
        and np.all(np.count_nonzero(unitary, axis=1) == 1)
    )
    if not is_unitary or not is_permutation:  # pragma: no cover - construction guard
        raise AssertionError("analytical Toffoli construction must be unitary permutation")

    # A new array is built on each call and marked read-only, preventing a
    # caller from accidentally changing the reference used in its own check.
    unitary.setflags(write=False)
    return unitary


def relative_phase_toffoli_impostor(
    *,
    phase_column: int = 3,
    num_qubits: int = TOFFOLI_NUM_QUBITS,
    controls: tuple[int, int] = TOFFOLI_CONTROLS,
    target: int = TOFFOLI_TARGET_QUBIT,
) -> np.ndarray:
    """Return an analytical CCX permutation with a -1 relative column phase.

    This is intentionally only an adversarial validation input; it is not
    labelled as a particular SDK's relative-phase Toffoli implementation.
    """

    _validate_benchmark_layout(
        num_qubits=num_qubits,
        controls=controls,
        target=target,
    )
    if isinstance(phase_column, bool) or not isinstance(phase_column, int):
        raise TypeError("phase_column must be an integer")
    dimension = 1 << num_qubits
    if phase_column < 0 or phase_column >= dimension:
        raise ValueError(f"phase_column must be in [0, {dimension})")
    impostor = np.array(
        toffoli_reference_unitary(
            num_qubits=num_qubits,
            controls=controls,
            target=target,
        ),
        copy=True,
    )
    impostor[:, phase_column] *= -1.0
    impostor.setflags(write=False)
    return impostor


def _copy_known_budget() -> ResourceBudget:
    """Return a fresh fixed budget without exposing the module constant."""

    return ResourceBudget(
        max_t_count=7,
        max_two_qubit_count=6,
        max_gates=15,
        max_depth=12,
    )


def build_known_toffoli_state() -> CircuitState:
    """Build the fixed witness only through public ``CircuitState`` calls."""

    state = CircuitState(CircuitDAG(TOFFOLI_NUM_QUBITS), _copy_known_budget())
    for gate_index, gate in enumerate(KNOWN_TOFFOLI_GATES, start=1):
        if not state.apply_gate(gate):
            raise RuntimeError(
                "known Toffoli gate was rejected by the public transition "
                f"at position {gate_index}: {gate!r}"
            )
        # Keep DAG integrity a per-prefix invariant, rather than a final-only
        # assertion that could obscure the rejecting transition.
        state.dag.validate()
    return state


def _coerce_gate_type(raw_gate_type: object) -> GateType:
    """Convert the compact public action representations to ``GateType``."""

    if isinstance(raw_gate_type, GateType):
        return raw_gate_type
    name = getattr(raw_gate_type, "name", raw_gate_type)
    if not isinstance(name, str):
        raise TypeError(f"unsupported action gate type {raw_gate_type!r}")
    try:
        return GateType[name.upper()]
    except KeyError as exc:
        raise ValueError(f"unsupported action gate type {name!r}") from exc


def gates_from_actions(actions: Sequence[object]) -> tuple[Gate, ...]:
    """Convert public search actions (or existing gates) to immutable gates.

    ``search.action.Action`` exposes ``gate_type`` and ``qubits``.  Supporting
    the older ``GateSpec`` shape as well keeps the benchmark useful for
    serialised action fixtures without introducing any search dependency.
    """

    if isinstance(actions, (str, bytes)):
        raise TypeError("actions must be a sequence of gate-like objects")

    converted: list[Gate] = []
    for index, action in enumerate(actions):
        if isinstance(action, Gate):
            converted.append(action)
            continue

        raw_gate_type = getattr(action, "gate_type", getattr(action, "gate", None))
        if raw_gate_type is None:
            raise TypeError(f"action at index {index} has no gate_type or gate")

        if hasattr(action, "qubits"):
            raw_qubits = getattr(action, "qubits")
        elif hasattr(action, "targets"):
            targets = tuple(getattr(action, "targets"))
            controls = getattr(action, "controls", None)
            raw_qubits = targets if controls is None else (*tuple(controls), *targets)
        else:
            raise TypeError(f"action at index {index} has no qubits or targets")

        if isinstance(raw_qubits, (str, bytes)):
            raise TypeError(f"action at index {index} has invalid qubits")
        try:
            qubits = tuple(raw_qubits)
        except TypeError as exc:
            raise TypeError(f"action at index {index} has non-iterable qubits") from exc
        converted.append(Gate(_coerce_gate_type(raw_gate_type), qubits))
    return tuple(converted)


def _gate_name(gate: object) -> str:
    """Read a normalized project gate name for JSON-ready diagnostics."""

    gate_type = getattr(gate, "gate_type", getattr(gate, "gate", None))
    name = getattr(gate_type, "name", gate_type)
    if not isinstance(name, str):
        raise TypeError(f"gate {gate!r} does not expose a supported gate type")
    return name.upper()


def _phase_aligned_matrix_error(candidate: np.ndarray, target: np.ndarray) -> float:
    """Return max entry error after removing the target-anchored global phase."""

    if candidate.shape != target.shape or candidate.ndim != 2:
        return float("inf")
    if not np.isfinite(candidate).all() or not np.isfinite(target).all():
        return float("inf")
    anchor = np.unravel_index(np.argmax(np.abs(target)), target.shape)
    denominator = target[anchor]
    if denominator == 0:
        return float("inf")
    ratio = candidate[anchor] / denominator
    if abs(ratio) == 0:
        return float("inf")
    phase = ratio / abs(ratio)
    return float(np.max(np.abs(candidate - phase * target)))


def _process_fidelity(candidate: np.ndarray, target: np.ndarray) -> float:
    """Compute the unitarily normalised process fidelity, or zero if malformed."""

    if candidate.shape != target.shape or candidate.ndim != 2:
        return 0.0
    if not np.isfinite(candidate).all() or not np.isfinite(target).all():
        return 0.0
    dimension = target.shape[0]
    value = abs(np.trace(target.conj().T @ candidate)) ** 2 / (dimension * dimension)
    # Candidate states are unitary, but trim insignificant floating-point
    # overshoot so the diagnostic retains the physical [0, 1] range.
    return float(min(1.0, max(0.0, value.real)))


def _truth_table_rows(
    candidate: np.ndarray,
    *,
    num_qubits: int,
    controls: tuple[int, int],
    target: int,
    matrix_atol: float,
    matrix_rtol: float,
) -> tuple[ToffoliTruthTableRow, ...]:
    """Materialise exhaustive basis diagnostics without phase quotienting."""

    dimension = 1 << num_qubits
    if candidate.shape != (dimension, dimension) or not np.isfinite(candidate).all():
        return ()

    rows: list[ToffoliTruthTableRow] = []
    for input_index in range(dimension):
        expected_output = _expected_index_unchecked(
            input_index,
            controls=controls,
            target=target,
        )
        column = candidate[:, input_index]
        probabilities = np.abs(column) ** 2
        expected_probability = float(probabilities[expected_output])
        off_target = np.delete(probabilities, expected_output)
        maximum_off_target_probability = float(np.max(off_target, initial=0.0))
        correct = bool(
            np.isclose(
                expected_probability,
                1.0,
                atol=matrix_atol,
                rtol=matrix_rtol,
            )
            and np.allclose(
                off_target,
                0.0,
                atol=matrix_atol,
                rtol=matrix_rtol,
            )
        )
        rows.append(
            ToffoliTruthTableRow(
                input_index=input_index,
                expected_output_index=expected_output,
                expected_output=expected_output,
                observed_output_index=int(np.argmax(probabilities)),
                expected_output_probability=expected_probability,
                maximum_off_target_probability=maximum_off_target_probability,
                amplitude=complex(column[expected_output]),
                correct=correct,
            )
        )
    return tuple(rows)


def _column_phases_are_consistent(
    rows: Sequence[ToffoliTruthTableRow],
    *,
    matrix_atol: float,
    matrix_rtol: float,
) -> bool:
    """Check that all expected basis amplitudes are one common phase."""

    if not rows or not all(row.correct for row in rows):
        return False
    amplitudes = np.asarray([row.amplitude for row in rows], dtype=np.complex128)
    if np.any(np.abs(amplitudes) <= matrix_atol):
        return False
    return bool(
        np.allclose(
            amplitudes,
            amplitudes[0],
            atol=matrix_atol,
            rtol=matrix_rtol,
        )
    )


def validate_toffoli_unitary(
    candidate: Any,
    *,
    num_qubits: int = TOFFOLI_NUM_QUBITS,
    controls: tuple[int, int] = TOFFOLI_CONTROLS,
    target: int = TOFFOLI_TARGET_QUBIT,
    matrix_atol: float = 1e-10,
    matrix_rtol: float = 1e-10,
) -> ToffoliUnitaryDiagnostics:
    """Validate a dense candidate against analytical CCX and phase diagnostics.

    This function is intentionally public so negative controls can demonstrate
    that a truth-table-preserving, relative-phase candidate is not an accepted
    exact Toffoli, without needing a synthetic ``CircuitState``.
    """

    # Keep the analytical oracle import-independent.  Dense equivalence is
    # deliberately brought in only by this consumer-side validation routine.
    from certification.simulator import equivalent_up_to_global_phase

    if matrix_atol < 0 or matrix_rtol < 0:
        raise ValueError("matrix_atol and matrix_rtol must be non-negative")
    normalized_controls, normalized_target = _validate_benchmark_layout(
        num_qubits=num_qubits,
        controls=controls,
        target=target,
    )
    target_unitary = toffoli_reference_unitary(
        num_qubits=num_qubits,
        controls=normalized_controls,
        target=normalized_target,
    )
    try:
        candidate_unitary = np.asarray(candidate, dtype=np.complex128)
    except (TypeError, ValueError):
        candidate_unitary = np.empty((0, 0), dtype=np.complex128)

    rows = _truth_table_rows(
        candidate_unitary,
        num_qubits=num_qubits,
        controls=normalized_controls,
        target=normalized_target,
        matrix_atol=matrix_atol,
        matrix_rtol=matrix_rtol,
    )
    return ToffoliUnitaryDiagnostics(
        global_phase_equivalent=equivalent_up_to_global_phase(
            candidate_unitary,
            target_unitary,
            atol=matrix_atol,
            rtol=matrix_rtol,
        ),
        max_phase_aligned_matrix_error=_phase_aligned_matrix_error(
            candidate_unitary,
            target_unitary,
        ),
        process_fidelity=_process_fidelity(candidate_unitary, target_unitary),
        truth_table_correct=len(rows) == (1 << num_qubits) and all(
            row.correct for row in rows
        ),
        column_phase_consistent=_column_phases_are_consistent(
            rows,
            matrix_atol=matrix_atol,
            matrix_rtol=matrix_rtol,
        ),
        truth_table=rows,
    )


def toffoli_resource_summary(state: CircuitState) -> dict[str, object]:
    """Return JSON-ready, deterministic resource diagnostics for a witness.

    Semantic success is deliberately *not* inferred here.  Resource accounting
    remains separately inspectable for a candidate that is semantically wrong,
    and semantic validation later records its own result independently.
    """

    dag = getattr(state, "dag", None)
    if dag is None or not hasattr(dag, "gates"):
        raise TypeError("state must expose an authoritative CircuitDAG")
    gates = tuple(dag.gates)
    gate_counts = {name: 0 for name in ("H", "S", "SDG", "T", "TDG", "X", "CNOT")}
    directed_cnot_counts = {
        f"{control}->{candidate_target}": 0
        for control in range(TOFFOLI_NUM_QUBITS)
        for candidate_target in range(TOFFOLI_NUM_QUBITS)
        if control != candidate_target
    }
    ordered_gates: list[dict[str, object]] = []
    for index, gate in enumerate(gates, start=1):
        name = _gate_name(gate)
        gate_counts[name] = gate_counts.get(name, 0) + 1
        qubits = tuple(getattr(gate, "qubits"))
        if name == "CNOT" and len(qubits) == 2:
            key = f"{qubits[0]}->{qubits[1]}"
            if key in directed_cnot_counts:
                directed_cnot_counts[key] += 1
        ordered_gates.append(
            {"index": index, "gate": name, "qubits": list(qubits)}
        )

    wire_depths = tuple(int(value) for value in getattr(state, "wire_depths", ()))
    expected_profile = {
        "num_gates": 15,
        "t_count": 7,
        "two_qubit_count": 6,
        "h_count": 2,
        "cnot_count": 6,
        "t_gate_count": 4,
        "tdg_gate_count": 3,
        "depth": 12,
        "wire_depths": (9, 11, 12),
    }
    actual_profile = {
        "num_gates": int(getattr(state, "num_gates")),
        "t_count": int(getattr(state, "t_count")),
        "two_qubit_count": int(getattr(state, "two_qubit_count")),
        "h_count": gate_counts.get("H", 0),
        "cnot_count": gate_counts.get("CNOT", 0),
        "t_gate_count": gate_counts.get("T", 0),
        "tdg_gate_count": gate_counts.get("TDG", 0),
        "depth": int(getattr(state, "depth")),
        "wire_depths": wire_depths,
    }
    resource_vector = list(state.resource_vector())
    continuation_interface = list(getattr(state, "continuation_interface", ()))
    dag_node_levels = [
        {
            "node_id": int(node_id),
            "level": int(dag.nodes[node_id].level),
            "gate": _gate_name(dag.nodes[node_id].gate),
            "qubits": list(dag.nodes[node_id].gate.qubits),
        }
        for node_id in dag.topological_order
    ]

    # All values are native JSON scalars, lists, and mappings.  The known
    # resource claims refer only to this fixed reference witness, never to a
    # lower-bound proof or a search result.
    return {
        **actual_profile,
        "wire_depths": list(wire_depths),
        "ancilla_count": 0,
        "continuation_model": "all-to-all, no-ancilla",
        "continuation_interface": continuation_interface,
        "gate_counts": gate_counts,
        "directed_cnot_counts": directed_cnot_counts,
        "ordered_gates": ordered_gates,
        "dag_node_levels": dag_node_levels,
        "resource_vector": resource_vector,
        "expected_fixed_witness_resources": {
            **expected_profile,
            "wire_depths": list(expected_profile["wire_depths"]),
        },
        "resource_accounting_correct": actual_profile == expected_profile,
        "matches_known_optimal_T_count": actual_profile["t_count"] == 7,
        "matches_known_optimal_CNOT_count": actual_profile["cnot_count"] == 6,
    }


def validate_exact_toffoli_state(
    state: CircuitState,
    *,
    controls: tuple[int, int] = TOFFOLI_CONTROLS,
    target: int = TOFFOLI_TARGET_QUBIT,
    matrix_atol: float = 1e-10,
    matrix_rtol: float = 1e-10,
    max_process_infidelity: float = 1e-12,
) -> ToffoliValidation:
    """Certify a candidate DAG against independent analytical CCX.

    The dense target is constructed analytically.  The candidate matrix is
    rebuilt from ``state.dag.gates``; the symbolic materialisation is then
    compared with that independent dense reconstruction as a separate check.
    Resource diagnostics are reported, but never substituted for semantics.
    """

    # These are intentionally local imports: the analytical oracle above is
    # never defined through, or coupled at import time to, a witness replay.
    from certification.base import CertStatus
    from certification.simulator import (
        SimulatorCertificationEngine,
        SynthesisTarget,
        equivalent_up_to_global_phase,
        unitary_from_gates,
    )

    if matrix_atol < 0 or matrix_rtol < 0:
        raise ValueError("matrix_atol and matrix_rtol must be non-negative")
    if (
        isinstance(max_process_infidelity, bool)
        or not isinstance(max_process_infidelity, (int, float))
        or not np.isfinite(max_process_infidelity)
        or max_process_infidelity < 0
    ):
        raise ValueError("max_process_infidelity must be a finite non-negative number")
    if not isinstance(state, CircuitState):
        raise TypeError("state must be a CircuitState")
    # The DAG is authoritative.  Refuse to certify a malformed witness rather
    # than silently using a traversal of a structure that violates its own
    # dependency invariants.
    state.dag.validate()

    normalized_controls, normalized_target = _validate_benchmark_layout(
        num_qubits=TOFFOLI_NUM_QUBITS,
        controls=controls,
        target=target,
    )
    target_unitary = toffoli_reference_unitary(
        controls=normalized_controls,
        target=normalized_target,
    )
    candidate_unitary = unitary_from_gates(state.dag.num_qubits, state.dag.gates)
    diagnostics = validate_toffoli_unitary(
        candidate_unitary,
        controls=normalized_controls,
        target=normalized_target,
        matrix_atol=matrix_atol,
        matrix_rtol=matrix_rtol,
    )
    certificate = SimulatorCertificationEngine(SynthesisTarget(target_unitary)).certify(state)
    symbolic_unitary = state.symbolic_unitary()
    symbolic_agrees_with_dense = equivalent_up_to_global_phase(
        symbolic_unitary,
        candidate_unitary,
        atol=matrix_atol,
        rtol=matrix_rtol,
    )

    resources = toffoli_resource_summary(state)
    # This aggregate remains a distinct result category within the report:
    # resource assertions above do not determine whether a candidate is
    # semantically accepted.  It records every required semantic layer with
    # the configured strict process-infidelity threshold for JSON consumers.
    resources["semantic_correct"] = bool(
        certificate.status is CertStatus.SUCCESS
        and diagnostics.global_phase_equivalent
        and diagnostics.process_fidelity >= 1.0 - max_process_infidelity
        and diagnostics.truth_table_correct
        and diagnostics.column_phase_consistent
        and symbolic_agrees_with_dense
    )
    resources["simulator_certification_status"] = certificate.status.name
    resources["matrix_atol"] = float(matrix_atol)
    resources["matrix_rtol"] = float(matrix_rtol)
    resources["configured_max_process_infidelity"] = float(max_process_infidelity)
    return ToffoliValidation(
        exact_certified=certificate.status is CertStatus.SUCCESS,
        global_phase_equivalent=diagnostics.global_phase_equivalent,
        max_phase_aligned_matrix_error=diagnostics.max_phase_aligned_matrix_error,
        process_fidelity=diagnostics.process_fidelity,
        truth_table_correct=diagnostics.truth_table_correct,
        column_phase_consistent=diagnostics.column_phase_consistent,
        symbolic_agrees_with_dense=symbolic_agrees_with_dense,
        resources=resources,
    )


__all__ = [
    "KNOWN_TOFFOLI_BUDGET",
    "KNOWN_TOFFOLI_GATES",
    "TOFFOLI_CONTROLS",
    "TOFFOLI_NUM_QUBITS",
    "TOFFOLI_TARGET_QUBIT",
    "ToffoliTruthTableRow",
    "ToffoliUnitaryDiagnostics",
    "ToffoliValidation",
    "build_known_toffoli_state",
    "expected_toffoli_basis_index",
    "gates_from_actions",
    "relative_phase_toffoli_impostor",
    "toffoli_reference_unitary",
    "toffoli_resource_summary",
    "validate_exact_toffoli_state",
    "validate_toffoli_unitary",
]
