"""Certify a fixed native Toffoli-3 witness and write deterministic evidence.

This is a reference-certification command, not a search or RL experiment.  It
builds the analytical CCX target independently, replays the known
Clifford+T witness through public ``CircuitState.apply_gate`` transitions, and
checks its authoritative DAG with the dense simulator and symbolic invariant.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from benchmarks.toffoli import (
    KNOWN_TOFFOLI_GATES,
    TOFFOLI_CONTROLS,
    TOFFOLI_NUM_QUBITS,
    TOFFOLI_TARGET_QUBIT,
    build_known_toffoli_state,
    expected_toffoli_basis_index,
    relative_phase_toffoli_impostor,
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
from enums import GateType
from reporting import save_toffoli_artifacts
from reporting.toffoli import stable_gate_witness_digest, stable_matrix_digest


MATRIX_ATOL = 1e-10
MATRIX_RTOL = 1e-10
MAX_PROCESS_INFIDELITY = 1e-12
_QUBIT_ORDER = "q0 is LSB; displayed bit strings are q2 q1 q0"


def _operation(gate: Gate, *, index: int) -> dict[str, object]:
    return {
        "index": index,
        "gate": gate.gate_type.name,
        "qubits": [int(qubit) for qubit in gate.qubits],
    }


def _bit_label(index: int) -> str:
    return format(index, f"0{TOFFOLI_NUM_QUBITS}b")


def _build_state(
    gates: Sequence[Gate],
    *,
    budget: ResourceBudget,
) -> CircuitState:
    """Replay a candidate only through public state/DAG transitions."""

    state = CircuitState(CircuitDAG(TOFFOLI_NUM_QUBITS), budget)
    for gate_index, gate in enumerate(gates, start=1):
        if not state.apply_gate(gate):
            raise RuntimeError(
                f"candidate gate {gate_index} was rejected by the public transition: "
                f"{gate!r}"
            )
        state.dag.validate()
    return state


def _permissive_budget(*, gate_count: int) -> ResourceBudget:
    """Return a finite public budget for deliberately incorrect controls."""

    return ResourceBudget(
        max_t_count=max(32, gate_count),
        max_two_qubit_count=max(32, gate_count),
        max_gates=max(64, gate_count),
        max_depth=max(64, gate_count),
    )


def _state_snapshot(state: CircuitState) -> tuple[object, ...]:
    """Capture every observable mutability target for a rejected continuation."""

    return (
        tuple(state.dag.gates),
        tuple(state.dag.topological_order),
        state.resource_vector(),
        state.depth,
        tuple(state.wire_depths),
        tuple(state.rotations),
        state.global_phase_eighths,
        tuple(state.continuation_interface),
    )


def _budget_boundary_result(name: str, budget: ResourceBudget) -> dict[str, object]:
    """Verify one fixed-witness budget failure is rejected atomically."""

    state = CircuitState(CircuitDAG(TOFFOLI_NUM_QUBITS), budget)
    for gate_index, gate in enumerate(KNOWN_TOFFOLI_GATES, start=1):
        before = _state_snapshot(state)
        accepted = state.apply_gate(gate)
        if not accepted:
            after = _state_snapshot(state)
            state.dag.validate()
            return {
                "passed": before == after,
                "resource": name,
                "rejected_gate_index": gate_index,
                "rejected_gate": repr(gate),
                "atomic_rejection": before == after,
                "state_num_gates_after_rejection": state.num_gates,
                "state_depth_after_rejection": state.depth,
            }
    # A supposed insufficient budget that accepts the complete witness is a
    # mandatory failure rather than a skipped validation.
    return {
        "passed": False,
        "resource": name,
        "reason": "insufficient budget unexpectedly accepted the full witness",
        "atomic_rejection": False,
    }


def _budget_boundary_controls() -> dict[str, dict[str, object]]:
    """Run all required fixed-witness resource-boundary controls."""

    return {
        "budget_t_count_6": _budget_boundary_result(
            "max_t_count",
            ResourceBudget(
                max_t_count=6,
                max_two_qubit_count=6,
                max_gates=15,
                max_depth=12,
            ),
        ),
        "budget_two_qubit_count_5": _budget_boundary_result(
            "max_two_qubit_count",
            ResourceBudget(
                max_t_count=7,
                max_two_qubit_count=5,
                max_gates=15,
                max_depth=12,
            ),
        ),
        "budget_gate_count_14": _budget_boundary_result(
            "max_gates",
            ResourceBudget(
                max_t_count=7,
                max_two_qubit_count=6,
                max_gates=14,
                max_depth=12,
            ),
        ),
        "budget_depth_11": _budget_boundary_result(
            "max_depth",
            ResourceBudget(
                max_t_count=7,
                max_two_qubit_count=6,
                max_gates=15,
                max_depth=11,
            ),
        ),
    }


def _wrong_gate_controls() -> dict[str, dict[str, object]]:
    """Build native negative witnesses and require dense exact rejection."""

    first_cnot = next(
        index
        for index, gate in enumerate(KNOWN_TOFFOLI_GATES)
        if gate.gate_type is GateType.CNOT
    )
    first_phase = next(
        index
        for index, gate in enumerate(KNOWN_TOFFOLI_GATES)
        if gate.gate_type in {GateType.T, GateType.TDG}
    )

    reversed_cnot = list(KNOWN_TOFFOLI_GATES)
    original_cnot = reversed_cnot[first_cnot]
    reversed_cnot[first_cnot] = Gate(
        GateType.CNOT,
        (original_cnot.qubits[1], original_cnot.qubits[0]),
    )
    missing_phase = list(KNOWN_TOFFOLI_GATES)
    deleted_phase = missing_phase.pop(first_phase)
    missing_final_h = list(KNOWN_TOFFOLI_GATES[:-1])
    wrong_target_label = [
        Gate(gate.gate_type, (1,))
        if gate.gate_type is GateType.H and gate.qubits == (TOFFOLI_TARGET_QUBIT,)
        else gate
        for gate in KNOWN_TOFFOLI_GATES
    ]

    controls: dict[str, tuple[Sequence[Gate], str]] = {
        "reversed_cnot": (reversed_cnot, f"reversed {original_cnot!r}"),
        "missing_phase_gate": (missing_phase, f"deleted {deleted_phase!r}"),
        "missing_final_h": (missing_final_h, "deleted final H(2)"),
        "wrong_target_label": (
            wrong_target_label,
            "replaced outer H(2) gates with H(1) while declared target remains q2",
        ),
    }
    result: dict[str, dict[str, object]] = {}
    for name, (gates, mutation) in controls.items():
        state = _build_state(gates, budget=_permissive_budget(gate_count=len(gates)))
        validation = validate_exact_toffoli_state(
            state,
            matrix_atol=MATRIX_ATOL,
            matrix_rtol=MATRIX_RTOL,
            max_process_infidelity=MAX_PROCESS_INFIDELITY,
        )
        result[name] = {
            "passed": not validation.exact_certified,
            "mutation": mutation,
            "exact_certified": bool(validation.exact_certified),
            "global_phase_equivalent": bool(validation.global_phase_equivalent),
            "simulator_certification_status": validation.resources.get(
                "simulator_certification_status"
            ),
        }
    return result


def _relative_phase_control(target_unitary: np.ndarray) -> dict[str, object]:
    """Prove a truth-table-preserving relative phase does not certify CCX.

    The dense impostor is analytical.  A native realization of the same
    matrix is additionally built through public state transitions so the
    repository's actual dense simulator certifier is exercised too.
    """

    analytical_impostor = relative_phase_toffoli_impostor()
    analytical = validate_toffoli_unitary(
        analytical_impostor,
        matrix_atol=MATRIX_ATOL,
        matrix_rtol=MATRIX_RTOL,
    )

    # X(2) H(2) CCX H(2) X(2) is a diagonal phase on input column 3.
    # Executing it before CCX yields CCX @ diag(1,1,1,-1,1,1,1,1), which is
    # exactly the analytical impostor rather than a merely similar circuit.
    native_impostor_gates: tuple[Gate, ...] = (
        Gate(GateType.X, (TOFFOLI_TARGET_QUBIT,)),
        Gate(GateType.H, (TOFFOLI_TARGET_QUBIT,)),
        *KNOWN_TOFFOLI_GATES,
        Gate(GateType.H, (TOFFOLI_TARGET_QUBIT,)),
        Gate(GateType.X, (TOFFOLI_TARGET_QUBIT,)),
        *KNOWN_TOFFOLI_GATES,
    )
    native_state = _build_state(
        native_impostor_gates,
        budget=_permissive_budget(gate_count=len(native_impostor_gates)),
    )
    native_unitary = unitary_from_gates(
        TOFFOLI_NUM_QUBITS, native_state.dag.gates
    )
    native_validation = validate_exact_toffoli_state(
        native_state,
        matrix_atol=MATRIX_ATOL,
        matrix_rtol=MATRIX_RTOL,
        max_process_infidelity=MAX_PROCESS_INFIDELITY,
    )
    native_matches_analytical = equivalent_up_to_global_phase(
        native_unitary,
        analytical_impostor,
        atol=MATRIX_ATOL,
        rtol=MATRIX_RTOL,
    )
    return {
        "passed": bool(
            analytical.truth_table_correct
            and not analytical.global_phase_equivalent
            and not analytical.column_phase_consistent
            and not native_validation.exact_certified
            and native_matches_analytical
        ),
        "truth_table_correct": bool(analytical.truth_table_correct),
        "global_phase_equivalent": bool(analytical.global_phase_equivalent),
        "column_phase_consistent": bool(analytical.column_phase_consistent),
        "process_fidelity": float(analytical.process_fidelity),
        "max_phase_aligned_matrix_error": float(
            analytical.max_phase_aligned_matrix_error
        ),
        "native_candidate_matches_analytical_impostor": bool(
            native_matches_analytical
        ),
        "exact_simulator_rejected_native_candidate": not native_validation.exact_certified,
        "simulator_certification_status": native_validation.resources.get(
            "simulator_certification_status"
        ),
        "phase_column": 3,
        "target_matrix_digest": stable_matrix_digest(target_unitary),
        "impostor_matrix_digest": stable_matrix_digest(analytical_impostor),
    }


def _global_phase_control(target_unitary: np.ndarray) -> dict[str, object]:
    """Confirm the configured global-phase quotient accepts one common phase."""

    candidate = np.exp(0.317j) * target_unitary
    diagnostics = validate_toffoli_unitary(
        candidate,
        matrix_atol=MATRIX_ATOL,
        matrix_rtol=MATRIX_RTOL,
    )
    return {
        "passed": bool(
            diagnostics.global_phase_equivalent
            and diagnostics.truth_table_correct
            and diagnostics.column_phase_consistent
            and diagnostics.process_fidelity >= 1.0 - MAX_PROCESS_INFIDELITY
        ),
        "global_phase_equivalent": bool(diagnostics.global_phase_equivalent),
        "truth_table_correct": bool(diagnostics.truth_table_correct),
        "column_phase_consistent": bool(diagnostics.column_phase_consistent),
        "process_fidelity": float(diagnostics.process_fidelity),
    }


def _truth_table_report_rows(
    candidate_unitary: np.ndarray,
) -> list[dict[str, object]]:
    """Create exhaustive, phase-aware CSV/JSON truth-table evidence."""

    diagnostics = validate_toffoli_unitary(
        candidate_unitary,
        matrix_atol=MATRIX_ATOL,
        matrix_rtol=MATRIX_RTOL,
    )
    rows = diagnostics.truth_table
    if len(rows) != 1 << TOFFOLI_NUM_QUBITS:
        raise RuntimeError("candidate did not produce exhaustive truth-table diagnostics")
    reference_amplitude = rows[0].amplitude
    if abs(reference_amplitude) <= MATRIX_ATOL:
        raise RuntimeError("candidate has no usable global-phase reference amplitude")
    common_phase = reference_amplitude / abs(reference_amplitude)

    report_rows: list[dict[str, object]] = []
    for row in rows:
        input_index = int(row.input_index)
        expected_output = int(row.expected_output_index)
        observed_output = int(row.observed_output_index)
        amplitude = complex(row.amplitude)
        report_rows.append(
            {
                "input_index": input_index,
                "input_bits_q2q1q0": _bit_label(input_index),
                "input_q0": (input_index >> 0) & 1,
                "input_q1": (input_index >> 1) & 1,
                "input_q2": (input_index >> 2) & 1,
                "expected_output_index": expected_output,
                "expected_output_bits_q2q1q0": _bit_label(expected_output),
                "candidate_output_index": observed_output,
                "candidate_output_bits_q2q1q0": _bit_label(observed_output),
                "probability": float(row.expected_output_probability),
                "maximum_off_target_probability": float(
                    row.maximum_off_target_probability
                ),
                "amplitude_real": float(amplitude.real),
                "amplitude_imag": float(amplitude.imag),
                "mapping_correct": bool(row.correct),
                "column_phase_matches_global": bool(
                    np.isclose(
                        amplitude,
                        common_phase,
                        atol=MATRIX_ATOL,
                        rtol=MATRIX_RTOL,
                    )
                ),
                "controls": list(TOFFOLI_CONTROLS),
                "target": TOFFOLI_TARGET_QUBIT,
                "qubit_order": _QUBIT_ORDER,
            }
        )
    return report_rows


def _positive_checks(
    *,
    target_unitary: np.ndarray,
    candidate_unitary: np.ndarray,
    validation: Any,
    resources: Mapping[str, object],
) -> dict[str, bool]:
    """Keep each mandatory acceptance criterion independently inspectable."""

    identity = np.eye(1 << TOFFOLI_NUM_QUBITS, dtype=np.complex128)
    target_is_unitary = bool(
        np.allclose(
            target_unitary.conj().T @ target_unitary,
            identity,
            atol=MATRIX_ATOL,
            rtol=MATRIX_RTOL,
        )
    )
    expected_mapping = all(
        expected_toffoli_basis_index(index)
        == (index ^ 4 if index in {3, 7} else index)
        for index in range(1 << TOFFOLI_NUM_QUBITS)
    )
    expected_resources = {
        "num_gates": 15,
        "t_count": 7,
        "two_qubit_count": 6,
        "h_count": 2,
        "cnot_count": 6,
        "t_gate_count": 4,
        "tdg_gate_count": 3,
        "depth": 12,
        "wire_depths": [9, 11, 12],
        "ancilla_count": 0,
        "continuation_model": "all-to-all, no-ancilla",
    }
    fixed_resources_match = all(
        resources.get(key) == value for key, value in expected_resources.items()
    )
    return {
        "analytical_target_is_unitary": target_is_unitary,
        "analytical_basis_mapping_is_lsb_ccx": expected_mapping,
        "candidate_dense_matrix_comes_from_authoritative_dag": bool(
            equivalent_up_to_global_phase(
                candidate_unitary,
                target_unitary,
                atol=MATRIX_ATOL,
                rtol=MATRIX_RTOL,
            )
        ),
        "exact_simulator_certified": bool(validation.exact_certified),
        "global_phase_equivalent": bool(validation.global_phase_equivalent),
        "strict_matrix_error": bool(
            validation.max_phase_aligned_matrix_error
            <= MATRIX_ATOL + MATRIX_RTOL
        ),
        "process_fidelity": bool(
            validation.process_fidelity >= 1.0 - MAX_PROCESS_INFIDELITY
        ),
        "truth_table_correct": bool(validation.truth_table_correct),
        "column_phase_consistent": bool(validation.column_phase_consistent),
        "symbolic_agrees_with_dense": bool(validation.symbolic_agrees_with_dense),
        "semantic_correct": bool(resources.get("semantic_correct", False)),
        "resource_accounting_correct": bool(
            resources.get("resource_accounting_correct", False)
        ),
        "fixed_resource_profile_matches": fixed_resources_match,
        "matches_published_t_lower_bound": bool(
            resources.get("matches_published_t_lower_bound", False)
        ),
        "matches_published_cnot_lower_bound": bool(
            resources.get("matches_published_cnot_lower_bound", False)
        ),
    }


def run_toffoli_certification(
    output_dir: str | Path,
    *,
    corrupt_mandatory_check: bool = False,
) -> dict[str, Any]:
    """Run deterministic known-witness certification and save all artifacts.

    ``corrupt_mandatory_check`` is an explicit test hook.  It does not alter
    the witness or hide failures; it proves that the CLI's success condition
    is tied to every mandatory check and consequently returns a nonzero exit
    status through :func:`main`.
    """

    target_unitary = toffoli_reference_unitary()
    candidate_state = build_known_toffoli_state()
    candidate_state.dag.validate()
    candidate_unitary = unitary_from_gates(
        TOFFOLI_NUM_QUBITS,
        candidate_state.dag.gates,
    )
    validation = validate_exact_toffoli_state(
        candidate_state,
        matrix_atol=MATRIX_ATOL,
        matrix_rtol=MATRIX_RTOL,
        max_process_infidelity=MAX_PROCESS_INFIDELITY,
    )
    resources = dict(toffoli_resource_summary(candidate_state))
    # The semantic outcome belongs with the core certification result.  The
    # resource helper returns only accounting facts, so preserve the core
    # semantic diagnostic after re-materialising the resource summary above.
    resources["semantic_correct"] = bool(
        validation.exact_certified
        and validation.global_phase_equivalent
        and validation.truth_table_correct
        and validation.column_phase_consistent
        and validation.symbolic_agrees_with_dense
        and validation.process_fidelity >= 1.0 - MAX_PROCESS_INFIDELITY
    )
    resources["configured_max_process_infidelity"] = MAX_PROCESS_INFIDELITY
    resources["simulator_certification_status"] = validation.resources.get(
        "simulator_certification_status"
    )

    positive_checks = _positive_checks(
        target_unitary=target_unitary,
        candidate_unitary=candidate_unitary,
        validation=validation,
        resources=resources,
    )
    if corrupt_mandatory_check:
        # Keep this explicit and reportable.  Mutating no circuit data makes
        # it safe for deterministic failure-path coverage.
        positive_checks["intentional_corrupt_mandatory_check"] = False

    negative_controls: dict[str, dict[str, object]] = {
        "relative_phase_truth_table_impostor": _relative_phase_control(target_unitary),
        "global_phase_positive_control": _global_phase_control(target_unitary),
        **_wrong_gate_controls(),
        **_budget_boundary_controls(),
    }
    truth_table = _truth_table_report_rows(candidate_unitary)
    semantic_validation = {
        **asdict(validation),
        "resources": None,
        "matrix_atol": MATRIX_ATOL,
        "matrix_rtol": MATRIX_RTOL,
        "max_process_infidelity": MAX_PROCESS_INFIDELITY,
    }
    # ``resources`` is separately included below so semantic diagnostics and
    # accounting cannot accidentally be conflated in a consumer's report.
    candidate = {
        "operations": [
            _operation(gate, index=index)
            for index, gate in enumerate(candidate_state.dag.gates, start=1)
        ],
        "exact_certified": bool(validation.exact_certified),
        "global_phase_equivalent": bool(validation.global_phase_equivalent),
        "max_phase_aligned_matrix_error": float(
            validation.max_phase_aligned_matrix_error
        ),
        "process_fidelity": float(validation.process_fidelity),
        "truth_table_correct": bool(validation.truth_table_correct),
        "column_phase_consistent": bool(validation.column_phase_consistent),
        "symbolic_agrees_with_dense": bool(validation.symbolic_agrees_with_dense),
        "semantic_validation": semantic_validation,
        "resource_validation": resources,
    }
    correct = bool(
        all(positive_checks.values())
        and all(control.get("passed", False) for control in negative_controls.values())
    )
    report: dict[str, Any] = {
        "correct": correct,
        "scope": "Known exact Toffoli witness certification; no synthesis search.",
        "target": {
            "controls": list(TOFFOLI_CONTROLS),
            "target": TOFFOLI_TARGET_QUBIT,
            "qubit_order": _QUBIT_ORDER,
            "matrix_digest": stable_matrix_digest(target_unitary),
            "gate_witness_digest": stable_gate_witness_digest(
                candidate_state.dag.gates
            ),
        },
        "candidate": candidate,
        "positive_checks": positive_checks,
        "truth_table": truth_table,
        "negative_controls": negative_controls,
    }
    report["artifacts"] = save_toffoli_artifacts(
        output_dir,
        report=report,
        candidate_state=candidate_state,
        target_unitary=target_unitary,
        candidate_unitary=candidate_unitary,
        controls=TOFFOLI_CONTROLS,
        target=TOFFOLI_TARGET_QUBIT,
        num_qubits=TOFFOLI_NUM_QUBITS,
    )
    return report


def main(argv: Iterable[str] | None = None) -> int:
    """Run the deterministic CLI and return a process-compatible exit code."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("outputs") / "toffoli-known",
        help="directory for deterministic JSON, CSV, SVG, and Markdown evidence",
    )
    parser.add_argument(
        "--corrupt-mandatory-check",
        action="store_true",
        help="test hook: deliberately fail one mandatory acceptance check",
    )
    args = parser.parse_args(argv)
    try:
        report = run_toffoli_certification(
            args.artifacts_dir,
            corrupt_mandatory_check=args.corrupt_mandatory_check,
        )
    except Exception as exc:  # artifact/validation failure must be nonzero
        print(f"Toffoli certification failed: {exc}")
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["correct"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())


__all__ = ["main", "run_toffoli_certification"]
