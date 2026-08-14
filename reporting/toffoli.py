"""Deterministic artifact writers for the known exact Toffoli-3 benchmark.

This module deliberately contains presentation and serialization only.  The
analytical target and all acceptance decisions remain in ``benchmarks.toffoli``
and the command-line runner, respectively.  In particular, the SVG is drawn
from the candidate state's authoritative DAG and is not used as evidence of
semantic correctness.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from circuit.gate import Gate
from reporting.artifacts import circuit_svg


_QUBIT_ORDER = "q0 is LSB; displayed bit strings are q2 q1 q0"


def _json_ready(value: Any) -> Any:
    """Convert the small set of project values used in reports to JSON data."""

    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Gate):
        return {
            "gate": value.gate_type.name,
            "qubits": [int(qubit) for qubit in value.qubits],
        }
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _bit_label(index: int, num_qubits: int) -> str:
    return format(index, f"0{num_qubits}b")


def _as_complex_matrix(matrix: Any, *, name: str, num_qubits: int) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.complex128)
    dimension = 1 << num_qubits
    if array.shape != (dimension, dimension):
        raise ValueError(
            f"{name} must have shape ({dimension}, {dimension}), got {array.shape!r}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite complex values")
    return array.copy()


def stable_matrix_digest(matrix: Any) -> str:
    """Return a platform-independent digest of a finite complex matrix.

    ``float.hex`` encodes the exact IEEE floating-point values without making
    a report digest sensitive to CSV/newline formatting or locale settings.
    """

    array = np.asarray(matrix, dtype=np.complex128)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("matrix digest requires a finite two-dimensional matrix")
    digest = hashlib.sha256()
    digest.update(f"shape={array.shape!r};dtype=complex128;".encode("ascii"))
    for value in array.ravel(order="C"):
        digest.update(float(value.real).hex().encode("ascii"))
        digest.update(b",")
        digest.update(float(value.imag).hex().encode("ascii"))
        digest.update(b";")
    return digest.hexdigest()


def stable_gate_witness_digest(gates: Sequence[Gate]) -> str:
    """Return a stable digest of the ordered, qubit-labelled gate witness."""

    operations = [
        {"gate": gate.gate_type.name, "qubits": [int(qubit) for qubit in gate.qubits]}
        for gate in gates
    ]
    encoded = json.dumps(
        operations, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _matrix_rows(
    matrix: np.ndarray,
    *,
    matrix_name: str,
    controls: Sequence[int],
    target: int,
    num_qubits: int,
) -> list[dict[str, Any]]:
    """Represent a matrix in a portable, unambiguous CSV schema."""

    digest = stable_matrix_digest(matrix)
    common = {
        "matrix": matrix_name,
        "matrix_digest": digest,
        "controls": json.dumps([int(qubit) for qubit in controls]),
        "target": int(target),
        "qubit_order": _QUBIT_ORDER,
    }
    rows: list[dict[str, Any]] = []
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            rows.append(
                {
                    **common,
                    "row_index": row_index,
                    "row_bits_q2q1q0": _bit_label(row_index, num_qubits),
                    "column_index": column_index,
                    "column_bits_q2q1q0": _bit_label(column_index, num_qubits),
                    # ``.17g`` round-trips every float64 while retaining the
                    # physically meaningful signed zero / tiny-error detail.
                    "real": format(float(value.real), ".17g"),
                    "imag": format(float(value.imag), ".17g"),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(fieldnames),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for field, value in row.items()
                }
            )


def _truth_table_fieldnames(rows: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    preferred = (
        "input_index",
        "input_bits_q2q1q0",
        "input_q0",
        "input_q1",
        "input_q2",
        "expected_output_index",
        "expected_output_bits_q2q1q0",
        "candidate_output_index",
        "candidate_output_bits_q2q1q0",
        "probability",
        "amplitude_real",
        "amplitude_imag",
        "mapping_correct",
        "column_phase_matches_global",
        "controls",
        "target",
        "qubit_order",
    )
    seen = set(preferred)
    extras = sorted({str(key) for row in rows for key in row} - seen)
    return (*preferred, *extras)


def _markdown_report(report: Mapping[str, Any], artifact_names: Mapping[str, str]) -> str:
    candidate = report.get("candidate", {})
    semantic = candidate.get("semantic_validation", {})
    resources = candidate.get("resource_validation", {})
    target = report.get("target", {})
    negative_controls = report.get("negative_controls", {})
    lines = [
        "# Known exact Toffoli-3 certification",
        "",
        f"- Correct: `{report.get('correct', False)}`",
        f"- Scope: {report.get('scope', '')}",
        f"- Controls / target: `{target.get('controls', [])}` / `{target.get('target')}`",
        f"- Qubit convention: `{target.get('qubit_order', _QUBIT_ORDER)}`",
        f"- Exact dense certification: `{semantic.get('exact_certified', False)}`",
        f"- Global-phase equivalence: `{semantic.get('global_phase_equivalent', False)}`",
        f"- Process fidelity: `{semantic.get('process_fidelity', 0.0):.16g}`",
        f"- Maximum phase-aligned matrix error: `{semantic.get('max_phase_aligned_matrix_error', 0.0):.16g}`",
        f"- Gates / T count / CNOT count / depth: `{resources.get('num_gates')}` / `{resources.get('t_count')}` / `{resources.get('two_qubit_count')}` / `{resources.get('depth')}`",
        "",
        "## Candidate operations",
        "",
        "```text",
        *[
            f"{index:02d}  {operation.get('gate')}({', '.join(str(q) for q in operation.get('qubits', []))})"
            for index, operation in enumerate(candidate.get("operations", []), start=1)
        ],
        "```",
        "",
        "## Negative controls",
        "",
    ]
    for name in sorted(negative_controls):
        result = negative_controls[name]
        if isinstance(result, Mapping):
            passed = result.get("passed", False)
            lines.append(f"- `{name}`: `{passed}`")
        else:
            lines.append(f"- `{name}`: `{result}`")
    lines.extend(
        [
            "",
            "## Files",
            "",
            *[f"- [{name}]({path})" for name, path in artifact_names.items()],
            "",
            "The circuit diagram is rendered from the actual authoritative candidate DAG. It is a review artifact, not evidence of semantic correctness.",
            "",
        ]
    )
    return "\n".join(lines)


def save_toffoli_artifacts(
    output_dir: str | Path,
    *,
    report: Mapping[str, Any],
    candidate_state: Any,
    target_unitary: Any,
    candidate_unitary: Any,
    controls: Sequence[int] = (0, 1),
    target: int = 2,
    num_qubits: int = 3,
) -> dict[str, str]:
    """Write all deterministic review artifacts for a known Toffoli witness.

    ``candidate_state`` is intentionally required rather than a loose gate
    tuple so the circuit SVG and DAG diagnostics always originate from the
    public state's authoritative ``CircuitDAG``.
    """

    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer")
    try:
        dag = candidate_state.dag
        dag.validate()
        gates = tuple(dag.gates)
    except AttributeError as exc:
        raise TypeError("candidate_state must expose an authoritative CircuitDAG") from exc

    candidate_matrix = _as_complex_matrix(
        candidate_unitary, name="candidate_unitary", num_qubits=num_qubits
    )
    target_matrix = _as_complex_matrix(
        target_unitary, name="target_unitary", num_qubits=num_qubits
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_path / "summary.json",
        "report": output_path / "summary.md",
        "truth_table": output_path / "truth_table.csv",
        "resource_summary": output_path / "resource_summary.json",
        "candidate_unitary": output_path / "candidate_unitary.csv",
        "target_unitary": output_path / "target_unitary.csv",
        "circuit_diagram": output_path / "circuit.svg",
    }
    artifact_names = {name: path.name for name, path in paths.items()}

    truth_table = [
        _json_ready(row) for row in report.get("truth_table", [])
    ]
    if len(truth_table) != 1 << num_qubits:
        raise ValueError("Toffoli report must contain one truth-table row per input")
    _write_csv(
        paths["truth_table"],
        truth_table,
        _truth_table_fieldnames(truth_table),
    )

    matrix_fieldnames = (
        "matrix",
        "matrix_digest",
        "controls",
        "target",
        "qubit_order",
        "row_index",
        "row_bits_q2q1q0",
        "column_index",
        "column_bits_q2q1q0",
        "real",
        "imag",
    )
    _write_csv(
        paths["candidate_unitary"],
        _matrix_rows(
            candidate_matrix,
            matrix_name="candidate",
            controls=controls,
            target=target,
            num_qubits=num_qubits,
        ),
        matrix_fieldnames,
    )
    _write_csv(
        paths["target_unitary"],
        _matrix_rows(
            target_matrix,
            matrix_name="analytical_ccx_target",
            controls=controls,
            target=target,
            num_qubits=num_qubits,
        ),
        matrix_fieldnames,
    )

    resource_summary = {
        "controls": [int(qubit) for qubit in controls],
        "target": int(target),
        "qubit_order": _QUBIT_ORDER,
        "candidate_resources": report.get("candidate", {}).get(
            "resource_validation", {}
        ),
        "ordered_operations": _json_ready(gates),
        "dag_nodes": [
            {
                "node_id": int(node.id),
                "gate": _json_ready(node.gate),
                "level": int(node.level),
                "parents": sorted(int(parent) for parent in node.parents),
                "children": sorted(int(child) for child in node.children),
            }
            for node in dag.topological_nodes()
        ],
        "candidate_matrix_digest": stable_matrix_digest(candidate_matrix),
        "target_matrix_digest": stable_matrix_digest(target_matrix),
        "gate_witness_digest": stable_gate_witness_digest(gates),
    }
    _write_json(paths["resource_summary"], resource_summary)

    paths["circuit_diagram"].write_text(
        circuit_svg(
            gates,
            num_qubits,
            heading="Known exact Toffoli-3 candidate circuit",
            description=(
                "Actual authoritative CircuitDAG order; q0 is LSB; "
                "controls q0, q1 and target q2."
            ),
            empty_heading="No candidate gates were available for certification.",
        ),
        encoding="utf-8",
    )

    summary = _json_ready(dict(report))
    summary["artifacts"] = artifact_names
    _write_json(paths["summary"], summary)
    paths["report"].write_text(
        _markdown_report(summary, artifact_names), encoding="utf-8"
    )
    return {name: str(path.resolve()) for name, path in paths.items()}


__all__ = [
    "save_toffoli_artifacts",
    "stable_gate_witness_digest",
    "stable_matrix_digest",
]
