"""Dependency-free post-processing for the GHZ-3 smoke run.

The project intentionally avoids a plotting dependency for this small,
reproducible benchmark.  JSON and CSV preserve machine-readable data, while
the SVG files can be opened in any browser without a Python environment.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from circuit.gate import Gate


_BLUE = "#2563eb"
_DARK = "#0f172a"
_GRID = "#cbd5e1"
_LIGHT = "#f8fafc"


def _basis_label(index: int, num_qubits: int) -> str:
    """Render a basis label in conventional q[n-1]...q[0] ket order."""

    return format(index, f"0{num_qubits}b")


def _svg(
    width: int,
    height: int,
    body: str,
    title: str,
    *,
    description: str = "Deterministic GHZ-3 smoke-test artifact.",
) -> str:
    escaped_title = html.escape(title)
    escaped_description = html.escape(description)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-labelledby="title desc">\n'
        f"  <title id=\"title\">{escaped_title}</title>\n"
        f"  <desc id=\"desc\">{escaped_description}</desc>\n"
        f'  <rect width="100%" height="100%" fill="{_LIGHT}"/>\n'
        f"{body}\n"
        "</svg>\n"
    )


def _probability_svg(probabilities: np.ndarray, num_qubits: int) -> str:
    width, height = 900, 460
    left, right, top, bottom = 72, 30, 64, 86
    chart_width = width - left - right
    chart_height = height - top - bottom
    bar_slot = chart_width / len(probabilities)
    max_probability = max(0.5, float(np.max(probabilities)))
    parts = [
        f'  <text x="{left}" y="32" font-family="sans-serif" font-size="22" fill="{_DARK}">GHZ-3 output probabilities</text>',
        f'  <line x1="{left}" y1="{top + chart_height}" x2="{width - right}" y2="{top + chart_height}" stroke="{_DARK}" stroke-width="2"/>',
        f'  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="{_DARK}" stroke-width="2"/>',
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + chart_height * (1.0 - fraction)
        value = max_probability * fraction
        parts.extend(
            [
                f'  <line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="{_GRID}" stroke-width="1"/>',
                f'  <text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13" fill="{_DARK}">{value:.2f}</text>',
            ]
        )
    for index, probability in enumerate(probabilities):
        x = left + index * bar_slot + bar_slot * 0.18
        bar_width = bar_slot * 0.64
        bar_height = chart_height * float(probability) / max_probability
        y = top + chart_height - bar_height
        color = _BLUE if probability > 1e-12 else "#cbd5e1"
        parts.extend(
            [
                f'  <rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" rx="3" fill="{color}"/>',
                f'  <text x="{x + bar_width / 2:.2f}" y="{top + chart_height + 24}" text-anchor="middle" font-family="monospace" font-size="14" fill="{_DARK}">{_basis_label(index, num_qubits)}</text>',
            ]
        )
    parts.append(
        f'  <text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="{_DARK}">Basis state |q2 q1 q0⟩ (q0 is the least-significant simulation bit)</text>'
    )
    # Keep the on-chart description ASCII so SVG labels remain portable across
    # terminals and viewers with different text encodings.
    parts[-1] = (
        f'  <text x="{width / 2}" y="{height - 20}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="14" fill="{_DARK}">Basis '
        "ordering q2 q1 q0 (q0 is the least-significant simulation bit)</text>"
    )
    return _svg(width, height, "\n".join(parts), "GHZ-3 output probability chart")


def _frontier_svg(
    trace: Sequence[Mapping[str, Any]],
    *,
    heading: str = "Frontier size during deterministic search",
) -> str:
    width, height = 900, 420
    left, right, top, bottom = 76, 32, 64, 72
    chart_width = width - left - right
    chart_height = height - top - bottom
    parts = [
        f'  <text x="{left}" y="32" font-family="sans-serif" font-size="22" fill="{_DARK}">{html.escape(heading)}</text>',
        f'  <line x1="{left}" y1="{top + chart_height}" x2="{width - right}" y2="{top + chart_height}" stroke="{_DARK}" stroke-width="2"/>',
        f'  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="{_DARK}" stroke-width="2"/>',
    ]
    if not trace:
        parts.append(
            f'  <text x="{width / 2}" y="{top + chart_height / 2}" text-anchor="middle" font-family="sans-serif" font-size="16" fill="{_DARK}">No search trace was collected.</text>'
        )
        return _svg(width, height, "\n".join(parts), "Empty frontier trace")

    x_values = [max(1, int(row.get("expansion", index + 1))) for index, row in enumerate(trace)]
    y_values = [max(0, int(row.get("frontier_size", 0))) for row in trace]
    max_x = max(x_values)
    max_y = max(1, max(y_values))
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + chart_height * (1.0 - fraction)
        value = int(round(max_y * fraction))
        parts.extend(
            [
                f'  <line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="{_GRID}" stroke-width="1"/>',
                f'  <text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13" fill="{_DARK}">{value}</text>',
            ]
        )
    points = []
    for x_value, y_value in zip(x_values, y_values):
        x = left + chart_width * (x_value - 1) / max(1, max_x - 1)
        y = top + chart_height * (1.0 - y_value / max_y)
        points.append(f"{x:.2f},{y:.2f}")
    parts.extend(
        [
            f'  <polyline points="{" ".join(points)}" fill="none" stroke="{_BLUE}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>',
            f'  <text x="{left}" y="{height - 24}" font-family="sans-serif" font-size="14" fill="{_DARK}">1</text>',
            f'  <text x="{width - right}" y="{height - 24}" text-anchor="end" font-family="sans-serif" font-size="14" fill="{_DARK}">{max_x}</text>',
            f'  <text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="{_DARK}">Expansion</text>',
        ]
    )
    return _svg(width, height, "\n".join(parts), heading)


def circuit_svg(
    gates: Sequence[Gate],
    num_qubits: int,
    *,
    heading: str | None = None,
    description: str | None = None,
    empty_heading: str | None = None,
) -> str:
    """Render a small dependency-free circuit diagram from concrete gates.

    The caller supplies the actual authoritative DAG gate sequence.  This is
    deliberately a presentation helper only: its SVG must never be used as
    semantic evidence for a candidate circuit.  The optional labels make the
    helper reusable by deterministic reference benchmarks without duplicating
    the GHZ-specific renderer.
    """

    gate_count = max(1, len(gates))
    width = max(760, 180 + 145 * gate_count)
    height = 120 + 92 * num_qubits
    left, top, spacing = 150, 86, 84
    wire_y = [top + qubit * spacing for qubit in range(num_qubits)]
    has_witness = bool(gates)
    rendered_heading = heading or (
        "Generated GHZ-3 circuit" if has_witness else "No certified GHZ-3 witness"
    )
    rendered_description = description or (
        "Circuit order is left to right; q0 is the least-significant simulation bit."
    )
    rendered_empty_heading = empty_heading or (
        "The search did not return a certified circuit."
    )
    parts = [
        f'  <text x="36" y="32" font-family="sans-serif" font-size="22" fill="{_DARK}">{html.escape(rendered_heading)}</text>',
        f'  <text x="36" y="54" font-family="sans-serif" font-size="13" fill="{_DARK}">{html.escape(rendered_description)}</text>',
    ]
    for qubit, y in enumerate(wire_y):
        label = f"q{qubit}" + (" (LSB)" if qubit == 0 else "")
        parts.extend(
            [
                f'  <text x="{left - 22}" y="{y + 5}" text-anchor="end" font-family="monospace" font-size="16" fill="{_DARK}">{label}</text>',
                f'  <line x1="{left}" y1="{y}" x2="{width - 32}" y2="{y}" stroke="{_DARK}" stroke-width="2"/>',
            ]
        )
    for index, gate in enumerate(gates):
        x = left + 78 + index * 145
        name = gate.gate_type.name
        if name == "CNOT":
            control, target = gate.qubits
            control_y, target_y = wire_y[control], wire_y[target]
            parts.extend(
                [
                    f'  <line x1="{x}" y1="{control_y}" x2="{x}" y2="{target_y}" stroke="{_DARK}" stroke-width="2"/>',
                    f'  <circle cx="{x}" cy="{control_y}" r="7" fill="{_DARK}"/>',
                    f'  <circle cx="{x}" cy="{target_y}" r="18" fill="{_LIGHT}" stroke="{_DARK}" stroke-width="2"/>',
                    f'  <line x1="{x - 12}" y1="{target_y}" x2="{x + 12}" y2="{target_y}" stroke="{_DARK}" stroke-width="2"/>',
                    f'  <line x1="{x}" y1="{target_y - 12}" x2="{x}" y2="{target_y + 12}" stroke="{_DARK}" stroke-width="2"/>',
                    f'  <text x="{x}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="{_DARK}">CNOT {control}-&gt;{target}</text>',
                ]
            )
        else:
            qubit = gate.qubits[0]
            y = wire_y[qubit]
            parts.extend(
                [
                    f'  <rect x="{x - 24}" y="{y - 24}" width="48" height="48" rx="5" fill="#dbeafe" stroke="{_BLUE}" stroke-width="2"/>',
                    f'  <text x="{x}" y="{y + 6}" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="bold" fill="{_DARK}">{html.escape(name)}</text>',
                ]
            )
    if not has_witness:
        parts.append(
            f'  <text x="{width / 2}" y="{height - 38}" text-anchor="middle" font-family="sans-serif" font-size="16" fill="{_DARK}">{html.escape(rendered_empty_heading)}</text>'
        )
    return _svg(
        width,
        height,
        "\n".join(parts),
        f"{rendered_heading} diagram",
        description=rendered_description,
    )


# Kept as a private compatibility alias for downstream code written against
# the initial GHZ artifact helper before it became a general circuit renderer.
_circuit_svg = circuit_svg


def _markdown_report(report: Mapping[str, Any], artifact_names: Mapping[str, str]) -> str:
    search = report.get("search", {})
    preparation = report.get("state_preparation", {})
    resources = preparation.get("resources", {})
    witness = search.get("witness", [])
    lines = [
        "# GHZ-3 state-preparation smoke report",
        "",
        f"- Correct: `{report.get('correct', False)}`",
        f"- Deterministic frontier search returned the target witness: `{search.get('certified', False)}`",
        f"- Search expansions: `{search.get('expansions', 0)}`",
        f"- State fidelity with analytical GHZ+: `{preparation.get('fidelity', 0.0):.16g}`",
        f"- Logical gates / CNOTs / T-count / depth: `{resources.get('num_gates')}` / `{resources.get('two_qubit_count')}` / `{resources.get('t_count')}` / `{resources.get('depth')}`",
        "",
        "## Generated witness",
        "",
        "```text",
        *[str(gate) for gate in witness],
        "```",
        "",
        "## Files",
        "",
    ]
    lines.extend(f"- [{key}]({name})" for key, name in artifact_names.items())
    lines.extend(
        [
            "",
            "This is a GHZ state-preparation smoke result. It does not claim a general learned-policy benchmark or prove full-unitary synthesis capability.",
            "",
        ]
    )
    return "\n".join(lines)


def save_ghz3_artifacts(
    output_dir: str | Path,
    *,
    report: Mapping[str, Any],
    gates: Sequence[Gate],
    statevector: Any,
    num_qubits: int = 3,
) -> dict[str, str]:
    """Save reviewable GHZ-3 data, charts, search trace, and circuit SVG.

    The return value maps stable artifact labels to absolute paths.  All
    inputs are serialised deterministically, making consecutive smoke runs
    directly comparable.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vector = np.asarray(statevector, dtype=np.complex128)
    dimension = 1 << num_qubits
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError("statevector must be a finite vector of length 2**num_qubits")
    probabilities = np.abs(vector) ** 2
    trace = list(report.get("search", {}).get("trace", []))

    paths = {
        "summary": output_path / "ghz3_summary.json",
        "statevector": output_path / "ghz3_statevector.json",
        "probabilities": output_path / "ghz3_probabilities.csv",
        "probability_chart": output_path / "ghz3_probabilities.svg",
        "frontier_chart": output_path / "ghz3_frontier_trace.svg",
        "circuit_diagram": output_path / "ghz3_circuit.svg",
        "search_trace": output_path / "ghz3_search_trace.csv",
        "report": output_path / "ghz3_report.md",
    }
    artifact_names = {name: path.name for name, path in paths.items()}

    state_rows = [
        {
            "basis": _basis_label(index, num_qubits),
            "real": float(amplitude.real),
            "imag": float(amplitude.imag),
            "probability": float(probabilities[index]),
        }
        for index, amplitude in enumerate(vector)
    ]
    paths["statevector"].write_text(
        json.dumps(state_rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with paths["probabilities"].open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("basis", "probability", "real", "imag"))
        writer.writeheader()
        writer.writerows(state_rows)
    with paths["search_trace"].open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "expansion",
            "selected_record_id",
            "frontier_size",
            "num_children",
            "num_accepted",
            "num_pruned",
            "reward",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in trace:
            writer.writerow({field: row.get(field) for field in fieldnames})

    paths["probability_chart"].write_text(
        _probability_svg(probabilities, num_qubits), encoding="utf-8"
    )
    paths["frontier_chart"].write_text(_frontier_svg(trace), encoding="utf-8")
    paths["circuit_diagram"].write_text(
        circuit_svg(gates, num_qubits), encoding="utf-8"
    )

    summary = dict(report)
    summary["artifacts"] = artifact_names
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["report"].write_text(
        _markdown_report(summary, artifact_names), encoding="utf-8"
    )
    return {name: str(path.resolve()) for name, path in paths.items()}


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a deterministic CSV even when a trace has no rows.

    The learned benchmark carries a few more diagnostics than the smoke
    baseline.  Deriving the columns from the complete trace means future
    additive diagnostics do not silently disappear from the review artifact.
    """

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
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


def _ghz3_rl_markdown_report(
    report: Mapping[str, Any], artifact_names: Mapping[str, str]
) -> str:
    """Build a concise, self-contained handoff for a learned GHZ run."""

    evaluation = report.get("evaluation", {})
    learning = report.get("learning", {})
    lines = [
        "# Learned frontier-record GHZ-3 report",
        "",
        f"- Correct: `{report.get('correct', False)}`",
        f"- Certified: `{evaluation.get('certified', False)}`",
        f"- Frozen learned-policy expansions: `{evaluation.get('expansions', 0)}`",
        f"- Zero-weight baseline expansions: `{report.get('zero_policy_expansions')}`",
        f"- Training episodes: `{learning.get('episodes')}`",
        f"- Target fingerprint: `{learning.get('target_fingerprint')}`",
        "",
        "## Certified witness",
        "",
        "```text",
        *[str(gate) for gate in evaluation.get("witness", [])],
        "```",
        "",
        "## Interpretation",
        "",
        "The policy selects persistent frontier records.  Each selected record is expanded by the search engine through every legal native gate; the policy neither emits gates nor injects a known GHZ witness.",
        "",
        "## Files",
        "",
    ]
    lines.extend(f"- [{key}]({name})" for key, name in artifact_names.items())
    lines.append("")
    return "\n".join(lines)


def save_ghz3_rl_artifacts(
    output_dir: str | Path,
    *,
    report: Mapping[str, Any],
    gates: Sequence[Gate],
    statevector: Any,
    num_qubits: int = 3,
) -> dict[str, str]:
    """Save data and dependency-free visual artifacts for learned GHZ-3.

    This intentionally uses different stable filenames from
    :func:`save_ghz3_artifacts`: the latter is a FIFO smoke-test result, while
    these files document a trained, frozen-policy evaluation.  In particular
    an empty ``gates`` list renders the explicit *no certified witness*
    diagram rather than substituting the known reference circuit.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vector = np.asarray(statevector, dtype=np.complex128)
    dimension = 1 << num_qubits
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError("statevector must be a finite vector of length 2**num_qubits")

    probabilities = np.abs(vector) ** 2
    evaluation = report.get("evaluation", {})
    trace = list(evaluation.get("trace", []))
    history = list(report.get("training_history", []))
    policy = dict(report.get("policy", {}))
    paths = {
        "summary": output_path / "summary.json",
        "training_history": output_path / "training_history.csv",
        "evaluation_trace": output_path / "evaluation_trace.csv",
        "policy_weights": output_path / "policy_weights.json",
        "statevector": output_path / "statevector.json",
        "probabilities": output_path / "probabilities.csv",
        "probability_chart": output_path / "probabilities.svg",
        "frontier_progress": output_path / "frontier_progress.svg",
        "circuit_diagram": output_path / "circuit.svg",
        "report": output_path / "README.md",
    }
    artifact_names = {name: path.name for name, path in paths.items()}

    state_rows = [
        {
            "basis": _basis_label(index, num_qubits),
            "real": float(amplitude.real),
            "imag": float(amplitude.imag),
            "probability": float(probabilities[index]),
        }
        for index, amplitude in enumerate(vector)
    ]
    paths["statevector"].write_text(
        json.dumps(state_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_rows_csv(paths["probabilities"], state_rows)
    _write_rows_csv(paths["training_history"], history)
    _write_rows_csv(paths["evaluation_trace"], trace)
    paths["policy_weights"].write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["probability_chart"].write_text(
        _probability_svg(probabilities, num_qubits), encoding="utf-8"
    )
    paths["frontier_progress"].write_text(
        _frontier_svg(
            trace,
            heading="Frontier size during frozen learned-policy evaluation",
        ),
        encoding="utf-8",
    )
    paths["circuit_diagram"].write_text(
        circuit_svg(gates, num_qubits), encoding="utf-8"
    )

    summary = dict(report)
    summary["artifacts"] = artifact_names
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["report"].write_text(
        _ghz3_rl_markdown_report(summary, artifact_names), encoding="utf-8"
    )
    return {name: str(path.resolve()) for name, path in paths.items()}


__all__ = ["circuit_svg", "save_ghz3_artifacts", "save_ghz3_rl_artifacts"]
