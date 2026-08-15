"""Dependency-free artifacts for the reference-only QFT-3 benchmark."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping

from benchmarks.qft import ControlledPhase, H, QFTReference, SWAP


_DARK = "#0f172a"
_BLUE = "#2563eb"
_LIGHT = "#f8fafc"
_AMBER = "#b45309"


def reference_circuit_svg(reference: QFTReference, *, heading: str) -> str:
    """Render high-level reference operations without implying a native witness."""

    gate_count = max(1, len(reference.operations))
    width = max(920, 250 + 125 * gate_count)
    height = 190 + 82 * reference.num_qubits
    left, top, spacing = 175, 112, 82
    wire_y = [top + qubit * spacing for qubit in range(reference.num_qubits)]
    parts = [
        f'  <rect width="100%" height="100%" fill="{_LIGHT}"/>',
        f'  <text x="34" y="34" font-family="sans-serif" font-size="22" fill="{_DARK}">{html.escape(heading)}</text>',
        f'  <text x="34" y="58" font-family="sans-serif" font-size="13" fill="{_AMBER}">SDK-neutral high-level reference; not a native-search witness</text>',
        f'  <text x="34" y="79" font-family="sans-serif" font-size="12" fill="{_DARK}">{html.escape(reference.permutation_convention)}; q0 is the least-significant basis bit</text>',
    ]
    for qubit, y in enumerate(wire_y):
        parts.extend(
            [
                f'  <text x="{left - 22}" y="{y + 5}" text-anchor="end" font-family="monospace" font-size="15" fill="{_DARK}">q{qubit}{" (LSB)" if qubit == 0 else ""}</text>',
                f'  <line x1="{left}" y1="{y}" x2="{width - 32}" y2="{y}" stroke="{_DARK}" stroke-width="2"/>',
            ]
        )

    for index, operation in enumerate(reference.operations):
        x = left + 62 + 125 * index
        if isinstance(operation, H):
            y = wire_y[operation.qubit]
            parts.extend(
                [
                    f'  <rect x="{x - 22}" y="{y - 22}" width="44" height="44" rx="5" fill="#dbeafe" stroke="{_BLUE}" stroke-width="2"/>',
                    f'  <text x="{x}" y="{y + 6}" text-anchor="middle" font-family="sans-serif" font-size="18" font-weight="bold" fill="{_DARK}">H</text>',
                ]
            )
        elif isinstance(operation, ControlledPhase):
            control_y = wire_y[operation.control]
            target_y = wire_y[operation.target]
            label = f"P({operation.angle_pi} pi)"
            parts.extend(
                [
                    f'  <line x1="{x}" y1="{control_y}" x2="{x}" y2="{target_y}" stroke="{_DARK}" stroke-width="2"/>',
                    f'  <circle cx="{x}" cy="{control_y}" r="7" fill="{_DARK}"/>',
                    f'  <rect x="{x - 38}" y="{target_y - 19}" width="76" height="38" rx="5" fill="#ede9fe" stroke="#7c3aed" stroke-width="2"/>',
                    f'  <text x="{x}" y="{target_y + 5}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="{_DARK}">{html.escape(label)}</text>',
                ]
            )
        elif isinstance(operation, SWAP):
            first_y = wire_y[operation.qubit_a]
            second_y = wire_y[operation.qubit_b]
            parts.append(
                f'  <line x1="{x}" y1="{first_y}" x2="{x}" y2="{second_y}" stroke="{_DARK}" stroke-width="2"/>'
            )
            for y in (first_y, second_y):
                parts.extend(
                    [
                        f'  <line x1="{x - 10}" y1="{y - 10}" x2="{x + 10}" y2="{y + 10}" stroke="{_DARK}" stroke-width="2"/>',
                        f'  <line x1="{x - 10}" y1="{y + 10}" x2="{x + 10}" y2="{y - 10}" stroke="{_DARK}" stroke-width="2"/>',
                    ]
                )

    if reference.omitted_operations:
        omitted = ", ".join(
            f"ControlledPhase({operation.angle_pi} pi; q{operation.control},q{operation.target})"
            for operation in reference.omitted_operations
        )
        parts.append(
            f'  <text x="34" y="{height - 28}" font-family="sans-serif" font-size="13" fill="{_AMBER}">Omitted by declared AQFT approximation: {html.escape(omitted)}</text>'
        )
    else:
        parts.append(
            f'  <text x="34" y="{height - 28}" font-family="sans-serif" font-size="13" fill="{_DARK}">All exact high-level QFT-3 operations are shown.</text>'
        )

    title = html.escape(heading)
    description = html.escape(
        "High-level QFT reference diagram; parameterized operations are not native search gates."
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">\n'
        f'  <title id="title">{title}</title>\n'
        f'  <desc id="desc">{description}</desc>\n'
        + "\n".join(parts)
        + "\n</svg>\n"
    )


def _markdown(report: Mapping[str, Any], names: Mapping[str, str]) -> str:
    exact = report["exact_qft3"]
    aqft = report["aqft3"]
    checks = report["checks"]
    lines = [
        "# QFT-3 reference and capability report",
        "",
        f"- Passed: `{report['passed']}`",
        f"- Exact native classification: `{exact['capability']['classification']}`",
        f"- Exact target submitted to native search: `{exact['native_search_target_created']}`",
        f"- AQFT-3 process fidelity: `{aqft['process_fidelity']:.16g}`",
        f"- AQFT-3 maximum matrix error: `{aqft['maximum_matrix_error']:.16g}`",
        "",
        "This benchmark validates analytical/reference QFT semantics and an exact-domain guard. It does not claim that RL or native search synthesized either diagram.",
        "",
        "## Acceptance checks",
        "",
        *[f"- {name}: `{passed}`" for name, passed in checks.items()],
        "",
        "## Files",
        "",
        *[f"- [{label}]({filename})" for label, filename in names.items()],
        "",
    ]
    return "\n".join(lines)


def save_qft_benchmark_artifacts(
    output_dir: str | Path,
    *,
    report: Mapping[str, Any],
    exact_reference: QFTReference,
    approximate_reference: QFTReference,
) -> dict[str, str]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_path / "summary.json",
        "report": output_path / "summary.md",
        "exact_qft3_diagram": output_path / "exact_qft3.svg",
        "aqft3_diagram": output_path / "aqft3.svg",
    }
    names = {label: path.name for label, path in paths.items()}
    summary = dict(report)
    summary["artifacts"] = names
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["report"].write_text(_markdown(summary, names), encoding="utf-8")
    paths["exact_qft3_diagram"].write_text(
        reference_circuit_svg(exact_reference, heading="Exact QFT-3 high-level reference"),
        encoding="utf-8",
    )
    paths["aqft3_diagram"].write_text(
        reference_circuit_svg(
            approximate_reference,
            heading="AQFT-3 high-level reference (pi/4 phase omitted)",
        ),
        encoding="utf-8",
    )
    return {label: str(path.resolve()) for label, path in paths.items()}


__all__ = ["reference_circuit_svg", "save_qft_benchmark_artifacts"]
