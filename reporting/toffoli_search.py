"""Deterministic artifacts for a learned Toffoli parity-network search.

This module is deliberately presentation-only.  In particular, it never
constructs, substitutes, or replays the Stage 2 reference Toffoli witness.
The circuit SVG is rendered only from gates supplied by the Stage 3 search
runner after its learned evaluation.  If no certified learned witness exists,
the SVG explicitly says so.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from circuit.gate import Gate
from reporting.artifacts import circuit_svg


_DARK = "#0f172a"
_GRID = "#cbd5e1"
_LIGHT = "#f8fafc"
_BLUE = "#2563eb"


def _json_ready(value: Any) -> Any:
    """Convert the runner's small numeric/dataclass surface to JSON values."""

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


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    value = _json_ready(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return value


def _mapping_rows(value: Any) -> list[Mapping[str, Any]]:
    """Return only mapping rows, preserving the runner's deterministic order."""

    value = _json_ready(value)
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _write_rows_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    default_fields: Sequence[str],
) -> None:
    """Write a stable CSV, including a useful header for an empty trace."""

    field_set = {str(field) for field in default_fields}
    for row in rows:
        field_set.update(str(field) for field in row)
    fieldnames = tuple(sorted(field_set))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    str(key): _csv_value(value)
                    for key, value in sorted(row.items(), key=lambda item: str(item[0]))
                }
            )


def _report_section(report: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = report.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _run_section(report: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Read a named run from either the compact or grouped runner surface."""

    direct = _report_section(report, key)
    if direct:
        return direct
    return _report_section(_report_section(report, "baselines"), key)


def _trace(report: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    section = _run_section(report, key)
    return _mapping_rows(section.get("trace", report.get(f"{key}_trace", ())))


def _gate_operations(gates: Sequence[Gate]) -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "gate": gate.gate_type.name,
            "qubits": [int(qubit) for qubit in gate.qubits],
        }
        for index, gate in enumerate(gates, start=1)
    ]


def _frontier_size_svg(trace: Sequence[Mapping[str, Any]]) -> str:
    """Render a small dependency-free learned-frontier chart."""

    width, height = 900, 420
    left, right, top, bottom = 76, 32, 64, 72
    chart_width = width - left - right
    chart_height = height - top - bottom
    parts = [
        f'  <text x="{left}" y="32" font-family="sans-serif" font-size="22" fill="{_DARK}">Learned evaluation frontier size</text>',
        f'  <line x1="{left}" y1="{top + chart_height}" x2="{width - right}" y2="{top + chart_height}" stroke="{_DARK}" stroke-width="2"/>',
        f'  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_height}" stroke="{_DARK}" stroke-width="2"/>',
    ]
    points: list[tuple[int, int]] = []
    for index, row in enumerate(trace, start=1):
        try:
            expansion = max(1, int(row.get("expansion", index)))
            frontier_size = max(0, int(row.get("frontier_size", 0)))
        except (TypeError, ValueError):
            continue
        points.append((expansion, frontier_size))
    if not points:
        parts.append(
            f'  <text x="{width / 2}" y="{top + chart_height / 2}" text-anchor="middle" font-family="sans-serif" font-size="16" fill="{_DARK}">No learned evaluation trace was recorded.</text>'
        )
    else:
        max_x = max(expansion for expansion, _ in points)
        max_y = max(1, max(size for _, size in points))
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = top + chart_height * (1.0 - fraction)
            parts.extend(
                [
                    f'  <line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="{_GRID}" stroke-width="1"/>',
                    f'  <text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13" fill="{_DARK}">{int(round(max_y * fraction))}</text>',
                ]
            )
        encoded_points = " ".join(
            f"{left + chart_width * (expansion - 1) / max(1, max_x - 1):.2f},{top + chart_height * (1.0 - size / max_y):.2f}"
            for expansion, size in points
        )
        parts.extend(
            [
                f'  <polyline points="{encoded_points}" fill="none" stroke="{_BLUE}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>',
                f'  <text x="{left}" y="{height - 24}" font-family="sans-serif" font-size="14" fill="{_DARK}">1</text>',
                f'  <text x="{width - right}" y="{height - 24}" text-anchor="end" font-family="sans-serif" font-size="14" fill="{_DARK}">{max_x}</text>',
                f'  <text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="14" fill="{_DARK}">Expansion</text>',
            ]
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">\n'
        '  <title id="title">Learned evaluation frontier size</title>\n'
        '  <desc id="desc">Frontier size over the learned Toffoli evaluation trace.</desc>\n'
        f'  <rect width="100%" height="100%" fill="{_LIGHT}"/>\n'
        + "\n".join(parts)
        + "\n</svg>\n"
    )


def _policy_weight_rows(policy: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Make the linear policy vector reviewable without assuming its schema."""

    raw_weights = _json_ready(policy.get("weights", ()))
    names = _json_ready(policy.get("feature_names", ()))
    name_list = names if isinstance(names, list) else []
    rows: list[dict[str, Any]] = []
    if isinstance(raw_weights, Mapping):
        for index, name in enumerate(sorted(str(key) for key in raw_weights), start=1):
            rows.append({"index": index, "feature": name, "weight": raw_weights[name]})
    elif isinstance(raw_weights, list):
        for index, weight in enumerate(raw_weights):
            feature = name_list[index] if index < len(name_list) else f"feature_{index}"
            rows.append({"index": index, "feature": feature, "weight": weight})
    return rows


def _metric_rows(value: Any) -> list[Mapping[str, Any]]:
    value = _json_ready(value)
    if isinstance(value, Mapping):
        rows = value.get("rows")
        if isinstance(rows, list):
            return _mapping_rows(rows)
        return [
            {"metric": str(name), "value": metric_value}
            for name, metric_value in sorted(value.items(), key=lambda item: str(item[0]))
        ]
    return _mapping_rows(value)


def _markdown_report(report: Mapping[str, Any], artifact_names: Mapping[str, str]) -> str:
    learned = _run_section(report, "learned") or _report_section(report, "evaluation")
    training = _report_section(report, "training") or _report_section(report, "learning")
    problem = _report_section(report, "problem")
    certified = bool(learned.get("certified", report.get("certified", False)))
    shown_witness = _json_ready(report.get("learned_witness", {}).get("operations", ()))
    lines = [
        "# Learned Toffoli parity-network search report",
        "",
        f"- All Stage 3 gates passed: `{bool(report.get('correct', report.get('all_stage3_gates_passed', False)))}`",
        f"- Certified learned witness: `{certified and bool(shown_witness)}`",
        f"- Search expansions: `{learned.get('expansions', report.get('expansions', 0))}`",
        f"- Training episodes: `{training.get('episodes', report.get('episodes', 0))}`",
        f"- Seed: `{training.get('seed', report.get('seed'))}`",
        f"- Problem: `{problem.get('name', 'ToffoliParityNetworkProblem')}`",
        "",
        "Exact Toffoli synthesis within the fixed seven-term CCZ parity-network normal form; not a proof of general unconstrained Clifford+T synthesis.",
        "",
        "## Learned witness",
        "",
        "```text",
    ]
    if certified and shown_witness:
        lines.extend(str(operation) for operation in shown_witness)
    else:
        lines.append("No certified learned witness was returned. No reference witness is shown.")
    lines.extend(
        [
            "```",
            "",
            "## Interpretation",
            "",
            "The learned policy schedules persistent frontier records. Gate expansion, parity-network progress, archive reduction, and certification remain deterministic search operations. The circuit diagram contains only the witness returned by the learned evaluation; it is never replaced with a known reference circuit.",
            "",
            "## Files",
            "",
            *[f"- [{name}]({path})" for name, path in artifact_names.items()],
            "",
        ]
    )
    return "\n".join(lines)


def save_toffoli_search_artifacts(
    output_dir: str | Path,
    *,
    report: Mapping[str, Any],
    learned_witness_gates: Sequence[Gate],
    seed: int = 23,
    num_qubits: int = 3,
) -> dict[str, str]:
    """Write the complete deterministic Stage 3 artifact contract.

    ``learned_witness_gates`` must be reconstructed from the learned
    evaluation's ``solution_node``.  The helper shows those gates only if the
    learned evaluation is certified.  Thus a caller cannot accidentally
    display an uncertified prefix or a substituted known witness on failure.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int) or num_qubits < 1:
        raise ValueError("num_qubits must be a positive integer")
    gates = tuple(learned_witness_gates)
    if any(not isinstance(gate, Gate) for gate in gates):
        raise TypeError("learned_witness_gates must contain Gate values")
    for gate in gates:
        if any(
            isinstance(qubit, bool)
            or not isinstance(qubit, int)
            or not 0 <= qubit < num_qubits
            for qubit in gate.qubits
        ):
            raise ValueError("learned witness contains a qubit outside the report register")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    learned = _run_section(report, "learned") or _report_section(report, "evaluation")
    policy = _report_section(report, "policy")
    learned_certified = bool(learned.get("certified", report.get("certified", False)))
    # This guards the artifact boundary: an unsuccessful run never gets a
    # partial candidate circuit, and this module has no Stage 2 fallback.
    shown_gates = gates if learned_certified and gates else ()
    learned_trace = _trace(report, "learned")
    if not learned_trace:
        learned_trace = _mapping_rows(learned.get("trace", report.get("evaluation_trace", ())))
    training_history = _mapping_rows(report.get("training_history", ()))
    truth_table = _mapping_rows(report.get("truth_table", ()))
    phase_identity = report.get(
        "phase_identity",
        {
            "passed": False,
            "available": False,
            "reason": "the Stage 3 runner did not report a learned-candidate phase check",
        },
    )
    resource_summary = report.get(
        "resource_summary",
        learned.get("resource_summary", learned.get("resources", {})),
    )
    search_metrics = _metric_rows(report.get("search_metrics", {}))

    paths = {
        "summary": output_path / "summary.json",
        "summary_markdown": output_path / "summary.md",
        "phase_identity": output_path / "phase_identity.json",
        "fifo_trace": output_path / "fifo_trace.csv",
        "uniform_trace": output_path / "uniform_trace.csv",
        "random_trace": output_path / f"random_trace_seed_{seed}.csv",
        "zero_policy_trace": output_path / "zero_policy_trace.csv",
        "training_history": output_path / "training_history.csv",
        "learned_trace": output_path / "learned_trace.csv",
        "policy": output_path / "policy.json",
        "policy_weights": output_path / "policy_weights.csv",
        "truth_table": output_path / "truth_table.csv",
        "resource_summary": output_path / "resource_summary.json",
        "search_metrics": output_path / "search_metrics.csv",
        "circuit": output_path / "circuit.svg",
        "frontier_size": output_path / "frontier_size.svg",
    }
    artifact_names = {name: path.name for name, path in paths.items()}
    trace_fields = (
        "expansion",
        "selected_record_id",
        "frontier_size",
        "num_children",
        "num_accepted",
        "num_pruned",
        "reward",
        "stage",
        "basis_rows",
        "emitted_terms",
        "is_terminal_candidate",
        "certified",
    )
    for path_key, report_key in (
        ("fifo_trace", "fifo"),
        ("uniform_trace", "uniform"),
        ("random_trace", "random"),
        ("zero_policy_trace", "zero_policy"),
    ):
        _write_rows_csv(paths[path_key], _trace(report, report_key), default_fields=trace_fields)
    _write_rows_csv(
        paths["training_history"],
        training_history,
        default_fields=("episode", "reward", "steps", "certified", "truncated", "epsilon"),
    )
    _write_rows_csv(paths["learned_trace"], learned_trace, default_fields=trace_fields)
    _write_rows_csv(
        paths["policy_weights"],
        _policy_weight_rows(policy),
        default_fields=("index", "feature", "weight"),
    )
    _write_rows_csv(
        paths["truth_table"],
        truth_table,
        default_fields=(
            "input_index",
            "input_bits",
            "expected_output_index",
            "expected_output_bits",
            "candidate_output_index",
            "candidate_output_bits",
            "mapping_correct",
            "phase_identity",
        ),
    )
    _write_rows_csv(paths["search_metrics"], search_metrics, default_fields=("metric", "value"))
    _write_json(paths["phase_identity"], phase_identity)
    _write_json(paths["policy"], policy)
    _write_json(paths["resource_summary"], resource_summary)
    paths["circuit"].write_text(
        circuit_svg(
            shown_gates,
            num_qubits,
            heading=(
                "Generated learned Toffoli witness"
                if shown_gates
                else "No certified learned Toffoli witness"
            ),
            description=(
                "Actual gate sequence reconstructed from the learned evaluation's "
                "solution node; q0 is the least-significant simulation bit."
                if shown_gates
                else "The learned evaluation returned no certified solution node; no "
                "reference Toffoli circuit is displayed."
            ),
            empty_heading="No certified learned witness was returned; no reference circuit is shown.",
        ),
        encoding="utf-8",
    )
    paths["frontier_size"].write_text(_frontier_size_svg(learned_trace), encoding="utf-8")

    summary = _json_ready(dict(report))
    summary["artifacts"] = artifact_names
    summary["learned_witness"] = {
        "certified": bool(shown_gates),
        "operations": _gate_operations(shown_gates),
        "source": "learned evaluation solution_node only",
    }
    _write_json(paths["summary"], summary)
    paths["summary_markdown"].write_text(
        _markdown_report(summary, artifact_names), encoding="utf-8"
    )
    return {name: str(path.resolve()) for name, path in paths.items()}


__all__ = ["save_toffoli_search_artifacts"]
