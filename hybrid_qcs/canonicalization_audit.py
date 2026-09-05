"""Bounded differential audit for the strengthened projective canonicalizer."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import statistics
import time
from typing import Iterable

import numpy as np

from .canonicalize import legacy_projective_key
from .canonicalize import clear_canonicalization_cache, projective_key
from .certify import gate_matrix
from .model import Budget, Gate, HybridState


def _dense_projective_fingerprint(
    matrix: np.ndarray,
    *,
    decimals: int = 11,
    tolerance: float = 1e-12,
) -> tuple[float, ...]:
    flat = matrix.ravel()
    pivot = next((value for value in flat if abs(value) > tolerance), 1.0 + 0.0j)
    normalized = matrix / (pivot / abs(pivot))
    real = np.round(normalized.real, decimals=decimals).ravel()
    imag = np.round(normalized.imag, decimals=decimals).ravel()
    return tuple(float(value) for value in np.concatenate((real, imag)))


def _weakly_dominates(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return all(a <= b for a, b in zip(left, right, strict=True))


def _pareto_survivors(
    rows: Iterable[tuple[tuple[object, ...], tuple[int, ...]]]
) -> int:
    archive: dict[tuple[object, ...], list[tuple[int, ...]]] = {}
    for key, resources in rows:
        group = archive.setdefault(key, [])
        if any(_weakly_dominates(existing, resources) for existing in group):
            continue
        group[:] = [
            existing
            for existing in group
            if not (_weakly_dominates(resources, existing) and resources != existing)
        ]
        group.append(resources)
    return sum(len(group) for group in archive.values())


def _enumerate(
    num_qubits: int,
    max_depth: int,
) -> list[dict[str, object]]:
    single = tuple(
        Gate(name, (q,))
        for q in range(num_qubits)
        for name in ("H", "S", "SDG", "T", "TDG")
    )
    cnot = tuple(
        Gate("CNOT", (control, target))
        for control in range(num_qubits)
        for target in range(num_qubits)
        if control != target
    )
    gates = (*single, *cnot)
    budget = Budget(max_depth, max_depth, max_depth, max_depth)
    identity = np.eye(1 << num_qubits, dtype=np.complex128)
    rows: list[dict[str, object]] = []

    frontier = [(HybridState.identity(num_qubits, budget), identity, tuple())]
    for depth in range(max_depth + 1):
        next_frontier = []
        for state, dense, sequence in frontier:
            rows.append(
                {
                    "depth": depth,
                    "sequence": tuple(gate.label() for gate in sequence),
                    "strong_key": state.canonical_key,
                    "legacy_key": legacy_projective_key(
                        num_qubits, state.tableau, state.rotations
                    ),
                    "dense_key": _dense_projective_fingerprint(dense),
                    "resources": state.resource_vector(),
                    "state": state,
                }
            )
            if depth == max_depth:
                continue
            for gate in gates:
                child = state.apply(gate, partial_order_reduction=False)
                if child is None:
                    continue
                next_frontier.append(
                    (child, gate_matrix(num_qubits, gate) @ dense, (*sequence, gate))
                )
        frontier = next_frontier
    return rows


def _key_timing(rows: list[dict[str, object]], repetitions: int = 5) -> dict[str, float]:
    legacy_samples: list[float] = []
    strong_samples: list[float] = []
    for _ in range(repetitions):
        clear_canonicalization_cache()
        started = time.perf_counter()
        for row in rows:
            state = row["state"]
            assert isinstance(state, HybridState)
            legacy_projective_key(state.num_qubits, state.tableau, state.rotations)
        legacy_samples.append(time.perf_counter() - started)

        clear_canonicalization_cache()
        started = time.perf_counter()
        for row in rows:
            state = row["state"]
            assert isinstance(state, HybridState)
            projective_key(state.num_qubits, state.tableau, state.rotations)
        strong_samples.append(time.perf_counter() - started)
    return {
        "legacy_key_seconds_median": statistics.median(legacy_samples),
        "strong_key_seconds_median": statistics.median(strong_samples),
        "key_time_ratio": statistics.median(strong_samples)
        / max(statistics.median(legacy_samples), 1e-12),
    }


def audit_width(num_qubits: int, max_depth: int) -> dict[str, object]:
    rows = _enumerate(num_qubits, max_depth)
    dense_to_legacy: dict[tuple[float, ...], set[tuple[object, ...]]] = {}
    dense_to_strong: dict[tuple[float, ...], set[tuple[object, ...]]] = {}
    strong_to_dense: dict[tuple[object, ...], set[tuple[float, ...]]] = {}
    for row in rows:
        dense_key = row["dense_key"]
        legacy_key = row["legacy_key"]
        strong_key = row["strong_key"]
        assert isinstance(dense_key, tuple)
        assert isinstance(legacy_key, tuple)
        assert isinstance(strong_key, tuple)
        dense_to_legacy.setdefault(dense_key, set()).add(legacy_key)
        dense_to_strong.setdefault(dense_key, set()).add(strong_key)
        strong_to_dense.setdefault(strong_key, set()).add(dense_key)

    legacy_excess = sum(max(0, len(keys) - 1) for keys in dense_to_legacy.values())
    strong_excess = sum(max(0, len(keys) - 1) for keys in dense_to_strong.values())
    legacy_rows = [(row["legacy_key"], row["resources"]) for row in rows]
    strong_rows = [(row["strong_key"], row["resources"]) for row in rows]
    result: dict[str, object] = {
        "num_qubits": num_qubits,
        "max_depth": max_depth,
        "enumerated_prefixes": len(rows),
        "dense_projective_classes": len(dense_to_legacy),
        "legacy_unique_keys": len({row["legacy_key"] for row in rows}),
        "strong_unique_keys": len({row["strong_key"] for row in rows}),
        "legacy_duplicate_excess": legacy_excess,
        "strong_duplicate_excess": strong_excess,
        "duplicate_excess_reduction_fraction": (
            0.0 if legacy_excess == 0 else 1.0 - strong_excess / legacy_excess
        ),
        "legacy_pareto_survivors": _pareto_survivors(legacy_rows),
        "strong_pareto_survivors": _pareto_survivors(strong_rows),
        "strong_key_safety_violations": sum(
            len(classes) > 1 for classes in strong_to_dense.values()
        ),
    }
    result.update(_key_timing(rows))
    return result


def run(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    widths = [audit_width(1, 5), audit_width(2, 3)]
    summary = {
        "schema": "hybrid-qcs-canonicalization-audit-v1",
        "canonicalizer": (
            "projective Clifford-tableau extraction of even Pauli rotations "
            "plus canonical commuting-word normalization"
        ),
        "claim_boundary": (
            "The dense fingerprint is a bounded differential-test oracle, not "
            "a production archive key. The symbolic canonicalizer remains sound "
            "and intentionally incomplete for general noncommuting Clifford+T identities."
        ),
        "widths": widths,
    }
    (output_dir / "canonicalization_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output_dir / "canonicalization_audit.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(widths[0]))
        writer.writeheader()
        writer.writerows(widths)

    try:
        import matplotlib.pyplot as plt

        labels = [f"{row['num_qubits']}q / depth {row['max_depth']}" for row in widths]
        x = np.arange(len(labels), dtype=float)
        bar_width = 0.36
        figure, axis = plt.subplots(figsize=(7.6, 4.6))
        axis.bar(
            x - bar_width / 2,
            [int(row["legacy_duplicate_excess"]) for row in widths],
            bar_width,
            label="Incremental key",
        )
        axis.bar(
            x + bar_width / 2,
            [int(row["strong_duplicate_excess"]) for row in widths],
            bar_width,
            label="Strengthened key",
        )
        axis.set_ylabel("Unresolved key excess within dense equivalence classes")
        axis.set_xticks(x, labels)
        axis.set_title("Bounded projective-canonicalization audit")
        axis.legend()
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / "canonicalization_audit.png", dpi=180)
        plt.close(figure)
        summary["plot"] = "canonicalization_audit.png"
        (output_dir / "canonicalization_audit.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
    except ImportError:
        summary["plot"] = None
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/canonicalization-audit"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
