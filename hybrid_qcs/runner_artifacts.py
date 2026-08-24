"""Machine-readable artifacts for a hybrid-QCS qualification campaign."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from .benchmarks import all_targets, structured_stress_targets
from .rl import FEATURE_NAMES
from .runner_support import _write_csv


def write_machine_artifacts(output: Path, context: dict[str, Any]) -> None:
    seed_results = list(context["seed_results"])
    baselines = list(context["baselines"])
    scaling = list(context["scaling"])
    component_timing = context["component_timing"]
    summary = {
        "success": bool(context["success"]),
        "hard_deadline_seconds": float(context["hard_deadline"]),
        "deadline_hit": bool(context["deadline_hit"]),
        "cpu_seconds": float(context["cpu_seconds"]),
        "wall_seconds": float(context["wall_seconds"]),
        "seeds": seed_results,
        "baselines": baselines,
        "scaling_profile": scaling,
        "component_timing": component_timing,
        "feature_names": FEATURE_NAMES,
        "scope": {
            "qubits": "1-3",
            "native_gate_set": ["H", "S", "SDG", "T", "TDG", "CNOT"],
            "frontier_state": [
                "persistent dependency DAG",
                "complete forward/inverse Clifford tableau",
                "ordered signed Pauli rotations",
                "global phase and resources",
            ],
            "rl_action": "select one complete persistent frontier record",
            "gate_legality": "target-independent exhaustive native expansion",
            "canonicalizer": "incremental conservative tableau/rotation key",
            "dense_evaluation": "terminal symbolic matches only",
            "qft2_benchmark": (
                "conventional exact forward QFT-2 with final native SWAP; "
                "unrestricted native search"
            ),
            "toffoli_benchmark": (
                "separate certified seven-term CCZ parity-network stress test"
            ),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    targets = [
        {
            "name": target.name,
            "split": target.split,
            "num_qubits": target.num_qubits,
            "budget": asdict(target.budget),
            "generator_length": target.generator_length,
            "target_digest": target.target_digest,
            "family": target.family,
            "convention": target.convention,
            "tableau_payload": [list(row) for row in target.tableau_payload],
            "rotation_payloads": [list(row) for row in target.rotation_payloads],
            "generator_witness_exposed_to_search": False,
        }
        for target in (*all_targets(), *structured_stress_targets())
    ]
    (output / "targets.json").write_text(
        json.dumps(targets, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(
        output / "training.csv",
        list(context["training_csv_rows"]),
        [
            "seed", "episode", "target", "success", "certified", "expansions",
            "total_reward", "mean_abs_td", "epsilon", "cpu_seconds", "wall_seconds",
        ],
    )
    _write_csv(
        output / "evaluations.csv",
        list(context["evaluation_csv_rows"]),
        [
            "seed", "target", "split", "success", "expansions", "frontier_peak",
            "maximum_matrix_error", "gate_count", "t_count", "cnot_count", "depth",
            "witness",
        ],
    )
    _write_csv(
        output / "structured_toffoli.csv",
        list(context["stress_csv_rows"]),
        [
            "seed", "target", "success", "normal_form_complete", "expansions",
            "frontier_peak", "maximum_matrix_error", "gate_count", "t_count",
            "cnot_count", "depth", "witness",
        ],
    )
    _write_csv(
        output / "timing_components.csv",
        [
            {
                "component": component,
                "nanoseconds": values["nanoseconds"],
                "seconds": values["seconds"],
                "share_of_instrumented_time": values["share_of_instrumented_time"],
            }
            for component, values in sorted(
                component_timing["components"].items(),
                key=lambda item: (-item[1]["nanoseconds"], item[0]),
            )
        ],
        ["component", "nanoseconds", "seconds", "share_of_instrumented_time"],
    )
    _write_csv(
        output / "scaling.csv",
        [
            {
                "cap": row["cap"],
                "success": row["success"],
                "expansions": row["expansions"],
                "cpu_seconds": row["cpu_seconds"],
                "wall_seconds": row["wall_seconds"],
                "frontier_peak": row["metrics"]["frontier_peak"],
                "generated": row["metrics"]["generated"],
            }
            for row in scaling
        ],
        [
            "cap", "success", "expansions", "cpu_seconds", "wall_seconds",
            "frontier_peak", "generated",
        ],
    )


__all__ = ["write_machine_artifacts"]
