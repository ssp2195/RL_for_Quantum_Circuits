"""Run the mixed Clifford+T continuation-cost crossover experiment."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import platform
import random
import statistics
import sys
from typing import Callable

import numpy as np

from .mixed_crossover import (
    GATE_FAMILIES,
    MixedEvaluationResult,
    evaluate_mixed_deferred_native,
    evaluate_mixed_eager_sarsa,
    evaluate_mixed_hierarchy,
    evaluate_mixed_target_potential,
    mixed_evaluation_targets,
    mixed_gate_library,
    mixed_training_targets,
    train_mixed_inner_bandit,
    train_mixed_outer_sarsa,
)
from .pauli import clear_algebra_caches


METHOD_LABELS = {
    "mixed_eager_outer_sarsa": "Eager outer SARSA",
    "mixed_deferred_outer_sarsa_native_order": "Deferred SARSA + native order",
    "mixed_deferred_outer_sarsa_inner_linucb": "Deferred SARSA + LinUCB",
    "mixed_eager_target_potential": "Eager target potential",
}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"cannot interpret {value!r} as a Boolean")


def _quartiles(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("quartiles require at least one value")
    if len(values) == 1:
        return values[0], values[0]
    cuts = statistics.quantiles(values, n=4, method="inclusive")
    return float(cuts[0]), float(cuts[2])


def _fmt_float(value: object, digits: int = 6) -> str:
    if value == "" or value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _fmt_count(value: object) -> str:
    if value == "" or value is None:
        return "—"
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else f"{numeric:.1f}"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), str(row["target"])), []).append(row)

    result: list[dict[str, object]] = []
    for (method, target), values in sorted(grouped.items()):
        successful = [value for value in values if _as_bool(value["success"])]
        observed_walls = [float(value["wall_seconds"]) for value in values]
        success_walls = [float(value["wall_seconds"]) for value in successful]
        observed_q1, observed_q3 = _quartiles(observed_walls)
        if success_walls:
            success_q1, success_q3 = _quartiles(success_walls)
        else:
            success_q1 = success_q3 = ""
        result.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "target": target,
                "num_qubits": int(values[0]["num_qubits"]),
                "native_actions": len(mixed_gate_library(int(values[0]["num_qubits"]))),
                "generator_length": int(values[0]["generator_length"]),
                "runs": len(values),
                "success_rate": sum(_as_bool(value["success"]) for value in values)
                / len(values),
                "median_observed_wall_seconds": statistics.median(observed_walls),
                "q1_observed_wall_seconds": observed_q1,
                "q3_observed_wall_seconds": observed_q3,
                "median_observed_cpu_seconds": statistics.median(
                    float(value["cpu_seconds"]) for value in values
                ),
                "median_observed_attempted_edges": statistics.median(
                    int(value["attempted_edges"]) for value in values
                ),
                "median_observed_outer_decisions": statistics.median(
                    int(value["outer_decisions"]) for value in values
                ),
                "median_observed_frontier_peak": statistics.median(
                    int(value["frontier_peak"]) for value in values
                ),
                "median_observed_policy_rows": statistics.median(
                    int(value["policy_rows"]) for value in values
                ),
                "median_success_wall_seconds": (
                    "" if not successful else statistics.median(success_walls)
                ),
                "q1_success_wall_seconds": success_q1,
                "q3_success_wall_seconds": success_q3,
                "median_success_cpu_seconds": (
                    ""
                    if not successful
                    else statistics.median(
                        float(value["cpu_seconds"]) for value in successful
                    )
                ),
                "median_attempted_edges": (
                    ""
                    if not successful
                    else statistics.median(
                        int(value["attempted_edges"]) for value in successful
                    )
                ),
                "median_outer_decisions": (
                    ""
                    if not successful
                    else statistics.median(
                        int(value["outer_decisions"]) for value in successful
                    )
                ),
                "median_frontier_peak": (
                    ""
                    if not successful
                    else statistics.median(
                        int(value["frontier_peak"]) for value in successful
                    )
                ),
                "median_policy_rows": (
                    ""
                    if not successful
                    else statistics.median(
                        int(value["policy_rows"]) for value in successful
                    )
                ),
                "stop_reasons": ";".join(
                    sorted({str(value["stop_reason"]) for value in values})
                ),
            }
        )
    return result


def _lookup(
    aggregate: list[dict[str, object]],
    method: str,
    target: str,
    field: str,
) -> float | None:
    for row in aggregate:
        if row["method"] == method and row["target"] == target:
            value = row[field]
            return None if value == "" else float(value)
    return None


def _make_plots(output: Path, aggregate: list[dict[str, object]]) -> list[str]:
    import matplotlib.pyplot as plt

    targets = sorted(
        {str(row["target"]) for row in aggregate},
        key=lambda target: next(
            int(row["num_qubits"])
            for row in aggregate
            if row["target"] == target
        ),
    )
    methods = tuple(METHOD_LABELS)
    x = np.arange(len(targets), dtype=float)
    width = 0.19
    paths: list[str] = []

    fig, ax = plt.subplots(figsize=(11, 5.8))
    for index, method in enumerate(methods):
        values = [
            _lookup(aggregate, method, target, "median_success_wall_seconds")
            for target in targets
        ]
        ax.bar(
            x + (index - 1.5) * width,
            [np.nan if value is None else value for value in values],
            width,
            label=METHOD_LABELS[method],
        )
    ax.set_yscale("log")
    ax.set_ylabel("Median wall time to certified circuit (s, log scale)")
    ax.set_xticks(
        x,
        [target.replace("heldout-", "") for target in targets],
        rotation=12,
    )
    ax.set_title("Mixed Clifford+T continuation-cost crossover")
    ax.legend()
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    path = output / "mixed_wall_time_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path.name)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    for index, method in enumerate(methods):
        values = [
            _lookup(aggregate, method, target, "median_observed_attempted_edges")
            for target in targets
        ]
        ax.bar(
            x + (index - 1.5) * width,
            [np.nan if value is None else value for value in values],
            width,
            label=METHOD_LABELS[method],
        )
    ax.set_yscale("log")
    ax.set_ylabel("Median exact continuation attempts (log scale)")
    ax.set_xticks(
        x,
        [target.replace("heldout-", "") for target in targets],
        rotation=12,
    )
    ax.set_title("Exact mixed-gate work before certification")
    ax.legend()
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    path = output / "mixed_edge_work_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path.name)

    speedups: list[tuple[str, str, float]] = []
    for target in targets:
        hierarchy = _lookup(
            aggregate,
            "mixed_deferred_outer_sarsa_inner_linucb",
            target,
            "median_success_wall_seconds",
        )
        if hierarchy is None or hierarchy <= 0:
            continue
        for baseline in (
            "mixed_eager_outer_sarsa",
            "mixed_deferred_outer_sarsa_native_order",
            "mixed_eager_target_potential",
        ):
            value = _lookup(
                aggregate, baseline, target, "median_success_wall_seconds"
            )
            if value is not None:
                speedups.append((target, METHOD_LABELS[baseline], value / hierarchy))
    if speedups:
        fig, ax = plt.subplots(figsize=(12, 6.2))
        ax.bar(np.arange(len(speedups)), [value for _, _, value in speedups])
        ax.axhline(1.0, linewidth=1.0)
        ax.set_yscale("log")
        ax.set_ylabel("Baseline median / hierarchy median (log scale)")
        ax.set_xticks(
            np.arange(len(speedups)),
            [
                f"{target.replace('heldout-', '')}\nvs {baseline}"
                for target, baseline, _ in speedups
            ],
            rotation=18,
        )
        ax.set_title("Hierarchy wall-time speedup against certified baselines")
        ax.grid(axis="y", which="both", alpha=0.25)
        fig.tight_layout()
        path = output / "mixed_hierarchical_speedup.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path.name)
    return paths


def _build_report(
    output: Path,
    aggregate: list[dict[str, object]],
    config: dict[str, object],
) -> str:
    targets = sorted(
        {str(row["target"]) for row in aggregate},
        key=lambda target: next(
            int(row["num_qubits"])
            for row in aggregate
            if row["target"] == target
        ),
    )
    lines = [
        "# Mixed Clifford+T continuation-cost crossover",
        "",
        "## Shortlisted problem",
        "",
        "The benchmark synthesizes short Clifford-frame signed phase-pair motifs over the complete native grammar `H, S, SDG, T, TDG, CNOT`. Each held-out target contains a nontrivial Clifford scaffold and both positive and negative non-Clifford phase injections. The hidden native witness is used only to construct and validate the target; it is not exposed to either scheduler.",
        "",
        "The old and new methods use the same frozen linear outer SARSA policy. The old method exhaustively attempts every native continuation after selecting a record. The new method retains pending exact continuations and uses a frozen disjoint linear LinUCB policy to rank a batch. A deferred native-order control isolates the benefit of lazy expansion from the benefit of learned continuation ordering.",
        "",
        "## Aggregate results",
        "",
        "| Target | Qubits | Actions/node | Method | Success consistency | Median observed wall (s) | Median certified wall (s) | Median attempted edges | Outer decisions | Peak frontier | Stop reason |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for target in targets:
        for row in aggregate:
            if row["target"] != target:
                continue
            lines.append(
                f"| {target} | {row['num_qubits']} | {row['native_actions']} | "
                f"{row['method_label']} | {float(row['success_rate']):.3f} | "
                f"{_fmt_float(row['median_observed_wall_seconds'])} | "
                f"{_fmt_float(row['median_success_wall_seconds'])} | "
                f"{_fmt_count(row['median_observed_attempted_edges'])} | "
                f"{_fmt_count(row['median_observed_outer_decisions'])} | "
                f"{_fmt_count(row['median_observed_frontier_peak'])} | "
                f"{row['stop_reasons']} |"
            )

    lines.extend(
        [
            "",
            "The repeated runs use the same frozen policies and targets; the success column therefore measures execution consistency, not independent-policy generalization uncertainty. Timings are medians, and the raw CSV retains every repetition.",
            "",
            "## Crossover interpretation",
            "",
        ]
    )
    for target in targets:
        hierarchy = _lookup(
            aggregate,
            "mixed_deferred_outer_sarsa_inner_linucb",
            target,
            "median_success_wall_seconds",
        )
        if hierarchy is None:
            lines.append(f"- **{target}:** the hierarchy did not certify within the bound.")
            continue
        eager = _lookup(
            aggregate,
            "mixed_eager_outer_sarsa",
            target,
            "median_success_wall_seconds",
        )
        native = _lookup(
            aggregate,
            "mixed_deferred_outer_sarsa_native_order",
            target,
            "median_success_wall_seconds",
        )
        target_potential = _lookup(
            aggregate,
            "mixed_eager_target_potential",
            target,
            "median_success_wall_seconds",
        )
        statements = []
        if eager is not None:
            statements.append(f"{eager / hierarchy:.2f}x versus eager outer SARSA")
        if native is not None:
            statements.append(f"{native / hierarchy:.2f}x versus deferred native order")
        if target_potential is not None:
            statements.append(
                f"{target_potential / hierarchy:.2f}x versus eager target potential"
            )
        hierarchy_edges = _lookup(
            aggregate,
            "mixed_deferred_outer_sarsa_inner_linucb",
            target,
            "median_observed_attempted_edges",
        )
        eager_edges = _lookup(
            aggregate,
            "mixed_eager_outer_sarsa",
            target,
            "median_observed_attempted_edges",
        )
        if hierarchy_edges is not None and eager_edges is not None and eager_edges > 0:
            reduction = 100.0 * (1.0 - hierarchy_edges / eager_edges)
            statements.append(f"{reduction:.1f}% fewer exact edge attempts than eager SARSA")
        if not statements:
            statements.append("no certified timing comparison is available")
        lines.append(f"- **{target}:** " + "; ".join(statements) + ".")

    faster_than_native = 0
    lower_edge_work_than_native = 0
    comparable_targets = 0
    for target in targets:
        hierarchy_wall = _lookup(
            aggregate,
            "mixed_deferred_outer_sarsa_inner_linucb",
            target,
            "median_success_wall_seconds",
        )
        native_wall = _lookup(
            aggregate,
            "mixed_deferred_outer_sarsa_native_order",
            target,
            "median_success_wall_seconds",
        )
        hierarchy_edges = _lookup(
            aggregate,
            "mixed_deferred_outer_sarsa_inner_linucb",
            target,
            "median_observed_attempted_edges",
        )
        native_edges = _lookup(
            aggregate,
            "mixed_deferred_outer_sarsa_native_order",
            target,
            "median_observed_attempted_edges",
        )
        if hierarchy_wall is not None and native_wall is not None:
            comparable_targets += 1
            faster_than_native += int(hierarchy_wall < native_wall)
        if hierarchy_edges is not None and native_edges is not None:
            lower_edge_work_than_native += int(hierarchy_edges < native_edges)

    lines.extend(
        [
            "",
            (
                f"In this recorded frozen-policy run, LinUCB used fewer exact "
                f"continuations than deferred native ordering on "
                f"{lower_edge_work_than_native}/{len(targets)} targets and was "
                f"faster in wall time on {faster_than_native}/{comparable_targets} "
                f"certified comparisons. These selected cases demonstrate a "
                f"continuation-cost crossover; they are not a universal or "
                f"statistical claim over arbitrary target distributions."
            ),
        ]
    )

    lines.extend(
        [
            "",
            "## Scope and claim boundary",
            "",
            "This is unrestricted search within the declared finite Clifford+T grammar and resource envelope; it is not a prescribed normal form. The target family is deliberately selected so that continuation cost grows with register width while exact solutions remain short enough to certify. It demonstrates a continuation-cost crossover, not universal superiority on arbitrary unitaries.",
            "",
            "The four-qubit case is a lower-width control. The five- and six-qubit cases test whether the saved symbolic transitions, canonical keys, archive operations, and frontier growth amortize the extra LinUCB context cost. The six-qubit target is OOD in width because outer and inner training use only four- and five-qubit targets.",
            "",
            "## Reproduction configuration",
            "",
            "```json",
            json.dumps(config, indent=2, sort_keys=True),
            "```",
        ]
    )
    text = "\n".join(lines) + "\n"
    (output / "REPORT.md").write_text(text, encoding="utf-8")
    return text


def _timed_call(call: Callable[[], MixedEvaluationResult]) -> MixedEvaluationResult:
    gc.collect()
    clear_algebra_caches()
    return call()


def run(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    training = mixed_training_targets()
    evaluation = mixed_evaluation_targets()
    training_keys = {target.canonical_key for target in training}
    if any(target.canonical_key in training_keys for target in evaluation):
        raise AssertionError("held-out mixed target duplicates a training target")

    outer = train_mixed_outer_sarsa(
        training,
        episodes=args.outer_episodes,
        seed=args.outer_seed,
        max_expansions=args.train_expansion_cap,
    )
    bandit = train_mixed_inner_bandit(
        training,
        episodes=args.inner_episodes,
        alpha=args.linucb_alpha,
    )

    policy_arrays: dict[str, np.ndarray] = {"outer_theta": outer.theta}
    for family in GATE_FAMILIES:
        policy_arrays[f"inner_a_inverse_{family}"] = bandit.a_inverse[family]
        policy_arrays[f"inner_b_{family}"] = bandit.b_vector[family]
        policy_arrays[f"inner_mean_{family}"] = bandit.posterior_mean(family)
    np.savez(output / "mixed_policies.npz", **policy_arrays)

    raw: list[dict[str, object]] = []
    rng = random.Random(args.order_seed)
    for target in evaluation:
        stress = target.num_qubits >= 6
        cap = args.stress_cap if stress else args.max_expansions
        wall_limit = args.stress_wall_limit if stress else args.wall_limit
        repeated_methods: list[tuple[str, Callable[[], MixedEvaluationResult]]] = [
            (
                "mixed_eager_outer_sarsa",
                lambda target=target, cap=cap, wall_limit=wall_limit: evaluate_mixed_eager_sarsa(
                    outer, target, max_expansions=cap, wall_limit=wall_limit
                ),
            ),
            (
                "mixed_deferred_outer_sarsa_native_order",
                lambda target=target, cap=cap, wall_limit=wall_limit: evaluate_mixed_deferred_native(
                    outer,
                    target,
                    batch_size=args.batch_size,
                    max_allocations=cap,
                    wall_limit=wall_limit,
                ),
            ),
            (
                "mixed_deferred_outer_sarsa_inner_linucb",
                lambda target=target, cap=cap, wall_limit=wall_limit: evaluate_mixed_hierarchy(
                    outer,
                    bandit,
                    target,
                    batch_size=args.batch_size,
                    max_allocations=cap,
                    wall_limit=wall_limit,
                ),
            ),
        ]
        target_potential_method = (
            "mixed_eager_target_potential",
            lambda target=target, cap=cap, wall_limit=wall_limit: evaluate_mixed_target_potential(
                target, max_expansions=cap, wall_limit=wall_limit
            ),
        )
        repetitions = args.stress_repetitions if stress else args.repetitions
        for repetition in range(repetitions):
            shuffled = list(repeated_methods)
            if repetition < args.target_potential_repetitions:
                shuffled.append(target_potential_method)
            rng.shuffle(shuffled)
            for _, evaluate in shuffled:
                result = _timed_call(evaluate)
                row = result.to_dict()
                row["repetition"] = repetition
                raw.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
            _write_csv(output / "mixed_raw_runs.partial.csv", raw)

    aggregate = _aggregate(raw)
    config: dict[str, object] = {
        "benchmark": "mixed-clifford-t-continuation-crossover-v1",
        "native_gate_set": list(GATE_FAMILIES),
        "training_widths": [4, 5],
        "heldout_widths": [4, 5, 6],
        "training_target_count": len(training),
        "heldout_target_count": len(evaluation),
        "outer_policy": "shared frozen linear semi-gradient SARSA",
        "inner_policy": "frozen disjoint linear LinUCB posterior mean",
        "outer_episodes": args.outer_episodes,
        "inner_episodes": args.inner_episodes,
        "outer_seed": args.outer_seed,
        "linucb_alpha_during_training": args.linucb_alpha,
        "linucb_alpha_during_evaluation": 0.0,
        "batch_size": args.batch_size,
        "train_expansion_cap": args.train_expansion_cap,
        "evaluation_cap": args.max_expansions,
        "stress_cap": args.stress_cap,
        "wall_limit_seconds": args.wall_limit,
        "stress_wall_limit_seconds": args.stress_wall_limit,
        "repetitions": args.repetitions,
        "stress_repetitions": args.stress_repetitions,
        "target_potential_repetitions": args.target_potential_repetitions,
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "environment": {
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "machine": platform.machine(),
        },
    }
    _write_csv(output / "mixed_raw_runs.csv", raw)
    _write_csv(output / "mixed_aggregate.csv", aggregate)
    (output / "mixed_raw_runs.partial.csv").unlink(missing_ok=True)
    plots = _make_plots(output, aggregate)
    summary = {
        "config": config,
        "aggregate": aggregate,
        "plots": plots,
    }
    (output / "mixed_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _build_report(output, aggregate, config)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/mixed-crossover")
    parser.add_argument("--outer-episodes", type=int, default=64)
    parser.add_argument("--inner-episodes", type=int, default=96)
    parser.add_argument("--outer-seed", type=int, default=11)
    parser.add_argument("--order-seed", type=int, default=20260904)
    parser.add_argument("--linucb-alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-expansion-cap", type=int, default=128)
    parser.add_argument("--max-expansions", type=int, default=2_048)
    parser.add_argument("--stress-cap", type=int, default=512)
    parser.add_argument("--wall-limit", type=float, default=5.0)
    parser.add_argument("--stress-wall-limit", type=float, default=2.0)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--stress-repetitions", type=int, default=3)
    parser.add_argument("--target-potential-repetitions", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
