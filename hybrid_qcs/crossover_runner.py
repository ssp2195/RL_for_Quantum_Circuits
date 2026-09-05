"""Run and report the eager-versus-hierarchical CNOT crossover experiment."""
from __future__ import annotations

import argparse
import csv
import gc
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
from typing import Any

import numpy as np

from .cnot_crossover import (
    EvaluationResult,
    crossover_evaluation_targets,
    crossover_training_targets,
    evaluate_eager_sarsa,
    evaluate_eager_target_potential,
    evaluate_hierarchy,
    train_deferred_outer,
    train_eager_outer,
    train_inner_bandit,
    unrestricted_toffoli_probe,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), str(row["target"])), []).append(row)
    result: list[dict[str, object]] = []
    for (method, target), values in sorted(grouped.items()):
        successes = [bool(value["success"]) for value in values]
        successful = [value for value in values if bool(value["success"])]
        result.append(
            {
                "method": method,
                "target": target,
                "num_qubits": int(values[0]["num_qubits"]),
                "generator_length": int(values[0]["generator_length"]),
                "runs": len(values),
                "success_rate": sum(successes) / len(successes),
                "median_observed_wall_seconds": statistics.median(
                    float(v["wall_seconds"]) for v in values
                ),
                "median_observed_cpu_seconds": statistics.median(
                    float(v["cpu_seconds"]) for v in values
                ),
                "median_wall_seconds": (
                    ""
                    if not successful
                    else statistics.median(float(v["wall_seconds"]) for v in successful)
                ),
                "median_cpu_seconds": (
                    ""
                    if not successful
                    else statistics.median(float(v["cpu_seconds"]) for v in successful)
                ),
                "median_outer_decisions": (
                    ""
                    if not successful
                    else statistics.median(int(v["outer_decisions"]) for v in successful)
                ),
                "median_attempted_edges": (
                    ""
                    if not successful
                    else statistics.median(int(v["attempted_edges"]) for v in successful)
                ),
                "median_generated": (
                    ""
                    if not successful
                    else statistics.median(int(v["generated"]) for v in successful)
                ),
                "median_frontier_peak": (
                    ""
                    if not successful
                    else statistics.median(int(v["frontier_peak"]) for v in successful)
                ),
                "stop_reasons": ";".join(sorted({str(v["stop_reason"]) for v in values})),
            }
        )
    return result


def _lookup(
    aggregate: list[dict[str, object]], method: str, target: str, field: str
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
        key=lambda name: next(
            int(row["num_qubits"])
            for row in aggregate
            if str(row["target"]) == name
        ),
    )
    methods = (
        "eager_outer_sarsa",
        "eager_target_potential",
        "deferred_outer_sarsa_inner_linucb",
    )
    labels = {
        "eager_outer_sarsa": "Eager outer SARSA",
        "eager_target_potential": "Eager target potential",
        "deferred_outer_sarsa_inner_linucb": "Deferred SARSA + LinUCB",
    }
    x = np.arange(len(targets), dtype=float)
    width = 0.24
    plot_paths: list[str] = []

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, method in enumerate(methods):
        values = [
            _lookup(aggregate, method, target, "median_wall_seconds")
            for target in targets
        ]
        heights = [np.nan if value is None else value for value in values]
        ax.bar(x + (index - 1) * width, heights, width, label=labels[method])
    ax.set_yscale("log")
    ax.set_ylabel("Median wall time to certification (s, log scale)")
    ax.set_xticks(x, [target.replace("heldout-", "") for target in targets], rotation=12)
    ax.set_title("Crossover benchmark: wall time")
    ax.legend()
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    path = output / "wall_time_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plot_paths.append(path.name)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for index, method in enumerate(methods):
        values = [
            _lookup(aggregate, method, target, "median_attempted_edges")
            for target in targets
        ]
        heights = [np.nan if value is None else value for value in values]
        ax.bar(x + (index - 1) * width, heights, width, label=labels[method])
    ax.set_yscale("log")
    ax.set_ylabel("Median exact continuation attempts (log scale)")
    ax.set_xticks(x, [target.replace("heldout-", "") for target in targets], rotation=12)
    ax.set_title("Exact work performed before certification")
    ax.legend()
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    path = output / "edge_work_comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    plot_paths.append(path.name)

    speedup_rows = []
    for target in targets:
        new_time = _lookup(
            aggregate,
            "deferred_outer_sarsa_inner_linucb",
            target,
            "median_wall_seconds",
        )
        if new_time is None or new_time <= 0:
            continue
        for method in ("eager_outer_sarsa", "eager_target_potential"):
            baseline = _lookup(aggregate, method, target, "median_wall_seconds")
            if baseline is not None:
                speedup_rows.append((target, labels[method], baseline / new_time))
    if speedup_rows:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        tick_labels = [
            f"{target.replace('heldout-', '')}\nvs {label}"
            for target, label, _ in speedup_rows
        ]
        ax.bar(np.arange(len(speedup_rows)), [row[2] for row in speedup_rows])
        ax.axhline(1.0, linewidth=1.0)
        ax.set_yscale("log")
        ax.set_ylabel("Wall-time speedup of hierarchy (log scale)")
        ax.set_xticks(np.arange(len(speedup_rows)), tick_labels, rotation=18)
        ax.set_title("Hierarchical speedup relative to eager baselines")
        ax.grid(axis="y", which="both", alpha=0.25)
        fig.tight_layout()
        path = output / "hierarchical_speedup.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        plot_paths.append(path.name)
    return plot_paths


def _report(
    output: Path,
    aggregate: list[dict[str, object]],
    toffoli: dict[str, object],
    config: dict[str, object],
) -> str:
    targets = sorted(
        {str(row["target"]) for row in aggregate},
        key=lambda name: next(
            int(row["num_qubits"])
            for row in aggregate
            if str(row["target"]) == name
        ),
    )
    lines = [
        "# Exact CNOT-network crossover report",
        "",
        "## Question",
        "",
        "Can fair deferred continuation scheduling with an outer linear SARSA policy and an inner linear LinUCB policy become faster than atomic eager frontier expansion as register width and branching increase?",
        "",
        "## Benchmark definition",
        "",
        "The search grammar contains every directed CNOT on the register. The target witness is used only to construct and validate the target unitary; it is not supplied to either policy. The evaluated targets are multi-wire permutations requiring two or three SWAP compositions. This is unrestricted CNOT-network synthesis, not unrestricted Clifford+T synthesis.",
        "",
        "The eager baseline selects a frontier record and immediately attempts all n(n-1) directed CNOT continuations. The hierarchical system selects a frontier record, ranks its pending continuations with a frozen linear contextual bandit, and processes at most four exact edges before the next outer SARSA decision.",
        "",
        "## Results",
        "",
        "| Target | Qubits | Method | Success | Median wall (s) | Median attempts | Outer decisions | Frontier peak |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for target in targets:
        for row in aggregate:
            if str(row["target"]) != target:
                continue
            wall = row["median_wall_seconds"]
            attempts = row["median_attempted_edges"]
            decisions = row["median_outer_decisions"]
            peak = row["median_frontier_peak"]
            lines.append(
                f"| {target} | {row['num_qubits']} | {row['method']} | "
                f"{float(row['success_rate']):.3f} | {wall} | {attempts} | "
                f"{decisions} | {peak} |"
            )
    lines.extend(["", "## Speedups", ""])
    for target in targets:
        new_time = _lookup(
            aggregate,
            "deferred_outer_sarsa_inner_linucb",
            target,
            "median_wall_seconds",
        )
        if new_time is None or new_time <= 0:
            lines.append(f"- **{target}:** hierarchy did not certify within the bound.")
            continue
        old_time = _lookup(
            aggregate, "eager_outer_sarsa", target, "median_wall_seconds"
        )
        tp_time = _lookup(
            aggregate, "eager_target_potential", target, "median_wall_seconds"
        )
        old_text = (
            "old eager SARSA did not certify within the bound"
            if old_time is None
            else f"{old_time / new_time:.2f}x versus eager SARSA"
        )
        tp_text = (
            "target potential did not certify within the bound"
            if tp_time is None
            else f"{tp_time / new_time:.2f}x versus eager target potential"
        )
        lines.append(f"- **{target}:** {old_text}; {tp_text}.")
    lines.extend(
        [
            "",
            "## Unrestricted Toffoli feasibility probe",
            "",
            "The analytical three-qubit Toffoli target was also passed to the ordinary unrestricted native HybridSearch rather than the structured parity-network adapter. The result was:",
            "",
            "```json",
            json.dumps(toffoli, indent=2, sort_keys=True),
            "```",
            "",
            "This probe is intentionally not used to claim a hierarchical speedup. If the unrestricted search does not find Toffoli at the cap, comparing two failed policies would not identify the benefit of deferred edge scheduling.",
            "",
            "## Interpretation",
            "",
            "The permutation family creates the intended crossover for a principled reason: branching grows as n(n-1), while an exact solution needs only a short sequence of useful edges. Eager expansion pays for all siblings at every selected record. Deferred expansion can terminate after processing a small fraction of those siblings.",
            "",
            "The experiment demonstrates an architectural crossover, not universal superiority. The inner context contains a cheap algebraic measure of how a candidate CNOT changes the induced binary linear map. A deterministic continuation rule can exploit the same structure, so the result must not be presented as proof that learning is essential. The relevant conclusion is that continuation-level control can become faster once exact sibling work dominates policy overhead.",
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


def run(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    training_targets = crossover_training_targets()
    evaluation_targets = crossover_evaluation_targets()
    # Run the bounded unrestricted-Toffoli probe before the wider-register
    # experiments populate large symbolic caches.
    toffoli = unrestricted_toffoli_probe(max_expansions=args.toffoli_cap)
    gc.collect()
    training_keys = {target.canonical_key for target in training_targets}
    if any(target.canonical_key in training_keys for target in evaluation_targets):
        raise AssertionError("evaluation target duplicates a training transformation")

    eager_policy = train_eager_outer(
        training_targets, episodes=args.outer_episodes, seed=args.eager_seed
    )
    bandit = train_inner_bandit(
        training_targets,
        episodes=args.inner_episodes,
        alpha=args.linucb_alpha,
    )
    deferred_policy = train_deferred_outer(
        training_targets,
        bandit,
        episodes=args.outer_episodes,
        seed=args.deferred_seed,
        batch_size=args.batch_size,
    )

    np.savez(
        output / "policies.npz",
        eager_outer_theta=eager_policy.theta,
        deferred_outer_theta=deferred_policy.theta,
        inner_a=bandit.a_matrix,
        inner_a_inverse=bandit.a_inverse,
        inner_b=bandit.b_vector,
        inner_posterior_mean=bandit.posterior_mean,
    )

    raw: list[dict[str, object]] = []
    rng = random.Random(args.order_seed)
    for target in evaluation_targets:
        repetitions = args.repetitions if target.num_qubits < 6 else 1
        run_cap = args.max_expansions if target.num_qubits < 6 else args.stress_cap
        run_wall_limit = args.wall_limit if target.num_qubits < 6 else args.stress_wall_limit
        methods = [
            (
                "eager_outer_sarsa",
                lambda target=target: evaluate_eager_sarsa(
                    eager_policy,
                    target,
                    max_expansions=run_cap,
                    wall_limit=run_wall_limit,
                ),
            ),
            (
                "eager_target_potential",
                lambda target=target: evaluate_eager_target_potential(
                    target,
                    max_expansions=run_cap,
                    wall_limit=run_wall_limit,
                ),
            ),
            (
                "deferred_outer_sarsa_inner_linucb",
                lambda target=target: evaluate_hierarchy(
                    deferred_policy,
                    bandit,
                    target,
                    batch_size=args.batch_size,
                    max_allocations=run_cap,
                    wall_limit=run_wall_limit,
                ),
            ),
        ]
        for repetition in range(repetitions):
            shuffled = list(methods)
            rng.shuffle(shuffled)
            for _, evaluate in shuffled:
                result: EvaluationResult = evaluate()
                row = result.to_dict()
                row["repetition"] = repetition
                raw.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                gc.collect()
        _write_csv(output / "raw_runs.partial.csv", raw)

    aggregate = _aggregate(raw)
    config: dict[str, object] = {
        "base_branch": "deferred-cost-aware-qcs-v1",
        "base_commit_observed_before_change": "9ff381b2feb4d1abfecf7431ce089eef1f43090e",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "outer_episodes": args.outer_episodes,
        "inner_episodes": args.inner_episodes,
        "batch_size": args.batch_size,
        "linucb_alpha": args.linucb_alpha,
        "max_expansions_or_allocations": args.max_expansions,
        "wall_limit_seconds_per_run": args.wall_limit,
        "stress_cap_6q": args.stress_cap,
        "stress_wall_limit_seconds": args.stress_wall_limit,
        "timed_repetitions_4q_5q": args.repetitions,
        "timed_repetitions_6q": 1,
        "training_target_count": len(training_targets),
        "evaluation_target_count": len(evaluation_targets),
        "environment": {
            "processor": platform.processor(),
            "machine": platform.machine(),
            "pid": os.getpid(),
        },
    }
    _write_csv(output / "raw_runs.csv", raw)
    _write_csv(output / "aggregate.csv", aggregate)
    plot_paths = _make_plots(output, aggregate)
    summary = {
        "config": config,
        "aggregate": aggregate,
        "unrestricted_toffoli_probe": toffoli,
        "plots": plot_paths,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _report(output, aggregate, toffoli, config)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/cnot-crossover")
    parser.add_argument("--outer-episodes", type=int, default=400)
    parser.add_argument("--inner-episodes", type=int, default=300)
    parser.add_argument("--eager-seed", type=int, default=11)
    parser.add_argument("--deferred-seed", type=int, default=17)
    parser.add_argument("--order-seed", type=int, default=20260904)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--linucb-alpha", type=float, default=0.5)
    parser.add_argument("--max-expansions", type=int, default=4_096)
    parser.add_argument("--wall-limit", type=float, default=15.0)
    parser.add_argument("--stress-cap", type=int, default=512)
    parser.add_argument("--stress-wall-limit", type=float, default=10.0)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--toffoli-cap", type=int, default=1_024)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
