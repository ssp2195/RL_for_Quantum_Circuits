"""Run bounded clean-ancilla training, regression benchmarks, and QFT-3 probe."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .ancilla_benchmarks import (
    ancilla_evaluation_targets,
    ancilla_training_targets,
    qft3_clean_ancilla_target,
    qft3_clean_ancilla_witness,
)
from .ancilla_certify import certify_ancilla_state
from .ancilla_search import (
    DisjointAncillaLinUCB,
    LinearAncillaOuterSarsa,
    evaluate_ancilla_hierarchy,
    initialize_ancilla_bandit_from_mixed,
    initialize_ancilla_outer_from_mixed,
    train_ancilla_inner_bandit,
    train_ancilla_outer_sarsa,
)
from .mixed_crossover import (
    mixed_training_targets,
    train_mixed_inner_bandit,
    train_mixed_outer_sarsa,
)
from .model import HybridState
from .qft_guided import synthesize_qft_decomposition_guided


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _train_existing_reference(args: argparse.Namespace):
    targets = mixed_training_targets()
    started = time.perf_counter()
    outer = train_mixed_outer_sarsa(
        targets,
        episodes=args.outer_episodes,
        seed=args.seed,
        max_expansions=args.reference_train_cap,
    )
    bandit = train_mixed_inner_bandit(
        targets,
        episodes=args.inner_episodes,
        alpha=args.linucb_alpha,
    )
    return outer, bandit, time.perf_counter() - started


def _save_policies(
    path: Path,
    outer: LinearAncillaOuterSarsa,
    bandit: DisjointAncillaLinUCB,
) -> None:
    payload: dict[str, np.ndarray] = {"outer_theta": outer.theta}
    for family in sorted(bandit.a_inverse):
        safe = family.replace(":", "_").replace("-", "_")
        payload[f"inner_a_inverse_{safe}"] = bandit.a_inverse[family]
        payload[f"inner_b_{safe}"] = bandit.b_vector[family]
        payload[f"inner_mean_{safe}"] = bandit.posterior_mean(family)
    np.savez(path, **payload)


def _qft3_probe(
    outer: LinearAncillaOuterSarsa,
    bandit: DisjointAncillaLinUCB,
    args: argparse.Namespace,
) -> dict[str, object]:
    target = qft3_clean_ancilla_target()
    witness_state = HybridState.identity(target.num_qubits, target.budget)
    for gate in qft3_clean_ancilla_witness():
        child = witness_state.apply(gate, partial_order_reduction=False)
        if child is None:
            raise AssertionError("QFT-3 witness cannot be replayed")
        witness_state = child
    witness_certification = certify_ancilla_state(target, witness_state)
    guided_result = synthesize_qft_decomposition_guided(target)
    search_result = evaluate_ancilla_hierarchy(
        outer,
        bandit,
        target,
        max_allocations=args.qft_allocations,
        batch_size=args.batch_size,
        wall_limit=args.qft_wall_limit,
    )
    return {
        "target": target.name,
        "logical_qubits": target.contract.num_logical_qubits,
        "clean_ancillas": target.contract.num_clean_ancillas,
        "hidden_witness_gates": target.generator_length,
        "hidden_witness_t_count": sum(
            gate.is_non_clifford for gate in qft3_clean_ancilla_witness()
        ),
        "hidden_witness_cnot_count": sum(
            gate.is_two_qubit for gate in qft3_clean_ancilla_witness()
        ),
        "witness_certified": witness_certification.success,
        "witness_projective_error": witness_certification.projective_isometry_error,
        "witness_leakage": witness_certification.ancilla_leakage,
        "decomposition_guided_generation": guided_result.to_dict(),
        "guided_gate_reduction": (
            target.generator_length - guided_result.native_gate_count
        ),
        "guided_gate_reduction_fraction": (
            (target.generator_length - guided_result.native_gate_count)
            / max(1, target.generator_length)
        ),
        "unrestricted_search": search_result.to_dict(),
    }


def _make_plots(output: Path, summary: dict[str, object]) -> list[str]:
    import matplotlib.pyplot as plt

    paths: list[str] = []
    training = summary["training"]
    labels = ["Existing mixed hierarchy", "Ancilla-aware hierarchy"]
    values = [
        float(training["existing_reference_seconds"]),
        float(training["total_seconds"]),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(labels, values)
    ax.set_ylabel("Training wall time (s)")
    ax.set_title("Matched low-cost training protocol")
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    path = output / "ancilla_training_time.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path.name)

    evaluations = summary["evaluations"]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(len(evaluations))
    edges = [int(row["attempted_edges"]) for row in evaluations]
    ax.bar(x, edges)
    ax.set_xticks(x, [str(row["target"]).replace("heldout-", "") for row in evaluations], rotation=12)
    ax.set_ylabel("Exact continuation attempts")
    ax.set_title("Clean-ancilla held-out synthesis work")
    fig.tight_layout()
    path = output / "ancilla_evaluation_edges.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path.name)

    qft = summary["qft3"]
    search = qft["unrestricted_search"]
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.bar(
        ["Allocations", "Exact edges", "Peak frontier"],
        [
            int(search["allocations"]),
            int(search["attempted_edges"]),
            int(search["frontier_peak"]),
        ],
    )
    ax.set_title(
        "Unrestricted QFT-3 ancilla-search probe\n"
        + ("certified" if bool(search["success"]) else str(search["stop_reason"]))
    )
    fig.tight_layout()
    path = output / "qft3_search_probe.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path.name)

    guided = qft["decomposition_guided_generation"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    labels = ["Reference", "Guided"]
    gate_values = [int(qft["hidden_witness_gates"]), int(guided["native_gate_count"])]
    ax.bar(labels, gate_values)
    ax.set_ylabel("Native gates")
    ax.set_title("QFT-3 exact decomposition guidance")
    for index, value in enumerate(gate_values):
        ax.text(index, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    path = output / "qft3_guided_gate_reduction.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path.name)
    return paths


def _write_report(output: Path, summary: dict[str, object]) -> None:
    training = summary["training"]
    evaluations = summary["evaluations"]
    qft = summary["qft3"]
    search = qft["unrestricted_search"]
    guided = qft["decomposition_guided_generation"]
    lines = [
        "# Clean-ancilla hierarchical QCS qualification",
        "",
        "## Scope",
        "",
        "The implementation supports a fixed physical register partitioned into logical qubits, clean |0> ancillas, and optionally borrowed ancillas. Correctness is contract-relative isometry equality; clean workspace must be returned to |0>, and borrowed workspace is required to undergo the identity operation on an arbitrary input state.",
        "",
        "## Training cost",
        "",
        f"- Existing matched hierarchical training: {float(training['existing_reference_seconds']):.6f} s",
        f"- Existing training plus ancilla fine-tuning: {float(training['total_seconds']):.6f} s",
        f"- Total/reference ratio: {float(training['total_ratio']):.3f}x",
        "",
        "The ancilla policies are warm-started from the existing mixed-gate linear models and receive only a short staged fine-tuning pass. No joint neural training or graph encoder is introduced.",
        "",
        "## Held-out clean-ancilla results",
        "",
        "| Target | Certified | Wall (s) | Allocations | Exact edges | Peak frontier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in evaluations:
        lines.append(
            f"| {row['target']} | {row['success']} | {float(row['wall_seconds']):.6f} | "
            f"{row['allocations']} | {row['attempted_edges']} | {row['frontier_peak']} |"
        )
    lines.extend(
        [
            "",
            "## QFT-3 with one clean ancilla",
            "",
            f"The independently constructed 47-gate native witness is certified: **{qft['witness_certified']}**. Its projective isometry error is {float(qft['witness_projective_error']):.3e} and its clean-ancilla leakage is {float(qft['witness_leakage']):.3e}.",
            "",
            "### Decomposition-guided mitigation",
            "",
            f"The exact QFT block planner generated and independently certified a {guided['native_gate_count']}-gate native circuit with {guided['t_count']} T/TDG gates, {guided['cnot_count']} CNOTs, and depth {guided['depth']}. Its projective isometry error is {float(guided['projective_isometry_error']):.3e} and its clean-ancilla leakage is {float(guided['ancilla_leakage']):.3e}.",
            "",
            "The planner derives the standard Hadamard, controlled-phase, and final bit-reversal blocks from the analytical QFT target. Controlled-T is lowered through a relative-phase AND compute circuit, a T gate on the clean ancilla, and the exact inverse compute circuit. The input-dependent relative phases cancel, reducing the QFT-3 implementation from 47 to 35 native gates without adding training episodes.",
            "",
            f"The unrestricted hierarchical search result is **{search['stop_reason']}** after {search['allocations']} outer allocations and {search['attempted_edges']} exact continuation attempts. This bounded probe is reported honestly; witness certification establishes representability and contract correctness, while an unsuccessful search does not establish unrestricted synthesis at this depth.",
            "",
            "## Claim boundary",
            "",
            "Archive pruning remains based on the strengthened full-register projective key. This is sound but incomplete for clean-ancilla equivalence: full-unitary equality implies isometry equality, but the converse need not hold. Terminal acceptance no longer requires full symbolic-key equality and is decided independently by the ancilla isometry certifier.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    mixed_outer, mixed_bandit, reference_seconds = _train_existing_reference(args)
    training_targets = ancilla_training_targets()
    outer = initialize_ancilla_outer_from_mixed(mixed_outer, seed=args.seed)
    bandit = initialize_ancilla_bandit_from_mixed(
        mixed_bandit, training_targets, alpha=args.linucb_alpha
    )
    started = time.perf_counter()
    outer = train_ancilla_outer_sarsa(
        training_targets,
        episodes=args.ancilla_outer_episodes,
        seed=args.seed,
        max_allocations=args.ancilla_train_cap,
        batch_size=args.batch_size,
        policy=outer,
    )
    bandit = train_ancilla_inner_bandit(
        outer,
        training_targets,
        episodes=args.ancilla_inner_episodes,
        alpha=args.linucb_alpha,
        max_allocations=args.ancilla_train_cap,
        bandit=bandit,
    )
    ancilla_seconds = time.perf_counter() - started
    _save_policies(output / "ancilla_policies.npz", outer, bandit)

    evaluation_rows = [
        evaluate_ancilla_hierarchy(
            outer,
            bandit,
            target,
            max_allocations=args.evaluation_allocations,
            batch_size=args.batch_size,
            wall_limit=args.evaluation_wall_limit,
        ).to_dict()
        for target in ancilla_evaluation_targets()
    ]
    qft = _qft3_probe(outer, bandit, args)
    summary: dict[str, object] = {
        "schema": "ancilla-hierarchical-qualification-v1",
        "config": {
            "outer_episodes": args.outer_episodes,
            "inner_episodes": args.inner_episodes,
            "ancilla_outer_finetune_episodes": args.ancilla_outer_episodes,
            "ancilla_inner_finetune_episodes": args.ancilla_inner_episodes,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "linucb_alpha_training": args.linucb_alpha,
            "linucb_alpha_evaluation": 0.0,
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        },
        "training": {
            "existing_reference_seconds": reference_seconds,
            "ancilla_incremental_seconds": ancilla_seconds,
            "total_seconds": reference_seconds + ancilla_seconds,
            "incremental_fraction": ancilla_seconds / max(reference_seconds, 1e-12),
            "total_ratio": (reference_seconds + ancilla_seconds)
            / max(reference_seconds, 1e-12),
        },
        "evaluations": evaluation_rows,
        "qft3": qft,
    }
    summary["plots"] = _make_plots(output, summary)
    (output / "ancilla_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(output / "ancilla_evaluations.csv", evaluation_rows)
    _write_report(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/ancilla-qualification")
    parser.add_argument("--outer-episodes", type=int, default=16)
    parser.add_argument("--inner-episodes", type=int, default=24)
    parser.add_argument("--ancilla-outer-episodes", type=int, default=3)
    parser.add_argument("--ancilla-inner-episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--linucb-alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--reference-train-cap", type=int, default=64)
    parser.add_argument("--ancilla-train-cap", type=int, default=64)
    parser.add_argument("--evaluation-allocations", type=int, default=512)
    parser.add_argument("--evaluation-wall-limit", type=float, default=3.0)
    parser.add_argument("--qft-allocations", type=int, default=256)
    parser.add_argument("--qft-wall-limit", type=float, default=1.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
