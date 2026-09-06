"""Train the compact hierarchy and synthesize the BNN verification phase oracle."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import time
from typing import Callable

import matplotlib.pyplot as plt

from .oracle_benchmarks import (
    bnn_verification_oracle_target,
    oracle_training_targets,
)
from .oracle_synthesis import (
    DisjointOracleLinUCB,
    LinearOracleOuterSarsa,
    OracleMacroDeferredSearch,
    OracleSynthesisResult,
    assemble_phase_oracle,
    certify_evaluator_witness,
    evaluate_oracle_hierarchy,
    oracle_inner_context,
    oracle_macro_library,
    save_oracle_policies,
    train_oracle_inner_bandit,
    train_oracle_outer_sarsa,
)


def _evaluate_control(
    method: str,
    outer: LinearOracleOuterSarsa,
    *,
    batch_size: int,
    max_allocations: int,
    max_edges: int,
    wall_limit: float,
    continuation_order: str,
) -> dict[str, object]:
    target = bnn_verification_oracle_target()
    environment = OracleMacroDeferredSearch(
        target,
        max_allocations=max_allocations,
        max_edges=max_edges,
        batch_size=batch_size,
    )
    start = time.perf_counter()
    while environment.frontier and environment.solution_record_id is None:
        if environment.allocations >= max_allocations:
            break
        if environment.edge_attempts >= max_edges:
            break
        if time.perf_counter() - start >= wall_limit:
            break
        if continuation_order == "target_potential":
            record = min(
                environment.open_records(),
                key=lambda item: (item.distance, item.resources.gate_count, item.record_id),
            )
            record_id = record.record_id
            tokens = sorted(
                environment.pending_tokens(record),
                key=lambda token: (
                    oracle_inner_context(
                        record, environment.actions[token], environment.target
                    )[2],
                    token,
                ),
            )[:batch_size]
        else:
            record_id, _, _ = outer.choose(
                environment.open_records(),
                target,
                0.0,
                profile=environment.profile,
            )
            record = environment.frontier[record_id]
            tokens = environment.pending_tokens(record)[:batch_size]
        environment.process_batch(record_id, tokens)

    elapsed = time.perf_counter() - start
    success = environment.solution_record_id is not None
    macro_witness: tuple[str, ...] = ()
    projective_error = exact_error = leakage = None
    if success:
        solution = environment.records[environment.solution_record_id]
        evaluator_state, evaluator_cert = certify_evaluator_witness(
            target, solution.macro_witness
        )
        _, _, phase_cert = assemble_phase_oracle(target, evaluator_state)
        success = bool(evaluator_cert.success and phase_cert.success)
        macro_witness = tuple(
            environment.actions[token].name for token in solution.macro_witness
        )
        projective_error = phase_cert.projective_isometry_error
        exact_error = phase_cert.exact_isometry_error
        leakage = phase_cert.ancilla_leakage
    return {
        "method": method,
        "success": success,
        "wall_seconds": elapsed,
        "allocations": environment.allocations,
        "attempted_macro_edges": environment.edge_attempts,
        "frontier_peak": environment.profile.frontier_peak,
        "projective_error": projective_error,
        "exact_error": exact_error,
        "ancilla_leakage": leakage,
        "macro_witness": "; ".join(macro_witness),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_method_comparison(output_dir: Path, rows: list[dict[str, object]]) -> None:
    methods = []
    wall = []
    edges = []
    for method in sorted({str(row["method"]) for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        methods.append(method.replace("oracle_", "").replace("_", "\n"))
        wall.append(statistics.median(float(row["wall_seconds"]) for row in subset))
        edges.append(statistics.median(int(row["attempted_macro_edges"]) for row in subset))

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(methods, wall)
    ax.set_ylabel("Median wall time (s)")
    ax.set_title("BNN verification-oracle generation")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "oracle_wall_time.png", dpi=220)
    fig.savefig(output_dir / "oracle_wall_time.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(methods, edges)
    ax.set_ylabel("Exact macro continuations")
    ax.set_title("Search work to certified phase oracle")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "oracle_macro_edges.png", dpi=220)
    fig.savefig(output_dir / "oracle_macro_edges.pdf")
    plt.close(fig)


def _plot_trace(output_dir: Path, result: OracleSynthesisResult) -> None:
    allocations = [int(row["allocation"]) for row in result.trace]
    best = [float(row["best_distance"]) for row in result.trace]
    frontier = [int(row["frontier_size"]) for row in result.trace]

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(allocations, best)
    ax.set_xlabel("Outer allocations")
    ax.set_ylabel("Best exact evaluator distance")
    ax.set_title("Hierarchical search trace")
    fig.tight_layout()
    fig.savefig(output_dir / "oracle_search_distance.png", dpi=220)
    fig.savefig(output_dir / "oracle_search_distance.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(allocations, frontier)
    ax.set_xlabel("Outer allocations")
    ax.set_ylabel("Active frontier records")
    ax.set_title("Contract-canonical frontier growth")
    fig.tight_layout()
    fig.savefig(output_dir / "oracle_frontier_trace.png", dpi=220)
    fig.savefig(output_dir / "oracle_frontier_trace.pdf")
    plt.close(fig)


def _report(
    output_dir: Path,
    summary: dict[str, object],
    hierarchical: OracleSynthesisResult,
    rows: list[dict[str, object]],
) -> None:
    phase = hierarchical.to_dict()["phase_oracle"]
    evaluator = hierarchical.to_dict()["evaluator"]
    lines = [
        "# Hierarchical BNN verification-oracle synthesis",
        "",
        "The experiment synthesizes the reversible predicate evaluator with outer linear SARSA and inner disjoint LinUCB over a generic NOT/CNOT/Toffoli macro grammar. The final phase oracle is formed through the universal compute-phase-uncompute identity and lowered to the unchanged native Clifford+T library.",
        "",
        "## Target",
        "",
        "- Inputs: `x1 x2 x3` on data qubits `q0 q1 q2`.",
        "- Predicate: `g(x)=1` only for `x=100`.",
        "- Flag: `q3`, initialized and restored to `|0>`.",
        "- Clean workspace: `q4`, initialized and restored to `|0>`.",
        "- Phase action: `|x> -> (-1)^g(x)|x>`.",
        "",
        "## Certified result",
        "",
        f"- Success: **{hierarchical.success}**",
        f"- Outer allocations: **{hierarchical.allocations}**",
        f"- Attempted macro continuations: **{hierarchical.attempted_macro_edges}**",
        f"- Peak frontier: **{hierarchical.frontier_peak}**",
        f"- Evaluator macro witness: `{'; '.join(hierarchical.macro_witness)}`",
        f"- Evaluator native gates: **{evaluator['native_gate_count']}**",
        f"- Complete phase-oracle native gates: **{phase['native_gate_count']}**",
        f"- Phase-oracle T/TDG count: **{phase['t_count']}**",
        f"- Phase-oracle CNOT count: **{phase['cnot_count']}**",
        f"- Exact isometry error: **{float(phase['exact_error']):.3e}**",
        f"- Clean-workspace leakage: **{float(phase['ancilla_leakage']):.3e}**",
        f"- Policy-training time: **{hierarchical.training_seconds:.6f} s**",
        "",
        "## Interpretation",
        "",
        "The learned search does not receive a target-specific circuit. It sees the truth table and a target-independent reversible macro grammar. Exact domain mappings provide the oracle-layer canonical key; every returned macro witness is lowered through the strengthened Clifford-tableau/Pauli-rotation canonicalizer and independently certified under the clean-ancilla isometry contract.",
        "",
        "The macro grammar is a restricted first implementation for small Boolean predicates. It is not a complete compiler for arbitrary large BNNs, and the results do not establish superiority over an exact target-potential scheduler.",
        "",
        "## Frozen comparison rows",
        "",
        "| Method | Success | Median wall time (s) | Macro edges | Allocations | Peak frontier |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in sorted({str(row["method"]) for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        lines.append(
            "| {} | {} | {:.6f} | {:.1f} | {:.1f} | {:.1f} |".format(
                method,
                all(bool(row["success"]) for row in subset),
                statistics.median(float(row["wall_seconds"]) for row in subset),
                statistics.median(int(row["attempted_macro_edges"]) for row in subset),
                statistics.median(int(row["allocations"]) for row in subset),
                statistics.median(int(row["frontier_peak"]) for row in subset),
            )
        )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/bnn-oracle")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--outer-episodes", type=int, default=2)
    parser.add_argument("--inner-episodes", type=int, default=4)
    parser.add_argument("--max-allocations", type=int, default=1_000)
    parser.add_argument("--max-edges", type=int, default=4_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--wall-limit", type=float, default=12.0)
    parser.add_argument("--timing-repeats", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curriculum = oracle_training_targets()[:4]
    training_start = time.perf_counter()
    outer = LinearOracleOuterSarsa(seed=args.seed, learning_rate=1e-4)
    outer = train_oracle_outer_sarsa(
        curriculum,
        episodes=args.outer_episodes,
        seed=args.seed,
        max_allocations=32,
        batch_size=2,
        policy=outer,
    )
    bandit = DisjointOracleLinUCB(alpha=0.05, regularization=1_000.0)
    bandit = train_oracle_inner_bandit(
        outer,
        curriculum,
        episodes=args.inner_episodes,
        alpha=0.05,
        max_allocations=32,
        bandit=bandit,
    )
    training_seconds = time.perf_counter() - training_start
    target = bnn_verification_oracle_target()

    hierarchical_results: list[OracleSynthesisResult] = []
    rows: list[dict[str, object]] = []
    for repeat in range(args.timing_repeats):
        result = evaluate_oracle_hierarchy(
            outer,
            bandit,
            target,
            max_allocations=args.max_allocations,
            max_edges=args.max_edges,
            batch_size=args.batch_size,
            wall_limit=args.wall_limit,
            training_seconds=training_seconds,
        )
        hierarchical_results.append(result)
        rows.append(
            {
                "repeat": repeat,
                "method": "oracle_hierarchical_sarsa_linucb",
                "success": result.success,
                "wall_seconds": result.wall_seconds,
                "allocations": result.allocations,
                "attempted_macro_edges": result.attempted_macro_edges,
                "frontier_peak": result.frontier_peak,
                "projective_error": result.phase_oracle_projective_error,
                "exact_error": result.phase_oracle_exact_error,
                "ancilla_leakage": result.phase_oracle_leakage,
                "macro_witness": "; ".join(result.macro_witness),
            }
        )
        rows.append(
            {
                "repeat": repeat,
                **_evaluate_control(
                    "oracle_deferred_fixed_order",
                    outer,
                    batch_size=args.batch_size,
                    max_allocations=args.max_allocations,
                    max_edges=args.max_edges,
                    wall_limit=args.wall_limit,
                    continuation_order="fixed",
                ),
            }
        )
        rows.append(
            {
                "repeat": repeat,
                **_evaluate_control(
                    "oracle_exact_target_potential",
                    outer,
                    batch_size=args.batch_size,
                    max_allocations=args.max_allocations,
                    max_edges=args.max_edges,
                    wall_limit=args.wall_limit,
                    continuation_order="target_potential",
                ),
            }
        )

    successful = [result for result in hierarchical_results if result.success]
    if not successful:
        raise RuntimeError("the hierarchical oracle synthesis did not certify")
    representative = min(successful, key=lambda result: result.wall_seconds)
    _write_csv(output_dir / "oracle_runs.csv", rows)
    _write_csv(output_dir / "oracle_search_trace.csv", list(representative.trace))
    truth_rows = [
        {
            "basis_index": basis,
            "x1x2x3": "".join(
                str((basis >> position) & 1)
                for position in range(target.spec.num_inputs)
            ),
            "g": target.spec.value(basis),
            "phase": -1 if target.spec.value(basis) else 1,
        }
        for basis in range(1 << target.spec.num_inputs)
    ]
    _write_csv(output_dir / "oracle_truth_table.csv", truth_rows)
    (output_dir / "oracle_result.json").write_text(
        json.dumps(representative.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "phase_oracle_native_witness.txt").write_text(
        "\n".join(representative.phase_oracle_witness) + "\n"
    )
    save_oracle_policies(output_dir / "oracle_policies.npz", outer, bandit)
    _plot_method_comparison(output_dir, rows)
    _plot_trace(output_dir, representative)

    summary = {
        "schema": "hierarchical-bnn-oracle-synthesis-v1",
        "target": {
            "name": target.spec.name,
            "marked_bitstrings": target.spec.marked_bitstrings,
            "truth_table": target.spec.truth_table,
            "data_qubits": target.layout.data_qubits,
            "flag_qubit": target.layout.flag_qubit,
            "work_qubits": target.layout.work_qubits,
        },
        "training": {
            "curriculum": [item.spec.name for item in curriculum],
            "outer_episodes": args.outer_episodes,
            "inner_episodes": args.inner_episodes,
            "seconds": training_seconds,
            "outer_updates": outer.updates,
            "inner_updates": bandit.updates,
        },
        "representative": representative.to_dict(),
        "timing_repeats": args.timing_repeats,
    }
    _report(output_dir, summary, representative, rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
