"""CLI for reproducible qualification, custom bounded targets and proof checking."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

from .model import Budget
from .oracle_synthesis import BooleanOracleSpec
from .resource_audit import verify_closed_cover, write_certificate
from .resource_domain import BoundedProblem, OBJECTIVES
from .resource_optimize import OptimizationLimits, sweep_clean_ancillas
from .resource_policy import OUTER_FEATURE_NAMES, INNER_FEATURE_NAMES, load_hierarchy, save_hierarchy
from .resource_search import WorkLimits, train_budgeted_hierarchy


def _json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def _commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=Path(__file__).resolve().parents[1], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _export_sweep(output, name, sweep):
    for row in sweep["rows"]:
        result = row["result"]
        certificates = result.pop("infeasibility_certificates")
        references = []
        for index, proof in enumerate(certificates):
            relative = Path("certificates") / f"{name}-a{row['available_clean_workspace']}-{index}-{proof['problem_digest'][:12]}.json"
            # Do not call a potentially expensive second verifier without limits.
            checked = verify_closed_cover(proof)
            if not checked["valid"]:
                raise RuntimeError(f"qualification proof check failed: {checked}")
            write_certificate(output / relative, proof)
            references.append({"path": str(relative), "verification": checked,
                               "problem_digest": proof["problem_digest"],
                               "payload_sha256": proof["payload_sha256"]})
        result["certificate_files"] = references
    return sweep


def training_curriculum():
    # Only 1-input truth tables: held-out two-/three-input functions are absent.
    problems = []
    for truth in ((0, 1), (1, 0), (1, 1)):
        for mode in ("evaluator", "direct_phase"):
            for work in (0, 1):
                for cap in (0, 1):
                    spec = BooleanOracleSpec("one-input-training", 1, truth)
                    problems.append(BoundedProblem(spec, Budget(cap, 2, 8, 8), 4, work, mode))
    # A deterministic interleaving varies mode/width/cap within a fixed campaign.
    order = np.random.default_rng(11).permutation(len(problems))
    return tuple(problems[int(i)] for i in order)


def run_qualification(output_dir: Path, *, seed: int = 11, training_episodes=(6, 6, 2),
                      limits: OptimizationLimits = OptimizationLimits(100_000, 512, 40_000, 15_000, 12., 12., 16)) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    curriculum = training_curriculum()
    outer, inner, training = train_budgeted_hierarchy(curriculum, episodes=training_episodes, seed=seed,
                                                     limits=WorkLimits(128, 512, 4., 4.))
    save_hierarchy(output_dir / "policy.json", outer, inner)
    _json(output_dir / "training.json", {"episodes": training, "curriculum": [p.manifest() for p in curriculum],
                                        "seed": seed, "training_work_limit": asdict(WorkLimits(128, 512, 4., 4.))})
    cases = (
        ("and-two-input-evaluator", BoundedProblem(BooleanOracleSpec("and", 2, (0, 0, 0, 1)), Budget(7, 6, 15, 15), 3)),
        ("xnor-two-input-evaluator", BoundedProblem(BooleanOracleSpec("xnor", 2, (1, 0, 0, 1)), Budget(0, 2, 6, 6), 3)),
        ("parity-two-input-direct-phase", BoundedProblem(BooleanOracleSpec("parity", 2, (0, 1, 1, 0)), Budget(0, 2, 4, 4), 4, mode="direct_phase")),
        ("conjunction-three-input-evaluator", BoundedProblem(BooleanOracleSpec("conjunction", 3, (0, 0, 0, 0, 0, 0, 0, 1)), Budget(21, 18, 45, 45), 3)),
        ("majority-three-input-evaluator", BoundedProblem(BooleanOracleSpec("majority", 3, (0, 0, 0, 1, 0, 1, 1, 1)), Budget(21, 18, 45, 45), 3)),
    )
    weight_snapshot = outer.weights.copy()
    response_snapshot = {key: val.copy() for key, val in inner.responses.items()}
    evaluations, summary = [], []
    for name, problem in cases:
        for scheduler in ("distance", "cost", "outer", "hierarchy"):
            # Majority exercises multi-marked logic at zero workspace. The
            # cubic conjunction also tests a bounded-domain workspace minimum.
            widths = (0,) if name == "majority-three-input-evaluator" else (0, 1, 2)
            sweep = sweep_clean_ancillas(problem, budgets=widths, outer=outer, inner=inner,
                                        scheduler=scheduler, limits=limits)
            evaluation = {"case": name, "scheduler": scheduler,
                          "sweep": _export_sweep(output_dir, f"{name}-{scheduler}", sweep)}
            evaluations.append(evaluation)
            for row in sweep["rows"]:
                result = row["result"]
                cert = result["incumbent"]
                resources = cert["resources"] if cert else {}
                summary.append({"case": name, "scheduler": scheduler,
                                "available_clean_workspace": row["available_clean_workspace"],
                                "used_clean_workspace": cert["used_clean_workspace"] if cert else None,
                                "status": result["status"], "reason": result["reason"],
                                **{key: resources.get(key) for key in OBJECTIVES},
                                "edges_including_verification": result["work_edges_including_verification"],
                                **result["timing"]})
    if not np.array_equal(outer.weights, weight_snapshot) or any(
            not np.array_equal(inner.responses[k], v) for k, v in response_snapshot.items()):
        raise AssertionError("evaluation changed frozen policies")
    payload = {"schema": "qcs-resource-qualification-v1", "source_commit": _commit(),
               "runtime": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
               "outer_feature_dimension": len(OUTER_FEATURE_NAMES), "inner_feature_dimension": len(INNER_FEATURE_NAMES),
               "training_episode_counts": list(training_episodes), "outer_updates": outer.updates,
               "inner_updates": inner.updates, "policies_frozen_during_evaluation": True,
               "training_test_separation": "one-input training; held-out two-/three-input truth tables",
               "limits_per_width_and_scheduler": asdict(limits), "evaluations": evaluations,
               "important_limits": ["small fixed-lowering evaluator / affine phase-network domains only",
                                    "no generic unrestricted Clifford+T optimality claim",
                                    "cooperative operation-boundary deadlines, not OS hard interrupts",
                                    "record cap, not a measured hard byte-memory cap",
                                    "qualification is not a statistical demonstration of learned superiority"]}
    _json(output_dir / "results.json", payload)
    with (output_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    status_counts = {status: sum(row["status"] == status for row in summary)
                     for status in ("optimal", "upper_bound", "infeasible", "unknown")}
    text = ["# Resource/ancilla optimality qualification", "",
            f"Source commit at execution: `{payload['source_commit']}`. Local modifications, if any, must be recorded by the delivery manifest.", "",
            f"{len(training)} staged training episodes; {len(summary)} frozen evaluations. Outcomes: {status_counts}.", "",
            "Every exported infeasibility certificate was checked independently. A timeout remains unknown or an upper bound.", "",
            "Objectives: T count, CNOT count, native depth, native gate count (lexicographic). Operation count is a finiteness cap, not the primary objective.", "",
            "Clean budgets use separate exact archives. The evaluator output bit is logical; it counts as an extra auxiliary when the evaluator is wrapped as a phase oracle.", "",
            "No full DAG graph encoder is used. Cached workspace/dependency summaries augment linear SARSA and disjoint LinUCB.", "",
            "See summary.csv for per-case resources and discovery/proof timings, training.json for the curriculum, and certificates/ for checkable closed covers.", "",
            "These small-domain checks do not establish policy superiority, unrestricted optimality, or a need for additional ancillas on these targets.", ""]
    (output_dir / "REPORT.md").write_text("\n".join(text))
    return payload


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    qualification = sub.add_parser("qualify", help="bounded held-out regression/qualification campaign")
    qualification.add_argument("--output-dir", type=Path, default=Path("outputs/resource-optimality"))
    qualification.add_argument("--seed", type=int, default=11)
    optimize = sub.add_parser("optimize", help="optimize a truth-table target within an explicit domain")
    optimize.add_argument("--truth-table", required=True, help="bits indexed by x=0,1,...; q0 is least-significant")
    optimize.add_argument("--mode", choices=("evaluator", "direct_phase"), default="evaluator")
    optimize.add_argument("--ancillas", default="0,1,2", help="comma-separated clean-workspace budgets")
    optimize.add_argument("--t-cap", type=int, default=7)
    optimize.add_argument("--cnot-cap", type=int, default=6)
    optimize.add_argument("--gate-cap", type=int, default=15)
    optimize.add_argument("--depth-cap", type=int, default=15)
    optimize.add_argument("--operation-cap", type=int, default=3)
    optimize.add_argument("--objectives", default=",".join(OBJECTIVES))
    optimize.add_argument("--scheduler", choices=("distance", "cost", "outer", "hierarchy"), default="hierarchy")
    optimize.add_argument("--policy", type=Path, help="versioned trained hierarchy; omitted means untrained linear warm start")
    optimize.add_argument("--total-edges", type=int, default=50_000)
    optimize.add_argument("--discovery-edges", type=int, default=1000)
    optimize.add_argument("--audit-edges", type=int, default=20_000)
    optimize.add_argument("--max-records", type=int, default=20_000)
    optimize.add_argument("--wall-seconds", type=float, default=60.)
    optimize.add_argument("--cpu-seconds", type=float, default=60.)
    optimize.add_argument("--output-dir", type=Path, default=Path("outputs/resource-custom"))
    verify = sub.add_parser("verify", help="independently check self-contained infeasibility certificates")
    verify.add_argument("certificates", nargs="+", type=Path)
    verify.add_argument("--max-edges", type=int, default=1_000_000)
    verify.add_argument("--max-records", type=int, default=100_000)
    verify.add_argument("--seconds", type=float, default=60.)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            good = True
            for path in args.certificates:
                checked = verify_closed_cover(json.loads(path.read_text()),
                                              limits=WorkLimits(args.max_edges, args.max_records, args.seconds, args.seconds))
                print(json.dumps({"file": str(path), **checked}, sort_keys=True))
                good &= checked["valid"]
            return 0 if good else 2
        if args.command == "qualify":
            result = run_qualification(args.output_dir, seed=args.seed)
            print(json.dumps({"output_dir": str(args.output_dir), "evaluations": len(result["evaluations"]),
                              "frozen": result["policies_frozen_during_evaluation"]}))
            return 0
        table = args.truth_table.strip()
        if len(table) not in (2, 4, 8) or any(ch not in "01" for ch in table):
            raise ValueError("truth table must contain 2, 4 or 8 binary digits")
        n = (len(table) - 1).bit_length()
        p = BoundedProblem(BooleanOracleSpec("custom", n, tuple(map(int, table))),
                           Budget(args.t_cap, args.cnot_cap, args.gate_cap, args.depth_cap),
                           args.operation_cap, mode=args.mode)
        outer, inner = load_hierarchy(args.policy) if args.policy else (None, None)
        limits = OptimizationLimits(args.total_edges, args.discovery_edges, args.audit_edges,
                                    args.max_records, args.wall_seconds, args.cpu_seconds)
        result = sweep_clean_ancillas(p, budgets=tuple(map(int, args.ancillas.split(","))),
                                     objectives=tuple(args.objectives.split(",")), outer=outer, inner=inner,
                                     scheduler=args.scheduler, limits=limits)
        result["policy_source"] = str(args.policy) if args.policy else "untrained linear warm start"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _export_sweep(args.output_dir, "custom", result)
        _json(args.output_dir / "results.json", result)
        print(json.dumps({"output_dir": str(args.output_dir), "statuses": [r["result"]["status"] for r in result["rows"]]}))
        return 0  # Unknown is an explicit scientific outcome, not a process error.
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
