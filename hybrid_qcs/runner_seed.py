"""Per-seed online training and frozen-policy evaluation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import time
from typing import Any

import numpy as np

from .rl import LinearSarsaRanker, train_online_sarsa
from .runner_support import (
    _deadline_reached,
    _evaluate_policy,
    _evaluate_structured_policy,
    _status,
)


def run_seed_set(
    args: argparse.Namespace,
    seeds: tuple[int, ...],
    target_train,
    target_validation,
    target_test,
    target_stress,
    cpu_start: float,
    wall_start: float,
    hard_deadline: float,
) -> dict[str, Any]:
    seed_results: list[dict[str, Any]] = []
    training_csv_rows: list[dict[str, Any]] = []
    evaluation_csv_rows: list[dict[str, Any]] = []
    stress_csv_rows: list[dict[str, Any]] = []

    for seed_index, seed in enumerate(seeds):
        if _deadline_reached(cpu_start, wall_start, hard_deadline):
            break
        policy = LinearSarsaRanker(seed=seed, learning_rate=args.learning_rate)
        remaining = max(
            0.001,
            hard_deadline
            - max(time.process_time() - cpu_start, time.perf_counter() - wall_start),
        )

        def progress(event: dict[str, Any]) -> None:
            _status({"seed": seed, **event})

        training = train_online_sarsa(
            policy,
            target_train,
            episodes=args.episodes,
            max_expansions=args.train_expansion_cap,
            deadline_seconds=remaining,
            progress=progress,
        )
        for log in training.episodes:
            training_csv_rows.append({"seed": seed, **asdict(log)})

        evaluations: list[dict[str, Any]] = []
        for target in (*target_validation, *target_test):
            if _deadline_reached(cpu_start, wall_start, hard_deadline):
                break
            result = _evaluate_policy(
                policy,
                target,
                max_expansions=args.eval_expansion_cap,
                should_stop=lambda: _deadline_reached(
                    cpu_start, wall_start, hard_deadline
                ),
            )
            evaluations.append(result)
            evaluation_csv_rows.append(
                {
                    "seed": seed,
                    "target": target.name,
                    "split": target.split,
                    "success": result["success"],
                    "expansions": result["search"]["expansions"],
                    "frontier_peak": result["search"]["frontier_peak"],
                    "maximum_matrix_error": (
                        ""
                        if result["certification"] is None
                        else result["certification"]["maximum_matrix_error"]
                    ),
                    "gate_count": (
                        ""
                        if result["certification"] is None
                        else result["certification"]["gate_count"]
                    ),
                    "t_count": (
                        ""
                        if result["certification"] is None
                        else result["certification"]["t_count"]
                    ),
                    "cnot_count": (
                        ""
                        if result["certification"] is None
                        else result["certification"]["cnot_count"]
                    ),
                    "depth": (
                        ""
                        if result["certification"] is None
                        else result["certification"]["depth"]
                    ),
                    "witness": (
                        ""
                        if result["certification"] is None
                        else "; ".join(result["certification"]["witness"])
                    ),
                }
            )
            _status(
                {
                    "phase": "evaluation",
                    "seed": seed,
                    "target": target.name,
                    "split": target.split,
                    "success": result["success"],
                    "expansions": result["search"]["expansions"],
                    "frontier_peak": result["search"]["frontier_peak"],
                    "cpu_seconds": time.process_time() - cpu_start,
                    "wall_seconds": time.perf_counter() - wall_start,
                }
            )

        stress_tests: list[dict[str, Any]] = []
        for target in target_stress:
            if _deadline_reached(cpu_start, wall_start, hard_deadline):
                break
            result = _evaluate_structured_policy(
                policy,
                target,
                max_expansions=args.toffoli_expansion_cap,
                should_stop=lambda: _deadline_reached(
                    cpu_start, wall_start, hard_deadline
                ),
            )
            stress_tests.append(result)
            stress_csv_rows.append(
                {
                    "seed": seed,
                    "target": target.name,
                    "success": result["success"],
                    "normal_form_complete": result["normal_form_complete"],
                    "expansions": result["search"]["expansions"],
                    "frontier_peak": result["search"]["frontier_peak"],
                    "maximum_matrix_error": (
                        ""
                        if result["certification"] is None
                        else result["certification"]["maximum_matrix_error"]
                    ),
                    "gate_count": (
                        ""
                        if result["certification"] is None
                        else result["certification"]["gate_count"]
                    ),
                    "t_count": (
                        ""
                        if result["certification"] is None
                        else result["certification"]["t_count"]
                    ),
                    "cnot_count": (
                        ""
                        if result["certification"] is None
                        else result["certification"]["cnot_count"]
                    ),
                    "depth": (
                        ""
                        if result["certification"] is None
                        else result["certification"]["depth"]
                    ),
                    "witness": (
                        ""
                        if result["certification"] is None
                        else "; ".join(result["certification"]["witness"])
                    ),
                }
            )
            _status(
                {
                    "phase": "structured_stress",
                    "seed": seed,
                    "target": target.name,
                    "success": result["success"],
                    "expansions": result["search"]["expansions"],
                    "frontier_peak": result["search"]["frontier_peak"],
                    "cpu_seconds": time.process_time() - cpu_start,
                    "wall_seconds": time.perf_counter() - wall_start,
                }
            )

        seed_success = (
            not training.deadline_hit
            and len(training.episodes) == args.episodes
            and all(log.success and log.certified for log in training.episodes)
            and len(evaluations) == len(target_validation) + len(target_test)
            and all(result["success"] for result in evaluations)
            and len(stress_tests) == len(target_stress)
            and all(result["success"] for result in stress_tests)
        )
        seed_results.append(
            {
                "seed": seed,
                "success": seed_success,
                "training": {
                    "episodes_completed": len(training.episodes),
                    "episodes_requested": args.episodes,
                    "successes": sum(log.success for log in training.episodes),
                    "total_expansions": training.total_expansions,
                    "cpu_seconds": training.cpu_seconds,
                    "wall_seconds": training.wall_seconds,
                    "deadline_hit": training.deadline_hit,
                    "profile_totals": training.profile_totals,
                    "peak_frontier": training.peak_frontier,
                    "maximum_rotation_length": training.maximum_rotation_length,
                },
                "evaluations": evaluations,
                "stress_tests": stress_tests,
                "policy": {
                    "weights": [float(value) for value in policy.theta],
                    "l2_norm": float(np.linalg.norm(policy.theta)),
                    "updates": policy.updates,
                    "feature_time_ns": policy.feature_time_ns,
                    "scoring_time_ns": policy.scoring_time_ns,
                },
            }
        )
        _status(
            {
                "phase": "seed_complete",
                "seed": seed,
                "success": seed_success,
                "cpu_seconds": time.process_time() - cpu_start,
                "wall_seconds": time.perf_counter() - wall_start,
                "completed_seeds": seed_index + 1,
                "total_seeds": len(seeds),
            }
        )

    return {
        "seed_results": seed_results,
        "training_csv_rows": training_csv_rows,
        "evaluation_csv_rows": evaluation_csv_rows,
        "stress_csv_rows": stress_csv_rows,
    }


__all__ = ["run_seed_set"]
