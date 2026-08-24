"""Hard-deadline multi-seed training, evaluation, certification, and profiling."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np

from .benchmarks import (
    all_targets,
    held_out_targets,
    structured_stress_targets,
    training_targets,
    validation_targets,
)
from .certify import certify_state
from .rl import FEATURE_NAMES, LinearSarsaRanker, train_online_sarsa
from .search import HybridSearch, SearchRecord
from .structured_toffoli import StructuredToffoliSearch, phase_identity_holds



from .runner_seed import run_seed_set
from .runner_support import (
    _component_timing,
    _deadline_reached,
    _evaluate_baseline,
    _evaluate_policy,
    _evaluate_structured_baseline,
    _evaluate_structured_policy,
    _profile_scaling,
    _status,
    _write_csv,
)
def execute_campaign(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    hard_deadline = float(args.deadline_seconds)
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("at least one training seed is required")

    _status(
        {
            "phase": "start",
            "deadline_seconds": hard_deadline,
            "seeds": seeds,
            "episodes_per_seed": args.episodes,
            "train_expansion_cap": args.train_expansion_cap,
            "eval_expansion_cap": args.eval_expansion_cap,
            "toffoli_expansion_cap": args.toffoli_expansion_cap,
        }
    )

    if not phase_identity_holds():
        raise AssertionError("the structured Toffoli phase identity is invalid")
    target_train = training_targets()
    target_validation = validation_targets()
    target_test = held_out_targets()
    target_stress = structured_stress_targets()
    seed_data = run_seed_set(
        args,
        seeds,
        target_train,
        target_validation,
        target_test,
        target_stress,
        cpu_start,
        wall_start,
        hard_deadline,
    )
    seed_results = seed_data["seed_results"]
    training_csv_rows = seed_data["training_csv_rows"]
    evaluation_csv_rows = seed_data["evaluation_csv_rows"]
    stress_csv_rows = seed_data["stress_csv_rows"]

    baselines: list[dict[str, Any]] = []
    if not _deadline_reached(cpu_start, wall_start, hard_deadline):
        benchmark_target = target_test[1]
        for name in ("symbolic_distance", "fifo"):
            result = _evaluate_baseline(
                benchmark_target,
                name=name,
                max_expansions=args.eval_expansion_cap,
                should_stop=lambda: _deadline_reached(
                    cpu_start, wall_start, hard_deadline
                ),
            )
            baselines.append(result)
            _status(
                {
                    "phase": "baseline",
                    "baseline": name,
                    "target": benchmark_target.name,
                    "success": result["success"],
                    "expansions": result["search"]["expansions"],
                }
            )

        for target in target_stress:
            for name in ("symbolic_distance", "fifo"):
                result = _evaluate_structured_baseline(
                    target,
                    name=name,
                    max_expansions=args.toffoli_expansion_cap,
                    should_stop=lambda: _deadline_reached(
                        cpu_start, wall_start, hard_deadline
                    ),
                )
                baselines.append(result)
                _status(
                    {
                        "phase": "structured_baseline",
                        "baseline": name,
                        "target": target.name,
                        "success": result["success"],
                        "expansions": result["search"]["expansions"],
                    }
                )

    scaling = []
    if not _deadline_reached(cpu_start, wall_start, hard_deadline):
        scaling = _profile_scaling(target_test[2], (32, 64, 128, 256, 512))

    cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start
    deadline_hit = max(cpu_seconds, wall_seconds) >= hard_deadline
    success = (
        not deadline_hit
        and len(seed_results) == len(seeds)
        and all(result["success"] for result in seed_results)
    )

    component_timing = _component_timing(seed_results, baselines, scaling)

    return {
        "output": output,
        "hard_deadline": hard_deadline,
        "cpu_start": cpu_start,
        "wall_start": wall_start,
        "seeds": seeds,
        "target_train": target_train,
        "target_validation": target_validation,
        "target_test": target_test,
        "target_stress": target_stress,
        "seed_results": seed_results,
        "training_csv_rows": training_csv_rows,
        "evaluation_csv_rows": evaluation_csv_rows,
        "stress_csv_rows": stress_csv_rows,
        "baselines": baselines,
        "scaling": scaling,
        "cpu_seconds": cpu_seconds,
        "wall_seconds": wall_seconds,
        "deadline_hit": deadline_hit,
        "success": success,
    }


__all__ = ["execute_campaign"]
