"""Frozen-policy, baseline, and scaling evaluations."""
from __future__ import annotations

from dataclasses import asdict
import time
from typing import Any, Callable

from .certify import certify_state
from .rl import LinearSarsaRanker
from .search import HybridSearch
from .structured_toffoli import StructuredToffoliSearch


def _evaluate_policy(
    policy: LinearSarsaRanker,
    target,
    *,
    max_expansions: int,
    should_stop: Callable[[], bool],
) -> dict[str, Any]:
    env = HybridSearch(target, max_expansions=max_expansions)
    feature_before = policy.feature_time_ns
    score_before = policy.scoring_time_ns
    state = env.run_scheduler(
        lambda records: policy.choose(records, target, 0.0)[0],
        should_stop=should_stop,
    )
    started = time.perf_counter_ns()
    certification = None if state is None else certify_state(target, state)
    return {
        "target": target.name,
        "split": target.split,
        "success": bool(certification and certification.success),
        "search": env.metrics(),
        "certification": None if certification is None else asdict(certification),
        "certification_time_ns": time.perf_counter_ns() - started,
        "policy_feature_time_ns": policy.feature_time_ns - feature_before,
        "policy_scoring_time_ns": policy.scoring_time_ns - score_before,
    }


def _evaluate_structured_policy(
    policy: LinearSarsaRanker,
    target,
    *,
    max_expansions: int,
    should_stop: Callable[[], bool],
) -> dict[str, Any]:
    env = StructuredToffoliSearch(target, max_expansions=max_expansions)
    feature_before = policy.feature_time_ns
    score_before = policy.scoring_time_ns
    state = env.run_scheduler(
        lambda records: policy.choose(records, target, 0.0)[0],
        should_stop=should_stop,
    )
    started = time.perf_counter_ns()
    certification = None if state is None else certify_state(target, state)
    progress = env.solution_progress()
    return {
        "target": target.name,
        "split": target.split,
        "family": target.family,
        "success": bool(
            certification and certification.success
            and progress is not None and progress.stage.value == "DONE"
        ),
        "search": env.metrics(),
        "normal_form_complete": bool(progress and progress.stage.value == "DONE"),
        "certification": None if certification is None else asdict(certification),
        "certification_time_ns": time.perf_counter_ns() - started,
        "policy_feature_time_ns": policy.feature_time_ns - feature_before,
        "policy_scoring_time_ns": policy.scoring_time_ns - score_before,
    }


def _evaluate_baseline(
    target,
    *,
    name: str,
    max_expansions: int,
    should_stop: Callable[[], bool],
) -> dict[str, Any]:
    env = HybridSearch(target, max_expansions=max_expansions)
    if name == "fifo":
        selector = lambda records: min(records, key=lambda record: record.record_id).record_id
    elif name == "symbolic_distance":
        selector = lambda records: min(
            records,
            key=lambda record: (
                record.symbolic_distance, record.state.gate_count,
                record.state.t_count, record.state.cnot_count, record.record_id,
            ),
        ).record_id
    else:
        raise ValueError(f"unsupported baseline {name!r}")
    state = env.run_scheduler(selector, should_stop=should_stop)
    started = time.perf_counter_ns()
    certification = None if state is None else certify_state(target, state)
    return {
        "baseline": name,
        "target": target.name,
        "success": bool(certification and certification.success),
        "search": env.metrics(),
        "certification": None if certification is None else asdict(certification),
        "certification_time_ns": time.perf_counter_ns() - started,
    }


def _evaluate_structured_baseline(
    target,
    *,
    name: str,
    max_expansions: int,
    should_stop: Callable[[], bool],
) -> dict[str, Any]:
    env = StructuredToffoliSearch(target, max_expansions=max_expansions)
    if name == "fifo":
        selector = lambda records: min(records, key=lambda record: record.record_id).record_id
    elif name == "symbolic_distance":
        selector = lambda records: min(
            records,
            key=lambda record: (
                record.symbolic_distance, record.state.gate_count,
                record.state.t_count, record.state.cnot_count, record.record_id,
            ),
        ).record_id
    else:
        raise ValueError(f"unsupported structured baseline {name!r}")
    state = env.run_scheduler(selector, should_stop=should_stop)
    started = time.perf_counter_ns()
    certification = None if state is None else certify_state(target, state)
    progress = env.solution_progress()
    return {
        "baseline": name,
        "target": target.name,
        "family": target.family,
        "success": bool(
            certification and certification.success
            and progress is not None and progress.stage.value == "DONE"
        ),
        "search": env.metrics(),
        "normal_form_complete": bool(progress and progress.stage.value == "DONE"),
        "certification": None if certification is None else asdict(certification),
        "certification_time_ns": time.perf_counter_ns() - started,
    }


def _profile_scaling(target, caps: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cap in caps:
        env = HybridSearch(target, max_expansions=cap)
        cpu = time.process_time()
        wall = time.perf_counter()
        state = env.run_scheduler(
            lambda records: min(records, key=lambda record: record.record_id).record_id
        )
        rows.append(
            {
                "cap": cap,
                "success": state is not None,
                "expansions": env.expansions,
                "cpu_seconds": time.process_time() - cpu,
                "wall_seconds": time.perf_counter() - wall,
                "metrics": env.metrics(),
            }
        )
    return rows


__all__ = [
    "_evaluate_baseline", "_evaluate_policy", "_evaluate_structured_baseline",
    "_evaluate_structured_policy", "_profile_scaling",
]
