"""Timing aggregation and CSV output for the qualification runner."""
from __future__ import annotations

from collections import Counter
import csv
from pathlib import Path
from typing import Any


def _component_timing(
    seed_results: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    scaling: list[dict[str, Any]],
) -> dict[str, Any]:
    totals: Counter[str] = Counter()

    def add_profile(profile: dict[str, Any]) -> None:
        for name, value in profile.items():
            if name.endswith("_ns"):
                totals[name] += int(value)

    for seed_result in seed_results:
        training = seed_result["training"]
        add_profile(dict(training.get("profile_totals", {})))
        policy = seed_result["policy"]
        totals["policy_feature_ns"] += int(policy["feature_time_ns"])
        totals["policy_scoring_ns"] += int(policy["scoring_time_ns"])
        for evaluation in (
            *seed_result["evaluations"],
            *seed_result.get("stress_tests", ()),
        ):
            add_profile(dict(evaluation["search"]["profile"]))
            totals["certification_ns"] += int(
                evaluation.get("certification_time_ns", 0)
            )

    for baseline in baselines:
        add_profile(dict(baseline["search"]["profile"]))
        totals["certification_ns"] += int(baseline.get("certification_time_ns", 0))
    for row in scaling:
        add_profile(dict(row["metrics"]["profile"]))

    timed_total = sum(totals.values())
    ordered = sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    components = {
        name: {
            "nanoseconds": value,
            "seconds": value / 1e9,
            "share_of_instrumented_time": (
                0.0 if timed_total == 0 else value / timed_total
            ),
        }
        for name, value in ordered
    }
    dominant_name, dominant_value = ordered[0] if ordered else (None, 0)
    return {
        "instrumented_total_nanoseconds": timed_total,
        "instrumented_total_seconds": timed_total / 1e9,
        "dominant_component": dominant_name,
        "dominant_component_seconds": dominant_value / 1e9,
        "dominant_component_share": (
            0.0 if timed_total == 0 else dominant_value / timed_total
        ),
        "components": components,
        "interpretation": (
            "Exclusive implementation timers identify engineering cost, not "
            "algorithmic convergence. Uninstrumented Python/control/reporting "
            "time is outside the denominator."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


__all__ = ["_component_timing", "_write_csv"]
