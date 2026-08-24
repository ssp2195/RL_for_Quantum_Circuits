"""Human-readable completion summary for the hybrid-QCS campaign."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .runner_support import _status


def write_human_summary(output: Path, context: dict[str, Any]) -> int:
    seeds = tuple(context["seeds"])
    seed_results = list(context["seed_results"])
    component_timing = context["component_timing"]
    training_episode_count = sum(
        seed["training"]["episodes_completed"] for seed in seed_results
    )
    training_success_count = sum(
        seed["training"]["successes"] for seed in seed_results
    )
    evaluation_count = sum(len(seed["evaluations"]) for seed in seed_results)
    evaluation_success_count = sum(
        int(evaluation["success"])
        for seed in seed_results
        for evaluation in seed["evaluations"]
    )
    stress_count = sum(len(seed.get("stress_tests", ())) for seed in seed_results)
    stress_success_count = sum(
        int(result["success"])
        for seed in seed_results
        for result in seed.get("stress_tests", ())
    )
    qft_evaluations = [
        evaluation
        for seed in seed_results
        for evaluation in seed["evaluations"]
        if evaluation["target"] == "heldout-qft2-exact"
    ]
    qft_success_count = sum(int(result["success"]) for result in qft_evaluations)
    success = bool(context["success"])
    deadline_hit = bool(context["deadline_hit"])
    cpu_seconds = float(context["cpu_seconds"])
    wall_seconds = float(context["wall_seconds"])
    hard_deadline = float(context["hard_deadline"])
    lines = [
        "# Hybrid Clifford+T 30-minute qualification",
        "",
        f"- Success: **{success}**",
        f"- CPU time: **{cpu_seconds:.6f} s**",
        f"- Wall time: **{wall_seconds:.6f} s**",
        f"- Hard limit: **{hard_deadline:.1f} s**",
        f"- Seeds completed: **{len(seed_results)}/{len(seeds)}**",
        f"- Training episodes certified: **{training_success_count}/{training_episode_count}**",
        f"- Frozen-policy evaluations certified: **{evaluation_success_count}/{evaluation_count}**",
        f"- Exact unrestricted QFT-2 evaluations certified: **{qft_success_count}/{len(qft_evaluations)}**",
        f"- Structured Toffoli stress tests certified: **{stress_success_count}/{stress_count}**",
        f"- Dominant measured component: **{component_timing['dominant_component']}** "
        f"({component_timing['dominant_component_seconds']:.6f} s; "
        f"{100.0 * component_timing['dominant_component_share']:.2f}% of "
        "instrumented component time)",
        "",
        "Every reported solution was reconstructed from its persistent DAG and independently dense-certified up to global phase.",
        "The hidden target-generator gate sequences are not retained in the search-facing target records.",
    ]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _status(
        {
            "phase": "complete",
            "success": success,
            "deadline_hit": deadline_hit,
            "cpu_seconds": cpu_seconds,
            "wall_seconds": wall_seconds,
            "seeds_completed": len(seed_results),
            "summary": str(output / "summary.json"),
        }
    )
    return 0 if success else 1


__all__ = ["write_human_summary"]
