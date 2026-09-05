#!/usr/bin/env python3
"""Regenerate article figures from the packaged CSV tables."""
from __future__ import annotations

from pathlib import Path
import csv
import json
import math

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)



# Ancilla-isometry qualification figures.
summary = json.loads((DATA / "ancilla_summary.json").read_text(encoding="utf-8"))
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
save(fig, "ancilla_training_time")

ancilla_rows = read_csv("ancilla_evaluations.csv")
fig, ax = plt.subplots(figsize=(9.0, 4.8))
x = np.arange(len(ancilla_rows))
edges = [int(row["attempted_edges"]) for row in ancilla_rows]
ax.bar(x, edges)
ax.set_xticks(
    x,
    [row["target"].replace("heldout-", "") for row in ancilla_rows],
    rotation=12,
)
ax.set_ylabel("Exact continuation attempts")
ax.set_title("Clean-ancilla held-out synthesis work")
save(fig, "ancilla_evaluation_edges")

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
save(fig, "qft3_search_probe")

# Outer-controller success at 256 expansions.
rows = read_csv("outer_scheduler_results.csv")
names = [row["scheduler"] for row in rows]
values = np.asarray([float(row["success"]) for row in rows])
lower = values - np.asarray([float(row["ci_low"]) for row in rows])
upper = np.asarray([float(row["ci_high"]) for row in rows]) - values
fig, ax = plt.subplots(figsize=(8.2, 4.1))
positions = np.arange(len(names))
ax.bar(positions, values, yerr=np.vstack([lower, upper]), capsize=3)
ax.set_xticks(positions, names, rotation=28, ha="right")
ax.set_ylim(0, 0.72)
ax.set_ylabel("Certified success at 256 expansions")
ax.grid(axis="y", alpha=0.25)
save(fig, "outer_success_256")

# Success by target stratum.
rows = read_csv("outer_success_by_stratum.csv")
names = [row["scheduler"] for row in rows]
series = ["easy", "medium", "hard_ood"]
labels = ["Easy", "Medium", "Hard/OOD"]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
x = np.arange(len(names))
width = 0.24
for index, (field, label) in enumerate(zip(series, labels, strict=True)):
    vals = [float(row[field]) for row in rows]
    ax.bar(x + (index - 1) * width, vals, width, label=label)
ax.set_xticks(x, names, rotation=28, ha="right")
ax.set_ylim(0, 1.08)
ax.set_ylabel("Certified success")
ax.legend(ncol=3, loc="upper center")
ax.grid(axis="y", alpha=0.25)
save(fig, "outer_success_strata")

# Canonicalization audit.
rows = read_csv("canonicalization_audit.csv")
labels = [f"{row['num_qubits']}q, depth <= {row['max_depth']}" for row in rows]
local = [int(row["legacy_duplicate_excess"]) for row in rows]
strong = [int(row["strong_duplicate_excess"]) for row in rows]
fig, ax = plt.subplots(figsize=(6.8, 3.8))
x = np.arange(len(labels))
width = 0.34
ax.bar(x - width / 2, local, width, label="Local incremental key")
ax.bar(x + width / 2, strong, width, label="Clifford-extracted key")
ax.set_xticks(x, labels)
ax.set_ylabel("Unresolved duplicate excess")
ax.legend()
ax.grid(axis="y", alpha=0.25)
save(fig, "canonical_duplicate_excess")

ratios = [float(row["strong_to_legacy_key_time_ratio"]) for row in rows]
fig, ax = plt.subplots(figsize=(6.8, 3.5))
ax.bar(labels, ratios)
ax.axhline(1.0, linewidth=1.0, linestyle="--")
ax.set_ylabel("Canonical-key time ratio")
ax.set_ylim(0, max(ratios) * 1.22)
ax.grid(axis="y", alpha=0.25)
save(fig, "canonical_key_cost")

# Mixed crossover figures.
rows = read_csv("mixed_crossover_results.csv")
target_order = [
    "heldout-mixed-frame-phase-4q",
    "heldout-mixed-basis-echo-5q",
    "heldout-mixed-frame-phase-6q-ood",
]
target_labels = ["4q frame-phase", "5q basis-echo", "6q frame-phase OOD"]
method_order = [
    "Deferred SARSA + LinUCB",
    "Deferred SARSA + native order",
    "Eager outer SARSA",
    "Eager target potential",
]
method_labels = [
    "Hierarchical SARSA + LinUCB",
    "Deferred SARSA + fixed order",
    "Eager outer SARSA",
    "Eager target potential",
]
lookup = {(row["target"], row["method"]): row for row in rows}
x = np.arange(len(target_order))
width = 0.2
fig, ax = plt.subplots(figsize=(9.0, 4.2))
for index, (method, label) in enumerate(zip(method_order, method_labels, strict=True)):
    vals = [float(lookup[(target, method)]["median_wall_seconds"]) for target in target_order]
    bars = ax.bar(x + (index - 1.5) * width, vals, width, label=label)
    for bar, target in zip(bars, target_order, strict=True):
        certified = lookup[(target, method)]["certified"].strip().lower() == "true"
        if not certified:
            bar.set_hatch("//")
ax.set_yscale("log")
ax.set_xticks(x, target_labels)
ax.set_ylabel("Median observed wall time (s, log scale)")
ax.legend(ncol=2, fontsize=8)
ax.grid(axis="y", alpha=0.25)
save(fig, "mixed_wall_time")

fig, ax = plt.subplots(figsize=(9.0, 4.2))
for index, (method, label) in enumerate(zip(method_order, method_labels, strict=True)):
    vals = [int(lookup[(target, method)]["attempted_edges"]) for target in target_order]
    ax.bar(x + (index - 1.5) * width, vals, width, label=label)
ax.set_yscale("log")
ax.set_xticks(x, target_labels)
ax.set_ylabel("Exact continuation attempts (log scale)")
ax.legend(ncol=2, fontsize=8)
ax.grid(axis="y", alpha=0.25)
save(fig, "mixed_edge_attempts")

# Speedup of the full hierarchy relative to certified eager outer SARSA.
speedups = []
reductions = []
for target in target_order:
    h = lookup[(target, "Deferred SARSA + LinUCB")]
    e = lookup[(target, "Eager outer SARSA")]
    speedups.append(float(e["median_wall_seconds"]) / float(h["median_wall_seconds"]))
    reductions.append(1.0 - int(h["attempted_edges"]) / int(e["attempted_edges"]))
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.bar(target_labels, speedups)
ax.axhline(1.0, linewidth=1.0, linestyle="--")
ax.set_ylabel("Wall-time speedup over eager outer SARSA")
ax.grid(axis="y", alpha=0.25)
save(fig, "hierarchical_speedup")

fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.bar(target_labels, [100.0 * value for value in reductions])
ax.set_ylabel("Reduction in exact continuation attempts (%)")
ax.set_ylim(0, 100)
ax.grid(axis="y", alpha=0.25)
save(fig, "edge_reduction_percent")

print(f"Wrote figures to {FIG}")
