"""Small linear SARSA/LinUCB policies with resource and workspace interactions.

Features are linear in learned parameters, not necessarily in raw resources.
A global cap alone would cancel between frontier candidates. Normalized usage
and progress/slack interactions make relative priorities budget-dependent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .resource_domain import BoundedProblem, CAP_FIELDS, OBJECTIVES, clean_mask, discrepancy

if TYPE_CHECKING:
    from .resource_search import Record

FAMILIES = ("X", "CNOT", "TOFFOLI", "T", "TDG", "S", "SDG")
OUTER_FEATURE_NAMES = (
    "bias", "distance", "available_clean_fraction", "logical_depth_mean",
    "workspace_depth_mean", "wire_depth_spread", "pending_fraction",
    *(f"{r}_usage" for r in OBJECTIVES), "operation_usage",
    *(f"distance_times_{r}_slack" for r in OBJECTIVES),
    *(f"clean_times_{r}_slack" for r in OBJECTIVES),
    "primary_usage", "distance_times_primary_slack",
    "distance_times_work_remaining", "primary_usage_times_work_remaining",
    "clean_times_work_remaining", "incumbent_gap", "distance_times_incumbent_gap",
    *(f"last_{family}" for family in FAMILIES),
)
INNER_FEATURE_NAMES = (
    "bias", "current_distance", "projected_distance", "distance_reduction",
    *(f"projected_{r}_usage" for r in OBJECTIVES),
    *(f"progress_times_{r}_slack" for r in OBJECTIVES),
    "projected_clean_fraction", "clean_release_fraction", "operand_depth_mean",
    "projected_depth_increase", "operand_overlap", "inverse_opportunity",
    "primary_projected_usage", "progress_times_primary_slack",
    "projected_distance_times_work_remaining", "workspace_target",
)
POLICY_SCHEMA = "resource-conditioned-linear-hierarchy-v1"


@dataclass(frozen=True, slots=True)
class PolicyContext:
    objective: str = "t_count"
    work_remaining: float = 1.0
    incumbent: dict | None = None

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError("unsupported resource objective")
        if not np.isfinite(self.work_remaining) or not 0 <= self.work_remaining <= 1:
            raise ValueError("work_remaining must be in [0,1]")


def outer_features(problem: BoundedProblem, record: Record, context: PolicyContext) -> np.ndarray:
    # Only context-independent summaries are cached. Pending work and incumbent
    # change without changing semantic equivalence and must not enter its key.
    key = (problem.digest, context.objective)
    cached = record.feature_cache.get(key)
    if cached is None:
        r = record.resources
        d = record.distance
        free = record.clean_count / (problem.clean_workspace + 1)
        scale = problem.budget.max_depth + 1
        logical = r.wire_depths[:problem.spec.num_inputs]
        work = [r.wire_depths[q] for q in problem.work_wires]
        usages = [getattr(r, name) / (getattr(problem.budget, CAP_FIELDS[name]) + 1)
                  for name in OBJECTIVES]
        slack = [(getattr(problem.budget, CAP_FIELDS[name]) - getattr(r, name)) /
                 (getattr(problem.budget, CAP_FIELDS[name]) + 1) for name in OBJECTIVES]
        primary = OBJECTIVES.index(context.objective)
        last = None if record.last_token is None else problem.operations[record.last_token].family
        cached = np.asarray([
            1., d, free, sum(logical) / (len(logical) * scale),
            sum(work) / (max(1, len(work)) * scale),
            (max(r.wire_depths) - min(r.wire_depths)) / scale, 0.,
            *usages, r.operations / (problem.max_operations + 1),
            *(d * s for s in slack), *(free * s for s in slack),
            usages[primary], d * slack[primary], 0., 0., 0., 0., 0.,
            *(float(last == family) for family in FAMILIES),
        ], dtype=np.float64)
        if cached.shape != (len(OUTER_FEATURE_NAMES),):
            raise AssertionError("outer feature schema mismatch")
        cached.setflags(write=False)
        record.feature_cache[key] = cached
    out = cached.copy()
    i = {name: j for j, name in enumerate(OUTER_FEATURE_NAMES)}
    out[i["pending_fraction"]] = record.pending.bit_count() / len(problem.operations)
    out[i["distance_times_work_remaining"]] = record.distance * context.work_remaining
    out[i["primary_usage_times_work_remaining"]] = out[i["primary_usage"]] * context.work_remaining
    out[i["clean_times_work_remaining"]] = out[i["available_clean_fraction"]] * context.work_remaining
    if context.incumbent is not None:
        ceiling = context.incumbent[context.objective]
        gap = (ceiling - getattr(record.resources, context.objective)) / (ceiling + 1)
        out[i["incumbent_gap"]] = gap
        out[i["distance_times_incumbent_gap"]] = record.distance * gap
    return out


def inner_features(problem: BoundedProblem, record: Record, token: int,
                   context: PolicyContext) -> np.ndarray:
    from .resource_domain import transition
    key = (problem.digest, context.objective, token)
    cached = record.inner_cache.get(key)
    if cached is None:
        op = problem.operations[token]
        projected = transition(record.image, op)
        r = record.resources.append(op)
        d = discrepancy(problem, projected)
        progress = record.distance - d
        free = clean_mask(problem, projected).bit_count() / (problem.clean_workspace + 1)
        usage = [getattr(r, name) / (getattr(problem.budget, CAP_FIELDS[name]) + 1)
                 for name in OBJECTIVES]
        slack = [(getattr(problem.budget, CAP_FIELDS[name]) - getattr(r, name)) /
                 (getattr(problem.budget, CAP_FIELDS[name]) + 1) for name in OBJECTIVES]
        previous = None if record.last_token is None else problem.operations[record.last_token]
        overlap = 0. if previous is None else len(set(previous.qubits) & set(op.qubits)) / len(op.qubits)
        inverses = {"X": "X", "CNOT": "CNOT", "TOFFOLI": "TOFFOLI",
                    "T": "TDG", "TDG": "T", "S": "SDG", "SDG": "S"}
        inverse = float(previous is not None and previous.qubits == op.qubits
                        and inverses[previous.family] == op.family)
        primary = OBJECTIVES.index(context.objective)
        scale = problem.budget.max_depth + 1
        cached = np.asarray([
            1., record.distance, d, progress, *usage, *(progress * s for s in slack),
            free, free - record.clean_count / (problem.clean_workspace + 1),
            sum(record.resources.wire_depths[q] for q in op.qubits) / (len(op.qubits) * scale),
            (r.depth - record.resources.depth) / scale, overlap, inverse,
            usage[primary], progress * slack[primary], 0., float(op.qubits[-1] in problem.work_wires),
        ], dtype=np.float64)
        if cached.shape != (len(INNER_FEATURE_NAMES),):
            raise AssertionError("inner feature schema mismatch")
        cached.setflags(write=False)
        record.inner_cache[key] = cached
    out = cached.copy()
    out[INNER_FEATURE_NAMES.index("projected_distance_times_work_remaining")] = out[2] * context.work_remaining
    return out


def _initial_outer_weights():
    weights = np.zeros(len(OUTER_FEATURE_NAMES))
    weights[OUTER_FEATURE_NAMES.index("distance")] = -1.
    return weights


@dataclass
class BudgetedSarsa:
    learning_rate: float = 1e-3
    seed: int = 11
    weights: np.ndarray = field(default_factory=_initial_outer_weights)
    updates: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        self.weights = np.asarray(self.weights, dtype=np.float64).copy()
        if self.weights.shape != (len(OUTER_FEATURE_NAMES),) or not np.isfinite(self.weights).all():
            raise ValueError("invalid outer weights/schema")
        self.rng = np.random.default_rng(self.seed)

    def choose(self, problem, records, context, epsilon=0.):
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0,1]")
        nodes = tuple(records)
        if not nodes:
            raise ValueError("empty frontier")
        matrix = np.vstack([outer_features(problem, r, context) for r in nodes])
        values = matrix @ self.weights
        if epsilon > 0 and self.rng.random() < epsilon:
            index = int(self.rng.integers(len(nodes)))
        else:
            best = float(np.max(values))
            tied = np.flatnonzero(np.isclose(values, best, atol=1e-12, rtol=0.))
            index = min(tied, key=lambda i: nodes[int(i)].record_id)
        return nodes[index].record_id, matrix[index].copy(), float(values[index])

    def value(self, problem, records, context) -> float:
        nodes = tuple(records)
        return max((float(outer_features(problem, r, context) @ self.weights) for r in nodes), default=0.)

    def update(self, features, value, reward, next_value=None) -> None:
        # Undiscounted bounded episodes (gamma=1); terminal bootstrap is zero.
        error = float(reward + (0. if next_value is None else next_value) - value)
        self.weights += self.learning_rate * error * features
        np.clip(self.weights, -50., 50., out=self.weights)
        self.updates += 1


@dataclass
class BudgetedLinUCB:
    alpha: float = 0.2
    regularization: float = 10.
    inverses: dict[str, np.ndarray] = field(default_factory=dict)
    responses: dict[str, np.ndarray] = field(default_factory=dict)
    updates: int = 0

    def __post_init__(self) -> None:
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("alpha must be non-negative and finite")
        if not np.isfinite(self.regularization) or self.regularization <= 0:
            raise ValueError("regularization must be positive and finite")
        for family in FAMILIES:
            self.inverses.setdefault(family, np.eye(len(INNER_FEATURE_NAMES)) / self.regularization)
            self.responses.setdefault(family, np.zeros(len(INNER_FEATURE_NAMES)))

    def score(self, family, features, explore=False) -> float:
        inverse = self.inverses[family]
        mean = float(features @ inverse @ self.responses[family])
        return mean + (self.alpha * np.sqrt(max(0., float(features @ inverse @ features))) if explore else 0.)

    def choose(self, problem, record, tokens, context, explore=False):
        rows = [(token, inner_features(problem, record, token, context)) for token in tokens]
        if not rows:
            raise ValueError("no pending continuations")
        token, features = max(rows, key=lambda row: (
            self.score(problem.operations[row[0]].family, row[1], explore), -row[0]))
        return token, features

    def update(self, family, features, response) -> None:
        inverse = self.inverses[family]
        projected = inverse @ features
        inverse = inverse - np.outer(projected, projected) / (1. + float(features @ projected))
        self.inverses[family] = (inverse + inverse.T) * .5
        self.responses[family] += float(response) * features
        self.updates += 1


def save_hierarchy(path: Path, outer: BudgetedSarsa, inner: BudgetedLinUCB) -> None:
    payload = {
        "schema": POLICY_SCHEMA, "outer_features": OUTER_FEATURE_NAMES,
        "inner_features": INNER_FEATURE_NAMES, "weights": outer.weights.tolist(),
        "learning_rate": outer.learning_rate, "seed": outer.seed, "rng": outer.rng.bit_generator.state,
        "outer_updates": outer.updates, "inner_updates": inner.updates,
        "alpha": inner.alpha, "regularization": inner.regularization,
        "inverses": {k: v.tolist() for k, v in inner.inverses.items()},
        "responses": {k: v.tolist() for k, v in inner.responses.items()},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load_hierarchy(path: Path) -> tuple[BudgetedSarsa, BudgetedLinUCB]:
    payload = json.loads(Path(path).read_text())
    if (payload.get("schema") != POLICY_SCHEMA or
            tuple(payload.get("outer_features", ())) != OUTER_FEATURE_NAMES or
            tuple(payload.get("inner_features", ())) != INNER_FEATURE_NAMES):
        raise ValueError("incompatible policy checkpoint schema")
    outer = BudgetedSarsa(payload["learning_rate"], payload["seed"], np.asarray(payload["weights"]))
    outer.rng.bit_generator.state = payload["rng"]
    outer.updates = payload["outer_updates"]
    inner = BudgetedLinUCB(payload["alpha"], payload["regularization"])
    d = len(INNER_FEATURE_NAMES)
    for family in FAMILIES:
        inverse = np.asarray(payload["inverses"][family], dtype=np.float64)
        response = np.asarray(payload["responses"][family], dtype=np.float64)
        if (inverse.shape != (d, d) or response.shape != (d,) or
                not np.isfinite(inverse).all() or not np.isfinite(response).all() or
                not np.allclose(inverse, inverse.T, atol=1e-10) or
                np.min(np.linalg.eigvalsh(inverse)) <= 0):
            raise ValueError("invalid LinUCB checkpoint parameters")
        inner.inverses[family] = inverse
        inner.responses[family] = response
    inner.updates = payload["inner_updates"]
    return outer, inner
