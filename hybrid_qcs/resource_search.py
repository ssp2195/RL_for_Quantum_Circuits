"""Persistent-record deferred search; learning controls order, never proofs.

Every resource-feasible operation remains pending until attempted or covered by
sound semantic/resource dominance. There are no history-dependent local masks:
that makes continuation simulation explicit for the independent closure audit.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Callable

import numpy as np

from .resource_domain import (
    BoundedProblem, CAP_FIELDS, ExactImage, Resources, certify_witness,
    clean_mask, discrepancy, dominates, nonnegative_integer, transition,
)
from .resource_policy import (
    BudgetedLinUCB, BudgetedSarsa, PolicyContext, inner_features, outer_features,
)


@dataclass(frozen=True, slots=True)
class WorkLimits:
    max_edges: int = 10_000
    max_records: int = 20_000
    wall_seconds: float = 30.
    cpu_seconds: float = 30.

    def __post_init__(self) -> None:
        nonnegative_integer(self.max_edges, "max_edges")
        nonnegative_integer(self.max_records, "max_records")
        if self.max_records < 1:
            raise ValueError("max_records must allow the root record")
        for name in ("wall_seconds", "cpu_seconds"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")


class WorkMeter:
    """Cooperative checks at individual-operation boundaries, not a hard OS kill."""
    def __init__(self, limits: WorkLimits, cancel: Callable[[], bool] | None = None):
        self.limits, self.cancel = limits, cancel
        self.wall_start, self.cpu_start = time.perf_counter(), time.process_time()
        self.edges = 0

    @property
    def wall(self) -> float:
        return time.perf_counter() - self.wall_start

    @property
    def cpu(self) -> float:
        return time.process_time() - self.cpu_start

    def reason(self) -> str | None:
        if self.cancel is not None and self.cancel():
            return "cancelled"
        if self.wall >= self.limits.wall_seconds:
            return "wall_limit"
        if self.cpu >= self.limits.cpu_seconds:
            return "cpu_limit"
        if self.edges >= self.limits.max_edges:
            return "edge_limit"
        return None


@dataclass(slots=True)
class Record:
    record_id: int
    image: ExactImage
    resources: Resources
    parent_id: int | None
    last_token: int | None
    pending: int
    distance: float
    clean_count: int
    feature_cache: dict = field(default_factory=dict)
    inner_cache: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    status: str
    reason: str
    problem_digest: str
    witness: dict | None
    edges: int
    records: int
    wall_seconds: float
    cpu_seconds: float
    profile: dict = field(default_factory=dict)
    rewards: list[dict] = field(default_factory=list)
    proof: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class DeferredResourceSearch:
    def __init__(self, problem: BoundedProblem, limits: WorkLimits):
        self.problem, self.limits = problem, limits
        self.records: dict[int, Record] = {}
        self.frontier: dict[int, Record] = {}
        self.archive: dict[ExactImage, list[int]] = {}
        self.halt: str | None = None
        self.peak_frontier = 0
        self.dominance_comparisons = 0
        self.insert(problem.root, Resources.zero(problem.width), None, None)

    def insert(self, image, resources, parent_id, last_token) -> Record | None:
        group = self.archive.get(image, [])
        for rid in group:
            self.dominance_comparisons += 1
            if dominates(self.records[rid].resources, resources):
                return None
        if len(self.records) >= self.limits.max_records:
            self.halt = "record_limit"
            return None
        survivors = []
        for rid in group:
            self.dominance_comparisons += 1
            if dominates(resources, self.records[rid].resources):
                self.frontier.pop(rid, None)
            else:
                survivors.append(rid)
        pending = 0
        if resources.operations < self.problem.max_operations:
            for token, op in enumerate(self.problem.operations):
                if resources.append(op).within(self.problem):
                    pending |= 1 << token
        record = Record(len(self.records), image, resources, parent_id, last_token,
                        pending, discrepancy(self.problem, image), clean_mask(self.problem, image).bit_count())
        self.records[record.record_id] = record
        self.archive[image] = [*survivors, record.record_id]
        if pending:
            self.frontier[record.record_id] = record
        self.peak_frontier = max(self.peak_frontier, len(self.frontier))
        return record

    def tokens(self, record: Record) -> tuple[int, ...]:
        mask, tokens = record.pending, []
        while mask:
            low = mask & -mask
            tokens.append(low.bit_length() - 1)
            mask ^= low
        return tuple(tokens)

    def witness(self, record: Record) -> tuple[int, ...]:
        tokens = []
        while record.parent_id is not None:
            tokens.append(record.last_token)
            record = self.records[record.parent_id]
        return tuple(reversed(tokens))

    def potential(self, objective: str) -> float:
        scale = getattr(self.problem.budget, CAP_FIELDS[objective]) + 1
        return -min((r.distance + .05 * getattr(r.resources, objective) / scale
                     for r in self.frontier.values()), default=0.)

    def step(self, record_id: int, token: int) -> Record | None:
        record = self.frontier[record_id]
        if not record.pending & (1 << token):
            raise ValueError("operation is not pending on selected persistent record")
        record.pending &= ~(1 << token)
        if not record.pending:
            self.frontier.pop(record_id)
        op = self.problem.operations[token]
        resources = record.resources.append(op)
        if not resources.within(self.problem):
            raise AssertionError("legal mask admitted a resource-infeasible child")
        return self.insert(transition(record.image, op), resources, record_id, token)


def run_budgeted_search(
    problem: BoundedProblem, *, outer: BudgetedSarsa | None = None,
    inner: BudgetedLinUCB | None = None, objective: str = "t_count",
    limits: WorkLimits = WorkLimits(), scheduler: str = "hierarchy",
    incumbent: dict | None = None, learn: str | None = None, epsilon: float = .1,
    fairness_period: int = 32, shaping_weight: float = 4., work_cost: float = .01,
    success_reward: float = 5., cancel: Callable[[], bool] | None = None,
) -> SearchResult:
    if objective not in CAP_FIELDS:
        raise ValueError("unsupported objective")
    if scheduler not in {"hierarchy", "outer", "distance", "cost"}:
        raise ValueError("unsupported scheduler")
    if learn not in {None, "outer", "inner"}:
        raise ValueError("learn must be None, outer or inner")
    if fairness_period < 1:
        raise ValueError("fairness_period must be positive")
    if work_cost < 0 or shaping_weight < 0:
        raise ValueError("work/shaping weights must be non-negative")
    outer = BudgetedSarsa() if outer is None else outer
    inner = BudgetedLinUCB() if inner is None else inner
    meter = WorkMeter(limits, cancel)
    environment = DeferredResourceSearch(problem, limits)
    logs, feature_seconds, transition_seconds = [], 0., 0.
    allocations, forced_choices = 0, 0

    def context():
        return PolicyContext(objective, max(0., 1. - meter.edges / max(1, limits.max_edges)), incumbent)

    def choose():
        nonlocal feature_seconds, forced_choices
        started = time.perf_counter()
        ctx = context()
        records = tuple(environment.frontier.values())
        forced = (allocations + 1) % fairness_period == 0
        if forced:
            record = min(records, key=lambda r: r.record_id)
            token = environment.tokens(record)[0]
            forced_choices += 1
        else:
            if scheduler == "distance":
                record = min(records, key=lambda r: (r.distance, r.resources.vector(), r.record_id))
            elif scheduler == "cost":
                record = min(records, key=lambda r: (getattr(r.resources, objective), r.resources.vector(), r.record_id))
            else:
                rid, _, _ = outer.choose(problem, records, ctx, epsilon if learn == "outer" else 0.)
                record = environment.records[rid]
            tokens = environment.tokens(record)
            if scheduler == "hierarchy" or learn == "inner":
                token, _ = inner.choose(problem, record, tokens, ctx, explore=learn == "inner")
            elif scheduler == "cost":
                token = min(tokens, key=lambda t: (getattr(record.resources.append(problem.operations[t]), objective), t))
            else:
                token = min(tokens, key=lambda t: (discrepancy(problem, transition(record.image, problem.operations[t])), t))
        features = outer_features(problem, record, ctx)
        response_features = inner_features(problem, record, token, ctx)
        feature_seconds += time.perf_counter() - started
        return record.record_id, token, features, response_features

    def finish(status, reason, certificate=None):
        return SearchResult(status, reason, problem.digest, certificate, meter.edges,
                            len(environment.records), meter.wall, meter.cpu,
                            {"frontier_peak": environment.peak_frontier,
                             "dominance_comparisons": environment.dominance_comparisons,
                             "feature_and_scoring_seconds": feature_seconds,
                             "transition_seconds": transition_seconds, "forced_choices": forced_choices,
                             "full_dag_policy_reconstructions": 0}, logs)

    if problem.root == problem.goal:
        cert = certify_witness(problem, ())
        return finish("feasible" if cert["success"] else "unknown", "root_certificate", cert)
    if not environment.frontier:
        return finish("unknown", "frontier_exhausted_requires_audit")
    if meter.reason():
        return finish("unknown", meter.reason())
    selection = choose()
    while environment.frontier:
        rid, token, features, response_features = selection
        before = environment.potential(objective)
        current_q = float(features @ outer.weights)  # recompute under current theta
        before_value = outer.value(problem, environment.frontier.values(), context()) if learn == "inner" else 0.
        reason = meter.reason()
        certificate = None
        attempted = 0
        if reason is None:
            started = time.perf_counter()
            child = environment.step(rid, token)
            transition_seconds += time.perf_counter() - started
            meter.edges += 1
            attempted = 1
            allocations += 1
            if child is not None and child.image == problem.goal:
                certificate = certify_witness(problem, environment.witness(child))
                reason = "certified" if certificate["success"] else "certification_failed"
            if reason is None:
                reason = environment.halt
            if reason is None and not environment.frontier:
                reason = "frontier_exhausted_requires_audit"
            if reason is None:
                reason = meter.reason()
        next_selection = None
        if reason is None:
            next_selection = choose()
            # Scoring itself counts against the classical time limit.
            reason = meter.reason()
        done = reason is not None
        after = 0. if done else environment.potential(objective)
        hit = certificate is not None and certificate["success"]
        base_reward = (success_reward if hit else 0.) - work_cost * attempted
        shaping = shaping_weight * (after - before)
        reward = base_reward + shaping
        logs.append({"record_id": rid, "token": token, "attempted_edges": attempted,
                     "base_reward": base_reward, "shaping": shaping, "reward": reward,
                     "potential_before": before, "potential_after": after, "terminal": done})
        if learn == "outer":
            next_value = None if done else float(next_selection[2] @ outer.weights)
            outer.update(features, current_q, reward, next_value)
        elif learn == "inner":
            after_value = 0. if done else outer.value(problem, environment.frontier.values(), context())
            # The outer weights are frozen during this stage. This is a target
            # for learning search order, NOT an admissible remaining-cost bound.
            inner.update(problem.operations[token].family, response_features,
                         base_reward + after_value - before_value)
        if done:
            return finish("feasible" if hit else "unknown", reason, certificate)
        selection = next_selection
    return finish("unknown", "frontier_exhausted_requires_audit")


def train_budgeted_hierarchy(
    curriculum: tuple[BoundedProblem, ...], *, episodes: tuple[int, int, int] = (6, 6, 2),
    seed: int = 11, limits: WorkLimits = WorkLimits(max_edges=128, max_records=512, wall_seconds=5., cpu_seconds=5.),
) -> tuple[BudgetedSarsa, BudgetedLinUCB, list[dict]]:
    if not curriculum:
        raise ValueError("training requires a nonempty budget/target curriculum")
    if len(episodes) != 3:
        raise ValueError("episodes must specify outer, inner and outer-adjustment counts")
    for value in episodes:
        nonnegative_integer(value, "episodes")
    outer, inner, logs = BudgetedSarsa(seed=seed), BudgetedLinUCB(), []
    index = 0
    for stage, count in zip(("outer", "inner", "outer_adjustment"), episodes, strict=True):
        frozen = outer.weights.copy() if stage == "inner" else None
        inner_before = inner.updates
        for episode in range(count):
            problem = curriculum[index % len(curriculum)]
            objective = ("t_count", "cnot_count", "depth", "gate_count")[index % 4]
            result = run_budgeted_search(problem, outer=outer, inner=inner,
                                        objective=objective, limits=limits,
                                        scheduler="outer" if stage == "outer" else "hierarchy",
                                        learn="inner" if stage == "inner" else "outer", epsilon=.2)
            logs.append({"stage": stage, "episode": episode, "problem_digest": problem.digest,
                         "objective": objective, "status": result.status, "reason": result.reason,
                         "edges": result.edges, "wall_seconds": result.wall_seconds,
                         "outer_updates": outer.updates, "inner_updates": inner.updates})
            index += 1
        if stage == "inner" and not np.array_equal(frozen, outer.weights):
            raise AssertionError("outer policy changed during frozen-value inner training")
        if stage != "inner" and inner.updates != inner_before:
            raise AssertionError("inner policy changed during outer training")
    return outer, inner, logs
