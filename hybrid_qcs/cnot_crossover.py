"""Exact CNOT-network crossover benchmark for eager versus deferred search.

The benchmark is deliberately narrower than unrestricted Clifford+T synthesis:
it uses the complete directed CNOT grammar on four to six qubits.  No target
witness is exposed to either policy.  The purpose is to create a controlled
regime in which an eager frontier expansion pays for every directed CNOT,
whereas a hierarchical scheduler can allocate a small exact continuation batch.

All semantic transitions, projective keys, Pareto dominance, resource checks,
and terminal certification continue to use the repository's exact hybrid
symbolic engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Iterable, Sequence

import numpy as np

from .benchmarks import SynthesisTarget, _target_from_hidden_gates, structured_toffoli_target
from .certify import certify_state, unitary_from_gates
from .model import Gate, HybridState, TransitionProfile
from .search import (
    HybridSearch,
    SearchRecord,
    _strictly_dominates,
    _weakly_dominates,
    symbolic_distance_components,
)


OUTER_FEATURE_NAMES = (
    "bias",
    "gate_fraction",
    "cnot_fraction",
    "depth_fraction",
    "mean_wire_depth_fraction",
    "maximum_wire_depth_fraction",
    "tableau_mismatch_fraction",
    "symbolic_distance_fraction",
    "pending_fraction",
    "register_fraction",
)
OUTER_FEATURE_DIM = len(OUTER_FEATURE_NAMES)

INNER_FEATURE_NAMES = (
    "bias",
    "linear_distance_before",
    "linear_distance_after",
    "linear_distance_reduction",
    "corrected_target_bits",
    "introduced_target_bits",
    "active_control_columns",
    "projected_depth_fraction",
    "last_gate_operand_overlap",
    "ordered_operand_code",
)
INNER_FEATURE_DIM = len(INNER_FEATURE_NAMES)


def cnot_gate_library(num_qubits: int) -> tuple[Gate, ...]:
    """Return every directed CNOT on a register in deterministic order."""

    if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
        raise TypeError("num_qubits must be an integer")
    if num_qubits < 2:
        raise ValueError("CNOT synthesis requires at least two qubits")
    return tuple(
        Gate("CNOT", (control, target))
        for control in range(num_qubits)
        for target in range(num_qubits)
        if control != target
    )


def swap_network(left: int, right: int) -> tuple[Gate, ...]:
    """Return the standard exact three-CNOT SWAP witness."""

    if left == right:
        raise ValueError("SWAP operands must be distinct")
    return (
        Gate("CNOT", (left, right)),
        Gate("CNOT", (right, left)),
        Gate("CNOT", (left, right)),
    )


def _target(
    name: str,
    split: str,
    num_qubits: int,
    gates: Iterable[Gate],
) -> SynthesisTarget:
    return _target_from_hidden_gates(
        name,
        split,
        num_qubits,
        tuple(gates),
        family="unrestricted-directed-cnot-network",
        convention="q0 is the least-significant basis bit",
    )


def crossover_training_targets() -> tuple[SynthesisTarget, ...]:
    """Training family: unseen-width-safe local and adjacent permutation motifs."""

    targets: list[SynthesisTarget] = []
    for num_qubits in (4, 5):
        index = 0
        for left in range(num_qubits):
            for right in range(left + 1, num_qubits):
                targets.append(
                    _target(
                        f"train-swap-{num_qubits}q-{index}",
                        "train",
                        num_qubits,
                        swap_network(left, right),
                    )
                )
                index += 1
        # Two adjacent transpositions create six-gate compositions without
        # exposing the disjoint-pair evaluation targets below.
        for start in range(num_qubits):
            middle = (start + 1) % num_qubits
            end = (start + 2) % num_qubits
            targets.append(
                _target(
                    f"train-adjacent-composition-{num_qubits}q-{start}",
                    "train",
                    num_qubits,
                    (
                        *swap_network(start, middle),
                        *swap_network(middle, end),
                    ),
                )
            )
    return tuple(targets)


def crossover_evaluation_targets() -> tuple[SynthesisTarget, ...]:
    """Held-out permutation targets of increasing width and composition length."""

    return (
        _target(
            "heldout-register-swap-4q",
            "test",
            4,
            (*swap_network(0, 2), *swap_network(1, 3)),
        ),
        _target(
            "heldout-partial-reversal-5q",
            "test",
            5,
            (*swap_network(0, 4), *swap_network(1, 3)),
        ),
        _target(
            "heldout-full-reversal-6q-ood",
            "ood",
            6,
            (
                *swap_network(0, 5),
                *swap_network(1, 4),
                *swap_network(2, 3),
            ),
        ),
    )


class EagerCnotSearch(HybridSearch):
    """Previous atomic-node semantics with the full directed CNOT grammar."""

    def __init__(
        self,
        target: SynthesisTarget,
        *,
        max_expansions: int = 4_096,
        shaping_weight: float = 0.5,
    ) -> None:
        if target.family != "unrestricted-directed-cnot-network":
            raise ValueError("EagerCnotSearch requires a CNOT-network target")
        super().__init__(
            target,
            max_expansions=max_expansions,
            shaping_weight=shaping_weight,
        )
        self.actions = cnot_gate_library(target.num_qubits)


@dataclass(slots=True)
class DeferredRecord:
    record_id: int
    state: HybridState
    symbolic_distance: int
    pending_mask: int
    allocations: int = 0
    features: object | None = None


@dataclass(frozen=True, slots=True)
class DeferredStep:
    selected_record_id: int
    tokens: tuple[int, ...]
    reward: float
    terminated: bool
    truncated: bool
    attempted_edges: int
    generated: int
    accepted: int
    rejected: int
    frontier_size: int
    solution_record_id: int | None


@dataclass(slots=True)
class DeferredProfile:
    transitions: TransitionProfile = field(default_factory=TransitionProfile)
    archive_lookups: int = 0
    dominance_comparisons: int = 0
    certification_calls: int = 0
    outer_rows_scored: int = 0
    inner_rows_scored: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            **self.transitions.to_dict(),
            "archive_lookups": self.archive_lookups,
            "dominance_comparisons": self.dominance_comparisons,
            "certification_calls": self.certification_calls,
            "outer_rows_scored": self.outer_rows_scored,
            "inner_rows_scored": self.inner_rows_scored,
        }


class DeferredCnotSearch:
    """Fair exact edge scheduler over persistent CNOT continuations."""

    def __init__(
        self,
        target: SynthesisTarget,
        *,
        max_allocations: int = 4_096,
        max_edges: int = 65_536,
        batch_size: int = 4,
        shaping_weight: float = 0.5,
        success_bonus: float = 20.0,
        failure_penalty: float = 20.0,
        fairness_start_k: int = 8,
    ) -> None:
        if target.family != "unrestricted-directed-cnot-network":
            raise ValueError("DeferredCnotSearch requires a CNOT-network target")
        if max_allocations <= 0 or max_edges <= 0 or batch_size <= 0:
            raise ValueError("search limits and batch_size must be positive")
        self.target = target
        self.actions = cnot_gate_library(target.num_qubits)
        self.full_pending_mask = (1 << len(self.actions)) - 1
        self.max_allocations = int(max_allocations)
        self.max_edges = int(max_edges)
        self.batch_size = int(batch_size)
        self.shaping_weight = float(shaping_weight)
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)
        self.fairness_start_k = int(fairness_start_k)
        self.records: dict[int, DeferredRecord] = {}
        self.frontier: dict[int, DeferredRecord] = {}
        self.pareto: dict[
            tuple[object, ...], list[tuple[tuple[int, ...], int]]
        ] = {}
        self.next_record_id = 0
        self.allocations = 0
        self.edge_attempts = 0
        self.generated = 0
        self.accepted = 0
        self.rejected = 0
        self.solution_record_id: int | None = None
        self.frontier_peak = 0
        self.profile = DeferredProfile()
        self.reset()

    def reset(self) -> None:
        self.records.clear()
        self.frontier.clear()
        self.pareto.clear()
        self.next_record_id = 0
        self.allocations = 0
        self.edge_attempts = 0
        self.generated = 0
        self.accepted = 0
        self.rejected = 0
        self.solution_record_id = None
        self.frontier_peak = 0
        self.profile = DeferredProfile()
        root_state = HybridState.identity(self.target.num_qubits, self.target.budget)
        root = self._new_record(root_state)
        self.frontier[root.record_id] = root
        self.pareto[root_state.canonical_key] = [
            (root_state.resource_vector(), root.record_id)
        ]
        self.frontier_peak = 1

    def _new_record(self, state: HybridState) -> DeferredRecord:
        distance = symbolic_distance_components(state, self.target)[0]
        record = DeferredRecord(
            self.next_record_id,
            state,
            distance,
            self.full_pending_mask,
        )
        self.next_record_id += 1
        self.records[record.record_id] = record
        return record

    def open_records(self) -> tuple[DeferredRecord, ...]:
        return tuple(self.frontier.values())

    def pending_tokens(self, record: DeferredRecord) -> tuple[int, ...]:
        return tuple(
            token
            for token in range(len(self.actions))
            if record.pending_mask & (1 << token)
        )

    def frontier_potential(self) -> float:
        if self.solution_record_id is not None:
            return 0.0
        if not self.frontier:
            return -float(4 * self.target.budget.max_gates + 1)
        return -float(
            min(record.symbolic_distance for record in self.frontier.values())
        )

    def _insert(self, state: HybridState) -> DeferredRecord | None:
        self.profile.archive_lookups += 1
        key = state.canonical_key
        resources = state.resource_vector()
        group = self.pareto.setdefault(key, [])
        for existing, _ in group:
            self.profile.dominance_comparisons += 1
            if _weakly_dominates(existing, resources):
                self.rejected += 1
                return None

        survivors: list[tuple[tuple[int, ...], int]] = []
        for existing, record_id in group:
            self.profile.dominance_comparisons += 1
            if _strictly_dominates(resources, existing):
                self.frontier.pop(record_id, None)
            else:
                survivors.append((existing, record_id))

        # A dominating replacement starts with its complete continuation set.
        # Processed bits of a dominated record are deliberately not inherited.
        record = self._new_record(state)
        survivors.append((resources, record.record_id))
        self.pareto[key] = survivors
        self.frontier[record.record_id] = record
        self.accepted += 1
        return record

    def _forced_fair_edge(self) -> tuple[int, int] | None:
        """Use a vanishing-density FIFO override at square allocation indices."""

        next_allocation = self.allocations + 1
        root = math.isqrt(next_allocation)
        if root < self.fairness_start_k or root * root != next_allocation:
            return None
        for record_id in sorted(self.frontier):
            record = self.frontier[record_id]
            tokens = self.pending_tokens(record)
            if tokens:
                return record_id, tokens[0]
        return None

    def process_batch(
        self,
        record_id: int,
        tokens: Sequence[int],
        *,
        allow_fairness_override: bool = True,
    ) -> DeferredStep:
        if self.solution_record_id is not None:
            raise RuntimeError("the search has already certified a solution")
        try:
            selected = self.frontier[int(record_id)]
        except KeyError as exc:
            raise KeyError(f"record {record_id} is not active") from exc

        requested = [int(token) for token in tokens]
        if allow_fairness_override:
            forced = self._forced_fair_edge()
            if forced is not None:
                forced_record, forced_token = forced
                selected = self.frontier[forced_record]
                record_id = forced_record
                requested = [forced_token]

        before = self.frontier_potential()
        selected.allocations += 1
        self.allocations += 1
        generated_now = 0
        accepted_now = 0
        attempted_now = 0
        processed: list[int] = []

        for token in requested[: self.batch_size]:
            if self.edge_attempts >= self.max_edges:
                break
            if token < 0 or token >= len(self.actions):
                raise IndexError(f"continuation token {token} is out of range")
            bit = 1 << token
            if not selected.pending_mask & bit:
                continue
            selected.pending_mask &= ~bit
            processed.append(token)
            self.edge_attempts += 1
            attempted_now += 1
            child_state = selected.state.apply(
                self.actions[token],
                partial_order_reduction=True,
                profile=self.profile.transitions,
            )
            if child_state is None:
                continue
            self.generated += 1
            generated_now += 1
            child = self._insert(child_state)
            if child is None:
                continue
            accepted_now += 1
            if child_state.canonical_key == self.target.canonical_key:
                self.profile.certification_calls += 1
                if certify_state(self.target, child_state).success:
                    self.solution_record_id = child.record_id
                    break

        if selected.pending_mask == 0:
            self.frontier.pop(selected.record_id, None)

        self.frontier_peak = max(self.frontier_peak, len(self.frontier))
        exhausted = not self.frontier and self.solution_record_id is None
        truncated = (
            self.solution_record_id is None
            and (
                self.allocations >= self.max_allocations
                or self.edge_attempts >= self.max_edges
            )
        )
        terminated = self.solution_record_id is not None or exhausted
        after = self.frontier_potential()
        reward = -float(attempted_now) + self.shaping_weight * (after - before)
        if self.solution_record_id is not None:
            reward += self.success_bonus
        elif exhausted or truncated:
            reward -= self.failure_penalty

        return DeferredStep(
            selected_record_id=selected.record_id,
            tokens=tuple(processed),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            attempted_edges=attempted_now,
            generated=generated_now,
            accepted=accepted_now,
            rejected=generated_now - accepted_now,
            frontier_size=len(self.frontier),
            solution_record_id=self.solution_record_id,
        )

    def solution_state(self) -> HybridState | None:
        if self.solution_record_id is None:
            return None
        return self.records[self.solution_record_id].state

    def metrics(self) -> dict[str, object]:
        return {
            "success": self.solution_record_id is not None,
            "allocations": self.allocations,
            "edge_attempts": self.edge_attempts,
            "generated": self.generated,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "frontier_peak": self.frontier_peak,
            "records": len(self.records),
            "profile": self.profile.to_dict(),
        }


def _outer_features(
    record: SearchRecord | DeferredRecord,
    target: SynthesisTarget,
    *,
    deferred: bool,
) -> np.ndarray:
    state = record.state
    n = state.num_qubits
    total, tableau_mismatch, _, _ = symbolic_distance_components(state, target)
    pending_fraction = 1.0
    if deferred:
        pending_mask = getattr(record, "pending_mask")
        pending_fraction = int(pending_mask).bit_count() / max(1, n * (n - 1))
    values = np.asarray(
        [
            1.0,
            state.gate_count / max(1, state.budget.max_gates),
            state.cnot_count / max(1, state.budget.max_cnot_count),
            state.depth / max(1, state.budget.max_depth),
            sum(state.wire_depths)
            / max(1, n * state.budget.max_depth),
            max(state.wire_depths, default=0)
            / max(1, state.budget.max_depth),
            tableau_mismatch / max(1, 2 * n),
            total / max(1, 4 * n),
            pending_fraction,
            n / 6.0,
        ],
        dtype=np.float64,
    )
    if values.shape != (OUTER_FEATURE_DIM,):
        raise AssertionError("outer feature schema has the wrong dimension")
    return values


@dataclass(slots=True)
class LinearOuterSarsa:
    learning_rate: float = 0.01
    gamma: float = 1.0
    seed: int = 0
    theta: np.ndarray = field(init=False, repr=False)
    rng: np.random.Generator = field(init=False, repr=False)
    rows_scored: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.theta = np.zeros(OUTER_FEATURE_DIM, dtype=np.float64)
        self.rng = np.random.default_rng(self.seed)

    def choose(
        self,
        records: Sequence[SearchRecord | DeferredRecord],
        target: SynthesisTarget,
        epsilon: float,
        *,
        deferred: bool,
    ) -> tuple[int, np.ndarray, float]:
        nodes = tuple(records)
        if not nodes:
            raise RuntimeError("cannot choose from an empty frontier")
        matrix = np.stack(
            [_outer_features(record, target, deferred=deferred) for record in nodes]
        )
        self.rows_scored += len(nodes)
        scores = matrix @ self.theta
        if self.rng.random() < epsilon:
            index = int(self.rng.integers(len(nodes)))
        else:
            best = float(np.max(scores))
            tied = np.flatnonzero(np.isclose(scores, best, rtol=0.0, atol=1e-12))
            index = min(
                (int(value) for value in tied),
                key=lambda value: nodes[value].record_id,
            )
        return nodes[index].record_id, matrix[index].copy(), float(scores[index])

    def update(
        self,
        features: np.ndarray,
        q_value: float,
        reward: float,
        next_q_value: float | None,
        *,
        duration: int = 1,
    ) -> float:
        target = reward
        if next_q_value is not None:
            target += (self.gamma ** max(1, int(duration))) * next_q_value
        td_error = float(target - q_value)
        self.theta += self.learning_rate * td_error * features
        np.clip(self.theta, -50.0, 50.0, out=self.theta)
        return td_error


def _x_masks(state: HybridState) -> tuple[int, ...]:
    return tuple(axis.x_mask for axis in state.tableau.forward_x)


def _target_x_masks(target: SynthesisTarget) -> tuple[int, ...]:
    return tuple(
        int(payload[0]) for payload in target.tableau_payload[: target.num_qubits]
    )


def linear_map_distance(state: HybridState, target: SynthesisTarget) -> int:
    """Hamming distance between computational-basis linear maps."""

    return sum(
        (current ^ desired).bit_count()
        for current, desired in zip(
            _x_masks(state), _target_x_masks(target), strict=True
        )
    )


def inner_context(
    record: DeferredRecord,
    gate: Gate,
    target: SynthesisTarget,
) -> np.ndarray:
    """Cheap action-conditioned context for a directed CNOT continuation."""

    if gate.name != "CNOT":
        raise ValueError("the crossover bandit supports CNOT actions only")
    state = record.state
    n = state.num_qubits
    control, target_wire = gate.qubits
    current_masks = _x_masks(state)
    desired_masks = _target_x_masks(target)
    before = 0
    after = 0
    corrected = 0
    introduced = 0
    active_columns = 0
    target_bit = 1 << target_wire
    control_bit = 1 << control
    for current, desired in zip(current_masks, desired_masks, strict=True):
        before += (current ^ desired).bit_count()
        updated = current
        if current & control_bit:
            active_columns += 1
            if bool(current & target_bit) != bool(desired & target_bit):
                corrected += 1
            else:
                introduced += 1
            updated ^= target_bit
        after += (updated ^ desired).bit_count()
    reduction = before - after
    layer = 1 + max(
        state.wire_depths[control], state.wire_depths[target_wire]
    )
    last = state.last_gate
    overlap = (
        0.0
        if last is None
        else len(set(last.qubits).intersection(gate.qubits)) / 2.0
    )
    values = np.asarray(
        [
            1.0,
            before / max(1, n * n),
            after / max(1, n * n),
            reduction / max(1, n),
            corrected / max(1, n),
            introduced / max(1, n),
            active_columns / max(1, n),
            layer / max(1, state.budget.max_depth),
            overlap,
            (control * n + target_wire) / max(1, n * n - 1),
        ],
        dtype=np.float64,
    )
    if values.shape != (INNER_FEATURE_DIM,):
        raise AssertionError("inner feature schema has the wrong dimension")
    return values


@dataclass(slots=True)
class LinearCnotLinUCB:
    """Linear contextual bandit with a rank-one inverse update."""

    alpha: float = 0.5
    regularization: float = 1.0
    a_matrix: np.ndarray = field(init=False, repr=False)
    a_inverse: np.ndarray = field(init=False, repr=False)
    b_vector: np.ndarray = field(init=False, repr=False)
    updates: int = field(init=False, default=0)
    rows_scored: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")
        self.a_matrix = np.eye(INNER_FEATURE_DIM) * self.regularization
        self.a_inverse = np.eye(INNER_FEATURE_DIM) / self.regularization
        self.b_vector = np.zeros(INNER_FEATURE_DIM, dtype=np.float64)

    @property
    def posterior_mean(self) -> np.ndarray:
        return self.a_inverse @ self.b_vector

    def choose(
        self,
        record: DeferredRecord,
        tokens: Sequence[int],
        actions: Sequence[Gate],
        target: SynthesisTarget,
        *,
        explore: bool,
    ) -> tuple[int, np.ndarray, float]:
        if not tokens:
            raise RuntimeError("cannot choose from an empty continuation set")
        beta = self.posterior_mean
        best: tuple[float, int, int, np.ndarray] | None = None
        for token in tokens:
            context = inner_context(record, actions[int(token)], target)
            mean = float(context @ beta)
            bonus = 0.0
            if explore:
                bonus = self.alpha * math.sqrt(
                    max(0.0, float(context @ self.a_inverse @ context))
                )
            candidate = (mean + bonus, -int(token), int(token), context)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        assert best is not None
        self.rows_scored += len(tokens)
        return best[2], best[3].copy(), best[0]

    def update(self, context: np.ndarray, reward: float) -> None:
        vector = np.asarray(context, dtype=np.float64)
        projected = self.a_inverse @ vector
        denominator = 1.0 + float(vector @ projected)
        self.a_inverse -= np.outer(projected, projected) / denominator
        self.a_matrix += np.outer(vector, vector)
        self.b_vector += float(reward) * vector
        self.updates += 1


def _epsilon(episode: int, episodes: int) -> float:
    fraction = episode / max(1, episodes - 1)
    return 0.32 + fraction * (0.02 - 0.32)


def train_eager_outer(
    targets: Sequence[SynthesisTarget],
    *,
    episodes: int = 400,
    seed: int = 11,
) -> LinearOuterSarsa:
    policy = LinearOuterSarsa(seed=seed)
    for episode in range(episodes):
        target = targets[episode % len(targets)]
        environment = EagerCnotSearch(target, max_expansions=256)
        epsilon = _epsilon(episode, episodes)
        record_id, features, q_value = policy.choose(
            environment.open_records(), target, epsilon, deferred=False
        )
        while True:
            transition = environment.expand(record_id)
            done = transition.terminated or transition.truncated
            if done:
                policy.update(features, q_value, transition.reward, None)
                break
            next_id, next_features, next_q = policy.choose(
                environment.open_records(), target, epsilon, deferred=False
            )
            policy.update(features, q_value, transition.reward, next_q)
            record_id, features = next_id, next_features
            q_value = float(features @ policy.theta)
    return policy


def train_inner_bandit(
    targets: Sequence[SynthesisTarget],
    *,
    episodes: int = 300,
    alpha: float = 0.5,
) -> LinearCnotLinUCB:
    """Train from chosen-action feedback under a fixed symbolic outer rule."""

    bandit = LinearCnotLinUCB(alpha=alpha)
    for episode in range(episodes):
        target = targets[episode % len(targets)]
        environment = DeferredCnotSearch(
            target,
            max_allocations=128,
            max_edges=256,
            batch_size=1,
            fairness_start_k=10_000,
        )
        while environment.frontier and environment.solution_record_id is None:
            record = min(
                environment.open_records(),
                key=lambda item: (
                    item.symbolic_distance,
                    item.state.gate_count,
                    item.record_id,
                ),
            )
            tokens = environment.pending_tokens(record)
            token, context, _ = bandit.choose(
                record,
                tokens,
                environment.actions,
                target,
                explore=True,
            )
            step = environment.process_batch(
                record.record_id,
                (token,),
                allow_fairness_override=False,
            )
            # Coordinate 3 is the normalized exact reduction in the induced
            # binary linear map.  It is immediately observed after one edge.
            reward = float(context[3]) - 0.005
            if step.solution_record_id is not None:
                reward += 2.0
            bandit.update(context, reward)
            if step.terminated or step.truncated:
                break
    return bandit


def _ordered_bandit_tokens(
    bandit: LinearCnotLinUCB,
    environment: DeferredCnotSearch,
    record: DeferredRecord,
    target: SynthesisTarget,
    batch_size: int,
) -> tuple[int, ...]:
    remaining = list(environment.pending_tokens(record))
    ordered: list[int] = []
    while remaining and len(ordered) < batch_size:
        token, _, _ = bandit.choose(
            record,
            remaining,
            environment.actions,
            target,
            explore=False,
        )
        ordered.append(token)
        remaining.remove(token)
    return tuple(ordered)


def train_deferred_outer(
    targets: Sequence[SynthesisTarget],
    bandit: LinearCnotLinUCB,
    *,
    episodes: int = 400,
    seed: int = 17,
    batch_size: int = 4,
) -> LinearOuterSarsa:
    policy = LinearOuterSarsa(seed=seed)
    for episode in range(episodes):
        target = targets[episode % len(targets)]
        environment = DeferredCnotSearch(
            target,
            max_allocations=256,
            max_edges=512,
            batch_size=batch_size,
            fairness_start_k=10_000,
        )
        epsilon = _epsilon(episode, episodes)
        record_id, features, q_value = policy.choose(
            environment.open_records(), target, epsilon, deferred=True
        )
        while True:
            record = environment.frontier[record_id]
            tokens = _ordered_bandit_tokens(
                bandit, environment, record, target, batch_size
            )
            step = environment.process_batch(
                record_id,
                tokens,
                allow_fairness_override=False,
            )
            done = step.terminated or step.truncated
            if done:
                policy.update(
                    features,
                    q_value,
                    step.reward,
                    None,
                    duration=max(1, step.attempted_edges),
                )
                break
            next_id, next_features, next_q = policy.choose(
                environment.open_records(), target, epsilon, deferred=True
            )
            policy.update(
                features,
                q_value,
                step.reward,
                next_q,
                duration=max(1, step.attempted_edges),
            )
            record_id, features = next_id, next_features
            q_value = float(features @ policy.theta)
    return policy


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    method: str
    target: str
    num_qubits: int
    generator_length: int
    success: bool
    certified: bool
    stop_reason: str
    wall_seconds: float
    cpu_seconds: float
    outer_decisions: int
    attempted_edges: int
    generated: int
    accepted: int
    rejected: int
    frontier_peak: int
    policy_rows: int
    maximum_matrix_error: float | None
    witness: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "target": self.target,
            "num_qubits": self.num_qubits,
            "generator_length": self.generator_length,
            "success": self.success,
            "certified": self.certified,
            "stop_reason": self.stop_reason,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "outer_decisions": self.outer_decisions,
            "attempted_edges": self.attempted_edges,
            "generated": self.generated,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "frontier_peak": self.frontier_peak,
            "policy_rows": self.policy_rows,
            "maximum_matrix_error": self.maximum_matrix_error,
            "witness": "; ".join(self.witness),
        }


def _finish_result(
    *,
    method: str,
    target: SynthesisTarget,
    state: HybridState | None,
    wall_seconds: float,
    cpu_seconds: float,
    stop_reason: str,
    outer_decisions: int,
    attempted_edges: int,
    generated: int,
    accepted: int,
    rejected: int,
    frontier_peak: int,
    policy_rows: int,
) -> EvaluationResult:
    certification = None if state is None else certify_state(target, state)
    return EvaluationResult(
        method=method,
        target=target.name,
        num_qubits=target.num_qubits,
        generator_length=target.generator_length,
        success=bool(certification and certification.success),
        certified=bool(certification and certification.success),
        stop_reason=("certified" if certification and certification.success else stop_reason),
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        outer_decisions=outer_decisions,
        attempted_edges=attempted_edges,
        generated=generated,
        accepted=accepted,
        rejected=rejected,
        frontier_peak=frontier_peak,
        policy_rows=policy_rows,
        maximum_matrix_error=(
            None if certification is None else certification.maximum_matrix_error
        ),
        witness=() if certification is None else certification.witness,
    )


def evaluate_eager_sarsa(
    policy: LinearOuterSarsa,
    target: SynthesisTarget,
    *,
    max_expansions: int = 4_096,
    wall_limit: float = 20.0,
) -> EvaluationResult:
    environment = EagerCnotSearch(target, max_expansions=max_expansions)
    rows_before = policy.rows_scored
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    stop_reason = "frontier_exhausted"
    while environment.frontier and environment.solution_record_id is None:
        if environment.expansions >= max_expansions:
            stop_reason = "expansion_cap"
            break
        if time.perf_counter() - wall_start >= wall_limit:
            stop_reason = "wall_limit"
            break
        record_id, _, _ = policy.choose(
            environment.open_records(), target, 0.0, deferred=False
        )
        transition = environment.expand(record_id)
        if transition.terminated:
            stop_reason = "frontier_exhausted"
            break
        if transition.truncated:
            stop_reason = "expansion_cap"
            break
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start
    state = environment.solution_state()
    return _finish_result(
        method="eager_outer_sarsa",
        target=target,
        state=state,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        stop_reason=stop_reason,
        outer_decisions=environment.expansions,
        attempted_edges=environment.profile.transitions.attempted,
        generated=environment.generated,
        accepted=environment.accepted,
        rejected=environment.rejected,
        frontier_peak=environment.frontier_peak,
        policy_rows=policy.rows_scored - rows_before,
    )


def evaluate_eager_target_potential(
    target: SynthesisTarget,
    *,
    max_expansions: int = 4_096,
    wall_limit: float = 20.0,
) -> EvaluationResult:
    environment = EagerCnotSearch(target, max_expansions=max_expansions)
    distance_cache: dict[tuple[object, ...], float] = {}
    rows_scored = 0

    def score(record: SearchRecord) -> float:
        key = record.state.canonical_key
        value = distance_cache.get(key)
        if value is None:
            candidate = unitary_from_gates(
                target.num_qubits, record.state.reconstruct_gates()
            )
            dimension = 1 << target.num_qubits
            overlap = np.trace(target.unitary.conj().T @ candidate)
            value = 1.0 - float(abs(overlap) ** 2 / (dimension * dimension))
            distance_cache[key] = value
        return value

    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    stop_reason = "frontier_exhausted"
    while environment.frontier and environment.solution_record_id is None:
        if environment.expansions >= max_expansions:
            stop_reason = "expansion_cap"
            break
        if time.perf_counter() - wall_start >= wall_limit:
            stop_reason = "wall_limit"
            break
        records = environment.open_records()
        rows_scored += len(records)
        selected = min(
            records,
            key=lambda record: (
                score(record),
                record.state.gate_count,
                record.state.t_count,
                record.state.cnot_count,
                record.record_id,
            ),
        )
        transition = environment.expand(selected.record_id)
        if transition.terminated:
            stop_reason = "frontier_exhausted"
            break
        if transition.truncated:
            stop_reason = "expansion_cap"
            break
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start
    return _finish_result(
        method="eager_target_potential",
        target=target,
        state=environment.solution_state(),
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        stop_reason=stop_reason,
        outer_decisions=environment.expansions,
        attempted_edges=environment.profile.transitions.attempted,
        generated=environment.generated,
        accepted=environment.accepted,
        rejected=environment.rejected,
        frontier_peak=environment.frontier_peak,
        policy_rows=rows_scored,
    )


def evaluate_hierarchy(
    outer_policy: LinearOuterSarsa,
    bandit: LinearCnotLinUCB,
    target: SynthesisTarget,
    *,
    batch_size: int = 4,
    max_allocations: int = 4_096,
    wall_limit: float = 20.0,
) -> EvaluationResult:
    environment = DeferredCnotSearch(
        target,
        max_allocations=max_allocations,
        max_edges=max_allocations * len(cnot_gate_library(target.num_qubits)),
        batch_size=batch_size,
    )
    outer_before = outer_policy.rows_scored
    inner_before = bandit.rows_scored
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    stop_reason = "frontier_exhausted"
    while environment.frontier and environment.solution_record_id is None:
        if environment.allocations >= max_allocations:
            stop_reason = "allocation_cap"
            break
        if time.perf_counter() - wall_start >= wall_limit:
            stop_reason = "wall_limit"
            break
        record_id, _, _ = outer_policy.choose(
            environment.open_records(), target, 0.0, deferred=True
        )
        record = environment.frontier[record_id]
        tokens = _ordered_bandit_tokens(
            bandit, environment, record, target, batch_size
        )
        step = environment.process_batch(record_id, tokens)
        if step.terminated:
            stop_reason = "frontier_exhausted"
            break
        if step.truncated:
            stop_reason = "allocation_cap"
            break
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start
    policy_rows = (
        outer_policy.rows_scored
        - outer_before
        + bandit.rows_scored
        - inner_before
    )
    return _finish_result(
        method="deferred_outer_sarsa_inner_linucb",
        target=target,
        state=environment.solution_state(),
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        stop_reason=stop_reason,
        outer_decisions=environment.allocations,
        attempted_edges=environment.edge_attempts,
        generated=environment.generated,
        accepted=environment.accepted,
        rejected=environment.rejected,
        frontier_peak=environment.frontier_peak,
        policy_rows=policy_rows,
    )


def unrestricted_toffoli_probe(
    *,
    max_expansions: int = 4_096,
) -> dict[str, object]:
    """Bounded unrestricted-native probe using the analytical Toffoli target."""

    target = structured_toffoli_target()
    environment = HybridSearch(target, max_expansions=max_expansions)
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    state = environment.run_scheduler(
        lambda records: min(
            records,
            key=lambda record: (
                record.symbolic_distance,
                record.state.gate_count,
                record.record_id,
            ),
        ).record_id
    )
    return {
        "target": target.name,
        "search_mode": "unrestricted native HybridSearch; no structured adapter",
        "success": state is not None,
        "certified": bool(state is not None and certify_state(target, state).success),
        "expansions": environment.expansions,
        "attempted_edges": environment.profile.transitions.attempted,
        "generated": environment.generated,
        "frontier_peak": environment.frontier_peak,
        "wall_seconds": time.perf_counter() - wall_start,
        "cpu_seconds": time.process_time() - cpu_start,
        "interpretation": (
            "This is a feasibility probe, not the crossover benchmark. "
            "Failure at the cap does not compare old and hierarchical policies."
        ),
    }


__all__ = [
    "DeferredCnotSearch",
    "DeferredRecord",
    "DeferredStep",
    "EagerCnotSearch",
    "EvaluationResult",
    "INNER_FEATURE_NAMES",
    "LinearCnotLinUCB",
    "LinearOuterSarsa",
    "OUTER_FEATURE_NAMES",
    "cnot_gate_library",
    "crossover_evaluation_targets",
    "crossover_training_targets",
    "evaluate_eager_sarsa",
    "evaluate_eager_target_potential",
    "evaluate_hierarchy",
    "inner_context",
    "linear_map_distance",
    "swap_network",
    "train_deferred_outer",
    "train_eager_outer",
    "train_inner_bandit",
    "unrestricted_toffoli_probe",
]
