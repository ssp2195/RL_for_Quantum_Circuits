"""Mixed Clifford+T continuation-cost crossover benchmark.

The benchmark uses the complete native grammar

    H, S, SDG, T, TDG, CNOT

on four to six qubits.  Targets are short Clifford-frame signed phase-pair
motifs: a Clifford scaffold establishes a nontrivial frame and two signed
Pauli rotations are then injected with T and T-dagger.  Hidden witnesses are
used only to construct and independently validate target transformations.

The eager baseline retains the previous action semantics: linear SARSA selects
one frontier record and the exact engine attempts every native continuation.
The deferred method uses the *same* frozen outer SARSA policy, while a disjoint
linear LinUCB model orders only the still-pending continuations.  This isolates
the continuation-cost effect without changing deterministic legality,
canonicalization, Pareto pruning, or terminal certification.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
import time
from typing import Iterable, Sequence

import numpy as np

from .benchmarks import SynthesisTarget, _target_from_hidden_gates
from .certify import certify_state, unitary_from_gates
from .model import Gate, HybridState, INVERSE_GATE, TransitionProfile, generate_gates
from .pauli import conjugate_by_gate
from .search import (
    HybridSearch,
    SearchRecord,
    _strictly_dominates,
    _weakly_dominates,
    symbolic_distance_components,
)

MIXED_FAMILY = "unrestricted-mixed-clifford-t-phase-pair"
GATE_FAMILIES = ("H", "S", "SDG", "T", "TDG", "CNOT")

OUTER_FEATURE_NAMES = (
    "bias",
    "t_fraction",
    "cnot_fraction",
    "gate_fraction",
    "depth_fraction",
    "mean_wire_depth_fraction",
    "max_wire_depth_fraction",
    "rotation_fraction",
    "anticommuting_pair_fraction",
    "mean_pauli_weight_fraction",
    "tableau_mismatch_fraction",
    "rotation_sequence_mismatch_fraction",
    "rotation_multiset_mismatch_fraction",
    "symbolic_distance_fraction",
    "target_rotation_fraction",
    "target_tableau_nonidentity_fraction",
    "last_H",
    "last_S",
    "last_SDG",
    "last_T",
    "last_TDG",
    "last_CNOT",
    "register_fraction",
)
OUTER_FEATURE_DIM = len(OUTER_FEATURE_NAMES)

INNER_CONTEXT_NAMES = (
    "bias",
    "symbolic_distance_fraction",
    "tableau_reduction_fraction",
    "signed_rotation_match",
    "axis_rotation_match",
    "rotation_deficit_fraction",
    "target_tableau_operand_mismatch_fraction",
    "remaining_t_slack",
    "remaining_cnot_slack",
    "remaining_gate_slack",
    "remaining_depth_slack",
    "last_gate_operand_overlap",
    "register_fraction",
)
INNER_CONTEXT_DIM = len(INNER_CONTEXT_NAMES)


def mixed_gate_library(num_qubits: int) -> tuple[Gate, ...]:
    """Complete deterministic native Clifford+T grammar."""

    return generate_gates(num_qubits)


def _mixed_target(
    name: str,
    split: str,
    num_qubits: int,
    gates: Iterable[Gate],
) -> SynthesisTarget:
    target = _target_from_hidden_gates(
        name,
        split,
        num_qubits,
        tuple(gates),
        family=MIXED_FAMILY,
        convention="q0 is the least-significant basis bit",
    )
    if not target.rotation_payloads:
        raise ValueError("mixed crossover targets must retain non-Clifford content")
    return target


def mixed_training_targets() -> tuple[SynthesisTarget, ...]:
    """Small exact curriculum spanning local mixed-gate interactions.

    The curriculum contains no held-out witness.  It deliberately exposes the
    learner to H/CNOT axis transport, S/SDG Clifford changes, and both signs of
    non-Clifford rotation on four- and five-qubit registers.
    """

    specifications: tuple[tuple[int, tuple[Gate, ...]], ...] = (
        (4, (Gate("H", (0,)), Gate("CNOT", (0, 1)), Gate("T", (1,)))),
        (4, (Gate("H", (2,)), Gate("CNOT", (2, 3)), Gate("TDG", (3,)))),
        (4, (Gate("S", (1,)), Gate("CNOT", (0, 2)), Gate("T", (2,)))),
        (4, (Gate("SDG", (3,)), Gate("CNOT", (1, 0)), Gate("TDG", (0,)))),
        (
            4,
            (
                Gate("H", (0,)), Gate("CNOT", (0, 2)), Gate("S", (3,)),
                Gate("T", (2,)),
            ),
        ),
        (
            4,
            (
                Gate("H", (3,)), Gate("CNOT", (3, 1)), Gate("SDG", (0,)),
                Gate("TDG", (1,)),
            ),
        ),
        (
            4,
            (
                Gate("CNOT", (0, 3)), Gate("H", (1,)), Gate("CNOT", (2, 3)),
                Gate("T", (2,)), Gate("TDG", (3,)),
            ),
        ),
        (
            4,
            (
                Gate("S", (2,)), Gate("CNOT", (1, 0)), Gate("H", (0,)),
                Gate("CNOT", (1, 0)), Gate("T", (3,)),
            ),
        ),
        (5, (Gate("H", (1,)), Gate("CNOT", (1, 4)), Gate("T", (4,)))),
        (5, (Gate("S", (4,)), Gate("CNOT", (0, 3)), Gate("TDG", (3,)))),
        (
            5,
            (
                Gate("H", (2,)), Gate("CNOT", (2, 0)), Gate("SDG", (4,)),
                Gate("T", (0,)),
            ),
        ),
        (
            5,
            (
                Gate("H", (4,)), Gate("CNOT", (4, 2)), Gate("S", (1,)),
                Gate("TDG", (2,)),
            ),
        ),
        (
            5,
            (
                Gate("CNOT", (0, 4)), Gate("H", (3,)), Gate("CNOT", (2, 1)),
                Gate("T", (4,)), Gate("TDG", (1,)),
            ),
        ),
        (
            5,
            (
                Gate("SDG", (0,)), Gate("CNOT", (3, 2)), Gate("H", (2,)),
                Gate("CNOT", (3, 2)), Gate("T", (1,)),
            ),
        ),
        (
            5,
            (
                Gate("H", (0,)), Gate("S", (3,)), Gate("CNOT", (0, 1)),
                Gate("CNOT", (2, 4)), Gate("T", (1,)), Gate("TDG", (4,)),
            ),
        ),
        (
            5,
            (
                Gate("H", (3,)), Gate("SDG", (1,)), Gate("CNOT", (3, 4)),
                Gate("CNOT", (0, 2)), Gate("TDG", (4,)), Gate("T", (2,)),
            ),
        ),
    )
    targets: list[SynthesisTarget] = []
    seen: set[tuple[object, ...]] = set()
    for index, (num_qubits, gates) in enumerate(specifications):
        target = _mixed_target(
            f"train-mixed-frame-phase-{num_qubits}q-{index}",
            "train",
            num_qubits,
            gates,
        )
        if target.canonical_key in seen:
            raise AssertionError("mixed training curriculum contains a duplicate")
        seen.add(target.canonical_key)
        targets.append(target)
    return tuple(targets)


def mixed_evaluation_targets() -> tuple[SynthesisTarget, ...]:
    """Held-out mixed Clifford+T targets with increasing native branching."""

    return (
        _mixed_target(
            "heldout-mixed-frame-phase-4q",
            "test",
            4,
            (
                Gate("H", (1,)),
                Gate("CNOT", (0, 3)),
                Gate("CNOT", (2, 3)),
                Gate("S", (2,)),
                Gate("T", (2,)),
                Gate("TDG", (3,)),
            ),
        ),
        _mixed_target(
            "heldout-mixed-basis-echo-5q",
            "test",
            5,
            (
                Gate("SDG", (4,)),
                Gate("CNOT", (1, 0)),
                Gate("H", (0,)),
                Gate("CNOT", (1, 0)),
                Gate("T", (3,)),
                Gate("TDG", (4,)),
            ),
        ),
        _mixed_target(
            "heldout-mixed-frame-phase-6q-ood",
            "ood",
            6,
            (
                Gate("CNOT", (2, 4)),
                Gate("CNOT", (3, 1)),
                Gate("H", (3,)),
                Gate("S", (0,)),
                Gate("T", (2,)),
                Gate("TDG", (0,)),
            ),
        ),
    )


class EagerMixedSearch(HybridSearch):
    """Previous atomic-node expansion over the complete mixed grammar."""

    def __init__(
        self,
        target: SynthesisTarget,
        *,
        max_expansions: int = 4_096,
        shaping_weight: float = 0.5,
    ) -> None:
        if target.family != MIXED_FAMILY:
            raise ValueError("EagerMixedSearch requires a mixed crossover target")
        super().__init__(
            target,
            max_expansions=max_expansions,
            shaping_weight=shaping_weight,
        )
        self.actions = mixed_gate_library(target.num_qubits)


def _cheap_legal_continuation(state: HybridState, gate: Gate) -> bool:
    """Exact target-independent legality check without symbolic child creation."""

    wire_parents = {state.wire_tails[q] for q in gate.qubits}
    if len(wire_parents) == 1:
        previous = next(iter(wire_parents))
        if (
            previous is not None
            and INVERSE_GATE[previous.gate.name] == gate.name
            and previous.gate.qubits == gate.qubits
        ):
            return False
    if state.tail is not None:
        last = state.tail.gate
        if set(last.qubits).isdisjoint(gate.qubits) and gate.sort_key() < last.sort_key():
            return False

    layer = 1 + max(state.wire_depths[q] for q in gate.qubits)
    return not (
        state.gate_count + 1 > state.budget.max_gates
        or layer > state.budget.max_depth
        or state.t_count + int(gate.is_non_clifford) > state.budget.max_t_count
        or state.cnot_count + int(gate.is_two_qubit) > state.budget.max_cnot_count
    )


@dataclass(slots=True)
class DeferredMixedRecord:
    record_id: int
    state: HybridState
    symbolic_distance: int
    pending_mask: int
    allocations: int = 0
    outer_features: np.ndarray | None = None
    context_cache: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DeferredMixedStep:
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
class DeferredMixedProfile:
    transitions: TransitionProfile = field(default_factory=TransitionProfile)
    archive_lookups: int = 0
    dominance_comparisons: int = 0
    certification_calls: int = 0
    outer_rows_scored: int = 0
    inner_rows_scored: int = 0
    context_rows_built: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            **self.transitions.to_dict(),
            "archive_lookups": self.archive_lookups,
            "dominance_comparisons": self.dominance_comparisons,
            "certification_calls": self.certification_calls,
            "outer_rows_scored": self.outer_rows_scored,
            "inner_rows_scored": self.inner_rows_scored,
            "context_rows_built": self.context_rows_built,
        }


class DeferredMixedSearch:
    """Fair deferred exact search over complete mixed native continuations."""

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
        if target.family != MIXED_FAMILY:
            raise ValueError("DeferredMixedSearch requires a mixed crossover target")
        if max_allocations <= 0 or max_edges <= 0 or batch_size <= 0:
            raise ValueError("search limits and batch size must be positive")
        self.target = target
        self.actions = mixed_gate_library(target.num_qubits)
        self.max_allocations = int(max_allocations)
        self.max_edges = int(max_edges)
        self.batch_size = int(batch_size)
        self.shaping_weight = float(shaping_weight)
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)
        self.fairness_start_k = int(fairness_start_k)
        self.records: dict[int, DeferredMixedRecord] = {}
        self.frontier: dict[int, DeferredMixedRecord] = {}
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
        self.profile = DeferredMixedProfile()
        self.reset()

    def _legal_mask(self, state: HybridState) -> int:
        mask = 0
        for token, gate in enumerate(self.actions):
            if _cheap_legal_continuation(state, gate):
                mask |= 1 << token
        return mask

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
        self.profile = DeferredMixedProfile()
        root_state = HybridState.identity(self.target.num_qubits, self.target.budget)
        root = self._new_record(root_state)
        self.frontier[root.record_id] = root
        self.pareto[root_state.canonical_key] = [
            (root_state.resource_vector(), root.record_id)
        ]
        self.frontier_peak = 1

    def _new_record(self, state: HybridState) -> DeferredMixedRecord:
        record = DeferredMixedRecord(
            record_id=self.next_record_id,
            state=state,
            symbolic_distance=symbolic_distance_components(state, self.target)[0],
            pending_mask=self._legal_mask(state),
        )
        self.next_record_id += 1
        self.records[record.record_id] = record
        return record

    def open_records(self) -> tuple[DeferredMixedRecord, ...]:
        return tuple(self.frontier.values())

    def pending_tokens(self, record: DeferredMixedRecord) -> tuple[int, ...]:
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
        return -float(min(record.symbolic_distance for record in self.frontier.values()))

    def _insert(self, state: HybridState) -> DeferredMixedRecord | None:
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

        record = self._new_record(state)
        survivors.append((resources, record.record_id))
        self.pareto[key] = survivors
        if record.pending_mask:
            self.frontier[record.record_id] = record
        self.accepted += 1
        return record

    def _forced_fair_edge(self) -> tuple[int, int] | None:
        next_allocation = self.allocations + 1
        root = math.isqrt(next_allocation)
        if root < self.fairness_start_k or root * root != next_allocation:
            return None
        for record_id in sorted(self.frontier):
            tokens = self.pending_tokens(self.frontier[record_id])
            if tokens:
                return record_id, tokens[0]
        return None

    def process_batch(
        self,
        record_id: int,
        tokens: Sequence[int],
        *,
        allow_fairness_override: bool = True,
    ) -> DeferredMixedStep:
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
                raise AssertionError("cheap legality mask admitted an invalid transition")
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

        return DeferredMixedStep(
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


def _identity_tableau_payload(n: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        [(1 << q, 0, 1) for q in range(n)]
        + [(0, 1 << q, 1) for q in range(n)]
    )


def mixed_outer_features(
    record: SearchRecord | DeferredMixedRecord,
    target: SynthesisTarget,
) -> np.ndarray:
    state = record.state
    total, tableau_mismatch, sequence_mismatch, multiset_mismatch = (
        symbolic_distance_components(state, target)
    )
    n = state.num_qubits
    maximum_pairs = max(1, math.comb(max(2, state.budget.max_t_count), 2))
    last_name = None if state.last_gate is None else state.last_gate.name
    target_nonidentity = sum(
        left != right
        for left, right in zip(
            target.tableau_payload,
            _identity_tableau_payload(n),
            strict=True,
        )
    )
    values = np.asarray(
        [
            1.0,
            state.t_count / max(1, state.budget.max_t_count),
            state.cnot_count / max(1, state.budget.max_cnot_count),
            state.gate_count / max(1, state.budget.max_gates),
            state.depth / max(1, state.budget.max_depth),
            sum(state.wire_depths) / max(1, n * state.budget.max_depth),
            max(state.wire_depths, default=0) / max(1, state.budget.max_depth),
            len(state.rotations) / max(1, state.budget.max_t_count),
            state.anticommuting_pairs / maximum_pairs,
            state.mean_pauli_weight / max(1, n),
            tableau_mismatch / max(1, 2 * n),
            sequence_mismatch
            / max(1, state.budget.max_t_count + len(target.rotation_payloads)),
            multiset_mismatch
            / max(1, state.budget.max_t_count + len(target.rotation_payloads)),
            total / max(1, 4 * n + 2 * state.budget.max_t_count),
            len(target.rotation_payloads) / max(1, state.budget.max_t_count),
            target_nonidentity / max(1, 2 * n),
            *(1.0 if last_name == name else 0.0 for name in GATE_FAMILIES),
            n / 6.0,
        ],
        dtype=np.float64,
    )
    if values.shape != (OUTER_FEATURE_DIM,):
        raise AssertionError("mixed outer feature schema has the wrong dimension")
    return values


@dataclass(slots=True)
class LinearMixedOuterSarsa:
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
        records: Sequence[SearchRecord | DeferredMixedRecord],
        target: SynthesisTarget,
        epsilon: float,
    ) -> tuple[int, np.ndarray, float]:
        nodes = tuple(records)
        if not nodes:
            raise RuntimeError("cannot choose from an empty frontier")
        rows: list[np.ndarray] = []
        for record in nodes:
            cached = getattr(record, "outer_features", None)
            if cached is None:
                cached = mixed_outer_features(record, target)
                if isinstance(record, DeferredMixedRecord):
                    record.outer_features = cached
                elif record.features is None:
                    record.features = cached
            rows.append(np.asarray(cached, dtype=np.float64))
        matrix = np.stack(rows)
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
        target_value = float(reward)
        if next_q_value is not None:
            target_value += (self.gamma ** max(1, int(duration))) * next_q_value
        td_error = target_value - float(q_value)
        self.theta += self.learning_rate * td_error * features
        np.clip(self.theta, -50.0, 50.0, out=self.theta)
        return float(td_error)


def _projected_tableau_mismatch(
    state: HybridState,
    gate: Gate,
    target: SynthesisTarget,
) -> int:
    """Exact forward-tableau mismatch after one Clifford, without a child state."""

    if not gate.is_clifford:
        return sum(
            left != right
            for left, right in zip(
                state.tableau.canonical_payload(), target.tableau_payload, strict=True
            )
        )
    current_images = state.tableau.forward_x + state.tableau.forward_z
    projected = (
        conjugate_by_gate(axis, gate.name, gate.qubits).axis_payload()
        for axis in current_images
    )
    return sum(
        left != right
        for left, right in zip(projected, target.tableau_payload, strict=True)
    )


def _candidate_rotation_payload(state: HybridState, gate: Gate) -> tuple[int, int, int]:
    if not gate.is_non_clifford:
        raise ValueError("rotation payload requires T or TDG")
    axis = state.tableau.inverse_z[gate.qubits[0]]
    turns = 1 if gate.name == "T" else -1
    if axis.sign < 0:
        turns = -turns
        axis = axis.positive_axis()
    return axis.x_mask, axis.z_mask, turns


def _operand_tableau_mismatch(
    state: HybridState,
    gate: Gate,
    target: SynthesisTarget,
) -> int:
    operand_mask = sum(1 << q for q in gate.qubits)
    mismatch = 0
    for current, desired in zip(
        state.tableau.canonical_payload(), target.tableau_payload, strict=True
    ):
        if current == desired:
            continue
        if ((current[0] | current[1] | desired[0] | desired[1]) & operand_mask) != 0:
            mismatch += 1
    return mismatch


def mixed_inner_context(
    record: DeferredMixedRecord,
    gate: Gate,
    target: SynthesisTarget,
) -> np.ndarray:
    """Interpretable action-conditioned context for one pending native edge."""

    state = record.state
    n = state.num_qubits
    total, tableau_before, _, _ = symbolic_distance_components(state, target)
    tableau_after = _projected_tableau_mismatch(state, gate, target)
    tableau_reduction = tableau_before - tableau_after

    current_payloads = tuple(rotation.canonical_payload() for rotation in state.rotations)
    remaining = Counter(target.rotation_payloads)
    remaining.subtract(Counter(current_payloads))
    signed_match = 0.0
    axis_match = 0.0
    if gate.is_non_clifford:
        candidate = _candidate_rotation_payload(state, gate)
        signed_match = 1.0 if remaining[candidate] > 0 else 0.0
        axis_match = 1.0 if any(
            count > 0 and token[:2] == candidate[:2]
            for token, count in remaining.items()
        ) else 0.0

    next_t = state.t_count + int(gate.is_non_clifford)
    next_cnot = state.cnot_count + int(gate.is_two_qubit)
    next_gate = state.gate_count + 1
    next_layer = 1 + max(state.wire_depths[q] for q in gate.qubits)
    last = state.last_gate
    overlap = (
        0.0
        if last is None
        else len(set(last.qubits).intersection(gate.qubits)) / max(1, len(gate.qubits))
    )
    values = np.asarray(
        [
            1.0,
            total / max(1, 4 * n + 2 * state.budget.max_t_count),
            tableau_reduction / max(1, 2 * n),
            signed_match,
            axis_match,
            max(0, len(target.rotation_payloads) - len(current_payloads))
            / max(1, state.budget.max_t_count),
            _operand_tableau_mismatch(state, gate, target) / max(1, 2 * n),
            (state.budget.max_t_count - next_t) / max(1, state.budget.max_t_count),
            (state.budget.max_cnot_count - next_cnot)
            / max(1, state.budget.max_cnot_count),
            (state.budget.max_gates - next_gate) / max(1, state.budget.max_gates),
            (state.budget.max_depth - next_layer) / max(1, state.budget.max_depth),
            overlap,
            n / 6.0,
        ],
        dtype=np.float64,
    )
    if values.shape != (INNER_CONTEXT_DIM,):
        raise AssertionError("mixed inner context schema has the wrong dimension")
    return values


@dataclass(slots=True)
class DisjointMixedLinUCB:
    """One small linear LinUCB model per native gate family."""

    alpha: float = 0.5
    regularization: float = 1.0
    a_inverse: dict[str, np.ndarray] = field(init=False, repr=False)
    b_vector: dict[str, np.ndarray] = field(init=False, repr=False)
    updates: int = field(init=False, default=0)
    rows_scored: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")
        self.a_inverse = {
            family: np.eye(INNER_CONTEXT_DIM, dtype=np.float64) / self.regularization
            for family in GATE_FAMILIES
        }
        self.b_vector = {
            family: np.zeros(INNER_CONTEXT_DIM, dtype=np.float64)
            for family in GATE_FAMILIES
        }

    def posterior_mean(self, family: str) -> np.ndarray:
        return self.a_inverse[family] @ self.b_vector[family]

    def choose(
        self,
        record: DeferredMixedRecord,
        tokens: Sequence[int],
        actions: Sequence[Gate],
        target: SynthesisTarget,
        *,
        explore: bool,
        profile: DeferredMixedProfile | None = None,
    ) -> tuple[int, np.ndarray, float]:
        if not tokens:
            raise RuntimeError("cannot choose from an empty continuation set")
        best: tuple[float, int, int, np.ndarray] | None = None
        for token_value in tokens:
            token = int(token_value)
            gate = actions[token]
            context = record.context_cache.get(token)
            if context is None:
                context = mixed_inner_context(record, gate, target)
                record.context_cache[token] = context
                if profile is not None:
                    profile.context_rows_built += 1
            mean = float(context @ self.posterior_mean(gate.name))
            bonus = 0.0
            if explore:
                inverse = self.a_inverse[gate.name]
                bonus = self.alpha * math.sqrt(
                    max(0.0, float(context @ inverse @ context))
                )
            candidate = (mean + bonus, -token, token, context)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        assert best is not None
        self.rows_scored += len(tokens)
        if profile is not None:
            profile.inner_rows_scored += len(tokens)
        return best[2], best[3].copy(), best[0]

    def rank(
        self,
        record: DeferredMixedRecord,
        tokens: Sequence[int],
        actions: Sequence[Gate],
        target: SynthesisTarget,
        *,
        limit: int,
        profile: DeferredMixedProfile | None = None,
    ) -> tuple[int, ...]:
        """Rank frozen-policy continuations with one score evaluation per edge."""

        if limit <= 0:
            return ()
        means = {family: self.posterior_mean(family) for family in GATE_FAMILIES}
        scored: list[tuple[float, int]] = []
        for token_value in tokens:
            token = int(token_value)
            gate = actions[token]
            context = record.context_cache.get(token)
            if context is None:
                context = mixed_inner_context(record, gate, target)
                record.context_cache[token] = context
                if profile is not None:
                    profile.context_rows_built += 1
            scored.append((float(context @ means[gate.name]), token))
        self.rows_scored += len(scored)
        if profile is not None:
            profile.inner_rows_scored += len(scored)
        scored.sort(key=lambda item: (-item[0], item[1]))
        return tuple(token for _, token in scored[:limit])

    def update(self, gate_name: str, context: np.ndarray, reward: float) -> None:
        vector = np.asarray(context, dtype=np.float64)
        inverse = self.a_inverse[gate_name]
        projected = inverse @ vector
        denominator = 1.0 + float(vector @ projected)
        self.a_inverse[gate_name] = inverse - np.outer(projected, projected) / denominator
        self.b_vector[gate_name] += float(reward) * vector
        self.updates += 1


def _epsilon(episode: int, episodes: int) -> float:
    fraction = episode / max(1, episodes - 1)
    return 0.30 + fraction * (0.03 - 0.30)


def train_mixed_outer_sarsa(
    targets: Sequence[SynthesisTarget],
    *,
    episodes: int = 240,
    seed: int = 11,
    max_expansions: int = 256,
) -> LinearMixedOuterSarsa:
    """Train one outer policy under the old eager expansion semantics."""

    if not targets:
        raise ValueError("outer training requires targets")
    policy = LinearMixedOuterSarsa(seed=seed, learning_rate=0.003)
    for episode in range(episodes):
        target = targets[episode % len(targets)]
        environment = EagerMixedSearch(target, max_expansions=max_expansions)
        epsilon = _epsilon(episode, episodes)
        record_id, features, q_value = policy.choose(
            environment.open_records(), target, epsilon
        )
        while True:
            transition = environment.expand(record_id)
            done = transition.terminated or transition.truncated
            if done:
                policy.update(features, q_value, transition.reward, None)
                break
            next_id, next_features, next_q = policy.choose(
                environment.open_records(), target, epsilon
            )
            policy.update(features, q_value, transition.reward, next_q)
            record_id, features = next_id, next_features
            q_value = float(features @ policy.theta)
    return policy


def train_mixed_inner_bandit(
    targets: Sequence[SynthesisTarget],
    *,
    episodes: int = 220,
    alpha: float = 0.5,
) -> DisjointMixedLinUCB:
    """Train chosen-action LinUCB under a fixed symbolic-distance outer rule."""

    if not targets:
        raise ValueError("inner training requires targets")
    bandit = DisjointMixedLinUCB(alpha=alpha)
    for episode in range(episodes):
        target = targets[episode % len(targets)]
        environment = DeferredMixedSearch(
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
            if not tokens:
                environment.frontier.pop(record.record_id, None)
                continue
            token, context, _ = bandit.choose(
                record,
                tokens,
                environment.actions,
                target,
                explore=True,
                profile=environment.profile,
            )
            before = record.symbolic_distance
            gate_name = environment.actions[token].name
            first_new_record = environment.next_record_id
            step = environment.process_batch(
                record.record_id,
                (token,),
                allow_fairness_override=False,
            )
            child_distances = [
                environment.records[record_id].symbolic_distance
                for record_id in range(first_new_record, environment.next_record_id)
            ]
            best_child = min(child_distances, default=before)
            scale = max(1, 4 * target.num_qubits + 2 * target.budget.max_t_count)
            reward = (before - best_child) / scale - 0.01
            if step.accepted == 0:
                reward -= 0.05
            if step.solution_record_id is not None:
                reward += 2.0
            bandit.update(gate_name, context, reward)
            if step.terminated or step.truncated:
                break
    return bandit


def _ordered_mixed_tokens(
    bandit: DisjointMixedLinUCB,
    environment: DeferredMixedSearch,
    record: DeferredMixedRecord,
    target: SynthesisTarget,
    batch_size: int,
) -> tuple[int, ...]:
    return bandit.rank(
        record,
        environment.pending_tokens(record),
        environment.actions,
        target,
        limit=batch_size,
        profile=environment.profile,
    )


@dataclass(frozen=True, slots=True)
class MixedEvaluationResult:
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
    context_rows_built: int
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
            "context_rows_built": self.context_rows_built,
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
    context_rows_built: int = 0,
) -> MixedEvaluationResult:
    certification = None if state is None else certify_state(target, state)
    certified = bool(certification and certification.success)
    return MixedEvaluationResult(
        method=method,
        target=target.name,
        num_qubits=target.num_qubits,
        generator_length=target.generator_length,
        success=certified,
        certified=certified,
        stop_reason="certified" if certified else stop_reason,
        wall_seconds=wall_seconds,
        cpu_seconds=cpu_seconds,
        outer_decisions=outer_decisions,
        attempted_edges=attempted_edges,
        generated=generated,
        accepted=accepted,
        rejected=rejected,
        frontier_peak=frontier_peak,
        policy_rows=policy_rows,
        context_rows_built=context_rows_built,
        maximum_matrix_error=(
            None if certification is None else certification.maximum_matrix_error
        ),
        witness=() if certification is None else certification.witness,
    )


def evaluate_mixed_eager_sarsa(
    policy: LinearMixedOuterSarsa,
    target: SynthesisTarget,
    *,
    max_expansions: int = 4_096,
    wall_limit: float = 20.0,
) -> MixedEvaluationResult:
    environment = EagerMixedSearch(target, max_expansions=max_expansions)
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
        record_id, _, _ = policy.choose(environment.open_records(), target, 0.0)
        transition = environment.expand(record_id)
        if transition.terminated:
            break
        if transition.truncated:
            stop_reason = "expansion_cap"
            break
    return _finish_result(
        method="mixed_eager_outer_sarsa",
        target=target,
        state=environment.solution_state(),
        wall_seconds=time.perf_counter() - wall_start,
        cpu_seconds=time.process_time() - cpu_start,
        stop_reason=stop_reason,
        outer_decisions=environment.expansions,
        attempted_edges=environment.profile.transitions.attempted,
        generated=environment.generated,
        accepted=environment.accepted,
        rejected=environment.rejected,
        frontier_peak=environment.frontier_peak,
        policy_rows=policy.rows_scored - rows_before,
    )


def evaluate_mixed_target_potential(
    target: SynthesisTarget,
    *,
    max_expansions: int = 4_096,
    wall_limit: float = 20.0,
) -> MixedEvaluationResult:
    environment = EagerMixedSearch(target, max_expansions=max_expansions)
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
            break
        if transition.truncated:
            stop_reason = "expansion_cap"
            break
    return _finish_result(
        method="mixed_eager_target_potential",
        target=target,
        state=environment.solution_state(),
        wall_seconds=time.perf_counter() - wall_start,
        cpu_seconds=time.process_time() - cpu_start,
        stop_reason=stop_reason,
        outer_decisions=environment.expansions,
        attempted_edges=environment.profile.transitions.attempted,
        generated=environment.generated,
        accepted=environment.accepted,
        rejected=environment.rejected,
        frontier_peak=environment.frontier_peak,
        policy_rows=rows_scored,
    )


def evaluate_mixed_deferred_native(
    outer_policy: LinearMixedOuterSarsa,
    target: SynthesisTarget,
    *,
    batch_size: int = 4,
    max_allocations: int = 4_096,
    wall_limit: float = 20.0,
) -> MixedEvaluationResult:
    """Deferred exact expansion with fixed native continuation order."""

    environment = DeferredMixedSearch(
        target,
        max_allocations=max_allocations,
        max_edges=max_allocations * len(mixed_gate_library(target.num_qubits)),
        batch_size=batch_size,
    )
    outer_before = outer_policy.rows_scored
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
        record_id, _, _ = outer_policy.choose(environment.open_records(), target, 0.0)
        record = environment.frontier[record_id]
        tokens = environment.pending_tokens(record)[:batch_size]
        if not tokens:
            environment.frontier.pop(record_id, None)
            continue
        step = environment.process_batch(record_id, tokens)
        if step.terminated:
            break
        if step.truncated:
            stop_reason = "allocation_cap"
            break
    return _finish_result(
        method="mixed_deferred_outer_sarsa_native_order",
        target=target,
        state=environment.solution_state(),
        wall_seconds=time.perf_counter() - wall_start,
        cpu_seconds=time.process_time() - cpu_start,
        stop_reason=stop_reason,
        outer_decisions=environment.allocations,
        attempted_edges=environment.edge_attempts,
        generated=environment.generated,
        accepted=environment.accepted,
        rejected=environment.rejected,
        frontier_peak=environment.frontier_peak,
        policy_rows=outer_policy.rows_scored - outer_before,
    )


def evaluate_mixed_hierarchy(
    outer_policy: LinearMixedOuterSarsa,
    bandit: DisjointMixedLinUCB,
    target: SynthesisTarget,
    *,
    batch_size: int = 4,
    max_allocations: int = 4_096,
    wall_limit: float = 20.0,
) -> MixedEvaluationResult:
    environment = DeferredMixedSearch(
        target,
        max_allocations=max_allocations,
        max_edges=max_allocations * len(mixed_gate_library(target.num_qubits)),
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
        record_id, _, _ = outer_policy.choose(environment.open_records(), target, 0.0)
        record = environment.frontier[record_id]
        tokens = _ordered_mixed_tokens(
            bandit, environment, record, target, batch_size
        )
        if not tokens:
            environment.frontier.pop(record_id, None)
            continue
        step = environment.process_batch(record_id, tokens)
        if step.terminated:
            break
        if step.truncated:
            stop_reason = "allocation_cap"
            break
    return _finish_result(
        method="mixed_deferred_outer_sarsa_inner_linucb",
        target=target,
        state=environment.solution_state(),
        wall_seconds=time.perf_counter() - wall_start,
        cpu_seconds=time.process_time() - cpu_start,
        stop_reason=stop_reason,
        outer_decisions=environment.allocations,
        attempted_edges=environment.edge_attempts,
        generated=environment.generated,
        accepted=environment.accepted,
        rejected=environment.rejected,
        frontier_peak=environment.frontier_peak,
        policy_rows=(
            outer_policy.rows_scored - outer_before
            + bandit.rows_scored - inner_before
        ),
        context_rows_built=environment.profile.context_rows_built,
    )


__all__ = [
    "DeferredMixedRecord",
    "DeferredMixedSearch",
    "DeferredMixedStep",
    "DisjointMixedLinUCB",
    "EagerMixedSearch",
    "GATE_FAMILIES",
    "INNER_CONTEXT_NAMES",
    "LinearMixedOuterSarsa",
    "MIXED_FAMILY",
    "MixedEvaluationResult",
    "evaluate_mixed_deferred_native",
    "evaluate_mixed_eager_sarsa",
    "evaluate_mixed_hierarchy",
    "evaluate_mixed_target_potential",
    "mixed_evaluation_targets",
    "mixed_gate_library",
    "mixed_inner_context",
    "mixed_outer_features",
    "mixed_training_targets",
    "train_mixed_inner_bandit",
    "train_mixed_outer_sarsa",
]
