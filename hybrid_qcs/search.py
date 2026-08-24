"""Deterministic exact frontier search over hybrid symbolic states."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import time
from typing import Callable, TYPE_CHECKING

from .model import Gate, HybridState, TransitionProfile, generate_gates

if TYPE_CHECKING:
    from .benchmarks import SynthesisTarget


@dataclass(slots=True)
class SearchRecord:
    record_id: int
    state: HybridState
    symbolic_distance: int
    expanded: bool = False
    features: object | None = None


@dataclass(frozen=True, slots=True)
class ExpansionResult:
    selected_record_id: int
    reward: float
    terminated: bool
    truncated: bool
    generated: int
    accepted: int
    rejected: int
    frontier_size: int
    solution_record_id: int | None


@dataclass(slots=True)
class SearchProfile:
    transitions: TransitionProfile = field(default_factory=TransitionProfile)
    archive_ns: int = 0
    frontier_snapshot_ns: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            **self.transitions.to_dict(),
            "archive_ns": self.archive_ns,
            "frontier_snapshot_ns": self.frontier_snapshot_ns,
        }


def symbolic_distance_components(
    state: HybridState, target: "SynthesisTarget"
) -> tuple[int, int, int, int]:
    current_tableau = state.tableau.canonical_payload()
    tableau_mismatch = sum(
        left != right
        for left, right in zip(current_tableau, target.tableau_payload, strict=True)
    )
    current_rotations = tuple(
        rotation.canonical_payload() for rotation in state.rotations
    )
    sequence_mismatch = abs(len(current_rotations) - len(target.rotation_payloads))
    sequence_mismatch += sum(
        left != right
        for left, right in zip(current_rotations, target.rotation_payloads)
    )
    current_counts = Counter(current_rotations)
    target_counts = Counter(target.rotation_payloads)
    multiset_mismatch = sum((current_counts - target_counts).values()) + sum(
        (target_counts - current_counts).values()
    )
    total = 2 * tableau_mismatch + sequence_mismatch + multiset_mismatch
    return total, tableau_mismatch, sequence_mismatch, multiset_mismatch


class HybridSearch:
    """Environment whose action selects one complete frontier record.

    Every selected record is expanded through all target-independent legal
    gates.  The policy never chooses a gate, canonicalization rule, or pruning
    decision.
    """

    def __init__(
        self,
        target: "SynthesisTarget",
        *,
        max_expansions: int = 512,
        shaping_weight: float = 0.5,
        success_bonus: float = 20.0,
        failure_penalty: float = 20.0,
        partial_order_reduction: bool = True,
    ) -> None:
        if max_expansions <= 0:
            raise ValueError("max_expansions must be positive")
        self.target = target
        self.max_expansions = int(max_expansions)
        self.shaping_weight = float(shaping_weight)
        self.success_bonus = float(success_bonus)
        self.failure_penalty = float(failure_penalty)
        self.partial_order_reduction = bool(partial_order_reduction)
        self.actions: tuple[Gate, ...] = generate_gates(target.num_qubits)
        self.records: dict[int, SearchRecord] = {}
        self.frontier: dict[int, SearchRecord] = {}
        self.pareto: dict[tuple[object, ...], list[tuple[tuple[int, ...], int]]] = {}
        self.next_record_id = 0
        self.expansions = 0
        self.generated = 0
        self.accepted = 0
        self.rejected = 0
        self.solution_record_id: int | None = None
        self.frontier_peak = 0
        self.profile = SearchProfile()
        self.max_rotation_length = 0
        self.rotation_length_sum = 0
        self.generated_states = 0
        self.reset()

    def reset(self) -> None:
        self.records.clear()
        self.frontier.clear()
        self.pareto.clear()
        self.next_record_id = 0
        self.expansions = 0
        self.generated = 0
        self.accepted = 0
        self.rejected = 0
        self.solution_record_id = None
        self.frontier_peak = 0
        self.profile = SearchProfile()
        self.max_rotation_length = 0
        self.rotation_length_sum = 0
        self.generated_states = 0
        root_state = HybridState.identity(self.target.num_qubits, self.target.budget)
        root = self._new_record(root_state)
        self.frontier[root.record_id] = root
        self.pareto[root_state.canonical_key] = [
            (root_state.resource_vector(), root.record_id)
        ]
        self.frontier_peak = 1
        if root_state.canonical_key == self.target.canonical_key:
            self.solution_record_id = root.record_id

    def _new_record(self, state: HybridState) -> SearchRecord:
        distance = symbolic_distance_components(state, self.target)[0]
        record = SearchRecord(self.next_record_id, state, distance)
        self.next_record_id += 1
        self.records[record.record_id] = record
        return record

    def open_records(self) -> tuple[SearchRecord, ...]:
        started = time.perf_counter_ns()
        # Records enter this dictionary in monotonically increasing ID order.
        # Removing dominated/expanded entries does not disturb the relative
        # order of survivors, so sorting the complete frontier at every RL
        # decision was redundant.
        records = tuple(self.frontier.values())
        self.profile.frontier_snapshot_ns += time.perf_counter_ns() - started
        return records

    def frontier_potential(self) -> float:
        if self.solution_record_id is not None:
            return 0.0
        if not self.frontier:
            return -float(4 * self.target.budget.max_gates + 1)
        return -float(min(record.symbolic_distance for record in self.frontier.values()))

    def _insert(self, state: HybridState) -> SearchRecord | None:
        started = time.perf_counter_ns()
        key = state.canonical_key
        resources = state.resource_vector()
        group = self.pareto.setdefault(key, [])
        if any(_weakly_dominates(existing, resources) for existing, _ in group):
            self.rejected += 1
            self.profile.archive_ns += time.perf_counter_ns() - started
            return None

        survivors: list[tuple[tuple[int, ...], int]] = []
        for existing, record_id in group:
            if _strictly_dominates(resources, existing):
                self.frontier.pop(record_id, None)
            else:
                survivors.append((existing, record_id))

        record = self._new_record(state)
        survivors.append((resources, record.record_id))
        self.pareto[key] = survivors
        self.frontier[record.record_id] = record
        self.accepted += 1
        self.profile.archive_ns += time.perf_counter_ns() - started
        return record

    def expand(self, record_id: int) -> ExpansionResult:
        if self.solution_record_id is not None:
            raise RuntimeError("the search episode has already terminated")
        before = self.frontier_potential()
        try:
            selected = self.frontier.pop(int(record_id))
        except KeyError as exc:
            raise KeyError(f"record {record_id} is not an active frontier action") from exc
        selected.expanded = True
        self.expansions += 1
        generated_now = accepted_now = 0
        terminal_candidates: list[int] = []

        for gate in self.actions:
            child_state = selected.state.apply(
                gate,
                partial_order_reduction=self.partial_order_reduction,
                profile=self.profile.transitions,
            )
            if child_state is None:
                continue
            generated_now += 1
            self.generated += 1
            self.generated_states += 1
            rotation_length = len(child_state.rotations)
            self.rotation_length_sum = rotation_length
            self.max_rotation_length = max(self.max_rotation_length, rotation_length)
            child = self._insert(child_state)
            if child is None:
                continue
            accepted_now += 1
            if child.state.canonical_key == self.target.canonical_key:
                terminal_candidates.append(child.record_id)

        if terminal_candidates:
            self.solution_record_id = min(terminal_candidates)
        self.frontier_peak = max(self.frontier_peak, len(self.frontier))
        exhausted = not self.frontier and self.solution_record_id is None
        truncated = self.expansions >= self.max_expansions and self.solution_record_id is None
        terminated = self.solution_record_id is not None or exhausted
        after = self.frontier_potential()
        reward = -1.0 + self.shaping_weight * (after - before)
        if self.solution_record_id is not None:
            reward += self.success_bonus
        elif exhausted or truncated:
            reward -= self.failure_penalty

        return ExpansionResult(
            selected_record_id=selected.record_id,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
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

    def run_scheduler(
        self,
        select: Callable[[tuple[SearchRecord, ...]], int],
        *,
        should_stop: Callable[[], bool] | None = None,
    ) -> HybridState | None:
        self.reset()
        while self.frontier and self.solution_record_id is None:
            if self.expansions >= self.max_expansions:
                break
            if should_stop is not None and should_stop():
                break
            record_id = int(select(self.open_records()))
            result = self.expand(record_id)
            if result.terminated or result.truncated:
                break
        return self.solution_state()

    def metrics(self) -> dict[str, int | float | bool | dict[str, int]]:
        mean_rotation_length = (
            self.rotation_length_sum / self.generated_states
            if self.generated_states
            else 0.0
        )
        return {
            "success": self.solution_record_id is not None,
            "expansions": self.expansions,
            "generated": self.generated,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "frontier_peak": self.frontier_peak,
            "records": len(self.records),
            "max_rotation_length": self.max_rotation_length,
            "mean_rotation_length": mean_rotation_length,
            "profile": self.profile.to_dict(),
        }


def _weakly_dominates(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return all(a <= b for a, b in zip(left, right, strict=True))


def _strictly_dominates(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return _weakly_dominates(left, right) and left != right

__all__ = [
    "ExpansionResult",
    "HybridSearch",
    "SearchProfile",
    "SearchRecord",
    "symbolic_distance_components",
]
