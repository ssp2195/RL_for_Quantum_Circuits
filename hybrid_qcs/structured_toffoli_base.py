"""Frontier search adapter for the certified Toffoli parity normal form."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import time
from typing import Callable

from .benchmarks import SynthesisTarget, structured_toffoli_target
from .model import Gate, HybridState, TransitionProfile
from .search import ExpansionResult, SearchProfile, SearchRecord, symbolic_distance_components
from .toffoli_normal_form import (
    CNOT_BASIS_DISTANCE_TO_IDENTITY,
    CORE_CNOT_LIMIT,
    CORE_PHASE_LIMIT,
    IDENTITY_BASIS_ROWS,
    NORMAL_FORM_CONTRACT,
    PHASE_TERM_ORDER,
    REQUIRED_PHASE_TERMS,
    ToffoliProgress,
    ToffoliStage,
    _FULL_EMITTED_TERMS,
    _TERM_BIT,
    _apply_cnot_to_rows,
    _core_can_reach_terminal,
    _remaining_outer_h,
    _resource_feasible,
    _strictly_dominates,
    _weakly_dominates,
    phase_identity_holds,
    phase_identity_rows,
)

@dataclass
class _StructuredToffoliBase:
    """Frontier-record search inside the certified Toffoli normal form."""

    target: SynthesisTarget = field(default_factory=structured_toffoli_target)
    max_expansions: int = 8_192
    shaping_weight: float = 0.5
    success_bonus: float = 20.0
    failure_penalty: float = 20.0

    def __post_init__(self) -> None:
        if self.target.family != "structured-toffoli-parity-network":
            raise ValueError("StructuredToffoliSearch requires the structured target")
        if self.max_expansions <= 0:
            raise ValueError("max_expansions must be positive")
        self.records: dict[int, SearchRecord] = {}
        self.frontier: dict[int, SearchRecord] = {}
        self.progress_by_record: dict[int, ToffoliProgress] = {}
        self.pareto: dict[
            tuple[object, ...], list[tuple[tuple[int, ...], int]]
        ] = {}
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
        self.normal_form_attempts = 0
        self.reset()

    def reset(self) -> None:
        self.records.clear()
        self.frontier.clear()
        self.progress_by_record.clear()
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
        self.normal_form_attempts = 0
        root_state = HybridState.identity(3, self.target.budget)
        progress = ToffoliProgress(ToffoliStage.PRE_H, IDENTITY_BASIS_ROWS, 0)
        root = self._new_record(root_state, progress)
        self.frontier[root.record_id] = root
        self.pareto[self._archive_key(root_state, progress)] = [
            (root_state.resource_vector(), root.record_id)
        ]
        self.frontier_peak = 1

    def _new_record(
        self, state: HybridState, progress: ToffoliProgress
    ) -> SearchRecord:
        generic_distance = symbolic_distance_components(state, self.target)[0]
        normal_form_distance = (
            CORE_PHASE_LIMIT
            - progress.emitted_count
            + CNOT_BASIS_DISTANCE_TO_IDENTITY[progress.basis_rows]
            + _remaining_outer_h(progress.stage)
        )
        record = SearchRecord(
            self.next_record_id,
            state,
            generic_distance + normal_form_distance,
        )
        self.next_record_id += 1
        self.records[record.record_id] = record
        self.progress_by_record[record.record_id] = progress
        return record

    @staticmethod
    def _archive_key(
        state: HybridState, progress: ToffoliProgress
    ) -> tuple[object, ...]:
        return (
            "structured-toffoli-hybrid-v1",
            state.canonical_key,
            progress.canonical_payload(),
        )

    def open_records(self) -> tuple[SearchRecord, ...]:
        started = time.perf_counter_ns()
        records = tuple(self.frontier.values())
        self.profile.frontier_snapshot_ns += time.perf_counter_ns() - started
        return records

    def frontier_potential(self) -> float:
        if self.solution_record_id is not None:
            return 0.0
        if not self.frontier:
            return -float(4 * self.target.budget.max_gates + 1)
        return -float(min(record.symbolic_distance for record in self.frontier.values()))

    def _insert(
        self, state: HybridState, progress: ToffoliProgress
    ) -> SearchRecord | None:
        started = time.perf_counter_ns()
        key = self._archive_key(state, progress)
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
        record = self._new_record(state, progress)
        survivors.append((resources, record.record_id))
        self.pareto[key] = survivors
        self.frontier[record.record_id] = record
        self.accepted += 1
        self.profile.archive_ns += time.perf_counter_ns() - started
        return record

    def _legal_transitions(
        self, state: HybridState, progress: ToffoliProgress
    ) -> tuple[tuple[Gate, ToffoliProgress], ...]:
        if not _resource_feasible(state, progress):
            return ()
        if progress.stage is ToffoliStage.PRE_H:
            return ((Gate("H", (2,)), ToffoliProgress(ToffoliStage.CORE, IDENTITY_BASIS_ROWS, 0)),)
        if progress.stage is ToffoliStage.POST_H:
            return ((Gate("H", (2,)), ToffoliProgress(ToffoliStage.DONE, IDENTITY_BASIS_ROWS, _FULL_EMITTED_TERMS)),)
        if progress.stage is ToffoliStage.DONE:
            return ()

        transitions: list[tuple[Gate, ToffoliProgress]] = []
        for qubit, mask in enumerate(progress.basis_rows):
            sign = REQUIRED_PHASE_TERMS.get(mask)
            term_index = _TERM_BIT.get(mask)
            if sign is None or term_index is None:
                continue
            bit = 1 << term_index
            if progress.emitted_terms & bit:
                continue
            emitted = progress.emitted_terms | bit
            stage = (
                ToffoliStage.POST_H
                if emitted == _FULL_EMITTED_TERMS
                and progress.basis_rows == IDENTITY_BASIS_ROWS
                else ToffoliStage.CORE
            )
            transitions.append(
                (
                    Gate("T" if sign > 0 else "TDG", (qubit,)),
                    ToffoliProgress(stage, progress.basis_rows, emitted),
                )
            )

        if state.cnot_count < CORE_CNOT_LIMIT:
            for control in range(3):
                for target in range(3):
                    if control == target:
                        continue
                    rows = _apply_cnot_to_rows(
                        progress.basis_rows, control, target
                    )
                    next_count = state.cnot_count + 1
                    if (
                        CNOT_BASIS_DISTANCE_TO_IDENTITY[rows]
                        > CORE_CNOT_LIMIT - next_count
                    ):
                        continue
                    if not _core_can_reach_terminal(
                        rows, progress.emitted_terms, next_count
                    ):
                        continue
                    transitions.append(
                        (
                            Gate("CNOT", (control, target)),
                            ToffoliProgress(
                                ToffoliStage.CORE,
                                rows,
                                progress.emitted_terms,
                            ),
                        )
                    )
        return tuple(transitions)



__all__ = ["_StructuredToffoliBase"]
