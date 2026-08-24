"""Public structured-Toffoli frontier-search adapter."""
from __future__ import annotations

from typing import Callable

from .model import HybridState
from .search import ExpansionResult, SearchRecord
from .structured_toffoli_base import _StructuredToffoliBase
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
    _resource_feasible,
    phase_identity_holds,
    phase_identity_rows,
)


class StructuredToffoliSearch(_StructuredToffoliBase):
    def expand(self, record_id: int) -> ExpansionResult:
        if self.solution_record_id is not None:
            raise RuntimeError("the structured stress test has already terminated")
        before = self.frontier_potential()
        try:
            selected = self.frontier.pop(int(record_id))
        except KeyError as exc:
            raise KeyError(f"record {record_id} is not an active frontier action") from exc
        selected.expanded = True
        progress = self.progress_by_record[selected.record_id]
        self.expansions += 1
        generated_now = accepted_now = 0
        terminal_candidates: list[int] = []

        transitions = self._legal_transitions(selected.state, progress)
        self.normal_form_attempts += len(transitions)
        for gate, child_progress in transitions:
            child_state = selected.state.apply(
                gate,
                partial_order_reduction=False,
                profile=self.profile.transitions,
            )
            if child_state is None or not _resource_feasible(child_state, child_progress):
                continue
            generated_now += 1
            self.generated += 1
            self.generated_states += 1
            rotation_length = len(child_state.rotations)
            self.rotation_length_sum += rotation_length
            self.max_rotation_length = max(self.max_rotation_length, rotation_length)
            child = self._insert(child_state, child_progress)
            if child is None:
                continue
            accepted_now += 1
            if child_progress.stage is ToffoliStage.DONE:
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

    def solution_progress(self) -> ToffoliProgress | None:
        if self.solution_record_id is None:
            return None
        return self.progress_by_record[self.solution_record_id]

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

    def metrics(self) -> dict[str, object]:
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
            "normal_form_attempts": self.normal_form_attempts,
            "normal_form_contract": NORMAL_FORM_CONTRACT,
            "basis_graph_vertices": len(CNOT_BASIS_DISTANCE_TO_IDENTITY),
            "profile": self.profile.to_dict(),
        }



__all__ = [
    "CNOT_BASIS_DISTANCE_TO_IDENTITY",
    "CORE_CNOT_LIMIT",
    "CORE_PHASE_LIMIT",
    "IDENTITY_BASIS_ROWS",
    "NORMAL_FORM_CONTRACT",
    "PHASE_TERM_ORDER",
    "REQUIRED_PHASE_TERMS",
    "StructuredToffoliSearch",
    "ToffoliProgress",
    "ToffoliStage",
    "phase_identity_holds",
    "phase_identity_rows",
]
