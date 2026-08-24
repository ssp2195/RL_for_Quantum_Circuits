"""Certified three-qubit Toffoli stress test in a CCZ parity normal form.

The learned action remains selection of one persistent frontier record.  The
normal-form engine, not the policy, enumerates all legal parity-phase and CNOT
continuations.  Every record still carries the full persistent DAG, Clifford
tableau, and ordered Pauli-rotation state used by the unrestricted synthesizer.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
import time
from types import MappingProxyType
from typing import Callable, Mapping

from .benchmarks import SynthesisTarget, structured_toffoli_target
from .model import Gate, HybridState, TransitionProfile
from .search import ExpansionResult, SearchProfile, SearchRecord, symbolic_distance_components


REQUIRED_PHASE_TERMS: Mapping[int, int] = MappingProxyType(
    {
        0b001: +1,
        0b010: +1,
        0b100: +1,
        0b011: -1,
        0b101: -1,
        0b110: -1,
        0b111: +1,
    }
)
PHASE_TERM_ORDER = tuple(sorted(REQUIRED_PHASE_TERMS))
_TERM_BIT = {mask: index for index, mask in enumerate(PHASE_TERM_ORDER)}
_FULL_EMITTED_TERMS = (1 << len(PHASE_TERM_ORDER)) - 1
IDENTITY_BASIS_ROWS = (0b001, 0b010, 0b100)
CORE_CNOT_LIMIT = 6
CORE_PHASE_LIMIT = 7
_UNREACHABLE_CNOT_COST = 1_000_000
NORMAL_FORM_CONTRACT = (
    "toffoli-ccz-seven-term-parity-network",
    "outer-H-q2",
    "phase-terms-once",
    "return-linear-basis-to-identity",
    "all-to-all-directed-cnot",
)


class ToffoliStage(str, Enum):
    PRE_H = "PRE_H"
    CORE = "CORE"
    POST_H = "POST_H"
    DONE = "DONE"


@dataclass(frozen=True, slots=True)
class ToffoliProgress:
    stage: ToffoliStage
    basis_rows: tuple[int, int, int]
    emitted_terms: int

    @property
    def emitted_count(self) -> int:
        return int(self.emitted_terms).bit_count()

    @property
    def complete_core(self) -> bool:
        return (
            self.emitted_terms == _FULL_EMITTED_TERMS
            and self.basis_rows == IDENTITY_BASIS_ROWS
        )

    def canonical_payload(self) -> tuple[object, ...]:
        return (
            self.stage.value,
            self.basis_rows,
            int(self.emitted_terms),
            NORMAL_FORM_CONTRACT,
        )


def _parity(mask: int, assignment: int) -> int:
    return (int(mask) & int(assignment)).bit_count() & 1


def phase_identity_rows() -> tuple[dict[str, int], ...]:
    """Exhaustively verify the seven-term phase identity for CCZ."""

    rows: list[dict[str, int]] = []
    for assignment in range(8):
        x0 = int(bool(assignment & 0b001))
        x1 = int(bool(assignment & 0b010))
        x2 = int(bool(assignment & 0b100))
        lhs = 4 * x0 * x1 * x2
        rhs = sum(
            coefficient * _parity(mask, assignment)
            for mask, coefficient in REQUIRED_PHASE_TERMS.items()
        )
        rows.append(
            {
                "assignment": assignment,
                "lhs_mod_8": lhs % 8,
                "rhs_mod_8": rhs % 8,
                "matches": int(lhs % 8 == rhs % 8),
            }
        )
    return tuple(rows)


def phase_identity_holds() -> bool:
    return all(bool(row["matches"]) for row in phase_identity_rows())


def _apply_cnot_to_rows(
    rows: tuple[int, int, int], control: int, target: int
) -> tuple[int, int, int]:
    if control == target or control not in range(3) or target not in range(3):
        raise ValueError("CNOT operands must be distinct q0..q2 wires")
    result = list(rows)
    result[target] ^= result[control]
    return tuple(result)  # type: ignore[return-value]


def _cnot_basis_distances() -> Mapping[tuple[int, int, int], int]:
    distances: dict[tuple[int, int, int], int] = {IDENTITY_BASIS_ROWS: 0}
    queue: deque[tuple[int, int, int]] = deque((IDENTITY_BASIS_ROWS,))
    while queue:
        rows = queue.popleft()
        distance = distances[rows] + 1
        for control in range(3):
            for target in range(3):
                if control == target:
                    continue
                child = _apply_cnot_to_rows(rows, control, target)
                if child not in distances:
                    distances[child] = distance
                    queue.append(child)
    if len(distances) != 168:
        raise AssertionError("the GL(3,F2) CNOT basis graph must have 168 states")
    return MappingProxyType(distances)


CNOT_BASIS_DISTANCE_TO_IDENTITY = _cnot_basis_distances()


@lru_cache(maxsize=262_144)
def _core_can_reach_terminal(
    basis_rows: tuple[int, int, int], emitted_terms: int, cnot_count: int
) -> bool:
    """Exact reachability in the finite declared normal-form graph."""

    if emitted_terms == _FULL_EMITTED_TERMS and basis_rows == IDENTITY_BASIS_ROWS:
        return True
    if cnot_count > CORE_CNOT_LIMIT:
        return False

    for mask in basis_rows:
        term_index = _TERM_BIT.get(mask)
        if term_index is None:
            continue
        bit = 1 << term_index
        if emitted_terms & bit:
            continue
        if _core_can_reach_terminal(basis_rows, emitted_terms | bit, cnot_count):
            return True

    if cnot_count >= CORE_CNOT_LIMIT:
        return False
    for control in range(3):
        for target in range(3):
            if control == target:
                continue
            child_rows = _apply_cnot_to_rows(basis_rows, control, target)
            next_count = cnot_count + 1
            if CNOT_BASIS_DISTANCE_TO_IDENTITY[child_rows] > CORE_CNOT_LIMIT - next_count:
                continue
            if _core_can_reach_terminal(child_rows, emitted_terms, next_count):
                return True
    return False


@lru_cache(maxsize=262_144)
def _minimum_additional_core_cnot(
    basis_rows: tuple[int, int, int], emitted_terms: int, cnot_count: int
) -> int:
    if emitted_terms == _FULL_EMITTED_TERMS and basis_rows == IDENTITY_BASIS_ROWS:
        return 0
    if cnot_count > CORE_CNOT_LIMIT:
        return _UNREACHABLE_CNOT_COST

    best = _UNREACHABLE_CNOT_COST
    for mask in basis_rows:
        term_index = _TERM_BIT.get(mask)
        if term_index is None:
            continue
        bit = 1 << term_index
        if emitted_terms & bit:
            continue
        best = min(
            best,
            _minimum_additional_core_cnot(
                basis_rows, emitted_terms | bit, cnot_count
            ),
        )

    if cnot_count >= CORE_CNOT_LIMIT:
        return best
    for control in range(3):
        for target in range(3):
            if control == target:
                continue
            child_rows = _apply_cnot_to_rows(basis_rows, control, target)
            next_count = cnot_count + 1
            if CNOT_BASIS_DISTANCE_TO_IDENTITY[child_rows] > CORE_CNOT_LIMIT - next_count:
                continue
            suffix = _minimum_additional_core_cnot(
                child_rows, emitted_terms, next_count
            )
            if suffix < _UNREACHABLE_CNOT_COST:
                best = min(best, 1 + suffix)
    return best


def _remaining_outer_h(stage: ToffoliStage) -> int:
    if stage is ToffoliStage.PRE_H:
        return 2
    if stage in {ToffoliStage.CORE, ToffoliStage.POST_H}:
        return 1
    return 0


def _resource_feasible(state: HybridState, progress: ToffoliProgress) -> bool:
    remaining_phases = CORE_PHASE_LIMIT - progress.emitted_count
    if state.t_count + remaining_phases > state.budget.max_t_count:
        return False

    if progress.stage is ToffoliStage.PRE_H:
        minimum_cnot = _minimum_additional_core_cnot(IDENTITY_BASIS_ROWS, 0, 0)
    elif progress.stage is ToffoliStage.DONE:
        minimum_cnot = 0
    else:
        minimum_cnot = _minimum_additional_core_cnot(
            progress.basis_rows, progress.emitted_terms, state.cnot_count
        )
    if minimum_cnot >= _UNREACHABLE_CNOT_COST:
        return False
    if state.cnot_count + minimum_cnot > state.budget.max_cnot_count:
        return False

    minimum_future_gates = (
        remaining_phases + minimum_cnot + _remaining_outer_h(progress.stage)
    )
    return state.gate_count + minimum_future_gates <= state.budget.max_gates


def _weakly_dominates(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return all(a <= b for a, b in zip(left, right, strict=True))


def _strictly_dominates(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return left != right and _weakly_dominates(left, right)


