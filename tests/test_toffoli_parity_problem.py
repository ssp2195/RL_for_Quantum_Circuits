"""Transition, continuation, and safe-pruning tests for Stage 3 Toffoli."""

from __future__ import annotations

from collections import deque
from functools import lru_cache

import pytest

from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from config import Config
from enums import GateType
from search.action import Action
from search.node import SearchNode
from search.problems.toffoli_parity import (
    CNOT_BASIS_DISTANCE_TO_IDENTITY,
    IDENTITY_BASIS_ROWS,
    InvalidToffoliParityPrefix,
    PHASE_TERM_ORDER,
    ToffoliParityNetworkProblem,
    ToffoliStage,
    _core_can_reach_terminal,
    analyze_toffoli_prefix,
)


_BUDGET = ResourceBudget(
    max_t_count=7,
    max_two_qubit_count=6,
    max_gates=15,
    max_depth=12,
)


def _state(*operations: tuple[GateType, tuple[int, ...]]) -> CircuitState:
    state = CircuitState(CircuitDAG(3), _BUDGET)
    for gate_type, qubits in operations:
        assert state.apply_gate(Gate(gate_type, qubits))
    return state


def _node(state: CircuitState) -> SearchNode:
    return SearchNode(priority=0.0, state=state)


def _known_normal_form_prefix(*, final_h: bool = False) -> CircuitState:
    """A test-only valid parity-network prefix, never used by expansion code."""

    operations = (
        (GateType.H, (2,)),
        (GateType.T, (0,)),
        (GateType.T, (1,)),
        (GateType.T, (2,)),
        (GateType.CNOT, (0, 1)),
        (GateType.TDG, (1,)),
        (GateType.CNOT, (0, 1)),
        (GateType.CNOT, (0, 2)),
        (GateType.TDG, (2,)),
        (GateType.CNOT, (1, 2)),
        (GateType.T, (2,)),
        (GateType.CNOT, (0, 2)),
        (GateType.TDG, (2,)),
        (GateType.CNOT, (1, 2)),
    )
    if final_h:
        operations += ((GateType.H, (2,)),)
    return _state(*operations)


def test_root_has_only_the_outer_h_transition_and_parent_is_unchanged():
    problem = ToffoliParityNetworkProblem()
    root_state = problem.initial_state(Config(3, _BUDGET))
    root = _node(root_state)

    children = problem.expand(root)

    assert root_state.dag.gates == []
    assert len(children) == 1
    assert children[0].action.gate_type is GateType.H
    assert children[0].action.qubits == (2,)
    assert problem.analyze(children[0].state).stage is ToffoliStage.CORE


def test_phase_and_cnot_transitions_are_derived_from_current_parity_rows():
    problem = ToffoliParityNetworkProblem()
    after_h = _state((GateType.H, (2,)))
    children = problem.expand(_node(after_h))
    actions = {(child.action.gate_type, child.action.qubits) for child in children}

    assert {(GateType.T, (0,)), (GateType.T, (1,)), (GateType.T, (2,))} <= actions
    assert not any(gate_type is GateType.TDG for gate_type, _ in actions)
    assert {(GateType.CNOT, (0, 1)), (GateType.CNOT, (1, 0))} <= actions

    cnot_state = _state((GateType.H, (2,)), (GateType.CNOT, (0, 1)))
    progress = analyze_toffoli_prefix(cnot_state)
    assert progress.basis_rows == (0b001, 0b011, 0b100)
    assert any(
        child.action == Action(GateType.TDG, (1,))
        for child in problem.expand(_node(cnot_state))
    )


def test_analyzer_rejects_wrong_sign_and_duplicate_phase_terms():
    with pytest.raises(InvalidToffoliParityPrefix, match="wrong required sign"):
        analyze_toffoli_prefix(_state((GateType.H, (2,)), (GateType.TDG, (0,))))
    with pytest.raises(InvalidToffoliParityPrefix, match="emitted twice"):
        analyze_toffoli_prefix(
            _state((GateType.H, (2,)), (GateType.T, (0,)), (GateType.T, (0,)))
        )
    with pytest.raises(InvalidToffoliParityPrefix, match="permits only"):
        analyze_toffoli_prefix(_state((GateType.H, (2,)), (GateType.S, (0,))))


def test_core_completion_requires_identity_basis_and_final_h_is_then_unique():
    problem = ToffoliParityNetworkProblem()
    state = _known_normal_form_prefix()
    progress = problem.analyze(state)

    assert progress.stage is ToffoliStage.POST_H
    assert progress.basis_rows == IDENTITY_BASIS_ROWS
    final_children = problem.expand(_node(state))
    assert [(child.action.gate_type, child.action.qubits) for child in final_children] == [
        (GateType.H, (2,))
    ]
    assert problem.is_terminal_candidate(final_children[0])

    incomplete = _state((GateType.H, (2,)), (GateType.CNOT, (0, 1)))
    assert all(child.action.gate_type is not GateType.H for child in problem.expand(_node(incomplete)))


def test_canonical_key_carries_full_normal_form_continuation_information():
    problem = ToffoliParityNetworkProblem()
    canonicalizer = problem.canonicalizer()
    left = _state((GateType.H, (2,)), (GateType.T, (0,)))
    right = _state((GateType.H, (2,)), (GateType.T, (1,)))

    left_key = canonicalizer.semantic_key(left)
    right_key = canonicalizer.semantic_key(right)
    assert left_key != right_key
    assert left_key[0] == "toffoli-parity-network-v1"
    payload = left_key[2]
    assert payload[0] == "CORE"
    assert payload[1] == IDENTITY_BASIS_ROWS
    assert payload[2] != 0


def test_exact_basis_graph_and_resource_lower_bounds_prune_only_impossible_budgets():
    assert len(CNOT_BASIS_DISTANCE_TO_IDENTITY) == 168
    assert CNOT_BASIS_DISTANCE_TO_IDENTITY[IDENTITY_BASIS_ROWS] == 0

    problem = ToffoliParityNetworkProblem()
    for budget in (
        ResourceBudget(6, 12, 15, 6),
        ResourceBudget(7, 12, 15, 5),
        ResourceBudget(7, 12, 14, 6),
    ):
        root = _node(problem.initial_state(Config(3, budget)))
        assert problem.expand(root) == []

    viable_root = _node(problem.initial_state(Config(3, _BUDGET)))
    assert len(problem.expand(viable_root)) == 1


def test_compact_reachability_prune_matches_an_independent_unpruned_graph():
    """Exhaustively prove the optional compact prune keeps every terminal.

    This reference graph intentionally does *not* use the production CNOT
    distance table or its recursive pruning routine.  It simply enumerates
    every phase emission exposed by a basis and every directed CNOT through
    the seven-phase/six-CNOT finite normal form, then computes whether a
    terminal suffix exists.  The production reachability prune must agree for
    every compact state reachable from the identity core.
    """

    term_bit = {mask: index for index, mask in enumerate(PHASE_TERM_ORDER)}
    full_emitted = (1 << len(PHASE_TERM_ORDER)) - 1

    def apply_cnot(
        rows: tuple[int, int, int], control: int, target: int
    ) -> tuple[int, int, int]:
        updated = list(rows)
        updated[target] ^= updated[control]
        return tuple(updated)  # type: ignore[return-value]

    def successors(
        rows: tuple[int, int, int], emitted: int, cnot_count: int
    ) -> tuple[tuple[tuple[int, int, int], int, int], ...]:
        result: list[tuple[tuple[int, int, int], int, int]] = []
        for mask in rows:
            bit_index = term_bit.get(mask)
            if bit_index is not None and not emitted & (1 << bit_index):
                result.append((rows, emitted | (1 << bit_index), cnot_count))
        if cnot_count < 6:
            for control in range(3):
                for target in range(3):
                    if control != target:
                        result.append(
                            (apply_cnot(rows, control, target), emitted, cnot_count + 1)
                        )
        return tuple(result)

    root = (IDENTITY_BASIS_ROWS, 0, 0)
    reachable = {root}
    queue = deque((root,))
    while queue:
        state = queue.popleft()
        for child in successors(*state):
            if child not in reachable:
                reachable.add(child)
                queue.append(child)

    @lru_cache(maxsize=None)
    def unpruned_can_reach_terminal(
        rows: tuple[int, int, int], emitted: int, cnot_count: int
    ) -> bool:
        if rows == IDENTITY_BASIS_ROWS and emitted == full_emitted:
            return True
        return any(
            unpruned_can_reach_terminal(*child)
            for child in successors(rows, emitted, cnot_count)
        )

    # 61,120 includes all normal-form compact states reachable under the raw
    # seven-phase/six-CNOT grammar; this guards the intended exhaustive scope.
    assert len(reachable) == 61_120
    assert all(
        _core_can_reach_terminal(rows, emitted, cnot_count)
        == unpruned_can_reach_terminal(rows, emitted, cnot_count)
        for rows, emitted, cnot_count in reachable
    )
