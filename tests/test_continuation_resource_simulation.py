from itertools import product

import pytest

from canonical.canonicalizer import Canonicalizer
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType


def _state(num_qubits: int, budget: ResourceBudget) -> CircuitState:
    return CircuitState(CircuitDAG(num_qubits), budget)


def _append_all(state: CircuitState, gates: tuple[Gate, ...]) -> bool:
    return all(state.apply_gate(gate) for gate in gates)


def _gate_library(num_qubits: int) -> tuple[Gate, ...]:
    gates = tuple(
        Gate(gate_type, (qubit,))
        for gate_type in (
            GateType.H,
            GateType.S,
            GateType.SDG,
            GateType.T,
            GateType.TDG,
            GateType.X,
        )
        for qubit in range(num_qubits)
    )
    if num_qubits == 2:
        gates += (
            Gate(GateType.CNOT, (0, 1)),
            Gate(GateType.CNOT, (1, 0)),
        )
    return gates


def _suffixes(num_qubits: int, maximum_length: int = 2):
    library = _gate_library(num_qubits)
    for length in range(maximum_length + 1):
        yield from product(library, repeat=length)


def _weakly_dominates(left: CircuitState, right: CircuitState) -> bool:
    return all(
        left_value <= right_value
        for left_value, right_value in zip(
            left.resource_vector(),
            right.resource_vector(),
        )
    )


def _identity_pair(kind: str) -> tuple[CircuitState, CircuitState]:
    if kind == "gate-depth":
        budget = ResourceBudget(3, 3, 3, 3)
        better = _state(2, budget)
        worse = _state(2, budget)
        assert _append_all(
            worse,
            (Gate(GateType.H, (0,)), Gate(GateType.H, (0,))),
        )
    elif kind == "t-count":
        budget = ResourceBudget(9, 10, 10, 3)
        better = _state(1, budget)
        worse = _state(1, budget)
        assert _append_all(
            worse,
            tuple(Gate(GateType.T, (0,)) for _ in range(8)),
        )
    elif kind == "two-qubit":
        budget = ResourceBudget(3, 4, 4, 3)
        better = _state(2, budget)
        worse = _state(2, budget)
        assert _append_all(
            worse,
            (
                Gate(GateType.CNOT, (0, 1)),
                Gate(GateType.CNOT, (0, 1)),
            ),
        )
    else:  # pragma: no cover - parametrization is closed
        raise AssertionError(kind)
    return better, worse


@pytest.mark.parametrize("kind", ("gate-depth", "t-count", "two-qubit"))
def test_exhaustive_small_suffixes_obey_one_sided_resource_simulation(
    kind: str,
) -> None:
    """Every suffix feasible from a dominated witness is feasible from its dominator.

    This is deliberately one-sided.  Equal semantic keys need not have equal
    remaining-budget languages because consumed resources live in the Pareto
    record.  Extension-monotone limits make the cheaper language a superset.
    """
    better, worse = _identity_pair(kind)
    canonicalizer = Canonicalizer()

    assert canonicalizer.semantic_key(better) == canonicalizer.semantic_key(worse)
    assert _weakly_dominates(better, worse)
    assert better.resource_vector() != worse.resource_vector()

    strict_language_extension_seen = False
    for suffix in _suffixes(better.dag.num_qubits):
        better_after = better.copy()
        worse_after = worse.copy()
        better_feasible = _append_all(better_after, suffix)
        worse_feasible = _append_all(worse_after, suffix)

        if worse_feasible:
            assert better_feasible
            assert _weakly_dominates(better_after, worse_after)
            # Equal semantics plus the exact same suffix is a right congruence.
            assert canonicalizer.semantic_key(
                better_after
            ) == canonicalizer.semantic_key(worse_after)
        elif better_feasible:
            strict_language_extension_seen = True

    assert strict_language_extension_seen


def test_per_wire_depth_profiles_can_be_incomparable_at_one_semantic_key() -> None:
    budget = ResourceBudget(0, 3, 4, 0)
    left = _state(2, budget)
    right = _state(2, budget)
    assert _append_all(left, (Gate(GateType.H, (0,)), Gate(GateType.H, (0,))))
    assert _append_all(right, (Gate(GateType.H, (1,)), Gate(GateType.H, (1,))))

    assert Canonicalizer().semantic_key(left) == Canonicalizer().semantic_key(right)
    assert left.wire_depths == (2, 0)
    assert right.wire_depths == (0, 2)
    assert not _weakly_dominates(left, right)
    assert not _weakly_dominates(right, left)
