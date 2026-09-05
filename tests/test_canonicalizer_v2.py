from __future__ import annotations

import itertools

import numpy as np

from hybrid_qcs.canonicalize import canonicalize_projective, legacy_projective_key
from hybrid_qcs.certify import equal_up_to_global_phase, gate_matrix, unitary_from_gates
from hybrid_qcs.model import Budget, Gate, HybridState
from hybrid_qcs.pauli import Pauli, conjugate_by_pauli_rotation
from hybrid_qcs.rotation import PauliRotation
from hybrid_qcs.tableau import CliffordTableau


def _state(gates: tuple[Gate, ...], n: int = 1) -> HybridState:
    state = HybridState.identity(n, Budget(16, 16, 16, 16))
    for gate in gates:
        child = state.apply(gate, partial_order_reduction=False)
        assert child is not None
        state = child
    return state


def test_even_pauli_rotation_conjugation_matches_dense_matrices() -> None:
    for n in (1, 2):
        axes = [
            Pauli(n, x_mask, z_mask, 1)
            for x_mask in range(1 << n)
            for z_mask in range(1 << n)
            if x_mask or z_mask
        ]
        paulis = [
            Pauli(n, x_mask, z_mask, sign)
            for x_mask, z_mask, sign in itertools.product(
                range(1 << n), range(1 << n), (1, -1)
            )
        ]
        for axis, pauli, turns in itertools.product(axes, paulis, (0, 2, 4, 6)):
            rotation = PauliRotation(axis, turns).to_matrix()
            expected = rotation @ pauli.to_matrix() @ rotation.conj().T
            actual = conjugate_by_pauli_rotation(pauli, axis, turns).to_matrix()
            assert np.allclose(actual, expected, atol=1e-10)


def test_tableau_right_multiplication_by_clifford_rotation_is_consistent() -> None:
    tableau = CliffordTableau.identity(3)
    for gate in (
        Gate("H", (0,)),
        Gate("S", (1,)),
        Gate("CNOT", (0, 2)),
    ):
        tableau = tableau.left_multiply(gate.name, gate.qubits)
    axis = Pauli(3, x_mask=0b101, z_mask=0b010, sign=1)
    result = tableau.right_multiply_pauli_rotation(axis, 2)
    result.validate()
    for generator in (
        *(Pauli.x_axis(3, q) for q in range(3)),
        *(Pauli.z_axis(3, q) for q in range(3)),
    ):
        expected = tableau.forward_conjugate(
            conjugate_by_pauli_rotation(generator, axis, 2)
        )
        assert result.forward_conjugate(generator) == expected


def test_stronger_key_absorbs_clifford_valued_t_rotations() -> None:
    equivalent_pairs = (
        (
            (Gate("T", (0,)), Gate("T", (0,))),
            (Gate("S", (0,)),),
        ),
        (
            (Gate("TDG", (0,)), Gate("TDG", (0,))),
            (Gate("SDG", (0,)),),
        ),
        (
            (Gate("T", (0,)),) * 4,
            (Gate("S", (0,)), Gate("S", (0,))),
        ),
        (
            (Gate("H", (0,)), Gate("T", (0,)), Gate("T", (0,))),
            (Gate("H", (0,)), Gate("S", (0,))),
        ),
    )
    for left_gates, right_gates in equivalent_pairs:
        left = _state(left_gates)
        right = _state(right_gates)
        dense_equal, error = equal_up_to_global_phase(
            unitary_from_gates(1, left_gates),
            unitary_from_gates(1, right_gates),
        )
        assert dense_equal, error
        assert left.canonical_key == right.canonical_key
        left_legacy = legacy_projective_key(1, left.tableau, left.rotations)
        right_legacy = legacy_projective_key(1, right.tableau, right.rotations)
        assert left_legacy != right_legacy


def test_embedded_clifford_rotation_is_extracted_across_odd_barrier() -> None:
    tableau = CliffordTableau.identity(1)
    word = (
        PauliRotation(Pauli.x_axis(1, 0), 1),
        PauliRotation(Pauli.z_axis(1, 0), 2),
    )
    canonical = canonicalize_projective(tableau, word)
    assert canonical.extracted_clifford_rotations == 1
    assert len(canonical.rotations) == 1
    assert canonical.rotations[0].quarter_turns % 2 == 1

    expected_tableau = tableau.right_multiply_pauli_rotation(
        Pauli.z_axis(1, 0), 2
    )
    expected_axis = conjugate_by_pauli_rotation(
        Pauli.x_axis(1, 0), Pauli.z_axis(1, 0), -2
    )
    expected = canonicalize_projective(
        expected_tableau,
        (PauliRotation(expected_axis, 1),),
    )
    assert canonical.payload == expected.payload


def test_equal_strong_keys_remain_projectively_equal_on_bounded_enumeration() -> None:
    gates = tuple(Gate(name, (0,)) for name in ("H", "S", "SDG", "T", "TDG"))
    seen: dict[tuple[object, ...], np.ndarray] = {}
    for length in range(5):
        for sequence in itertools.product(gates, repeat=length):
            state = _state(tuple(sequence))
            dense = unitary_from_gates(1, sequence)
            previous = seen.get(state.canonical_key)
            if previous is None:
                seen[state.canonical_key] = dense
            else:
                equal, error = equal_up_to_global_phase(dense, previous)
                assert equal, (sequence, error)


def test_stronger_key_enables_terminal_certification_across_t2_and_s() -> None:
    from hybrid_qcs.benchmarks import _target_from_hidden_gates
    from hybrid_qcs.certify import certify_state

    target = _target_from_hidden_gates(
        "test-t2-target",
        "test",
        1,
        (Gate("T", (0,)), Gate("T", (0,))),
    )
    # The target metadata is constructed from T;T, while the candidate uses
    # the projectively equivalent single Clifford gate S.
    candidate = HybridState.identity(1, target.budget)
    child = candidate.apply(Gate("S", (0,)), partial_order_reduction=False)
    assert child is not None
    candidate = child
    result = certify_state(target, candidate)
    assert result.success
    assert result.symbolic_match


def test_stronger_key_allows_pareto_pruning_of_t2_by_s() -> None:
    from hybrid_qcs.benchmarks import _target_from_hidden_gates
    from hybrid_qcs.search import HybridSearch

    target = _target_from_hidden_gates(
        "test-s-pareto-target",
        "test",
        1,
        (Gate("T", (0,)), Gate("T", (0,))),
    )
    env = HybridSearch(target, max_expansions=8, partial_order_reduction=False)
    root = HybridState.identity(1, target.budget)
    s_state = root.apply(Gate("S", (0,)), partial_order_reduction=False)
    assert s_state is not None
    assert env._insert(s_state) is not None

    first_t = root.apply(Gate("T", (0,)), partial_order_reduction=False)
    assert first_t is not None
    second_t = first_t.apply(Gate("T", (0,)), partial_order_reduction=False)
    assert second_t is not None
    assert s_state.canonical_key == second_t.canonical_key
    assert env._insert(second_t) is None
