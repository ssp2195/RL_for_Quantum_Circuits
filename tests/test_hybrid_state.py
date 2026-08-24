from __future__ import annotations

import random

import numpy as np

from hybrid_qcs.certify import equal_up_to_global_phase, gate_matrix, unitary_from_gates
from hybrid_qcs.model import Budget, Gate, HybridState, generate_gates
from hybrid_qcs.rotation import PauliRotation, insert_rotation, normalize_word_reference


def test_random_hybrid_invariant_matches_dense_circuit() -> None:
    rng = random.Random(17)
    for n in (1, 2, 3):
        gates = generate_gates(n)
        for _trial in range(40):
            state = HybridState.identity(n, Budget(7, 7, 7, 7))
            full = np.eye(1 << n, dtype=np.complex128)
            clifford = np.eye(1 << n, dtype=np.complex128)
            for _step in range(7):
                gate = rng.choice(gates)
                child = state.apply(gate, partial_order_reduction=False)
                assert child is not None
                state = child
                matrix = gate_matrix(n, gate)
                full = matrix @ full
                if gate.is_clifford:
                    clifford = matrix @ clifford
                symbolic = (
                    np.exp(1j * np.pi * state.global_phase_eighths / 8.0)
                    * clifford
                )
                for rotation in state.rotations:
                    symbolic = symbolic @ rotation.to_matrix()
                match, error = equal_up_to_global_phase(symbolic, full)
                assert match, error
            state.validate()


def test_persistent_dag_shares_prefix_and_materializes_exact_dependencies() -> None:
    budget = Budget(2, 2, 5, 5)
    root = HybridState.identity(3, budget)
    first = root.apply(Gate("H", (0,)), partial_order_reduction=False)
    assert first is not None
    left = first.apply(Gate("T", (0,)), partial_order_reduction=False)
    right = first.apply(Gate("CNOT", (0, 1)), partial_order_reduction=False)
    assert left is not None and right is not None
    assert left.tail is not None and right.tail is not None
    assert left.tail.previous is first.tail
    assert right.tail.previous is first.tail
    left.materialize_dag().validate()
    right.materialize_dag().validate()


def test_incremental_rotation_insertion_matches_reference() -> None:
    budget = Budget(8, 0, 8, 8)
    state = HybridState.identity(1, budget)
    raw: list[PauliRotation] = []
    for gate in (
        Gate("T", (0,)),
        Gate("H", (0,)),
        Gate("T", (0,)),
        Gate("H", (0,)),
        Gate("TDG", (0,)),
        Gate("T", (0,)),
    ):
        if gate.is_non_clifford:
            from hybrid_qcs.pauli import Pauli
            axis = state.tableau.inverse_conjugate(Pauli.z_axis(1, 0))
            raw.insert(0, PauliRotation(axis, 1 if gate.name == "T" else -1))
        child = state.apply(gate, partial_order_reduction=False)
        assert child is not None
        state = child
    reference, _delta = normalize_word_reference(raw)
    assert state.rotations == reference


def test_equal_canonical_keys_never_hide_dense_inequality_on_random_sample() -> None:
    rng = random.Random(29)
    seen: dict[tuple[object, ...], np.ndarray] = {}
    budget = Budget(5, 5, 5, 5)
    gates = generate_gates(2)
    for _ in range(600):
        state = HybridState.identity(2, budget)
        for _step in range(rng.randint(0, 5)):
            child = state.apply(rng.choice(gates), partial_order_reduction=False)
            assert child is not None
            state = child
        dense = unitary_from_gates(2, state.reconstruct_gates())
        previous = seen.get(state.canonical_key)
        if previous is not None:
            match, error = equal_up_to_global_phase(dense, previous)
            assert match, error
        else:
            seen[state.canonical_key] = dense
