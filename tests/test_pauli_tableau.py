from __future__ import annotations

import itertools

import numpy as np

from hybrid_qcs.certify import gate_matrix
from hybrid_qcs.model import Gate
from hybrid_qcs.pauli import Pauli, conjugate_by_gate
from hybrid_qcs.tableau import CliffordTableau


def test_every_small_pauli_gate_conjugation_matches_dense_matrix() -> None:
    for n in (1, 2, 3):
        gates = [
            Gate(name, (q,))
            for q in range(n)
            for name in ("H", "S", "SDG")
        ]
        gates += [
            Gate("CNOT", (control, target))
            for control in range(n)
            for target in range(n)
            if control != target
        ]
        for gate in gates:
            matrix = gate_matrix(n, gate)
            for x_mask, z_mask, sign in itertools.product(
                range(1 << n), range(1 << n), (1, -1)
            ):
                axis = Pauli(n, x_mask, z_mask, sign)
                expected = matrix @ axis.to_matrix() @ matrix.conj().T
                actual = conjugate_by_gate(axis, gate.name, gate.qubits).to_matrix()
                assert np.allclose(actual, expected, atol=1e-10)


def test_tableau_stores_exact_direct_inverse_map_without_history() -> None:
    tableau = CliffordTableau.identity(3)
    for gate in (
        Gate("H", (0,)),
        Gate("S", (1,)),
        Gate("CNOT", (0, 2)),
        Gate("SDG", (2,)),
        Gate("CNOT", (2, 1)),
    ):
        tableau = tableau.left_multiply(gate.name, gate.qubits)
    tableau.validate()
    for q in range(3):
        for generator in (Pauli.x_axis(3, q), Pauli.z_axis(3, q)):
            assert tableau.inverse_conjugate(tableau.forward_conjugate(generator)) == generator
