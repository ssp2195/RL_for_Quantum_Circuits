"""Independent truth-table coverage for the public Toffoli reference oracle.

The reference matrix is deliberately tested from its basis-state action rather
than by comparing it to the Clifford+T decomposition.  That keeps this oracle
independent of the candidate witness used by the certification benchmark.
"""

from __future__ import annotations

import numpy as np

from benchmarks.toffoli import (
    TOFFOLI_CONTROLS,
    TOFFOLI_NUM_QUBITS,
    TOFFOLI_TARGET_QUBIT,
    expected_toffoli_basis_index,
    toffoli_reference_unitary,
)


def _analytical_toffoli_output(input_index: int) -> int:
    """Return the classical Toffoli output in the repository bit ordering."""

    controls_are_one = all(
        (input_index >> control) & 1 for control in TOFFOLI_CONTROLS
    )
    return input_index ^ ((1 << TOFFOLI_TARGET_QUBIT) if controls_are_one else 0)


def test_public_basis_index_oracle_matches_the_analytical_toffoli_truth_table():
    dimension = 1 << TOFFOLI_NUM_QUBITS

    assert TOFFOLI_NUM_QUBITS == 3
    assert len(TOFFOLI_CONTROLS) == 2
    assert TOFFOLI_TARGET_QUBIT not in TOFFOLI_CONTROLS

    for input_index in range(dimension):
        assert expected_toffoli_basis_index(input_index) == _analytical_toffoli_output(
            input_index
        )


def test_reference_unitary_is_an_exact_permutation_truth_table_and_is_self_inverse():
    reference = toffoli_reference_unitary()
    dimension = 1 << TOFFOLI_NUM_QUBITS

    assert reference.shape == (dimension, dimension)
    assert np.isfinite(reference).all()

    for input_index in range(dimension):
        expected_output = expected_toffoli_basis_index(input_index)
        expected_column = np.zeros(dimension, dtype=np.complex128)
        expected_column[expected_output] = 1.0
        np.testing.assert_allclose(reference[:, input_index], expected_column, atol=1e-12)

    # Toffoli is an involution, so this checks the oracle from a second
    # direction without consulting the known Clifford+T witness.
    np.testing.assert_allclose(reference @ reference, np.eye(dimension), atol=1e-12)
    np.testing.assert_allclose(reference.conj().T @ reference, np.eye(dimension), atol=1e-12)
