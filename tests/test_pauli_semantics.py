import numpy as np

from algebra.pauli import PauliAxis, conjugate_axis_by_gate
from algebra.pauli_rotation import PauliRotation, normalize_rotation_word


def test_symplectic_commutation_and_weight_are_exact_bit_operations():
    x0 = PauliAxis.x_axis(2, 0)
    z0 = PauliAxis.z_axis(2, 0)
    z1 = PauliAxis.z_axis(2, 1)

    assert x0.anticommutes_with(z0)
    assert x0.commutes_with(z1)
    assert PauliAxis(2, 0b11, 0b10).weight == 2


def test_signed_pauli_conjugation_has_the_required_h_and_s_signs():
    y = PauliAxis.y_axis(1, 0)
    x = PauliAxis.x_axis(1, 0)

    assert conjugate_axis_by_gate(y, "H", (0,)) == y.negated()
    assert conjugate_axis_by_gate(x, "S", (0,)) == y
    assert conjugate_axis_by_gate(y, "S", (0,)) == x.negated()
    assert conjugate_axis_by_gate(x, "SDG", (0,)) == y.negated()
    assert conjugate_axis_by_gate(y, "SDG", (0,)) == x


def test_cnot_generates_multi_qubit_pauli_axes():
    assert conjugate_axis_by_gate(PauliAxis.x_axis(2, 0), "CNOT", (0, 1)) == PauliAxis(
        2, 0b11, 0
    )
    assert conjugate_axis_by_gate(PauliAxis.z_axis(2, 1), "CNOT", (0, 1)) == PauliAxis(
        2, 0, 0b11
    )


def test_rotation_word_merges_cancels_and_tracks_global_phase_exactly():
    z = PauliAxis.z_axis(1, 0)
    word, phase = normalize_rotation_word(
        [PauliRotation(z, 1), PauliRotation(z.negated(), 1)]
    )
    assert word == ()
    assert phase == 0

    word, phase = normalize_rotation_word([PauliRotation(z, 9)])
    assert word == (PauliRotation(z, 1),)
    assert phase == 8  # Rz(theta + 2pi) = -Rz(theta)


def test_normalizer_reorders_only_commuting_rotations():
    x = PauliAxis.x_axis(1, 0)
    z = PauliAxis.z_axis(1, 0)
    z0 = PauliAxis.z_axis(2, 0)
    z1 = PauliAxis.z_axis(2, 1)

    commuting_a, _ = normalize_rotation_word([PauliRotation(z1, 1), PauliRotation(z0, 1)])
    commuting_b, _ = normalize_rotation_word([PauliRotation(z0, 1), PauliRotation(z1, 1)])
    assert commuting_a == commuting_b

    anti_a, _ = normalize_rotation_word([PauliRotation(x, 1), PauliRotation(z, 1)])
    anti_b, _ = normalize_rotation_word([PauliRotation(z, 1), PauliRotation(x, 1)])
    assert anti_a != anti_b


def test_rotation_dense_helper_uses_the_article_convention():
    z = PauliAxis.z_axis(1, 0)
    expected = np.diag([np.exp(-1j * np.pi / 8), np.exp(1j * np.pi / 8)])
    np.testing.assert_allclose(PauliRotation(z, 1).to_matrix(), expected)
