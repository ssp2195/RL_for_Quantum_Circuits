from fractions import Fraction

import numpy as np

from benchmarks.qft import (
    ControlledPhase,
    ExactCapability,
    analytical_qft_matrix,
    aqft3_metrics,
    aqft3_reference,
    assess_native_exact_capability,
    bit_reversal_matrix,
    declared_qft_target_matrix,
    prepare_native_exact_search,
    qft_reference,
    reference_unitary,
)
from enums import GateType
from search.action_space import generate_actions


def test_qft3_analytical_matrix_and_inverse_are_independent_closed_forms():
    forward = analytical_qft_matrix(3)
    inverse = analytical_qft_matrix(3, inverse=True)
    assert forward.shape == (8, 8)
    assert not forward.flags.writeable
    assert np.allclose(inverse, forward.conj().T, atol=1e-12)
    assert np.allclose(forward.conj().T @ forward, np.eye(8), atol=1e-12)

    # Phase-sensitive column checks use the analytical formula directly and
    # therefore cannot pass merely because a generated QFT is self-inverted.
    output_indices = np.arange(8)
    for basis_index in (0, 1, 3, 7):
        expected = np.exp(2j * np.pi * output_indices * basis_index / 8) / np.sqrt(8)
        assert np.allclose(forward[:, basis_index], expected, atol=1e-12)


def test_reference_operations_match_analytical_forward_and_inverse_qft():
    for num_qubits in (1, 2, 3):
        for inverse in (False, True):
            reference = qft_reference(num_qubits, inverse=inverse)
            assert np.allclose(
                reference_unitary(reference),
                analytical_qft_matrix(num_qubits, inverse=inverse),
                atol=2e-12,
            )


def test_swap_omission_has_an_explicit_distinct_permutation_target():
    forward = qft_reference(3, include_final_swaps=False)
    inverse = qft_reference(3, inverse=True, include_final_swaps=False)
    reversal = bit_reversal_matrix(3)
    fourier = analytical_qft_matrix(3)

    assert forward.permutation_convention == "bit-reversed-output"
    assert inverse.permutation_convention == "bit-reversed-input"
    assert np.allclose(declared_qft_target_matrix(forward), reversal @ fourier)
    assert np.allclose(declared_qft_target_matrix(inverse), fourier.conj().T @ reversal)
    assert np.allclose(reference_unitary(forward), declared_qft_target_matrix(forward))
    assert np.allclose(reference_unitary(inverse), declared_qft_target_matrix(inverse))
    assert not np.allclose(reference_unitary(forward), fourier)


def test_qft3_exact_native_submission_is_blocked_with_machine_reason():
    one_qubit = assess_native_exact_capability(qft_reference(1))
    two_qubit = assess_native_exact_capability(qft_reference(2))
    three_qubit = assess_native_exact_capability(qft_reference(3))

    assert one_qubit.classification is ExactCapability.EXACT_NATIVE
    assert two_qubit.classification is ExactCapability.EXACT_DECOMPOSABLE
    assert three_qubit.classification is ExactCapability.APPROXIMATION_REQUIRED
    assert not three_qubit.exact_search_allowed
    assert three_qubit.reason_code == "reference_contains_unlowered_controlled_phase_angles"
    assert [issue.operation.angle_pi for issue in three_qubit.issues] == [Fraction(1, 4)]

    request = prepare_native_exact_search(qft_reference(3))
    assert not request.accepted
    assert request.target is None
    assert request.capability.classification is ExactCapability.APPROXIMATION_REQUIRED


def test_aqft3_is_separate_and_reports_omission_and_fidelity_metrics():
    reference = aqft3_reference()
    assert reference.mode == "approximate"
    assert len(reference.omitted_operations) == 1
    assert isinstance(reference.omitted_operations[0], ControlledPhase)
    assert reference.omitted_operations[0].angle_pi == Fraction(1, 4)
    assert all(
        not (
            isinstance(operation, ControlledPhase)
            and abs(operation.angle_pi) == Fraction(1, 4)
        )
        for operation in reference.operations
    )

    metrics = aqft3_metrics()
    inverse_metrics = aqft3_metrics(inverse=True)
    no_swap_metrics = aqft3_metrics(include_final_swaps=False)
    expected_process_fidelity = (10 + 3 * np.sqrt(2)) / 16
    assert np.isclose(metrics.process_fidelity, expected_process_fidelity, atol=1e-12)
    assert np.isclose(inverse_metrics.process_fidelity, metrics.process_fidelity, atol=1e-12)
    assert np.isclose(no_swap_metrics.process_fidelity, metrics.process_fidelity, atol=1e-12)
    assert 0.0 < metrics.maximum_matrix_error < 1.0
    assert np.isclose(
        inverse_metrics.maximum_matrix_error,
        metrics.maximum_matrix_error,
        atol=1e-12,
    )
    assert len(metrics.state_fidelities) >= 4
    assert all(0.0 <= metric.fidelity <= 1.0 for metric in metrics.state_fidelities)
    metadata = metrics.metadata()
    assert metadata["benchmark"] == "AQFT-3"
    assert metadata["acceptance_mode"] == "approximate-metrics-only"
    assert metadata["reference"]["omitted_operations"][0]["angle_pi"] == "1/4"
    assert no_swap_metrics.reference.permutation_convention == "bit-reversed-output"


def test_reference_parameterized_gates_do_not_enter_native_search_grammar():
    assert {gate.name for gate in GateType} == {
        "H",
        "S",
        "SDG",
        "T",
        "TDG",
        "X",
        "CNOT",
    }
    native_action_names = {action.gate_type.name for action in generate_actions(3)}
    assert native_action_names == {"H", "S", "SDG", "T", "TDG", "CNOT"}
    assert "ControlledPhase" not in native_action_names
    assert "SWAP" not in native_action_names
