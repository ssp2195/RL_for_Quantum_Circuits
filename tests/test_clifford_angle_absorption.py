import random

import numpy as np
import pytest

from algebra.pauli import PauliAxis
from algebra.pauli_rotation import (
    PauliRotation,
    normalize_clifford_semantics,
)
from algebra.tableau import CliffordFrame
from canonical.canonicalizer import Canonicalizer
from certification.simulator import (
    equivalent_up_to_global_phase,
    unitary_from_gates,
)
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType


def _state(num_qubits: int = 1) -> CircuitState:
    return CircuitState(
        CircuitDAG(num_qubits),
        ResourceBudget(
            max_t_count=32,
            max_depth=32,
            max_gates=32,
            max_two_qubit_count=32,
        ),
    )


def _append(state: CircuitState, gate_type: GateType, qubits: tuple[int, ...]) -> None:
    assert state.apply_gate(Gate(gate_type, qubits))


@pytest.mark.parametrize(
    ("non_clifford", "clifford"),
    (
        (GateType.T, GateType.S),
        (GateType.TDG, GateType.SDG),
    ),
)
def test_equal_t_pair_is_absorbed_into_the_clifford_frame(
    non_clifford: GateType,
    clifford: GateType,
) -> None:
    pair = _state()
    _append(pair, non_clifford, (0,))
    _append(pair, non_clifford, (0,))

    reference = _state()
    _append(reference, clifford, (0,))

    assert len(pair.rotations) == 1
    canonicalizer = Canonicalizer()
    assert canonicalizer.semantic_key(pair) == canonicalizer.semantic_key(reference)
    # The explicit flag is a genuine ablation: the raw exact word keeps the
    # emergent Clifford factor separate from the frame when absorption is off.
    unabsorbed = Canonicalizer(absorb_clifford_angles=False)
    assert unabsorbed.semantic_key(pair) != unabsorbed.semantic_key(reference)
    np.testing.assert_allclose(
        pair.symbolic_unitary(),
        unitary_from_gates(1, pair.dag.gates),
        atol=1e-10,
    )


def test_t_and_inverse_cancel_including_literal_phase() -> None:
    cancelled = _state()
    _append(cancelled, GateType.T, (0,))
    _append(cancelled, GateType.TDG, (0,))
    identity = _state()

    literal = Canonicalizer(phase_sensitive=True)
    assert cancelled.rotations == ()
    assert cancelled.global_phase_eighths == 0
    assert literal.semantic_key(cancelled) == literal.semantic_key(identity)

    phase_shifted = identity.copy()
    phase_shifted.global_phase_eighths = 8
    assert Canonicalizer().semantic_key(identity) == Canonicalizer().semantic_key(
        phase_shifted
    )
    assert literal.semantic_key(identity) != literal.semantic_key(phase_shifted)


def test_absorption_conjugates_crossed_axis_without_anticommuting_swap() -> None:
    x = PauliAxis.x_axis(1, 0)
    z = PauliAxis.z_axis(1, 0)
    raw_frame = CliffordFrame(1)
    raw_word = (
        PauliRotation(x, 1),
        PauliRotation(z, 1),
        PauliRotation(z, 1),
    )

    normalized_frame, normalized_word, normalized_phase = (
        normalize_clifford_semantics(raw_frame, raw_word)
    )

    # X anticommutes with the emergent K=R_Z(pi/2).  Moving K to the frame
    # therefore transports X to K^dagger X K = -Y; it does not swap X and Z.
    assert normalized_word == (
        PauliRotation(PauliAxis.y_axis(1, 0), -1),
    )
    assert normalized_phase == 0

    raw_unitary = raw_frame.to_unitary()
    for rotation in raw_word:
        raw_unitary = raw_unitary @ rotation.to_matrix()
    normalized_unitary = normalized_frame.to_unitary()
    for rotation in normalized_word:
        normalized_unitary = normalized_unitary @ rotation.to_matrix()
    normalized_unitary *= np.exp(1j * normalized_phase * np.pi / 8)
    np.testing.assert_allclose(normalized_unitary, raw_unitary, atol=1e-10)


def test_absorption_repeats_to_a_fixed_point() -> None:
    x = PauliAxis.x_axis(1, 0)
    z = PauliAxis.z_axis(1, 0)
    frame, word, _ = normalize_clifford_semantics(
        CliffordFrame(1),
        (
            PauliRotation(x, 1),
            PauliRotation(x, 1),
            PauliRotation(z, 1),
            PauliRotation(z, 1),
        ),
    )

    assert word == ()
    assert frame.canonical_payload() != CliffordFrame(1).canonical_payload()


def test_seeded_short_clifford_t_witnesses_preserve_exact_dense_semantics() -> None:
    rng = random.Random(20260815)
    canonicalizer = Canonicalizer()

    for num_qubits in (1, 2):
        unitary_by_key: dict[tuple[object, ...], np.ndarray] = {}
        choices = [
            (gate_type, (qubit,))
            for gate_type in (
                GateType.H,
                GateType.S,
                GateType.SDG,
                GateType.T,
                GateType.TDG,
                GateType.X,
            )
            for qubit in range(num_qubits)
        ]
        if num_qubits == 2:
            choices.extend(
                (
                    (GateType.CNOT, (0, 1)),
                    (GateType.CNOT, (1, 0)),
                )
            )

        for _ in range(40):
            state = _state(num_qubits)
            for _ in range(rng.randint(0, 12)):
                gate_type, qubits = rng.choice(choices)
                _append(state, gate_type, qubits)

            witness = unitary_from_gates(num_qubits, state.dag.gates)
            np.testing.assert_allclose(state.symbolic_unitary(), witness, atol=1e-9)
            normalized_frame, normalized_word, normalized_phase = (
                normalize_clifford_semantics(
                    state.frame,
                    state.rotations,
                    state.global_phase_eighths,
                )
            )
            assert all(rotation.quarter_turns % 2 for rotation in normalized_word)
            normalized_unitary = normalized_frame.to_unitary()
            for rotation in normalized_word:
                normalized_unitary = normalized_unitary @ rotation.to_matrix()
            normalized_unitary *= np.exp(1j * normalized_phase * np.pi / 8)
            np.testing.assert_allclose(normalized_unitary, witness, atol=1e-9)

            # Appending H twice is a concrete identity suffix.  It must merge
            # projectively after frame normalization, and any such merge is
            # checked against the independent dense witness.
            equivalent = state.copy()
            qubit = rng.randrange(num_qubits)
            _append(equivalent, GateType.H, (qubit,))
            _append(equivalent, GateType.H, (qubit,))
            assert canonicalizer.semantic_key(state) == canonicalizer.semantic_key(
                equivalent
            )
            assert equivalent_up_to_global_phase(
                unitary_from_gates(num_qubits, state.dag.gates),
                unitary_from_gates(num_qubits, equivalent.dag.gates),
            )

            # Every collision found in the seeded sample (including the
            # guaranteed H;H variant) must be a true dense-unitary merge.
            for candidate in (state, equivalent):
                key = canonicalizer.semantic_key(candidate)
                candidate_unitary = unitary_from_gates(
                    num_qubits,
                    candidate.dag.gates,
                )
                previous = unitary_by_key.setdefault(key, candidate_unitary)
                assert equivalent_up_to_global_phase(candidate_unitary, previous)
