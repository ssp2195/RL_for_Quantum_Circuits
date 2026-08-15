"""A complete projective Clifford operator frame.

Unlike the original incomplete tableau, this frame stores signed images of
*both* ``X_i`` and ``Z_i`` for every qubit.  Those 2n images determine a
Clifford operation up to global phase, which is exactly the equivalence class
used by default canonicalisation.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from algebra.pauli import (
    PauliAxis,
    conjugate_axis_by_gate,
    multiply_binary_paulis,
    transform_axis,
)


_INVERSE_GATE = {
    "H": "H",
    "S": "SDG",
    "SDG": "S",
    "X": "X",
    "CNOT": "CNOT",
}

_PAULI_ROTATION = "PAULI_ROTATION"


def conjugate_axis_by_pauli_rotation(
    axis: PauliAxis,
    rotation_axis: PauliAxis,
    quarter_turns: int,
) -> PauliAxis:
    """Return exact conjugation of ``axis`` by a Clifford Pauli rotation.

    ``quarter_turns`` must be even, so the angle is an integer multiple of
    ``pi/2``.  For anticommuting Hermitian Paulis the +pi/2 case is
    ``R_P(pi/2) Q R_P(pi/2)^dagger = -i P Q``.  Binary Pauli phases preserve
    every sign introduced by that identity.
    """
    if not isinstance(axis, PauliAxis) or not isinstance(rotation_axis, PauliAxis):
        raise TypeError("Pauli rotation conjugation requires PauliAxis values")
    if axis.num_qubits != rotation_axis.num_qubits:
        raise ValueError("Pauli axes must have the same number of qubits")
    if isinstance(quarter_turns, bool) or not isinstance(quarter_turns, int):
        raise TypeError("quarter_turns must be an integer")
    if quarter_turns % 2:
        raise ValueError("only Clifford-valued Pauli rotations may conjugate a frame")
    if axis.commutes_with(rotation_axis):
        return axis

    turns = quarter_turns % 8
    if turns == 0:
        return axis
    if turns == 4:
        return axis.negated()

    x_mask, z_mask, phase = multiply_binary_paulis(
        rotation_axis.x_mask,
        rotation_axis.z_mask,
        rotation_axis.binary_phase,
        axis.x_mask,
        axis.z_mask,
        axis.binary_phase,
    )
    # +pi/2 contributes -i P Q; -pi/2 contributes +i P Q.
    phase = (phase + (-1 if turns == 2 else 1)) & 3
    return PauliAxis.from_binary_phase(axis.num_qubits, x_mask, z_mask, phase)


class CliffordFrame:
    """Exact Clifford conjugation data modulo global phase.

    Calling ``apply_G`` models a left multiplication ``C <- G C``, matching
    the circuit convention used by :class:`circuit.circuit_state.CircuitState`.
    ``inverse_conjugate(P)`` then computes ``C† P C`` for a signed Pauli axis.
    """

    def __init__(self, num_qubits: int):
        if isinstance(num_qubits, bool) or not isinstance(num_qubits, int):
            raise TypeError("num_qubits must be an integer")
        if num_qubits < 0:
            raise ValueError("num_qubits must be non-negative")
        self.n = num_qubits
        self.x_images: Tuple[PauliAxis, ...] = tuple(
            PauliAxis.x_axis(num_qubits, qubit) for qubit in range(num_qubits)
        )
        self.z_images: Tuple[PauliAxis, ...] = tuple(
            PauliAxis.z_axis(num_qubits, qubit) for qubit in range(num_qubits)
        )
        # Entries are in concrete execution order.  Native Clifford gates
        # store their qubits; absorbed Pauli rotations store the immutable
        # ``(n, x, z, sign, quarter_turns)`` payload.  History is excluded from
        # the projective payload but provides an exact literal lift and inverse
        # transport implementation.
        self._history: list[tuple[str, Tuple[int, ...]]] = []

    @property
    def num_qubits(self) -> int:
        return self.n

    # ---------- Clifford updates ----------

    def _left_multiply(self, name: str, qubits: Sequence[int]) -> None:
        qubits = tuple(qubits)
        self._validate_gate(name, qubits)
        self.x_images = tuple(
            conjugate_axis_by_gate(axis, name, qubits) for axis in self.x_images
        )
        self.z_images = tuple(
            conjugate_axis_by_gate(axis, name, qubits) for axis in self.z_images
        )
        self._history.append((name.upper(), qubits))

    def apply_H(self, qubit: int) -> None:
        self._left_multiply("H", (qubit,))

    def apply_S(self, qubit: int) -> None:
        self._left_multiply("S", (qubit,))

    def apply_SDG(self, qubit: int) -> None:
        self._left_multiply("SDG", (qubit,))

    # Legacy spelling used in some notebooks/callers.
    apply_Sdg = apply_SDG

    def apply_X(self, qubit: int) -> None:
        self._left_multiply("X", (qubit,))

    def apply_CNOT(self, control: int, target: int) -> None:
        self._left_multiply("CNOT", (control, target))

    def right_multiply_pauli_rotation(
        self,
        axis: PauliAxis,
        quarter_turns: int,
    ) -> None:
        """Absorb ``R_axis(theta)`` via exact right multiplication ``C <- C K``.

        Full ``2*pi`` turns must already have been moved to the caller's
        scalar phase.  Accepting only the normalizer's reduced representatives
        prevents that scalar ``-1`` from being hidden by this projective frame.
        """
        self._validate_axis(axis)
        if isinstance(quarter_turns, bool) or not isinstance(quarter_turns, int):
            raise TypeError("quarter_turns must be an integer")
        if quarter_turns not in (-2, 2, 4):
            raise ValueError(
                "absorbed rotations must be reduced Clifford turns -2, 2, or 4"
            )

        old_x_images = self.x_images
        old_z_images = self.z_images

        def through_old_frame(generator: PauliAxis) -> PauliAxis:
            rotated = conjugate_axis_by_pauli_rotation(
                generator,
                axis,
                quarter_turns,
            )
            return transform_axis(rotated, old_x_images, old_z_images)

        self.x_images = tuple(
            through_old_frame(PauliAxis.x_axis(self.n, qubit))
            for qubit in range(self.n)
        )
        self.z_images = tuple(
            through_old_frame(PauliAxis.z_axis(self.n, qubit))
            for qubit in range(self.n)
        )
        # If the previous frame history executes as K0 then C, the new
        # product C K executes K first.  Hence insertion at the beginning.
        self._history.insert(
            0,
            (_PAULI_ROTATION, (*axis.canonical_payload(), quarter_turns)),
        )

    def left_multiply_gate(self, gate_or_name, qubits: Sequence[int] | None = None) -> None:
        """Dispatch a Gate-like object or a name/operand tuple to an update."""
        if qubits is None:
            gate_type = getattr(gate_or_name, "gate_type", gate_or_name)
            qubits = getattr(gate_or_name, "qubits", None)
            if qubits is None:
                raise TypeError("gate-like input must expose qubits")
        else:
            gate_type = gate_or_name
        name = getattr(gate_type, "name", gate_type)
        if not isinstance(name, str):
            raise TypeError("gate name must be a string or enum")
        self._left_multiply(name.upper(), tuple(qubits))

    # ---------- Pauli transport ----------

    def forward_conjugate(self, axis: PauliAxis) -> PauliAxis:
        """Compute ``C P C†`` from the complete generator image tableau."""
        self._validate_axis(axis)
        return transform_axis(axis, self.x_images, self.z_images)

    def inverse_conjugate(self, axis: PauliAxis) -> PauliAxis:
        """Compute ``C† P C`` exactly through the inverse gate history."""
        self._validate_axis(axis)
        result = axis
        for name, payload in reversed(self._history):
            if name == _PAULI_ROTATION:
                n, x_mask, z_mask, sign, quarter_turns = payload
                rotation_axis = PauliAxis(n, x_mask, z_mask, sign)
                result = conjugate_axis_by_pauli_rotation(
                    result,
                    rotation_axis,
                    -quarter_turns,
                )
            else:
                result = conjugate_axis_by_gate(
                    result,
                    _INVERSE_GATE[name],
                    payload,
                )
        return result

    # More verbose aliases make call sites/readers explicit about direction.
    conjugate = forward_conjugate
    conjugate_by_inverse = inverse_conjugate

    # ---------- Identity/copy/payload ----------

    def canonical_payload(self) -> Tuple[object, ...]:
        return (
            self.n,
            tuple(axis.canonical_payload() for axis in self.x_images),
            tuple(axis.canonical_payload() for axis in self.z_images),
        )

    stable_payload = canonical_payload

    def phase_sensitive_payload(self) -> Tuple[tuple[str, Tuple[int, ...]], ...]:
        """A conservative exact lift of the projective frame.

        Signed generator images determine a Clifford only up to global phase.
        The ordered exact operation history (native gates plus absorbed Pauli
        rotations) is therefore retained *only* for the optional literal-phase
        key.  It may fail to merge equal literal phases reached through
        different Clifford words, but it can never merge different phases
        merely because their projective frames match.
        """
        return tuple(self._history)

    def copy(self) -> "CliffordFrame":
        clone = CliffordFrame(self.n)
        clone.x_images = tuple(self.x_images)
        clone.z_images = tuple(self.z_images)
        clone._history = list(self._history)
        return clone

    def to_unitary(self):
        """Dense matrix for small-instance tests and diagnostics only.

        The operational frame itself remains binary/symbolic; this helper
        deliberately imports the independent simulator lazily to avoid making
        dense matrices part of the synthesis-state representation.
        """
        from types import SimpleNamespace

        import numpy as np

        from algebra.pauli_rotation import PauliRotation
        from certification.simulator import unitary_from_gates

        unitary = np.eye(1 << self.n, dtype=np.complex128)
        for name, payload in self._history:
            if name == _PAULI_ROTATION:
                n, x_mask, z_mask, sign, quarter_turns = payload
                operation = PauliRotation(
                    PauliAxis(n, x_mask, z_mask, sign),
                    quarter_turns,
                ).to_matrix()
            else:
                operation = unitary_from_gates(
                    self.n,
                    [SimpleNamespace(gate_type=name, qubits=payload)],
                )
            unitary = operation @ unitary
        return unitary

    def _validate_axis(self, axis: PauliAxis) -> None:
        if not isinstance(axis, PauliAxis):
            raise TypeError("Clifford transport requires a PauliAxis")
        if axis.num_qubits != self.n:
            raise ValueError("Pauli axis has the wrong number of qubits")

    def _validate_gate(self, name: str, qubits: Tuple[int, ...]) -> None:
        normalized = name.upper()
        arity = 2 if normalized == "CNOT" else 1
        if normalized not in _INVERSE_GATE:
            raise ValueError(f"unsupported Clifford gate {name!r}")
        if len(qubits) != arity or len(set(qubits)) != len(qubits):
            raise ValueError(f"{normalized} has invalid operands {qubits!r}")
        if any(
            isinstance(qubit, bool) or not isinstance(qubit, int) or not 0 <= qubit < self.n
            for qubit in qubits
        ):
            raise ValueError(f"{normalized} has out-of-range operands {qubits!r}")

    def __repr__(self) -> str:
        return (
            f"CliffordFrame(n={self.n}, x_images={self.x_images!r}, "
            f"z_images={self.z_images!r})"
        )


# Existing imports keep working, but the old class's incomplete X/Z matrices
# are intentionally replaced by a full frame.
CliffordTableau = CliffordFrame


__all__ = [
    "CliffordFrame",
    "CliffordTableau",
    "conjugate_axis_by_pauli_rotation",
]
