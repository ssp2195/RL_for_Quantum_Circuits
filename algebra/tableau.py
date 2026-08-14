"""A complete projective Clifford operator frame.

Unlike the original incomplete tableau, this frame stores signed images of
*both* ``X_i`` and ``Z_i`` for every qubit.  Those 2n images determine a
Clifford operation up to global phase, which is exactly the equivalence class
used by default canonicalisation.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

from algebra.pauli import PauliAxis, conjugate_axis_by_gate, transform_axis


_INVERSE_GATE = {
    "H": "H",
    "S": "SDG",
    "SDG": "S",
    "X": "X",
    "CNOT": "CNOT",
}


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
        # History is not part of the canonical payload.  It is an efficient,
        # exact way to evaluate the inverse transport requested during a T
        # update; two equivalent histories still have identical images.
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
        for name, qubits in reversed(self._history):
            result = conjugate_axis_by_gate(result, _INVERSE_GATE[name], qubits)
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
        The ordered primitive history is therefore retained *only* for the
        optional literal-phase key.  It may fail to merge equal literal phases
        reached through different Clifford words, but it can never merge
        different phases merely because their projective frames match.
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

        from certification.simulator import unitary_from_gates

        gates = [
            SimpleNamespace(gate_type=name, qubits=qubits)
            for name, qubits in self._history
        ]
        return unitary_from_gates(self.n, gates)

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


__all__ = ["CliffordFrame", "CliffordTableau"]
