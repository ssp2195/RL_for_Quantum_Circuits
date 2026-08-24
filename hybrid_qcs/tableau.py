"""Compact exact Clifford tableau with direct forward and inverse maps."""
from __future__ import annotations

from dataclasses import dataclass

from .pauli import (
    INVERSE_CLIFFORD_GATE,
    Pauli,
    conjugate_by_gate,
    transform_pauli,
)


@dataclass(frozen=True, slots=True)
class CliffordTableau:
    """Signed images of all X/Z generators under C and C†.

    Forward images determine the projective Clifford operator.  Inverse images
    are stored as an execution accelerator so T/T† transport is independent of
    witness length.  Gate history belongs only to the persistent circuit DAG.
    """

    num_qubits: int
    forward_x: tuple[Pauli, ...]
    forward_z: tuple[Pauli, ...]
    inverse_x: tuple[Pauli, ...]
    inverse_z: tuple[Pauli, ...]

    @classmethod
    def identity(cls, num_qubits: int) -> "CliffordTableau":
        x_images = tuple(Pauli.x_axis(num_qubits, q) for q in range(num_qubits))
        z_images = tuple(Pauli.z_axis(num_qubits, q) for q in range(num_qubits))
        return cls(num_qubits, x_images, z_images, x_images, z_images)

    def forward_conjugate(self, axis: Pauli) -> Pauli:
        self._validate_axis(axis)
        return transform_pauli(axis, self.forward_x, self.forward_z)

    def inverse_conjugate(self, axis: Pauli) -> Pauli:
        self._validate_axis(axis)
        return transform_pauli(axis, self.inverse_x, self.inverse_z)

    def left_multiply(self, gate_name: str, qubits: tuple[int, ...]) -> "CliffordTableau":
        """Return the exact tableau for ``C' = G C``."""
        name = str(gate_name).upper()
        if name not in INVERSE_CLIFFORD_GATE:
            raise ValueError(f"unsupported Clifford gate {gate_name!r}")

        forward_x = tuple(
            conjugate_by_gate(axis, name, qubits) for axis in self.forward_x
        )
        forward_z = tuple(
            conjugate_by_gate(axis, name, qubits) for axis in self.forward_z
        )
        inverse_name = INVERSE_CLIFFORD_GATE[name]
        inverse_x = tuple(
            self.inverse_conjugate(
                conjugate_by_gate(
                    Pauli.x_axis(self.num_qubits, q), inverse_name, qubits
                )
            )
            for q in range(self.num_qubits)
        )
        inverse_z = tuple(
            self.inverse_conjugate(
                conjugate_by_gate(
                    Pauli.z_axis(self.num_qubits, q), inverse_name, qubits
                )
            )
            for q in range(self.num_qubits)
        )
        return CliffordTableau(
            self.num_qubits,
            forward_x,
            forward_z,
            inverse_x,
            inverse_z,
        )

    def canonical_payload(self) -> tuple[tuple[int, int, int], ...]:
        """Complete projective Clifford identity; inverse images are redundant."""
        return tuple(axis.axis_payload() for axis in self.forward_x + self.forward_z)

    def validate(self) -> None:
        for q in range(self.num_qubits):
            for generator in (
                Pauli.x_axis(self.num_qubits, q),
                Pauli.z_axis(self.num_qubits, q),
            ):
                if self.inverse_conjugate(self.forward_conjugate(generator)) != generator:
                    raise AssertionError("forward and inverse tableau maps disagree")

    def _validate_axis(self, axis: Pauli) -> None:
        if not isinstance(axis, Pauli) or axis.num_qubits != self.num_qubits:
            raise ValueError("Pauli axis has the wrong register width")


__all__ = ["CliffordTableau"]
