"""Exact signed-Hermitian Pauli primitives.

The binary convention is

``P(x, z, sign) = sign * i**popcount(x & z) * X**x * Z**z``.

It represents ``Y`` without a floating point phase and makes the usual
symplectic commutation test a pair of integer bit operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple


def _parity(mask: int) -> int:
    return mask.bit_count() & 1


@dataclass(frozen=True, slots=True)
class PauliAxis:
    """A signed Hermitian tensor-product Pauli operator.

    ``x_mask`` and ``z_mask`` use qubit ``0`` as the least-significant bit.
    The explicit sign is essential: Clifford conjugation can map, for example,
    ``Y`` to ``-Y``.
    """

    num_qubits: int
    x_mask: int = 0
    z_mask: int = 0
    sign: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.num_qubits, bool) or not isinstance(self.num_qubits, int):
            raise TypeError("num_qubits must be an integer")
        if self.num_qubits < 0:
            raise ValueError("num_qubits must be non-negative")
        for name, value in (("x_mask", self.x_mask), ("z_mask", self.z_mask)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0 or value >= (1 << self.num_qubits):
                raise ValueError(f"{name} has a bit outside the register")
        if self.sign not in (-1, 1):
            raise ValueError("PauliAxis.sign must be -1 or +1")

    # ---------- Constructors ----------

    @classmethod
    def identity(cls, num_qubits: int, sign: int = 1) -> "PauliAxis":
        return cls(num_qubits, 0, 0, sign)

    @classmethod
    def x_axis(cls, num_qubits: int, qubit: int) -> "PauliAxis":
        return cls._single(num_qubits, qubit, x=True, z=False)

    @classmethod
    def y_axis(cls, num_qubits: int, qubit: int) -> "PauliAxis":
        return cls._single(num_qubits, qubit, x=True, z=True)

    @classmethod
    def z_axis(cls, num_qubits: int, qubit: int) -> "PauliAxis":
        return cls._single(num_qubits, qubit, x=False, z=True)

    # Short aliases used by callers that prefer ``z_on(q)`` terminology.
    x_on = x_axis
    y_on = y_axis
    z_on = z_axis

    @classmethod
    def _single(
        cls, num_qubits: int, qubit: int, *, x: bool, z: bool
    ) -> "PauliAxis":
        if isinstance(qubit, bool) or not isinstance(qubit, int):
            raise TypeError("qubit must be an integer")
        if qubit < 0 or qubit >= num_qubits:
            raise ValueError(f"qubit {qubit} is outside a {num_qubits}-qubit register")
        mask = 1 << qubit
        return cls(num_qubits, mask if x else 0, mask if z else 0)

    # ---------- Structural properties ----------

    @property
    def weight(self) -> int:
        return (self.x_mask | self.z_mask).bit_count()

    @property
    def is_identity(self) -> bool:
        return self.x_mask == 0 and self.z_mask == 0

    @property
    def binary_phase(self) -> int:
        """Phase ``p`` in ``i**p X**x Z**z`` (modulo four)."""
        return (self.x_mask & self.z_mask).bit_count() + (0 if self.sign == 1 else 2)

    def canonical_payload(self) -> Tuple[int, int, int, int]:
        return (self.num_qubits, self.x_mask, self.z_mask, self.sign)

    payload = canonical_payload

    def sort_key(self) -> Tuple[int, int]:
        """Sign-free deterministic key used after rotation sign normalisation."""
        return (self.x_mask, self.z_mask)

    # ---------- Exact Pauli algebra ----------

    def symplectic_inner(self, other: "PauliAxis") -> int:
        self._same_size(other)
        return _parity(
            (self.x_mask & other.z_mask) ^ (self.z_mask & other.x_mask)
        )

    def commutes_with(self, other: "PauliAxis") -> bool:
        return self.symplectic_inner(other) == 0

    def anticommutes_with(self, other: "PauliAxis") -> bool:
        return not self.commutes_with(other)

    def negated(self) -> "PauliAxis":
        return PauliAxis(self.num_qubits, self.x_mask, self.z_mask, -self.sign)

    def with_sign(self, sign: int) -> "PauliAxis":
        return PauliAxis(self.num_qubits, self.x_mask, self.z_mask, sign)

    def positive_axis(self) -> "PauliAxis":
        return self if self.sign == 1 else self.negated()

    def multiply(self, other: "PauliAxis") -> "PauliAxis":
        """Multiply commuting axes and return their signed Hermitian product.

        The product of anticommuting Hermitian Paulis is anti-Hermitian and
        cannot be represented by ``PauliAxis`` alone, so that case raises.
        """
        self._same_size(other)
        if not self.commutes_with(other):
            raise ValueError("the product of anticommuting axes is not Hermitian")
        x_mask, z_mask, phase = multiply_binary_paulis(
            self.x_mask,
            self.z_mask,
            self.binary_phase,
            other.x_mask,
            other.z_mask,
            other.binary_phase,
        )
        return PauliAxis.from_binary_phase(self.num_qubits, x_mask, z_mask, phase)

    def _same_size(self, other: "PauliAxis") -> None:
        if not isinstance(other, PauliAxis):
            raise TypeError("Pauli operation requires another PauliAxis")
        if self.num_qubits != other.num_qubits:
            raise ValueError("Pauli axes must have the same number of qubits")

    @classmethod
    def from_binary_phase(
        cls,
        num_qubits: int,
        x_mask: int,
        z_mask: int,
        phase: int,
    ) -> "PauliAxis":
        """Build a Hermitian signed axis from ``i**phase X**x Z**z``."""
        base_phase = (x_mask & z_mask).bit_count() & 3
        delta = (int(phase) - base_phase) & 3
        if delta == 0:
            sign = 1
        elif delta == 2:
            sign = -1
        else:
            raise ValueError("binary Pauli phase is not Hermitian")
        return cls(num_qubits, x_mask, z_mask, sign)

    def to_matrix(self):
        """Materialise a dense matrix for tiny-instance tests/debugging only."""
        import numpy as np

        local = {
            (0, 0): np.eye(2, dtype=np.complex128),
            (1, 0): np.array([[0, 1], [1, 0]], dtype=np.complex128),
            (0, 1): np.array([[1, 0], [0, -1]], dtype=np.complex128),
            (1, 1): np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        }
        matrix = np.array([[1.0 + 0.0j]])
        # Kronecker's left-most factor is the most-significant wire, whereas
        # this codebase represents q=0 with the least-significant bit.
        for qubit in reversed(range(self.num_qubits)):
            local_key = ((self.x_mask >> qubit) & 1, (self.z_mask >> qubit) & 1)
            matrix = np.kron(matrix, local[local_key])
        return self.sign * matrix

    def __str__(self) -> str:
        labels = []
        for qubit in range(self.num_qubits):
            x = (self.x_mask >> qubit) & 1
            z = (self.z_mask >> qubit) & 1
            labels.append({(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}[(x, z)])
        prefix = "" if self.sign == 1 else "-"
        return prefix + "".join(reversed(labels))


def multiply_binary_paulis(
    left_x: int,
    left_z: int,
    left_phase: int,
    right_x: int,
    right_z: int,
    right_phase: int,
) -> Tuple[int, int, int]:
    """Multiply two phase-aware binary Paulis without assuming commutation."""
    phase = (
        int(left_phase)
        + int(right_phase)
        + 2 * _parity(left_z & right_x)
    ) & 3
    return left_x ^ right_x, left_z ^ right_z, phase


def transform_axis(
    axis: PauliAxis,
    x_images: Sequence[PauliAxis],
    z_images: Sequence[PauliAxis],
) -> PauliAxis:
    """Transport ``axis`` through a Clifford specified by generator images.

    ``x_images[q]`` is the image of ``X_q`` and ``z_images[q]`` the image of
    ``Z_q``.  The original binary phase and generator order are retained, so
    signs such as ``H Y H = -Y`` emerge exactly rather than by special cases.
    """
    n = axis.num_qubits
    if len(x_images) != n or len(z_images) != n:
        raise ValueError("one X and Z image is required per qubit")
    if any(image.num_qubits != n for image in (*x_images, *z_images)):
        raise ValueError("generator images have incompatible dimensions")

    x_mask = z_mask = phase = 0
    for qubit in range(n):
        if (axis.x_mask >> qubit) & 1:
            image = x_images[qubit]
            x_mask, z_mask, phase = multiply_binary_paulis(
                x_mask,
                z_mask,
                phase,
                image.x_mask,
                image.z_mask,
                image.binary_phase,
            )
    for qubit in range(n):
        if (axis.z_mask >> qubit) & 1:
            image = z_images[qubit]
            x_mask, z_mask, phase = multiply_binary_paulis(
                x_mask,
                z_mask,
                phase,
                image.x_mask,
                image.z_mask,
                image.binary_phase,
            )

    phase = (phase + axis.binary_phase) & 3
    return PauliAxis.from_binary_phase(n, x_mask, z_mask, phase)


def _elementary_images(
    num_qubits: int,
    name: str,
    qubits: Sequence[int],
) -> tuple[tuple[PauliAxis, ...], tuple[PauliAxis, ...]]:
    """Generator images for an elementary self-inverse / Clifford gate."""
    x_images = [PauliAxis.x_axis(num_qubits, q) for q in range(num_qubits)]
    z_images = [PauliAxis.z_axis(num_qubits, q) for q in range(num_qubits)]
    normalized = name.upper()

    if normalized == "H":
        (qubit,) = tuple(qubits)
        x_images[qubit], z_images[qubit] = z_images[qubit], x_images[qubit]
    elif normalized == "S":
        (qubit,) = tuple(qubits)
        x_images[qubit] = PauliAxis.y_axis(num_qubits, qubit)
    elif normalized in {"SDG", "S_DAG", "SDAG"}:
        (qubit,) = tuple(qubits)
        x_images[qubit] = PauliAxis.y_axis(num_qubits, qubit).negated()
    elif normalized == "X":
        (qubit,) = tuple(qubits)
        z_images[qubit] = z_images[qubit].negated()
    elif normalized == "CNOT":
        control, target = tuple(qubits)
        x_images[control] = PauliAxis(
            num_qubits,
            (1 << control) | (1 << target),
            0,
        )
        z_images[target] = PauliAxis(
            num_qubits,
            0,
            (1 << control) | (1 << target),
        )
    else:
        raise ValueError(f"unsupported Clifford gate {name!r}")
    return tuple(x_images), tuple(z_images)


def conjugate_axis_by_gate(
    axis: PauliAxis,
    name: str,
    qubits: Sequence[int],
) -> PauliAxis:
    """Return ``G axis G†`` for one supported Clifford gate ``G``."""
    images = _elementary_images(axis.num_qubits, name, qubits)
    return transform_axis(axis, *images)


__all__ = [
    "PauliAxis",
    "conjugate_axis_by_gate",
    "multiply_binary_paulis",
    "transform_axis",
]
