"""Bit-packed signed Hermitian Pauli operators.

The convention is

    P(x,z,s) = s i^popcount(x & z) X^x Z^z,

with qubit 0 stored in the least-significant bit.  All search-time algebra is
integer and exact; dense matrices are confined to tests and certification.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence


def _parity(mask: int) -> int:
    return int(mask).bit_count() & 1


@dataclass(frozen=True, slots=True)
class Pauli:
    num_qubits: int
    x_mask: int = 0
    z_mask: int = 0
    sign: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.num_qubits, bool) or not isinstance(self.num_qubits, int):
            raise TypeError("num_qubits must be an integer")
        if self.num_qubits < 0:
            raise ValueError("num_qubits must be non-negative")
        limit = 1 << self.num_qubits
        for name, value in (("x_mask", self.x_mask), ("z_mask", self.z_mask)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0 or value >= limit:
                raise ValueError(f"{name} has a bit outside the register")
        if self.sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1")

    @classmethod
    def identity(cls, n: int) -> "Pauli":
        return cls(n)

    @classmethod
    def x_axis(cls, n: int, q: int) -> "Pauli":
        _validate_qubit(n, q)
        return cls(n, 1 << q, 0)

    @classmethod
    def y_axis(cls, n: int, q: int) -> "Pauli":
        _validate_qubit(n, q)
        bit = 1 << q
        return cls(n, bit, bit)

    @classmethod
    def z_axis(cls, n: int, q: int) -> "Pauli":
        _validate_qubit(n, q)
        return cls(n, 0, 1 << q)

    @property
    def weight(self) -> int:
        return (self.x_mask | self.z_mask).bit_count()

    @property
    def is_identity(self) -> bool:
        return self.x_mask == 0 and self.z_mask == 0

    @property
    def binary_phase(self) -> int:
        return ((self.x_mask & self.z_mask).bit_count() + (0 if self.sign == 1 else 2)) & 3

    def commutes_with(self, other: "Pauli") -> bool:
        self._same_width(other)
        return _parity((self.x_mask & other.z_mask) ^ (self.z_mask & other.x_mask)) == 0

    def anticommutes_with(self, other: "Pauli") -> bool:
        return not self.commutes_with(other)

    def positive_axis(self) -> "Pauli":
        return self if self.sign == 1 else Pauli(
            self.num_qubits, self.x_mask, self.z_mask, 1
        )

    def negated(self) -> "Pauli":
        return Pauli(self.num_qubits, self.x_mask, self.z_mask, -self.sign)

    def sort_key(self) -> tuple[int, int]:
        return self.x_mask, self.z_mask

    def axis_payload(self) -> tuple[int, int, int]:
        return self.x_mask, self.z_mask, self.sign

    def payload(self) -> tuple[int, int, int, int]:
        return self.num_qubits, self.x_mask, self.z_mask, self.sign

    def _same_width(self, other: "Pauli") -> None:
        if not isinstance(other, Pauli) or self.num_qubits != other.num_qubits:
            raise ValueError("Pauli operators have incompatible widths")

    @classmethod
    def from_binary_phase(
        cls, n: int, x_mask: int, z_mask: int, phase: int
    ) -> "Pauli":
        base = (x_mask & z_mask).bit_count() & 3
        delta = (int(phase) - base) & 3
        if delta == 0:
            sign = 1
        elif delta == 2:
            sign = -1
        else:
            raise ValueError("binary phase is not Hermitian")
        return cls(n, x_mask, z_mask, sign)

    def to_matrix(self):
        import numpy as np

        local = {
            (0, 0): np.eye(2, dtype=np.complex128),
            (1, 0): np.array([[0, 1], [1, 0]], dtype=np.complex128),
            (0, 1): np.array([[1, 0], [0, -1]], dtype=np.complex128),
            (1, 1): np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        }
        result = np.array([[1.0 + 0.0j]])
        for q in reversed(range(self.num_qubits)):
            key = ((self.x_mask >> q) & 1, (self.z_mask >> q) & 1)
            result = np.kron(result, local[key])
        return self.sign * result


def _validate_qubit(n: int, q: int) -> None:
    if isinstance(q, bool) or not isinstance(q, int) or not 0 <= q < n:
        raise ValueError(f"qubit {q!r} is outside a {n}-qubit register")


def multiply_binary_paulis(
    left_x: int,
    left_z: int,
    left_phase: int,
    right_x: int,
    right_z: int,
    right_phase: int,
) -> tuple[int, int, int]:
    phase = (
        int(left_phase)
        + int(right_phase)
        + 2 * _parity(left_z & right_x)
    ) & 3
    return left_x ^ right_x, left_z ^ right_z, phase


def transform_pauli(
    pauli: Pauli,
    x_images: Sequence[Pauli],
    z_images: Sequence[Pauli],
) -> Pauli:
    """Transport ``pauli`` through a Clifford generator-image map.

    The public wrapper accepts any finite sequence, while the cached kernel
    operates on immutable tuples.  A three-qubit run repeatedly transports the
    same small set of Pauli generators through the same tableaus, so this cache
    removes a large amount of duplicate integer algebra without changing the
    mathematical state or key.
    """

    return _transform_pauli_cached(pauli, tuple(x_images), tuple(z_images))


@lru_cache(maxsize=262_144)
def _transform_pauli_cached(
    pauli: Pauli,
    x_images: tuple[Pauli, ...],
    z_images: tuple[Pauli, ...],
) -> Pauli:
    n = pauli.num_qubits
    if len(x_images) != n or len(z_images) != n:
        raise ValueError("one X and Z image is required per qubit")
    x_mask = z_mask = phase = 0
    for q in range(n):
        if (pauli.x_mask >> q) & 1:
            image = x_images[q]
            x_mask, z_mask, phase = multiply_binary_paulis(
                x_mask, z_mask, phase,
                image.x_mask, image.z_mask, image.binary_phase,
            )
    for q in range(n):
        if (pauli.z_mask >> q) & 1:
            image = z_images[q]
            x_mask, z_mask, phase = multiply_binary_paulis(
                x_mask, z_mask, phase,
                image.x_mask, image.z_mask, image.binary_phase,
            )
    phase = (phase + pauli.binary_phase) & 3
    return Pauli.from_binary_phase(n, x_mask, z_mask, phase)


@lru_cache(maxsize=None)
def _elementary_images(
    n: int, name: str, qubits: tuple[int, ...]
) -> tuple[tuple[Pauli, ...], tuple[Pauli, ...]]:
    xs = [Pauli.x_axis(n, q) for q in range(n)]
    zs = [Pauli.z_axis(n, q) for q in range(n)]
    operands = tuple(int(q) for q in qubits)
    name = name.upper()
    if name == "H":
        (q,) = operands
        xs[q], zs[q] = zs[q], xs[q]
    elif name == "S":
        (q,) = operands
        xs[q] = Pauli.y_axis(n, q)
    elif name == "SDG":
        (q,) = operands
        xs[q] = Pauli.y_axis(n, q).negated()
    elif name == "CNOT":
        control, target = operands
        xs[control] = Pauli(n, (1 << control) | (1 << target), 0)
        zs[target] = Pauli(n, 0, (1 << control) | (1 << target))
    else:
        raise ValueError(f"unsupported Clifford gate {name!r}")
    return tuple(xs), tuple(zs)


def conjugate_by_gate(pauli: Pauli, name: str, qubits: Sequence[int]) -> Pauli:
    return _conjugate_by_gate_cached(pauli, str(name).upper(), tuple(qubits))


@lru_cache(maxsize=131_072)
def _conjugate_by_gate_cached(
    pauli: Pauli,
    name: str,
    qubits: tuple[int, ...],
) -> Pauli:
    return transform_pauli(pauli, *_elementary_images(pauli.num_qubits, name, qubits))


def conjugate_by_pauli_rotation(
    pauli: Pauli,
    axis: Pauli,
    quarter_turns: int,
) -> Pauli:
    """Conjugate a Pauli by an even-quarter-turn Pauli rotation.

    The unitary is

    ``R_axis(k*pi/4) = exp(-i k*pi axis/8)``.

    Even values of ``k`` are Clifford operations, so conjugation maps every
    Hermitian Pauli to another signed Hermitian Pauli.  This helper is used by
    the stronger projective canonicalizer to move Clifford-valued Pauli
    rotations into the Clifford tableau without introducing dense matrices.
    """

    if not isinstance(pauli, Pauli) or not isinstance(axis, Pauli):
        raise TypeError("pauli and axis must be Pauli instances")
    if pauli.num_qubits != axis.num_qubits:
        raise ValueError("Pauli operators have incompatible widths")
    if isinstance(quarter_turns, bool) or not isinstance(quarter_turns, int):
        raise TypeError("quarter_turns must be an integer")
    if axis.sign < 0:
        axis = axis.positive_axis()
        quarter_turns = -quarter_turns
    turns = quarter_turns % 8
    if turns & 1:
        raise ValueError("only Clifford-valued even quarter turns are supported")
    return _conjugate_by_pauli_rotation_cached(pauli, axis, turns)


@lru_cache(maxsize=262_144)
def _conjugate_by_pauli_rotation_cached(
    pauli: Pauli,
    axis: Pauli,
    turns_mod_8: int,
) -> Pauli:
    if turns_mod_8 == 0 or pauli.commutes_with(axis):
        return pauli
    if turns_mod_8 == 4:
        return pauli.negated()

    # For anticommuting P,Q,
    #   R_P(pi/2) Q R_P(-pi/2) = -i P Q,
    #   R_P(-pi/2) Q R_P(pi/2) = +i P Q.
    x_mask, z_mask, phase = multiply_binary_paulis(
        axis.x_mask,
        axis.z_mask,
        axis.binary_phase,
        pauli.x_mask,
        pauli.z_mask,
        pauli.binary_phase,
    )
    scalar_phase = 3 if turns_mod_8 == 2 else 1
    return Pauli.from_binary_phase(
        pauli.num_qubits,
        x_mask,
        z_mask,
        (phase + scalar_phase) & 3,
    )


def clear_algebra_caches() -> None:
    """Clear performance caches for deterministic microbenchmarks/tests."""

    _transform_pauli_cached.cache_clear()
    _elementary_images.cache_clear()
    _conjugate_by_gate_cached.cache_clear()
    _conjugate_by_pauli_rotation_cached.cache_clear()



INVERSE_CLIFFORD_GATE = {"H": "H", "S": "SDG", "SDG": "S", "CNOT": "CNOT"}


__all__ = [
    "INVERSE_CLIFFORD_GATE",
    "Pauli",
    "clear_algebra_caches",
    "conjugate_by_gate",
    "conjugate_by_pauli_rotation",
    "multiply_binary_paulis",
    "transform_pauli",
]
