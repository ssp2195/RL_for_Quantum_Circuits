"""Exact quarter-turn rotations about signed Pauli axes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

from algebra.pauli import PauliAxis


@dataclass(frozen=True, slots=True)
class PauliRotation:
    """``R_axis(quarter_turns * pi / 4)`` without floating point angles."""

    axis: PauliAxis
    quarter_turns: int

    def __post_init__(self) -> None:
        if isinstance(self.quarter_turns, bool) or not isinstance(self.quarter_turns, int):
            raise TypeError("quarter_turns must be an integer")

    @property
    def is_identity(self) -> bool:
        return self.quarter_turns == 0

    def canonical_payload(self) -> Tuple[int, int, int, int, int]:
        return (*self.axis.canonical_payload(), self.quarter_turns)

    payload = canonical_payload

    def with_turns(self, quarter_turns: int) -> "PauliRotation":
        return PauliRotation(self.axis, quarter_turns)

    def sign_normalized(self) -> "PauliRotation":
        """Use a positive axis via ``R_-P(theta) = R_P(-theta)``."""
        if self.axis.sign == 1:
            return self
        return PauliRotation(self.axis.positive_axis(), -self.quarter_turns)

    def to_matrix(self):
        """Materialise this rotation for tiny-instance differential tests."""
        import numpy as np

        theta = self.quarter_turns * np.pi / 4.0
        dimension = 1 << self.axis.num_qubits
        return (
            np.cos(theta / 2.0) * np.eye(dimension, dtype=np.complex128)
            - 1j * np.sin(theta / 2.0) * self.axis.to_matrix()
        )


def _reduce_turns(rotation: PauliRotation) -> tuple[PauliRotation | None, int]:
    """Reduce an angle modulo 2*pi and return its global-phase adjustment."""
    rotation = rotation.sign_normalized()
    turns = rotation.quarter_turns
    # R_I(k*pi/4) is a pure global phase exp(-i*k*pi/8), not a meaningful
    # rotation-word element.  Removing it also makes phase-sensitive and
    # phase-quotient canonical modes agree on the structural word.
    if rotation.axis.is_identity:
        return None, -turns
    remainder = turns % 8
    # A symmetric-ish range (-3, 4] keeps signs deterministic.  Both +/-4
    # represent a pi rotation; selecting +4 avoids a second representation.
    if remainder > 4:
        remainder -= 8
    full_turns = (turns - remainder) // 8
    phase_delta_eighths = 8 * full_turns
    if remainder == 0:
        return None, phase_delta_eighths
    return PauliRotation(rotation.axis, remainder), phase_delta_eighths


def normalize_rotation_word(
    rotations: Iterable[PauliRotation],
) -> tuple[tuple[PauliRotation, ...], int]:
    """Conservatively normalise an ordered Pauli-rotation word.

    This routine uses only exact identities:

    * sign transport, ``R_-P(a) = R_P(-a)``;
    * same-axis fusion;
    * deletion of zero rotations; and
    * adjacent swaps when (and only when) the two axes commute.

    It intentionally does not claim complete equality for arbitrary words
    containing anticommuting axes.  Such incompleteness leaves extra search
    states but cannot cause a false canonical merge.
    """
    word: list[PauliRotation] = []
    phase_delta = 0
    for rotation in rotations:
        if not isinstance(rotation, PauliRotation):
            raise TypeError("rotation words must contain PauliRotation values")
        reduced, delta = _reduce_turns(rotation)
        phase_delta += delta
        if reduced is not None:
            word.append(reduced)

    # Bubble commuting rotations into a deterministic order.  When equal
    # axes meet they fuse, perhaps creating another cancellation opportunity,
    # so repeat until no exact local rewrite applies.
    changed = True
    while changed:
        changed = False
        index = 0
        while index < len(word) - 1:
            left, right = word[index], word[index + 1]
            if left.axis == right.axis:
                merged, delta = _reduce_turns(
                    PauliRotation(left.axis, left.quarter_turns + right.quarter_turns)
                )
                phase_delta += delta
                if merged is None:
                    del word[index : index + 2]
                    index = max(0, index - 1)
                else:
                    word[index : index + 2] = [merged]
                    index = max(0, index - 1)
                changed = True
                continue

            if (
                left.axis.commutes_with(right.axis)
                and left.axis.sort_key() > right.axis.sort_key()
            ):
                word[index], word[index + 1] = right, left
                changed = True
                index = max(0, index - 1)
                continue
            index += 1

    return tuple(word), phase_delta


def normalize_semantics(
    rotations: Iterable[PauliRotation],
    global_phase_eighths: int = 0,
) -> tuple[tuple[PauliRotation, ...], int]:
    """Normalise a rotation word and fold its exact phase into ``phi``."""
    word, phase_delta = normalize_rotation_word(rotations)
    return word, (int(global_phase_eighths) + phase_delta) % 16


__all__ = [
    "PauliRotation",
    "normalize_rotation_word",
    "normalize_semantics",
]
