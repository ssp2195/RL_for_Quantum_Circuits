"""Ordered exact Pauli rotations with incremental conservative normalization."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .pauli import Pauli


@dataclass(frozen=True, slots=True)
class PauliRotation:
    """``R_P(k*pi/4) = exp(-i k*pi P/8)``."""

    axis: Pauli
    quarter_turns: int

    def __post_init__(self) -> None:
        if isinstance(self.quarter_turns, bool) or not isinstance(self.quarter_turns, int):
            raise TypeError("quarter_turns must be an integer")

    def canonical_payload(self) -> tuple[int, int, int]:
        axis = self.axis.positive_axis()
        turns = self.quarter_turns if self.axis.sign > 0 else -self.quarter_turns
        return axis.x_mask, axis.z_mask, turns

    def to_matrix(self) -> np.ndarray:
        theta = self.quarter_turns * np.pi / 4.0
        dimension = 1 << self.axis.num_qubits
        return (
            np.cos(theta / 2.0) * np.eye(dimension, dtype=np.complex128)
            - 1j * np.sin(theta / 2.0) * self.axis.to_matrix()
        )


def reduce_rotation(rotation: PauliRotation) -> tuple[PauliRotation | None, int]:
    """Normalize sign/angle and return a global-phase change in eighth-turns."""
    axis = rotation.axis
    turns = rotation.quarter_turns
    if axis.sign < 0:
        axis = axis.positive_axis()
        turns = -turns
    if axis.is_identity:
        return None, -turns
    remainder = turns % 8
    if remainder > 4:
        remainder -= 8
    full_turns = (turns - remainder) // 8
    phase_delta = 8 * full_turns
    if remainder == 0:
        return None, phase_delta
    return PauliRotation(axis, remainder), phase_delta


def insert_rotation(
    word: tuple[PauliRotation, ...], rotation: PauliRotation
) -> tuple[tuple[PauliRotation, ...], int]:
    """Insert one new leftmost factor in O(len(word)).

    Only exact local identities are used: sign normalization, angle reduction,
    same-axis fusion, and swaps across commuting factors.  The new factor stops
    at the first anticommuting barrier.  Existing factors are not globally
    rescanned.
    """
    reduced, phase_delta = reduce_rotation(rotation)
    if reduced is None:
        return word, phase_delta
    items = list(word)
    position = 0
    while position < len(items):
        current = items[position]
        if not reduced.axis.commutes_with(current.axis):
            break
        if reduced.axis == current.axis:
            merged, delta = reduce_rotation(
                PauliRotation(
                    reduced.axis,
                    reduced.quarter_turns + current.quarter_turns,
                )
            )
            phase_delta += delta
            if merged is None:
                del items[position]
            else:
                items[position] = merged
            return tuple(items), phase_delta
        if current.axis.sort_key() < reduced.axis.sort_key():
            position += 1
            continue
        break
    items.insert(position, reduced)
    return tuple(items), phase_delta


def normalize_word_reference(
    rotations: Iterable[PauliRotation],
) -> tuple[tuple[PauliRotation, ...], int]:
    """Slow reference normalizer used only by bounded differential tests."""
    word: tuple[PauliRotation, ...] = ()
    phase = 0
    values = tuple(rotations)
    for rotation in reversed(values):
        word, delta = insert_rotation(word, rotation)
        phase += delta
    return word, phase


def anticommuting_pair_count(word: tuple[PauliRotation, ...]) -> int:
    return sum(
        not left.axis.commutes_with(right.axis)
        for index, left in enumerate(word)
        for right in word[index + 1 :]
    )


def mean_pauli_weight(word: tuple[PauliRotation, ...]) -> float:
    if not word:
        return 0.0
    return float(sum(rotation.axis.weight for rotation in word) / len(word))


__all__ = [
    "PauliRotation",
    "anticommuting_pair_count",
    "insert_rotation",
    "mean_pauli_weight",
    "normalize_word_reference",
    "reduce_rotation",
]
