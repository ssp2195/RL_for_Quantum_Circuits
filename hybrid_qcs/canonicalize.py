"""Deterministic projective canonicalization for hybrid Clifford+T states.

The execution representation deliberately remains incremental: a complete
forward/inverse Clifford tableau followed by an ordered Pauli-rotation word.
This module computes a stronger *view* used for semantic keys.  It therefore
improves duplicate and resource-Pareto pruning without altering the persistent
witness or the transition algebra used to reconstruct a circuit.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from .pauli import conjugate_by_pauli_rotation
from .rotation import PauliRotation, normalize_word_reference
from .tableau import CliffordTableau


@dataclass(frozen=True, slots=True)
class ProjectiveCanonicalization:
    """Canonicalized projective view of one hybrid symbolic state."""

    tableau: CliffordTableau
    rotations: tuple[PauliRotation, ...]
    extracted_clifford_rotations: int
    normalization_passes: int
    discarded_global_phase_eighths: int

    @property
    def payload(self) -> tuple[object, ...]:
        return (
            "hybrid-clifford-pauli-projective-v2",
            self.tableau.num_qubits,
            self.tableau.canonical_payload(),
            tuple(rotation.canonical_payload() for rotation in self.rotations),
        )


def legacy_projective_key(
    num_qubits: int,
    tableau: CliffordTableau,
    rotations: Iterable[PauliRotation],
) -> tuple[object, ...]:
    """Return the previous conservative key for differential validation."""

    values = tuple(rotations)
    return (
        "hybrid-clifford-pauli-incremental-v1",
        num_qubits,
        tableau.canonical_payload(),
        tuple(rotation.canonical_payload() for rotation in values),
    )


def canonicalize_projective(
    tableau: CliffordTableau,
    rotations: Iterable[PauliRotation],
) -> ProjectiveCanonicalization:
    """Compute a stronger sound projective normal view.

    The procedure repeatedly applies two exact operations:

    1. normalize the rotation word under sign reduction, angle reduction,
       same-axis fusion, and commuting swaps;
    2. extract every even-quarter-turn rotation, which is Clifford-valued,
       into the tableau.  A Clifford rotation embedded inside the word is
       moved left by conjugating all earlier Pauli axes by its inverse.

    The procedure is deterministic and terminating because each extraction
    strictly decreases the word length.  It is intentionally not claimed to
    be a complete normal form for arbitrary Clifford+T identities involving
    noncommuting odd rotations.
    """

    if not isinstance(tableau, CliffordTableau):
        raise TypeError("tableau must be a CliffordTableau")
    values = tuple(rotations)
    if any(rotation.axis.num_qubits != tableau.num_qubits for rotation in values):
        raise ValueError("rotation width does not match the tableau")

    return _canonicalize_projective_cached(tableau, values)


@lru_cache(maxsize=262_144)
def _canonicalize_projective_cached(
    tableau: CliffordTableau,
    values: tuple[PauliRotation, ...],
) -> ProjectiveCanonicalization:
    # The execution word is already maintained in the legacy incremental
    # normal order.  If all remaining rotations are odd quarter turns, the
    # strengthened view is identical and no second pass is needed.
    if all(rotation.quarter_turns % 2 != 0 for rotation in values):
        return ProjectiveCanonicalization(
            tableau=tableau,
            rotations=values,
            extracted_clifford_rotations=0,
            normalization_passes=0,
            discarded_global_phase_eighths=0,
        )

    word, phase = normalize_word_reference(values)
    working_tableau = tableau
    extracted = 0
    passes = 1

    while True:
        index = next(
            (
                position
                for position, rotation in enumerate(word)
                if rotation.quarter_turns % 2 == 0
            ),
            None,
        )
        if index is None:
            break

        clifford_rotation = word[index]
        axis = clifford_rotation.axis
        turns = clifford_rotation.quarter_turns

        # R_1 ... R_{i-1} K = K (K^dag R_1 K) ... (K^dag R_{i-1} K).
        conjugated_prefix = tuple(
            PauliRotation(
                conjugate_by_pauli_rotation(
                    rotation.axis,
                    axis,
                    -turns,
                ),
                rotation.quarter_turns,
            )
            for rotation in word[:index]
        )
        working_tableau = working_tableau.right_multiply_pauli_rotation(
            axis,
            turns,
        )
        word, phase_delta = normalize_word_reference(
            (*conjugated_prefix, *word[index + 1 :])
        )
        phase += phase_delta
        extracted += 1
        passes += 1

    if any(rotation.quarter_turns % 2 == 0 for rotation in word):
        raise AssertionError("projective canonicalizer left a Clifford rotation")
    return ProjectiveCanonicalization(
        tableau=working_tableau,
        rotations=word,
        extracted_clifford_rotations=extracted,
        normalization_passes=passes,
        discarded_global_phase_eighths=phase % 16,
    )


def projective_key(
    num_qubits: int,
    tableau: CliffordTableau,
    rotations: Iterable[PauliRotation],
) -> tuple[object, ...]:
    """Return the strengthened sound projective key."""

    if num_qubits != tableau.num_qubits:
        raise ValueError("num_qubits does not match the tableau")
    return canonicalize_projective(tableau, rotations).payload


def clear_canonicalization_cache() -> None:
    """Clear the bounded projective-view cache for deterministic benchmarks."""

    _canonicalize_projective_cached.cache_clear()


__all__ = [
    "ProjectiveCanonicalization",
    "canonicalize_projective",
    "clear_canonicalization_cache",
    "legacy_projective_key",
    "projective_key",
]
