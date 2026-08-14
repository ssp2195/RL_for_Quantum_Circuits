"""Continuation-safe canonical keys for Clifford+T search records.

The digest exposed by :meth:`identity_hash` is deliberately only an index.
Search code must compare :meth:`semantic_key` values, never hashes, before
merging records.  This keeps a hypothetical digest collision from becoming a
semantic error.
"""

from __future__ import annotations

from typing import Any, Tuple

from algebra.pauli_rotation import normalize_rotation_word
from canonical.hash import stable_hash
from circuit.circuit_state import CircuitState


class Canonicalizer:
    """Build immutable, exact symbolic keys for a circuit state.

    The default is physical unitary identity, i.e. global phase is quotiented
    out.  ``phase_sensitive=True`` is useful for diagnostics and tests but is
    intentionally not the default synthesis equivalence relation.
    """

    SCHEMA = "clifford-pauli-rotation-v1"

    def __init__(self, phase_sensitive: bool = False):
        self.phase_sensitive = phase_sensitive

    # ---------- Public API ----------

    def semantic_key(
        self,
        state: CircuitState,
        *,
        include_global_phase: bool | None = None,
    ) -> Tuple[Any, ...]:
        """Return the full immutable semantic payload, not a digest."""
        if include_global_phase is None:
            include_global_phase = self.phase_sensitive

        frame = getattr(state, "frame", None)
        if frame is None:
            # This fallback only supports an empty pre-migration state.  New
            # CircuitState instances always install a complete frame.
            frame_payload: Tuple[Any, ...] = ("identity", state.dag.num_qubits)
        elif hasattr(frame, "canonical_payload"):
            frame_payload = tuple(frame.canonical_payload())
        elif hasattr(frame, "stable_payload"):
            frame_payload = tuple(frame.stable_payload())
        else:  # pragma: no cover - defensive compatibility boundary
            raise TypeError("Clifford frame does not expose an immutable payload")

        normalized_rotations, phase_delta = normalize_rotation_word(
            getattr(state, "rotations", ())
        )
        rotations = tuple(
            self._rotation_payload(rotation)
            for rotation in normalized_rotations
        )

        interface = self._continuation_interface(state)
        key: Tuple[Any, ...] = (
            self.SCHEMA,
            state.dag.num_qubits,
            frame_payload,
            rotations,
            interface,
        )

        if include_global_phase:
            # Frame images are projective.  Add an exact (though deliberately
            # conservative) primitive-frame lift so literal mode cannot merge
            # two raw Clifford words whose only difference is global phase.
            frame_lift = (
                tuple(frame.phase_sensitive_payload())
                if frame is not None and hasattr(frame, "phase_sensitive_payload")
                else ()
            )
            # e^(i p pi/8) has period 16.
            key += (
                frame_lift,
                (int(getattr(state, "global_phase_eighths", 0)) + phase_delta) % 16,
            )
        return key

    # Alias with a name that makes its purpose clear to archive callers.
    identity_key = semantic_key

    def identity_hash(self, state: CircuitState) -> str:
        return stable_hash(self._serialize(self.semantic_key(state)))

    def resource_hash(self, state: CircuitState) -> str:
        return stable_hash(self._serialize(self._resource_canonical_form(state)))

    def combined_hash(self, state: CircuitState) -> str:
        return stable_hash(
            self._serialize((self.semantic_key(state), self._resource_canonical_form(state)))
        )

    # ---------- Compatibility helpers ----------

    def _identity_canonical_form(self, state: CircuitState) -> Tuple[Any, ...]:
        return self.semantic_key(state)

    def _resource_canonical_form(self, state: CircuitState) -> Tuple[int, ...]:
        """A continuation-monotone resource vector.

        Per-wire depth, rather than a scalar depth alone, is retained because
        a suffix can run in parallel with work on other wires.
        """
        wire_depths = tuple(int(depth) for depth in getattr(state, "wire_depths", ()))
        if not wire_depths:
            # Compatibility for a state constructed by old external code.
            wire_depths = tuple([int(getattr(state, "depth", 0))] * state.dag.num_qubits)
        return (
            int(getattr(state, "t_count", 0)),
            int(getattr(state, "two_qubit_count", 0)),
            int(getattr(state, "num_gates", 0)),
            *wire_depths,
        )

    # ---------- Payload helpers ----------

    @staticmethod
    def _rotation_payload(rotation: Any) -> Tuple[Any, ...]:
        if hasattr(rotation, "canonical_payload"):
            return tuple(rotation.canonical_payload())
        if hasattr(rotation, "payload"):
            return tuple(rotation.payload())
        axis = rotation.axis
        axis_payload = (
            int(axis.num_qubits),
            int(axis.x_mask),
            int(axis.z_mask),
            int(axis.sign),
        )
        quarter_turns = int(getattr(rotation, "quarter_turns"))
        return axis_payload + (quarter_turns,)

    @staticmethod
    def _continuation_interface(state: CircuitState) -> Tuple[Any, ...]:
        explicit = getattr(state, "continuation_interface", None)
        if explicit is not None:
            return tuple(explicit)

        budget = state.budget
        # The initial repository has logical labelled qubits, all-to-all
        # CNOTs, no ancillas, and a fixed gate grammar.  Budget *limits* are
        # part of the instance, while consumed resources remain in the Pareto
        # record instead of this semantic key.
        return (
            "all-to-all",
            "no-ancilla",
            int(budget.max_t_count),
            int(budget.max_depth),
            int(budget.max_gates),
            None if budget.max_two_qubit_count is None else int(budget.max_two_qubit_count),
        )

    @staticmethod
    def _serialize(canonical: Tuple[Any, ...]) -> str:
        return repr(canonical)
