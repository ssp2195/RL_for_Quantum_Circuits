from typing import Tuple

from circuit.circuit_state import CircuitState
from canonical.hash import stable_hash


class Canonicalizer:
    """
    Two-level hashing:
        1. Identity hash (unitary equivalence)
        2. Resource hash (cost signature)
    """

    # ---------- Public API ----------

    def identity_hash(self, state: CircuitState) -> str:
        canonical = self._identity_canonical_form(state)
        return stable_hash(self._serialize(canonical))

    def resource_hash(self, state: CircuitState) -> str:
        canonical = self._resource_canonical_form(state)
        return stable_hash(self._serialize(canonical))

    def combined_hash(self, state: CircuitState) -> str:
        """
        Useful for debugging / indexing:
        combines identity + resource
        """
        return stable_hash(
            self.identity_hash(state) + self.resource_hash(state)
        )

    # ---------- Identity Canonicalization ----------

    def _identity_canonical_form(self, state: CircuitState) -> Tuple:
        """
        ONLY functional behavior (unitary)
        """
        phase_repr = self._canonicalize_phase(state)

        # Tableau intentionally ignored (Stage 4 simplification)
        return (phase_repr,)

    # ---------- Resource Canonicalization ----------

    def _resource_canonical_form(self, state: CircuitState) -> Tuple:
        """
        ONLY cost / structure (NOT equivalence)
        """
        return (
            state.t_count,
            state.depth,
            state.num_gates,
        )

    # ---------- Phase Polynomial ----------

    def _canonicalize_phase(self, state: CircuitState) -> Tuple:
        pp = state.phase_poly

        if pp is None or not pp.terms:
            return ()

        terms = sorted(
            (mask, coeff % 8)
            for mask, coeff in pp.terms.items()
            if coeff % 8 != 0
        )

        return tuple(terms)

    # ---------- Serialization ----------

    def _serialize(self, canonical: Tuple) -> str:
        return repr(canonical)
