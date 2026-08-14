"""Backward-compatible adapter for the repository's unrestricted gate search."""

from __future__ import annotations

from collections.abc import Mapping

from canonical.canonicalizer import Canonicalizer
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from search.action_space import generate_actions
from search.expansion import expand_node
from search.node import SearchNode


class NativeGateSearchProblem:
    """Preserve the historical all-native-gates expansion behavior exactly."""

    name = "native-gate-search"
    schema_version = "native-gate-search-v1"

    def initial_state(self, config: object) -> CircuitState:
        return CircuitState(CircuitDAG(int(getattr(config, "num_qubits"))), getattr(config, "budget"))

    def analyze(self, state: CircuitState) -> tuple[object, ...]:
        """Native search has no additional continuation state beyond semantics."""

        return ()

    def expand(self, node: SearchNode) -> list[SearchNode]:
        return expand_node(
            node,
            generate_actions(node.state.dag.num_qubits),
            policy=None,
        )

    def canonicalizer(self, *, phase_sensitive: bool = False) -> Canonicalizer:
        return Canonicalizer(phase_sensitive=phase_sensitive)

    def is_terminal_candidate(self, node: SearchNode) -> bool:
        """Keep generic behavior: every native child is dense-certified."""

        return True

    def metadata(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "continuation_model": "all-native-clifford-t",
        }
