"""Small protocol shared by native and constrained search problems.

The frontier, archive, and :class:`~search.node.SearchNode` remain the common
search machinery.  A problem only owns the deterministic parts that genuinely
depend on a constrained synthesis language: initial witness construction,
prefix analysis, child generation, and continuation-safe identity.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Hashable, Protocol, runtime_checkable

if TYPE_CHECKING:
    from circuit.circuit_state import CircuitState
    from search.node import SearchNode


@runtime_checkable
class SearchProblem(Protocol):
    """Deterministic synthesis-problem boundary used by frontier search.

    Implementations must never use an RL score to decide which children are
    legal.  A policy ranks persistent frontier records *after* this protocol
    has enumerated their valid one-gate continuations.
    """

    name: str
    schema_version: str

    def initial_state(self, config: object) -> "CircuitState":
        """Return the authoritative empty witness for one search episode."""

    def analyze(self, state: "CircuitState") -> Hashable:
        """Return immutable continuation information derived from the DAG."""

    def expand(self, node: "SearchNode") -> list["SearchNode"]:
        """Enumerate deterministic, legal one-gate child witnesses."""

    def canonicalizer(self, *, phase_sensitive: bool = False) -> object:
        """Return an archive canonicalizer safe for this problem's suffixes."""

    def is_terminal_candidate(self, node: "SearchNode") -> bool:
        """Return whether this structural child warrants dense certification."""

    def metadata(self) -> Mapping[str, object]:
        """Return stable, JSON-friendly schema and continuation metadata."""
