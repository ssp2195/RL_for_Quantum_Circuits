from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, List, Optional
import numpy as np

if TYPE_CHECKING:
    from circuit.circuit_state import CircuitState


@dataclass(order=True)
class SearchNode:
    """A concrete circuit witness in the search tree.

    ``record_id`` is assigned by :class:`search.frontier.Frontier` when the
    node enters its Pareto archive.  It deliberately identifies one resource
    witness, rather than the semantic state itself: several incomparable
    records may have the same semantic key.
    """

    priority: float
    state: CircuitState = field(compare=False)
    parent: Optional["SearchNode"] = field(default=None, compare=False)
    action: Optional[object] = field(default=None, compare=False)

    # cached features (for RL later)
    features: Optional[np.ndarray] = field(default=None, compare=False)

    # Assigned on archive insertion.  ``expanded`` mirrors the archive record
    # for callers that operate on the legacy node-only frontier interface.
    record_id: Optional[int] = field(default=None, compare=False)
    expanded: bool = field(default=False, compare=False)

    def iter_path(self) -> Iterator["SearchNode"]:
        """Yield root-to-self nodes for this concrete witness."""
        reverse_path: List[SearchNode] = []
        current: Optional[SearchNode] = self
        while current is not None:
            reverse_path.append(current)
            current = current.parent
        yield from reversed(reverse_path)

    def reconstruct_actions(self) -> List[object]:
        """Return the ordered gate actions that produced this node."""
        return [
            node.action
            for node in self.iter_path()
            if node.action is not None
        ]

    # A short alias is convenient for certifiers/exporters that only need the
    # witness, not the intermediate nodes.
    witness_actions = reconstruct_actions

    def __repr__(self):
        record = "-" if self.record_id is None else self.record_id
        return (
            f"Node(record_id={record}, priority={self.priority:.4f}, "
            f"expanded={self.expanded}, state={self.state})"
        )
