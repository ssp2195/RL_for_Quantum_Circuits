from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from circuit.circuit_state import CircuitState


@dataclass(order=True)
class SearchNode:
    """
    Node in search tree
    """

    priority: float
    state: CircuitState = field(compare=False)
    parent: Optional["SearchNode"] = field(default=None, compare=False)
    action: Optional[object] = field(default=None, compare=False)

    # cached features (for RL later)
    features: Optional[np.ndarray] = field(default=None, compare=False)

    def __repr__(self):
        return f"Node(priority={self.priority:.4f}, state={self.state})"
