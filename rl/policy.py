"""A small, transparent semi-gradient SARSA baseline."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from circuit.circuit_state import CircuitState
from rl.features import extract_features, feature_dimension
from search.node import SearchNode


class LinearQPolicy:
    """Shared linear scorer for persistent frontier records.

    The historical class name is retained for callers, but its TD update is
    on-policy SARSA rather than max-bootstrap Q-learning.
    """

    def __init__(
        self,
        feature_dim: Optional[int] = None,
        lr: float = 1e-3,
        gamma: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.feature_dim = feature_dimension() if feature_dim is None else int(feature_dim)
        self.theta = np.zeros(self.feature_dim, dtype=np.float64)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)

    def q_value(
        self,
        state: CircuitState,
        frontier: Optional[Iterable[object]] = None,
    ) -> float:
        features = extract_features(state, frontier)
        if features.shape[0] != self.theta.shape[0]:
            raise ValueError(
                f"feature dimension changed from {self.theta.shape[0]} to {features.shape[0]}"
            )
        return float(np.dot(self.theta, features))

    def node_value(
        self,
        node: SearchNode,
        frontier: Optional[Iterable[object]] = None,
    ) -> float:
        return self.q_value(node.state, frontier)

    def evaluate_nodes(self, nodes: Sequence[SearchNode]):
        return [(node, self.node_value(node, nodes)) for node in nodes]

    @staticmethod
    def _stable_id(node: SearchNode) -> int:
        return int(getattr(node, "record_id", 0) or 0)

    def select_node(
        self,
        nodes: Sequence[SearchNode],
        epsilon: float = 0.1,
    ) -> Optional[SearchNode]:
        """Choose a frontier record with reproducible epsilon-greedy ties."""
        if not nodes:
            return None
        if self.rng.random() < epsilon:
            return nodes[int(self.rng.integers(len(nodes)))]

        values = [(self.node_value(node, nodes), self._stable_id(node), node) for node in nodes]
        # Stable record IDs make ordering changes in a heap irrelevant.
        return max(values, key=lambda entry: (entry[0], -entry[1]))[2]

    def update(
        self,
        state: CircuitState,
        reward: float,
        next_frontier: Optional[Sequence[SearchNode]] = None,
        done: bool = False,
        *,
        next_node: Optional[SearchNode] = None,
        frontier: Optional[Sequence[SearchNode]] = None,
    ) -> float:
        """Apply one semi-gradient SARSA(0) update and return its TD error.

        ``next_node`` must be the actual next epsilon-greedy choice made by
        the behavior policy.  For legacy callers that omit it, the update is
        terminal rather than silently reverting to max-bootstrap Q-learning.
        """
        current_frontier = frontier if frontier is not None else [state]
        phi = extract_features(state, current_frontier).astype(np.float64)
        q_value = float(np.dot(self.theta, phi))

        if done or next_node is None:
            target = float(reward)
        else:
            target = float(reward) + self.gamma * self.node_value(next_node, next_frontier)

        td_error = target - q_value
        self.theta += self.lr * td_error * phi
        return float(td_error)

    def score_state(self, state: CircuitState) -> float:
        """Lower queue priorities correspond to higher learned values."""
        return -self.q_value(state)


# Explicit alias for readers who prefer the learner's actual algorithm name.
LinearSarsaPolicy = LinearQPolicy
