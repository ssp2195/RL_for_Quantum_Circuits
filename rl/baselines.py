"""Additional transparent article baselines over persistent frontier records."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from circuit.circuit_state import CircuitState
from rl.policy import LinearQPolicy
from search.node import SearchNode


class LinearExpectedSarsaPolicy(LinearQPolicy):
    """Linear Expected-SARSA using the same epsilon-greedy record policy."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._behavior_epsilon = 0.1

    def select_node(
        self, nodes: Sequence[SearchNode], epsilon: float = 0.1
    ) -> Optional[SearchNode]:
        self._behavior_epsilon = float(epsilon)
        return super().select_node(nodes, epsilon=epsilon)

    def expected_value(self, nodes: Sequence[SearchNode]) -> float:
        if not nodes:
            return 0.0
        epsilon = float(np.clip(self._behavior_epsilon, 0.0, 1.0))
        values = np.asarray([self.node_value(node, nodes) for node in nodes], dtype=float)
        stable_ids = [self._stable_id(node) for node in nodes]
        greedy_index = max(
            range(len(nodes)), key=lambda index: (values[index], -stable_ids[index])
        )
        probabilities = np.full(len(nodes), epsilon / len(nodes), dtype=float)
        probabilities[greedy_index] += 1.0 - epsilon
        return float(np.dot(probabilities, values))

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
        current_frontier = frontier if frontier is not None else [state]
        phi = self._features(state, current_frontier).astype(np.float64)
        q_value = float(np.dot(self.theta, phi))
        bootstrap = (
            0.0
            if done or not next_frontier
            else self.expected_value(tuple(next_frontier))
        )
        td_error = float(reward) + self.gamma * bootstrap - q_value
        self.theta += self.lr * td_error * phi
        return float(td_error)

    def metadata(self) -> dict[str, object]:
        metadata = super().metadata()
        metadata["algorithm"] = "linear-semi-gradient-expected-sarsa(0)"
        return metadata


class LinearContextualBanditPolicy(LinearQPolicy):
    """One-step linear ranking baseline with no temporal bootstrap."""

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
        current_frontier = frontier if frontier is not None else [state]
        phi = self._features(state, current_frontier).astype(np.float64)
        q_value = float(np.dot(self.theta, phi))
        td_error = float(reward) - q_value
        self.theta += self.lr * td_error * phi
        return float(td_error)

    def metadata(self) -> dict[str, object]:
        metadata = super().metadata()
        metadata["algorithm"] = "linear-contextual-bandit"
        metadata["discount"] = 0.0
        return metadata


__all__ = ["LinearContextualBanditPolicy", "LinearExpectedSarsaPolicy"]
