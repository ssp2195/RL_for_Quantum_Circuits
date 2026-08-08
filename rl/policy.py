import numpy as np
from typing import List

from rl.features import extract_features
from search.node import SearchNode
from circuit.circuit_state import CircuitState


class LinearQPolicy:
    """
    Linear value/ranking function over frontier states:
        V(s) = θ^T φ(s)

    The RL agent does NOT choose gates. It ranks frontier nodes
    and selects the most promising one for symbolic expansion.
    """

    def __init__(self, feature_dim: int, lr: float = 0.01, gamma: float = 0.99):
        self.theta = np.zeros(feature_dim, dtype=np.float32)

        self.lr = lr
        self.gamma = gamma

    # =========================================================
    # Value estimation
    # =========================================================

    def q_value(self, state: CircuitState) -> float:
        features = extract_features(state)
        return float(np.dot(self.theta, features))

    def node_value(self, node: SearchNode) -> float:
        return self.q_value(node.state)

    # =========================================================
    # Frontier ranking / selection
    # =========================================================

    def evaluate_nodes(self, nodes: List[SearchNode]):
        """
        Returns list of (node, q_value) for every frontier node.
        """
        return [(node, self.node_value(node)) for node in nodes]

    def select_node(self, nodes: List[SearchNode], epsilon=0.1):
        """
        ε-greedy selection of a frontier node to expand.
        Returns the selected SearchNode, or None if empty.
        """
        if not nodes:
            return None

        if np.random.rand() < epsilon:
            return nodes[np.random.randint(len(nodes))]

        return max(nodes, key=lambda n: self.node_value(n))

    # =========================================================
    # TD Update over the frontier MDP
    # =========================================================

    def update(self, state, reward, next_frontier, done):
        """
        TD(0):
            θ ← θ + α [ r + γ max_{s'∈F_{t+1}} V(s') - V(s*) ] φ(s*)
        """
        phi_s = extract_features(state)
        q_s = np.dot(self.theta, phi_s)

        if done or not next_frontier:
            target = reward
        else:
            target = reward + self.gamma * max(
                self.node_value(n) for n in next_frontier
            )

        td_error = target - q_s

        self.theta += self.lr * td_error * phi_s

    # =========================================================
    # Node scoring (priority, lower = more promising)
    # =========================================================

    def score_state(self, state):
        """
        Higher V → better → lower priority value
        """
        return -self.q_value(state)
