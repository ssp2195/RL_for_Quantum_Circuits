import numpy as np
from typing import List

from rl.features import extract_features
from search.action import Action
from circuit.circuit_state import CircuitState


class LinearQPolicy:
    """
    Linear Q-function:
        Q(s, a) = θ^T φ(s')
    """

    def __init__(self, feature_dim: int, lr: float = 0.01, gamma: float = 0.99):
        self.theta = np.zeros(feature_dim, dtype=np.float32)

        self.lr = lr
        self.gamma = gamma

    # =========================================================
    # Q-value
    # =========================================================

    def q_value(self, state: CircuitState) -> float:
        features = extract_features(state)
        return float(np.dot(self.theta, features))

    # =========================================================
    # Evaluate actions
    # =========================================================

    def evaluate_actions(self, parent_state, actions: List[Action]):
        """
        Returns list of (action, q_value, next_state)
        """
        results = []

        for action in actions:
            next_state = parent_state.copy()

            from circuit.gate import Gate
            gate = Gate(action.gate_type, action.qubits)

            if not next_state.apply_gate(gate):
                continue

            q = self.q_value(next_state)

            results.append((action, q, next_state))

        return results

    # =========================================================
    # Action selection (ε-greedy)
    # =========================================================

    def select_action(self, state, actions, epsilon=0.1):
        candidates = self.evaluate_actions(state, actions)

        if not candidates:
            return None

        # exploration
        if np.random.rand() < epsilon:
            idx = np.random.randint(len(candidates))
            return candidates[idx]

        # exploitation
        return max(candidates, key=lambda x: x[1])

    # =========================================================
    # TD Update
    # =========================================================

    def update(self, state, reward, next_state, done):
        """
        TD(0):
            θ ← θ + α [r + γ max_a Q(s') - Q(s)] φ(s)
        """

        phi_s = extract_features(state)
        q_s = np.dot(self.theta, phi_s)

        if done or next_state is None:
            target = reward
        else:
            q_next = self.q_value(next_state)
            target = reward + self.gamma * q_next

        td_error = target - q_s

        self.theta += self.lr * td_error * phi_s

    # =========================================================
    # Node scoring (for frontier integration)
    # =========================================================

    def score_state(self, state):
        """
        Higher Q → better → lower priority value
        """
        return -self.q_value(state)

    
    # Reward function
    def compute_reward(state, cert_result):
        if cert_result.status.name == "SUCCESS":
            return +10.0

        if cert_result.status.name == "FAILURE":
            return -1.0

        # INCONCLUSIVE
        return -0.1
