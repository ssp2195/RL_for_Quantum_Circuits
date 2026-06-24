import numpy as np
import gymnasium as gym
from gymnasium import spaces

from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from ckt_types import ResourceBudget

from search.node import SearchNode
from search.frontier import Frontier
from search.action_space import generate_actions
from search.expansion import expand_node

from rl.features import extract_features
from canonical.canonicalizer import Canonicalizer


class CircuitSynthesisEnv(gym.Env):
    """
    Gymnasium environment for circuit synthesis
    """

    def __init__(self, config, certification_engine, policy=None):
        super().__init__()

        self.config = config
        self.cert_engine = certification_engine
        self.policy = policy  # optional (used for priority)

        # ---------- Action space ----------
        self.actions = generate_actions(config.num_qubits)
        self.action_space = spaces.Discrete(len(self.actions))

        # ---------- Observation space ----------
        self.feature_dim = 12  # from Stage 5
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.feature_dim,),
            dtype=np.float32
        )

        # ---------- Internal state ----------
        self.frontier = None
        self.current_node = None
        self.steps = 0

        self.canonicalizer = Canonicalizer()

    # =========================================================
    # Reset
    # =========================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0

        # initial state
        dag = CircuitDAG(self.config.num_qubits)
        state = CircuitState(dag, self.config.budget)

        root = SearchNode(priority=0.0, state=state)

        # initialize frontier
        self.frontier = Frontier()
        self.frontier.push(root)

        self.current_node = root

        obs = extract_features(state)

        return obs, {}

    # =========================================================
    # Step
    # =========================================================

    def step(self, action_idx):
        self.steps += 1

        action = self.actions[action_idx]

        parent_state = self.current_node.state
        next_state = parent_state.copy()

        from circuit.gate import Gate
        gate = Gate(action.gate_type, action.qubits)

        success = next_state.apply_gate(gate)

        if not success:
            # invalid action penalty
            reward = -1.0
            done = False
            return extract_features(parent_state), reward, done, False, {}

        # ---------- Certification ----------
        cert_result = self.cert_engine.certify(next_state)

        # ---------- Reward ----------
        reward = self._compute_reward(next_state, cert_result)

        # ---------- Create node ----------
        priority = (
            self.policy.score_state(next_state)
            if self.policy is not None
            else next_state.t_count + next_state.depth
        )

        node = SearchNode(
            priority=priority,
            state=next_state,
            parent=self.current_node,
            action=action,
        )

        # ---------- Push to frontier ----------
        inserted = self.frontier.push(node)

        # ---------- Transition ----------
        self.current_node = node

        # ---------- Done conditions ----------
        done = (
            cert_result.status.name == "SUCCESS"
            or self.steps >= self.config.max_steps
        )

        obs = extract_features(next_state)

        return obs, reward, done, False, {
            "cert_status": cert_result.status.name,
            "inserted": inserted
        }

    # =========================================================
    # Reward Function
    # =========================================================

    def _compute_reward(self, state, cert_result):
        if cert_result.status.name == "SUCCESS":
            return 10.0

        if cert_result.status.name == "FAILURE":
            reward = -1.0
        else:
            reward = -0.1

        # resource penalties
        reward -= 0.05 * state.t_count
        reward -= 0.01 * state.depth

        return reward

    # =========================================================
    # Render (optional)
    # =========================================================

    def render(self):
        print(self.current_node.state)
