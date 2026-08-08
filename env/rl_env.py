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
    Gymnasium environment for RL-guided frontier search.

    The RL action selects a frontier node for symbolic expansion;
    the symbolic engine deterministically generates all successors.
    """

    def __init__(self, config, certification_engine, policy=None):
        super().__init__()

        self.config = config
        self.cert_engine = certification_engine
        self.policy = policy  # optional (used for child priority)

        # ---------- Symbolic gate library ----------
        self.actions = generate_actions(config.num_qubits)

        # ---------- Action space ----------
        # Index into the current frontier node list.
        self.action_space = spaces.Discrete(config.max_frontier)

        # ---------- Observation space ----------
        self.feature_dim = 12
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.feature_dim,),
            dtype=np.float32,
        )

        # ---------- Internal state ----------
        self.frontier = None
        self.current_node = None
        self.steps = 0

        self.canonicalizer = Canonicalizer()

        # ---------- Reward weights (Stage 10) ----------
        self.alpha = 0.1     # resource improvement
        self.beta = 0.5      # dominated states pruned
        self.gamma_r = 10.0  # certification success
        self.delta = 1.0     # dead-end expansions

    # =========================================================
    # Helpers
    # =========================================================

    def _cost(self, state) -> float:
        return state.t_count + state.depth + state.num_gates

    def current_nodes(self):
        """Snapshot of the frontier node list."""
        return list(self.frontier.heap)

    # =========================================================
    # Reset (Stage 1)
    # =========================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0

        dag = CircuitDAG(self.config.num_qubits)
        state = CircuitState(dag, self.config.budget)

        root = SearchNode(priority=0.0, state=state)

        self.frontier = Frontier()
        self.frontier.push(root)

        self.current_node = root

        obs = extract_features(state)

        return obs, {}

    # =========================================================
    # Step (Stages 3-5, 10-11)
    # =========================================================

    def step(self, node_idx):
        self.steps += 1

        nodes = self.current_nodes()

        if node_idx >= len(nodes):
            reward = -1.0
            done = self.frontier.is_empty() or self.steps >= self.config.max_steps
            return extract_features(self.current_node.state), reward, done, False, {}

        node = nodes[node_idx]
        self.frontier.remove(node)
        self.current_node = node

        parent_cost = self._cost(node.state)

        # ---------- Symbolic expansion (Stage 5) ----------
        children = expand_node(node, self.actions, self.policy)

        survivors = []
        num_pruned = 0
        num_dead = 0
        num_certified = 0
        delta_j = 0.0

        for child in children:
            child_state = child.state

            # ---------- Certification (Stage 9) ----------
            cert_result = self.cert_engine.certify(child_state)

            if cert_result.status.name == "SUCCESS":
                num_certified += 1
                continue

            if cert_result.status.name == "FAILURE":
                num_dead += 1
                continue

            # ---------- INCONCLUSIVE: canonicalize, prune, insert ----------
            inserted = self.frontier.push(child)

            if inserted:
                survivors.append(child)
                delta_j += parent_cost - self._cost(child_state)
            else:
                num_pruned += 1

        # ---------- Reward (Stage 10) ----------
        reward = (
            self.alpha * delta_j
            + self.beta * num_pruned
            + self.gamma_r * (num_certified > 0)
            - self.delta * num_dead
        )

        # ---------- Done conditions (Stage 13) ----------
        done = (
            num_certified > 0
            or self.frontier.is_empty()
            or self.steps >= self.config.max_steps
        )

        obs = extract_features(node.state)

        return obs, reward, done, False, {
            "cert_status": "SUCCESS" if num_certified > 0 else "NONE",
            "num_certified": num_certified,
            "num_pruned": num_pruned,
            "num_dead": num_dead,
            "num_children": len(children),
            "frontier_size": len(self.frontier.heap),
        }

    # =========================================================
    # Render (optional)
    # =========================================================

    def render(self):
        print(self.current_node.state)
