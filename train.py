import numpy as np
from collections import defaultdict

from rl.policy import LinearQPolicy
from env.rl_env import CircuitSynthesisEnv


class Trainer:
    def __init__(self, env, feature_dim=12):
        self.env = env
        self.policy = LinearQPolicy(feature_dim)

        # attach policy to env (used for child priorities)
        self.env.policy = self.policy

        # visitation counts for exploration bonus
        self.visit_counts = defaultdict(int)

        # canonicalizer (for identity hash)
        self.canonicalizer = env.canonicalizer

        # hyperparameters
        self.epsilon = 0.2
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.995

        self.reward_clip = 10.0
        self.exploration_beta = 0.1

    # =========================================================
    # Training loop
    # =========================================================

    def train(self, num_episodes=100):
        for ep in range(num_episodes):

            state_features, _ = self.env.reset()
            done = False

            total_reward = 0.0
            steps = 0
            certified = False

            while not done:
                nodes = self.env.current_nodes()

                if not nodes:
                    break

                # ---------- Frontier node selection (Stage 4) ----------
                selected = self.policy.select_node(nodes, epsilon=self.epsilon)

                if selected is None:
                    break

                node_idx = next(
                    i for i, n in enumerate(nodes) if n is selected
                )
                current_state = selected.state

                # ---------- Step (expand selected node) ----------
                next_obs, reward, done, _, info = self.env.step(node_idx)

                # ---------- Exploration bonus ----------
                identity = self.canonicalizer.identity_hash(current_state)
                self.visit_counts[identity] += 1

                bonus = self.exploration_beta / np.sqrt(self.visit_counts[identity])

                reward += bonus

                # ---------- Reward clipping ----------
                reward = np.clip(reward, -self.reward_clip, self.reward_clip)

                # ---------- TD Update (Stage 12) ----------
                next_frontier = self.env.current_nodes()

                self.policy.update(
                    state=current_state,
                    reward=reward,
                    next_frontier=next_frontier,
                    done=done,
                )

                total_reward += reward
                steps += 1

                if info.get("num_certified", 0) > 0:
                    certified = True

            # ---------- Epsilon decay ----------
            self.epsilon = max(
                self.min_epsilon,
                self.epsilon * self.epsilon_decay
            )

            # ---------- Logging ----------
            print(
                f"Episode {ep:03d} | "
                f"Reward: {total_reward:.3f} | "
                f"Steps: {steps} | "
                f"Certified: {certified} | "
                f"Epsilon: {self.epsilon:.3f}"
            )
