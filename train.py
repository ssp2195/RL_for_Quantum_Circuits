"""Training helper for the transparent linear SARSA frontier scheduler."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np

from rl.policy import LinearQPolicy


class Trainer:
    def __init__(self, env, policy: Optional[LinearQPolicy] = None):
        self.env = env
        if policy is None:
            policy = LinearQPolicy(
                feature_dim=env.feature_dim,
                gamma=env.config.discount,
                seed=getattr(env.config, "seed", None),
            )
        self.policy = policy
        self.env.policy = self.policy
        self.visit_counts = defaultdict(int)
        self.canonicalizer = env.canonicalizer
        self.epsilon = 0.2
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.995
        self.reward_clip = 10.0
        self.exploration_beta = 0.1

    def _behavior_node(self, nodes):
        """Mirror the environment's optional fair-scheduling override."""
        interval = max(0, int(getattr(self.env.config, "fairness_interval", 0)))
        if interval and (self.env.steps + 1) % interval == 0:
            return min(nodes, key=lambda node: int(node.record_id or 0)) if nodes else None
        return self.policy.select_node(nodes, epsilon=self.epsilon)

    def train(self, num_episodes: int = 100) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for episode in range(num_episodes):
            _, reset_info = self.env.reset(
                seed=getattr(self.env.config, "seed", None) if episode == 0 else None
            )
            terminated = bool(reset_info.get("initial_certified", False))
            truncated = False
            total_reward = 0.0
            steps = 0
            selected = None

            while not (terminated or truncated):
                nodes_before = self.env.current_nodes()
                if selected is None or not any(node is selected for node in nodes_before):
                    selected = self._behavior_node(nodes_before)
                if selected is None:
                    break

                selected_index = nodes_before.index(selected)
                _, reward, terminated, truncated, info = self.env.step(selected_index)

                # Fairness may have expanded another record; learn from the
                # actual environment transition rather than the proposed one.
                chosen_id = info.get("selected_record_id")
                actual = next(
                    (node for node in nodes_before if node.record_id == chosen_id),
                    selected,
                )
                identity = self.canonicalizer.identity_hash(actual.state)
                self.visit_counts[identity] += 1
                reward += self.exploration_beta / np.sqrt(self.visit_counts[identity])
                reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

                next_nodes = self.env.current_nodes()
                next_node = None
                if not (terminated or truncated):
                    next_node = self._behavior_node(next_nodes)

                self.policy.update(
                    state=actual.state,
                    reward=reward,
                    next_frontier=next_nodes,
                    done=terminated or truncated,
                    next_node=next_node,
                    frontier=nodes_before,
                )
                total_reward += reward
                steps += 1
                # SARSA's bootstrap action is the behavior action used on the
                # next iteration (unless the fairness interleave supersedes it).
                selected = next_node

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            result = {
                "episode": episode,
                "reward": total_reward,
                "steps": steps,
                "certified": self.env.solution_node is not None,
                "truncated": truncated,
                "epsilon": self.epsilon,
            }
            history.append(result)
            print(
                f"Episode {episode:03d} | Reward: {total_reward:.3f} | "
                f"Steps: {steps} | Certified: {result['certified']} | "
                f"Epsilon: {self.epsilon:.3f}"
            )
        return history
