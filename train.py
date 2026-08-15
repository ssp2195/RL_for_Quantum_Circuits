"""Training helper for the transparent linear SARSA frontier scheduler."""

from __future__ import annotations

from collections import defaultdict
import time
from typing import Any, Optional

import numpy as np

from config import normalize_reward_mode
from rl.policy import LinearQPolicy


class Trainer:
    def __init__(self, env, policy: Optional[LinearQPolicy] = None):
        self.env = env
        # ``target_context`` is owned by the environment because it is
        # derived from the same exact target used by certification.  Bind the
        # scorer before any rollout so observations, Q-values, and SARSA
        # updates all use one feature schema.  Target-free environments keep
        # the historical 16-coordinate representation.
        target_context = getattr(env, "_feature_target_context", None)
        feature_provider = getattr(env, "feature_provider", None)
        if policy is None:
            if feature_provider is None:
                policy = LinearQPolicy(
                    feature_dim=env.feature_dim,
                    gamma=env.config.discount,
                    seed=getattr(env.config, "seed", None),
                    target_context=target_context,
                )
            else:
                policy = LinearQPolicy(
                    feature_provider=feature_provider,
                    gamma=env.config.discount,
                    seed=getattr(env.config, "seed", None),
                )
        elif feature_provider is not None:
            bind_feature_provider = getattr(policy, "bind_feature_provider", None)
            if not callable(bind_feature_provider):
                raise ValueError("policy does not support the environment feature provider")
            bind_feature_provider(feature_provider)
        elif target_context is not None:
            policy.bind_target_context(target_context)
        bind_policy = getattr(env, "_bind_policy_target_context", None)
        if callable(bind_policy):
            bind_policy(policy)
        self.policy = policy
        self.env.policy = self.policy
        self.visit_counts = defaultdict(int)
        self.canonicalizer = env.canonicalizer
        self.epsilon = 0.2
        self.min_epsilon = 0.05
        self.epsilon_decay = 0.995
        self.last_training_runtime_seconds = 0.0
        self.reward_mode = normalize_reward_mode(
            getattr(env.config, "reward_mode", "legacy_archive_shaping")
        )
        # Clipping would change Equation (24) when an exhausted frontier must
        # receive a failure correction larger than ten.  Legacy/task-shaped
        # modes retain their historical trainer-side bound.
        self.reward_clip = (
            None
            if self.reward_mode
            in {
                "expansion_cost",
                "expansion_cost_plus_visit_bonus",
                "article_v1_expansion_potential",
            }
            else 10.0
        )
        # The target-progress environment already supplies an explicit
        # potential-shaped reward.  Do not silently add the legacy visit
        # bonus to that objective; callers can still opt in manually.
        self.exploration_beta = (
            0.0
            if (
                self.reward_mode in {"target_progress_shaping", "expansion_cost"}
                or self.reward_mode == "article_v1_expansion_potential"
                or getattr(env, "reward_model", None) is not None
            )
            else 0.1
        )

    def _behavior_node(self, nodes):
        """Mirror the environment's optional fair-scheduling override."""
        interval = max(0, int(getattr(self.env.config, "fairness_interval", 0)))
        if interval and (self.env.steps + 1) % interval == 0:
            return min(nodes, key=lambda node: int(node.record_id or 0)) if nodes else None
        return self.policy.select_node(nodes, epsilon=self.epsilon)

    def _behavior_node_from_batch(self, nodes, batch):
        """Select from the exact frozen Article V1 decision snapshot."""

        interval = max(0, int(getattr(self.env.config, "fairness_interval", 0)))
        if interval and (self.env.steps + 1) % interval == 0:
            return min(nodes, key=lambda node: int(node.record_id or 0)) if nodes else None
        return self.policy.select_from_batch(batch, epsilon=self.epsilon)

    def train(self, num_episodes: int = 100) -> list[dict[str, Any]]:
        training_started = time.perf_counter()
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
            selected_features = None
            td_errors: list[float] = []

            while not (terminated or truncated):
                nodes_before = self.env.current_nodes()
                if selected is None or not any(node is selected for node in nodes_before):
                    if self.reward_mode == "article_v1_expansion_potential":
                        decision_batch = self.policy.build_feature_batch(nodes_before)
                        selected = self._behavior_node_from_batch(
                            nodes_before, decision_batch
                        )
                        selected_features = (
                            None
                            if selected is None
                            else decision_batch.features_for(selected)
                        )
                    else:
                        selected = self._behavior_node(nodes_before)
                        selected_features = None
                if selected is None:
                    break

                if (
                    self.reward_mode == "article_v1_expansion_potential"
                    and selected_features is None
                ):
                    decision_batch = self.policy.build_feature_batch(nodes_before)
                    selected_features = decision_batch.features_for(selected)

                # The core action is a persistent record identity.  Positional
                # Gym actions are only a compatibility adapter and are never
                # used by the reference trainer.
                _, reward, terminated, truncated, info = self.env.select_record(
                    int(selected.record_id)
                )

                # Fairness may have expanded another record; learn from the
                # actual environment transition rather than the proposed one.
                chosen_id = info.get("selected_record_id")
                actual = next(
                    (node for node in nodes_before if node.record_id == chosen_id),
                    selected,
                )
                if not info.get("selected_by_fairness", False) and actual is not selected:
                    raise RuntimeError(
                        "environment expanded a different record than the behavior policy selected"
                    )
                identity = self.canonicalizer.semantic_key(actual.state)
                self.visit_counts[identity] += 1
                if self.exploration_beta:
                    reward += self.exploration_beta / np.sqrt(
                        self.visit_counts[identity]
                    )
                reward = float(reward)
                if self.reward_clip is not None:
                    reward = float(
                        np.clip(reward, -self.reward_clip, self.reward_clip)
                    )
                if (
                    self.reward_mode == "article_v1_expansion_potential"
                    and actual is not selected
                ):
                    # Fairness is disabled in frozen article evaluation, but
                    # retain a correct legacy override path during training.
                    selected_features = decision_batch.features_for(actual)

                next_nodes = self.env.current_nodes()
                next_node = None
                next_features = None
                if not (terminated or truncated):
                    if self.reward_mode == "article_v1_expansion_potential":
                        next_batch = self.policy.build_feature_batch(next_nodes)
                        next_node = self._behavior_node_from_batch(next_nodes, next_batch)
                        next_features = (
                            None
                            if next_node is None
                            else next_batch.features_for(next_node)
                        )
                    else:
                        next_node = self._behavior_node(next_nodes)

                if self.reward_mode == "article_v1_expansion_potential":
                    assert selected_features is not None
                    td_error = self.policy.update_from_features(
                        current_features=selected_features,
                        reward=reward,
                        next_features=next_features,
                        done=terminated or truncated,
                    )
                else:
                    td_error = self.policy.update(
                        state=actual.state,
                        reward=reward,
                        next_frontier=next_nodes,
                        done=terminated or truncated,
                        next_node=next_node,
                        frontier=nodes_before,
                    )
                td_errors.append(float(td_error))
                total_reward += reward
                steps += 1
                # SARSA's bootstrap action is the behavior action used on the
                # next iteration (unless the fairness interleave supersedes it).
                selected = next_node
                selected_features = next_features

            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            result = {
                "episode": episode,
                "reward": total_reward,
                "steps": steps,
                "certified": self.env.solution_node is not None,
                "truncated": truncated,
                "epsilon": self.epsilon,
                "reward_mode": self.reward_mode,
                "reward_coefficients": self.env.reward_spec(),
                "learning_rate": float(self.policy.lr),
                "discount": float(self.policy.gamma),
                "exploration_beta": float(self.exploration_beta),
                "mean_absolute_td_error": float(
                    np.mean(np.abs(td_errors)) if td_errors else 0.0
                ),
                "max_absolute_td_error": float(
                    np.max(np.abs(td_errors)) if td_errors else 0.0
                ),
                "weight_norm": float(np.linalg.norm(self.policy.theta)),
                "policy_weight_digest": self.policy.weight_digest(),
                # Keep nondeterministic wall-clock counters out of the
                # reproducible learning history.  Publication runners collect
                # those counters separately from the environment.
                "search_metrics": {
                    name: value
                    for name, value in dict(
                        getattr(self.env, "search_metrics", {})
                    ).items()
                    if not name.endswith("_time_ns")
                },
            }
            history.append(result)
            print(
                f"Episode {episode:03d} | Reward: {total_reward:.3f} | "
                f"Steps: {steps} | Certified: {result['certified']} | "
                f"Epsilon: {self.epsilon:.3f}"
            )
        # Wall time is deliberately kept outside the deterministic episode
        # history so same-seed histories remain byte-for-byte reproducible.
        self.last_training_runtime_seconds = float(time.perf_counter() - training_started)
        return history
