"""Training helper for the transparent linear SARSA frontier scheduler."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

import numpy as np

from config import normalize_reward_mode
from rl.policy import LinearQPolicy


def _freeze_boundary_value(value: object) -> object:
    """Copy common JSON/RNG values into an immutable callback snapshot."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return _freeze_boundary_value(value.item())
    if isinstance(value, np.ndarray):
        return tuple(_freeze_boundary_value(item) for item in value.tolist())
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_boundary_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_boundary_value(item) for item in value)
    raise TypeError(
        f"trainer boundary state contains unsupported {type(value).__name__} value"
    )


def _rng_state(generator: object) -> Mapping[str, object]:
    bit_generator = getattr(generator, "bit_generator", None)
    state = getattr(bit_generator, "state", None)
    if not isinstance(state, Mapping):
        return MappingProxyType({})
    frozen = _freeze_boundary_value(state)
    assert isinstance(frozen, Mapping)
    return frozen


def _scalar_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    """Keep only portable scalar diagnostics; large witnesses stay outside events."""

    result: dict[str, object] = {}
    for key, value in values.items():
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (str, bool, int, float)):
            result[str(key)] = value
    return MappingProxyType(result)


@dataclass(frozen=True, slots=True)
class TrainerBoundaryEvent:
    """Immutable safe-boundary callback state for progress and recovery adapters."""

    boundary: str
    episode_index: int
    episode_count: int
    expansion: int
    expansion_cap: int
    selected_record_id: int | None
    selected_features: tuple[float, ...] | None
    reward: float | None
    terminated: bool
    truncated: bool
    td_error: float | None
    next_record_id: int | None
    next_features: tuple[float, ...] | None
    total_reward: float
    epsilon: float
    policy_weights_after_update: tuple[float, ...]
    policy_weight_digest_after_update: str
    policy_rng_state: Mapping[str, object]
    environment_rng_state: Mapping[str, object]
    frontier_revision: int
    frontier_active_record_ids: tuple[int, ...]
    transition_info: Mapping[str, object]
    search_metrics: Mapping[str, object]
    episode_result: Mapping[str, object]
    safe_sarsa_boundary: bool = True

    def __post_init__(self) -> None:
        if self.boundary not in {"expansion", "episode_end"}:
            raise ValueError("trainer boundary must be expansion or episode_end")
        if self.safe_sarsa_boundary is not True:
            raise ValueError("trainer callback is valid only at a safe SARSA boundary")


TrainerBoundaryCallback = Callable[[TrainerBoundaryEvent], None]


@dataclass(frozen=True, slots=True)
class TrainerEpisodeResume:
    """Validated continuation cursor after a runner replays an episode journal."""

    episode_index: int
    expansion: int
    selected_record_id: int
    selected_features: tuple[float, ...]
    total_reward: float
    td_errors: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.episode_index, bool)
            or not isinstance(self.episode_index, int)
            or self.episode_index < 0
        ):
            raise ValueError("resume episode index must be a nonnegative integer")
        if (
            isinstance(self.expansion, bool)
            or not isinstance(self.expansion, int)
            or self.expansion < 0
        ):
            raise ValueError("resume expansion must be a nonnegative integer")
        if (
            isinstance(self.selected_record_id, bool)
            or not isinstance(self.selected_record_id, int)
            or self.selected_record_id < 0
        ):
            raise ValueError("resume selected record ID must be nonnegative")
        if not self.selected_features or not all(
            np.isfinite(value) for value in self.selected_features
        ):
            raise ValueError("resume selected features must be finite and nonempty")
        if not np.isfinite(self.total_reward) or not all(
            np.isfinite(value) for value in self.td_errors
        ):
            raise ValueError("resume reward/TD history must be finite")


class Trainer:
    def __init__(
        self,
        env,
        policy: Optional[LinearQPolicy] = None,
        *,
        progress_callback: TrainerBoundaryCallback | None = None,
        checkpoint_callback: TrainerBoundaryCallback | None = None,
    ):
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
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable or None")
        if checkpoint_callback is not None and not callable(checkpoint_callback):
            raise TypeError("checkpoint_callback must be callable or None")
        self.progress_callback = progress_callback
        self.checkpoint_callback = checkpoint_callback
        # Callback overhead is engineering time and is deliberately kept out of
        # deterministic episode history and environment feature/step timers.
        self.progress_callback_time_ns = 0
        self.checkpoint_callback_time_ns = 0

    @staticmethod
    def _feature_tuple(features: object | None) -> tuple[float, ...] | None:
        if features is None:
            return None
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("trainer callback features must be a finite vector")
        return tuple(float(value) for value in values)

    def _boundary_event(
        self,
        *,
        boundary: str,
        episode_index: int,
        episode_count: int,
        expansion: int,
        selected_record_id: int | None,
        selected_features: object | None,
        reward: float | None,
        terminated: bool,
        truncated: bool,
        td_error: float | None,
        next_record_id: int | None,
        next_features: object | None,
        total_reward: float,
        transition_info: Mapping[str, object],
        episode_result: Mapping[str, object] | None = None,
    ) -> TrainerBoundaryEvent:
        frontier = getattr(self.env, "frontier", None)
        active_record_ids = getattr(frontier, "active_record_ids", None)
        record_ids = (
            tuple(int(value) for value in active_record_ids())
            if callable(active_record_ids)
            else tuple(
                int(node.record_id)
                for node in self.env.current_nodes()
                if node.record_id is not None
            )
        )
        revision = int(getattr(frontier, "revision", 0))
        environment_rng = getattr(self.env, "__dict__", {}).get("_np_random")
        metrics = dict(getattr(self.env, "search_metrics", {}))
        return TrainerBoundaryEvent(
            boundary=boundary,
            episode_index=int(episode_index),
            episode_count=int(episode_count),
            expansion=int(expansion),
            expansion_cap=int(getattr(self.env.config, "max_steps", expansion)),
            selected_record_id=(
                None if selected_record_id is None else int(selected_record_id)
            ),
            selected_features=self._feature_tuple(selected_features),
            reward=None if reward is None else float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            td_error=None if td_error is None else float(td_error),
            next_record_id=None if next_record_id is None else int(next_record_id),
            next_features=self._feature_tuple(next_features),
            total_reward=float(total_reward),
            epsilon=float(self.epsilon),
            policy_weights_after_update=tuple(
                float(value) for value in np.asarray(self.policy.theta, dtype=np.float64)
            ),
            policy_weight_digest_after_update=str(self.policy.weight_digest()),
            policy_rng_state=_rng_state(getattr(self.policy, "rng", None)),
            environment_rng_state=_rng_state(environment_rng),
            frontier_revision=revision,
            frontier_active_record_ids=record_ids,
            transition_info=_scalar_mapping(transition_info),
            search_metrics=_scalar_mapping(metrics),
            episode_result=(
                MappingProxyType({})
                if episode_result is None
                else _freeze_boundary_value(episode_result)
            ),
        )

    def _notify_boundary(self, event: TrainerBoundaryEvent) -> None:
        # Checkpoint first so a progress adapter can report the newly committed
        # path.  Any callback exception aborts training rather than pretending a
        # required operability artifact succeeded.
        if self.checkpoint_callback is not None:
            started = time.perf_counter_ns()
            try:
                self.checkpoint_callback(event)
            finally:
                self.checkpoint_callback_time_ns += time.perf_counter_ns() - started
        if self.progress_callback is not None:
            started = time.perf_counter_ns()
            try:
                self.progress_callback(event)
            finally:
                self.progress_callback_time_ns += time.perf_counter_ns() - started

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
        select_compact = getattr(self.policy, "select_from_compact_batch", None)
        if callable(select_compact) and hasattr(batch, "frontier_nodes"):
            return select_compact(batch, epsilon=self.epsilon)
        return self.policy.select_from_batch(batch, epsilon=self.epsilon)

    def _article_decision_batch(self):
        current_records = getattr(self.env, "current_records", None)
        build_compact = getattr(self.policy, "build_compact_decision_batch", None)
        if callable(current_records) and callable(build_compact):
            return build_compact(current_records())
        return self.policy.build_feature_batch(self.env.current_nodes())

    def _frozen_features(self, batch, node):
        features_from_batch = getattr(
            self.policy, "features_from_decision_batch", None
        )
        if callable(features_from_batch):
            return features_from_batch(batch, node)
        return batch.features_for(node)

    def train(
        self,
        num_episodes: int = 100,
        *,
        start_episode: int = 0,
        resume_episode: TrainerEpisodeResume | None = None,
    ) -> list[dict[str, Any]]:
        if (
            isinstance(num_episodes, bool)
            or not isinstance(num_episodes, int)
            or num_episodes < 1
        ):
            raise ValueError("num_episodes must be a positive integer")
        if (
            isinstance(start_episode, bool)
            or not isinstance(start_episode, int)
            or start_episode < 0
            or start_episode > num_episodes
        ):
            raise ValueError("start_episode must be between zero and num_episodes")
        if resume_episode is not None and resume_episode.episode_index != start_episode:
            raise ValueError("resume episode index must equal start_episode")
        training_started = time.perf_counter()
        history: list[dict[str, Any]] = []
        for episode in range(start_episode, num_episodes):
            if resume_episode is not None and episode == start_episode:
                reset_info: Mapping[str, object] = {}
                terminated = False
                truncated = False
                total_reward = float(resume_episode.total_reward)
                steps = int(resume_episode.expansion)
                selected = next(
                    (
                        node
                        for node in self.env.current_nodes()
                        if node.record_id == resume_episode.selected_record_id
                    ),
                    None,
                )
                if selected is None:
                    raise ValueError("resume selected record is not open")
                selected_features = np.asarray(
                    resume_episode.selected_features, dtype=np.float64
                )
                if selected_features.shape != self.policy.theta.shape:
                    raise ValueError("resume selected features have the wrong dimension")
                td_errors = [float(value) for value in resume_episode.td_errors]
                resume_episode = None
            else:
                _, reset_info = self.env.reset(
                    seed=getattr(self.env.config, "seed", None) if episode == 0 else None
                )
                terminated = bool(reset_info.get("initial_certified", False))
                truncated = False
                total_reward = 0.0
                steps = 0
                selected = None
                selected_features = None
                td_errors = []
            last_boundary: TrainerBoundaryEvent | None = None

            while not (terminated or truncated):
                nodes_before = self.env.current_nodes()
                if selected is None or not any(node is selected for node in nodes_before):
                    if self.reward_mode == "article_v1_expansion_potential":
                        decision_batch = self._article_decision_batch()
                        selected = self._behavior_node_from_batch(
                            nodes_before, decision_batch
                        )
                        selected_features = (
                            None
                            if selected is None
                            else self._frozen_features(decision_batch, selected)
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
                    decision_batch = self._article_decision_batch()
                    selected_features = self._frozen_features(
                        decision_batch, selected
                    )

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
                    selected_features = self._frozen_features(decision_batch, actual)

                next_nodes = self.env.current_nodes()
                next_node = None
                next_features = None
                if not (terminated or truncated):
                    if self.reward_mode == "article_v1_expansion_potential":
                        next_batch = self._article_decision_batch()
                        next_node = self._behavior_node_from_batch(next_nodes, next_batch)
                        next_features = (
                            None
                            if next_node is None
                            else self._frozen_features(next_batch, next_node)
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
                if self.progress_callback is not None or self.checkpoint_callback is not None:
                    last_boundary = self._boundary_event(
                        boundary="expansion",
                        episode_index=episode,
                        episode_count=num_episodes,
                        expansion=steps,
                        selected_record_id=(
                            None if actual.record_id is None else int(actual.record_id)
                        ),
                        selected_features=selected_features,
                        reward=reward,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        td_error=float(td_error),
                        next_record_id=(
                            None
                            if next_node is None or next_node.record_id is None
                            else int(next_node.record_id)
                        ),
                        next_features=next_features,
                        total_reward=total_reward,
                        transition_info=info,
                    )
                    self._notify_boundary(last_boundary)
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
            if self.progress_callback is not None or self.checkpoint_callback is not None:
                if last_boundary is None:
                    episode_boundary = self._boundary_event(
                        boundary="episode_end",
                        episode_index=episode,
                        episode_count=num_episodes,
                        expansion=steps,
                        selected_record_id=None,
                        selected_features=None,
                        reward=None,
                        terminated=bool(terminated),
                        truncated=bool(truncated),
                        td_error=None,
                        next_record_id=None,
                        next_features=None,
                        total_reward=total_reward,
                        transition_info=reset_info,
                        episode_result=result,
                    )
                else:
                    # Epsilon has now advanced to the value used by the next
                    # episode.  Every other field is the already-frozen final
                    # transition snapshot.
                    episode_boundary = replace(
                        last_boundary,
                        boundary="episode_end",
                        epsilon=float(self.epsilon),
                        policy_rng_state=_rng_state(getattr(self.policy, "rng", None)),
                        episode_result=_freeze_boundary_value(result),
                    )
                self._notify_boundary(episode_boundary)
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
