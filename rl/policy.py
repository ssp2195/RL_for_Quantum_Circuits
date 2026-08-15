"""A small, transparent semi-gradient SARSA frontier-record scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Optional, Sequence, TYPE_CHECKING

import numpy as np

from circuit.circuit_state import CircuitState
from rl.features import (
    FeatureProvider,
    LegacyFeatureProvider,
    validate_feature_provider,
)
from search.node import SearchNode

if TYPE_CHECKING:
    from rl.target_context import DenseTargetContext


@dataclass(frozen=True, slots=True)
class PolicyFeatureBatch:
    """Frozen feature matrix for one immutable frontier decision snapshot."""

    nodes: tuple[SearchNode, ...]
    features: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.array(self.features, dtype=np.float64, copy=True)
        if matrix.ndim != 2 or matrix.shape[0] != len(self.nodes):
            raise ValueError("feature batch must have one row per frontier node")
        if not np.isfinite(matrix).all():
            raise ValueError("feature batch must contain only finite values")
        matrix.setflags(write=False)
        object.__setattr__(self, "features", matrix)

    def features_for(self, node: SearchNode) -> np.ndarray:
        """Return the frozen row for ``node`` using object identity."""

        for index, candidate in enumerate(self.nodes):
            if candidate is node:
                return self.features[index]
        raise KeyError("node is not part of this frontier feature snapshot")


def _context_fingerprint(target_context: Optional[object]) -> Optional[str]:
    if target_context is None:
        return None
    fingerprint = getattr(target_context, "fingerprint", None)
    return None if fingerprint is None else str(fingerprint)


def _context_schema_version(target_context: Optional[object]) -> Optional[str]:
    if target_context is None:
        return None
    schema_version = getattr(target_context, "schema_version", None)
    return None if schema_version is None else str(schema_version)


def _context_binding_digest(target_context: Optional[object]) -> Optional[str]:
    """Digest every context property that can alter an extracted feature.

    A target matrix fingerprint alone is insufficient because the same target
    can be scored with a different phase mode or target-progress weight tuple.
    Keep this small, explicit binding alongside the policy rather than relying
    on an object's identity or mutable cache contents.
    """

    if target_context is None:
        return None
    weights = getattr(target_context, "weights", None)
    weight_values = tuple(
        (name, float(getattr(weights, name)))
        for name in ("process_fidelity", "support_match", "entanglement_match")
        if weights is not None and hasattr(weights, name)
    )
    payload = repr(
        (
            _context_fingerprint(target_context),
            _context_schema_version(target_context),
            getattr(target_context, "num_qubits", None),
            getattr(target_context, "phase_mode", None),
            weight_values,
        )
    ).encode("utf-8")
    return f"sha256:{sha256(payload).hexdigest()}"


def _freeze_metadata(value: object) -> object:
    """Return a deterministic representation of provider-binding metadata."""

    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_metadata(item))
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_metadata(item) for item in value)
    return repr(value)


def _feature_provider_binding_digest(provider: FeatureProvider) -> str:
    """Bind explicit policy weights to all provider-visible metadata."""

    metadata = dict(provider.metadata())
    payload = repr(
        (
            str(provider.schema_version),
            int(provider.dimension),
            tuple(provider.names),
            _freeze_metadata(metadata),
        )
    ).encode("utf-8")
    return f"sha256:{sha256(payload).hexdigest()}"


def _feature_provider_target_fingerprint(provider: FeatureProvider) -> Optional[str]:
    """Read the immutable target binding advertised by an explicit provider."""

    fingerprint = dict(provider.metadata()).get("target_fingerprint")
    if fingerprint is None:
        return None
    value = str(fingerprint)
    return value or None


class LinearQPolicy:
    """Shared linear scorer for persistent frontier records.

    The historical class name is retained for callers, but its TD update is
    on-policy SARSA rather than max-bootstrap Q-learning.  A policy can be
    explicitly bound to a dense target context; every score and update then
    uses the same target-aware extractor.  It never consumes a gate action.
    """

    def __init__(
        self,
        feature_dim: Optional[int] = None,
        lr: float = 1e-3,
        gamma: float = 1.0,
        seed: Optional[int] = None,
        target_context: Optional["DenseTargetContext"] = None,
        feature_provider: Optional[FeatureProvider] = None,
    ):
        if feature_provider is not None and target_context is not None:
            raise ValueError(
                "pass either target_context or feature_provider, not both; "
                "use a TargetContextFeatureProvider for an explicit adapter"
            )
        if feature_provider is None:
            provider: FeatureProvider = LegacyFeatureProvider(target_context)
            explicit_provider = False
        else:
            provider = validate_feature_provider(feature_provider)
            explicit_provider = True

        expected_dim = int(provider.dimension)
        if feature_dim is None:
            feature_dim = expected_dim
        elif int(feature_dim) != expected_dim:
            context_description = (
                "target-aware" if target_context is not None else provider.schema_version
            )
            raise ValueError(
                f"feature_dim={feature_dim} does not match the {context_description} "
                f"extractor dimension {expected_dim}"
            )

        self.feature_dim = int(feature_dim)
        self.theta = np.zeros(self.feature_dim, dtype=np.float64)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.feature_evaluation_count = 0
        self.feature_time_ns = 0
        self.ranking_time_ns = 0
        self.target_context: Optional["DenseTargetContext"] = target_context
        self.feature_provider: FeatureProvider = provider
        self._explicit_feature_provider = explicit_provider
        self._feature_provider_binding_digest = (
            _feature_provider_binding_digest(provider) if explicit_provider else None
        )
        self._bound_feature_provider_target_fingerprint = (
            _feature_provider_target_fingerprint(provider)
            if explicit_provider
            else None
        )
        self._bound_target_fingerprint = _context_fingerprint(target_context)
        self._bound_target_context_schema_version = _context_schema_version(target_context)
        self._bound_target_context_digest = _context_binding_digest(target_context)

    @property
    def target_fingerprint(self) -> Optional[str]:
        """Fingerprint of the immutable target binding for this policy.

        Legacy policies continue to expose the dense target-context fingerprint.
        An explicit problem feature provider instead owns the target binding,
        so expose the value captured when that provider was bound rather than
        reporting an unhelpful ``None`` to checkpoint/report callers.
        """

        return (
            self._bound_target_fingerprint
            or self._bound_feature_provider_target_fingerprint
        )

    @property
    def target_context_schema_version(self) -> Optional[str]:
        """Schema version of the context used to build target-aware inputs."""

        return self._bound_target_context_schema_version

    @property
    def target_context_binding_digest(self) -> Optional[str]:
        """Digest that binds the policy to target and potential configuration."""

        return self._bound_target_context_digest

    @property
    def feature_provider_binding_digest(self) -> Optional[str]:
        """Metadata digest for an explicitly supplied feature provider."""

        return self._feature_provider_binding_digest

    @property
    def feature_schema_version(self) -> str:
        """Feature schema used by the policy's weight vector."""

        return self.feature_provider.schema_version

    def bind_target_context(self, target_context: "DenseTargetContext") -> None:
        """Bind an initially target-free, zero-weight policy to one target.

        This is useful when an environment constructs its target context after
        a caller has constructed a default policy.  Non-zero weights cannot be
        silently reinterpreted under another schema or target, so rebinding
        them raises a clear error.  Equivalent contexts with the same
        fingerprint and context schema may be rebound safely.
        """

        if target_context is None:
            raise TypeError("bind_target_context requires a DenseTargetContext")

        if self._explicit_feature_provider:
            raise ValueError(
                "cannot bind a target_context to a policy with an explicit "
                "feature_provider; create a TargetContextFeatureProvider instead"
            )

        candidate_provider = LegacyFeatureProvider(target_context)
        expected_dim = candidate_provider.dimension
        fingerprint = _context_fingerprint(target_context)
        schema_version = _context_schema_version(target_context)
        binding_digest = _context_binding_digest(target_context)
        if self.target_context is not None:
            if (
                fingerprint != self._bound_target_fingerprint
                or schema_version != self._bound_target_context_schema_version
                or binding_digest != self._bound_target_context_digest
            ):
                raise ValueError(
                    "policy is already bound to a different target context; "
                    "create a new policy for a different target"
                )
            if self.feature_dim != expected_dim:
                raise ValueError("bound target context has an incompatible feature dimension")
            self.target_context = target_context
            self.feature_provider = candidate_provider
            return

        if self.feature_dim != expected_dim:
            if np.any(self.theta != 0.0):
                raise ValueError(
                    "cannot bind a non-zero target-free weight vector to the "
                    "target-aware feature schema"
                )
            self.feature_dim = expected_dim
            self.theta = np.zeros(self.feature_dim, dtype=np.float64)

        self.target_context = target_context
        self.feature_provider = candidate_provider
        self._bound_target_fingerprint = fingerprint
        self._bound_target_context_schema_version = schema_version
        self._bound_target_context_digest = binding_digest

    def bind_feature_provider(self, feature_provider: FeatureProvider) -> None:
        """Bind an opt-in provider without silently reinterpreting weights.

        An equivalent provider may replace an already bound explicit provider.
        Moving between provider schemas is allowed only while the policy has a
        zero weight vector, matching the established target-context binding
        safety rule.
        """

        provider = validate_feature_provider(feature_provider)
        expected_dim = int(provider.dimension)
        binding_digest = _feature_provider_binding_digest(provider)

        if self._explicit_feature_provider:
            if binding_digest == self._feature_provider_binding_digest:
                if self.feature_dim != expected_dim:
                    raise ValueError("equivalent feature provider has incompatible dimension")
                self.feature_provider = provider
                return
            if np.any(self.theta != 0.0):
                raise ValueError(
                    "cannot bind a non-zero policy to a different feature provider"
                )
        elif np.any(self.theta != 0.0):
            raise ValueError(
                "cannot replace a non-zero legacy policy with an explicit "
                "feature provider"
            )

        if self.feature_dim != expected_dim:
            if np.any(self.theta != 0.0):
                raise ValueError(
                    "cannot bind a non-zero weight vector to a different feature schema"
                )
            self.feature_dim = expected_dim
            self.theta = np.zeros(self.feature_dim, dtype=np.float64)

        self.feature_provider = provider
        self._explicit_feature_provider = True
        self._feature_provider_binding_digest = binding_digest
        self._bound_feature_provider_target_fingerprint = (
            _feature_provider_target_fingerprint(provider)
        )
        self.target_context = None
        self._bound_target_fingerprint = None
        self._bound_target_context_schema_version = None
        self._bound_target_context_digest = None

    def weight_digest(self) -> str:
        """Return a stable digest that binds weights to their feature schema."""

        digest = sha256()
        digest.update(b"linear-sarsa-policy-v1\0")
        digest.update(self.feature_schema_version.encode("utf-8"))
        digest.update(b"\0")
        digest.update((self.target_fingerprint or "").encode("utf-8"))
        digest.update(b"\0")
        binding_digest = (
            self.target_context_binding_digest
            or self.feature_provider_binding_digest
            or ""
        )
        digest.update(binding_digest.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(self.feature_dim).encode("ascii"))
        digest.update(b"\0")
        # Explicit little-endian conversion avoids host byte-order ambiguity
        # in JSON reports and saved checkpoint metadata.
        weights = np.ascontiguousarray(self.theta, dtype="<f8")
        digest.update(weights.tobytes(order="C"))
        return f"sha256:{digest.hexdigest()}"

    def metadata(self) -> dict[str, object]:
        """Return checkpoint/report metadata for the active target binding."""

        metadata = dict(self.feature_provider.metadata())
        provider_target_fingerprint = metadata.get("target_fingerprint")
        metadata.update(
            {
                "algorithm": "linear-semi-gradient-sarsa(0)",
                "learning_rate": self.lr,
                "discount": self.gamma,
                "seed": self.seed,
                # An explicit problem provider owns its target binding.  Keep
                # that metadata rather than replacing it with the absence of
                # a DenseTargetContext.
                "target_fingerprint": self.target_fingerprint
                if self.target_fingerprint is not None
                else provider_target_fingerprint,
                "target_context_schema_version": self.target_context_schema_version,
                "target_context_binding_digest": self.target_context_binding_digest,
                "weight_digest": self.weight_digest(),
            }
        )
        if self.feature_provider_binding_digest is not None:
            metadata["feature_provider_binding_digest"] = (
                self.feature_provider_binding_digest
            )
        return metadata

    def checkpoint_payload(
        self,
        *,
        reward_schema_version: str,
        target_metric_schema_version: str,
        certification_schema_version: str,
        epsilon_schedule: Mapping[str, float],
        code_commit_sha: str,
        dirty_worktree: bool,
        profile_name: str,
    ) -> dict[str, object]:
        """Return a schema-bound, JSON-ready linear-policy checkpoint."""

        if not isinstance(dirty_worktree, bool):
            raise TypeError("dirty_worktree must be a bool")
        required_strings = {
            "reward_schema_version": reward_schema_version,
            "target_metric_schema_version": target_metric_schema_version,
            "certification_schema_version": certification_schema_version,
            "code_commit_sha": code_commit_sha,
            "profile_name": profile_name,
        }
        if any(not isinstance(value, str) or not value for value in required_strings.values()):
            raise ValueError("checkpoint schema and provenance strings must be non-empty")
        provider_metadata = dict(self.feature_provider.metadata())
        ordered_names = tuple(str(name) for name in self.feature_provider.names)
        payload: dict[str, object] = {
            "checkpoint_schema": "linear-frontier-policy-checkpoint-v1",
            "profile_name": profile_name,
            "algorithm": self.metadata()["algorithm"],
            "feature_schema_version": self.feature_schema_version,
            "ordered_feature_names": list(ordered_names),
            "feature_dimension": self.feature_dim,
            "reward_schema_version": reward_schema_version,
            "target_metric_schema_version": target_metric_schema_version,
            "certification_schema_version": certification_schema_version,
            "target_fingerprint": self.target_fingerprint
            or provider_metadata.get("target_fingerprint"),
            "target_context_binding_digest": self.target_context_binding_digest
            or self.feature_provider_binding_digest,
            "learning_rate": self.lr,
            "discount": self.gamma,
            "epsilon_schedule": {
                str(key): float(value) for key, value in epsilon_schedule.items()
            },
            "training_seed": self.seed,
            "weight_digest": self.weight_digest(),
            "code_commit_sha": code_commit_sha,
            "dirty_worktree": dirty_worktree,
            "weights": [float(value) for value in self.theta],
        }
        return payload

    def save_checkpoint(self, path: str | Path, **metadata: Any) -> dict[str, object]:
        """Atomically write a schema-bound JSON checkpoint."""

        payload = self.checkpoint_payload(**metadata)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
        return payload

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
        *,
        feature_provider: FeatureProvider,
        expected_profile_name: Optional[str] = None,
        expected_reward_schema: Optional[str] = None,
        expected_target_metric_schema: Optional[str] = None,
        expected_certification_schema: Optional[str] = None,
    ) -> tuple["LinearQPolicy", dict[str, object]]:
        """Load weights only when every declared schema matches exactly."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("checkpoint_schema") != "linear-frontier-policy-checkpoint-v1":
            raise ValueError("unsupported policy checkpoint schema")
        provider = validate_feature_provider(feature_provider)
        checks = {
            "feature_schema_version": provider.schema_version,
            "feature_dimension": int(provider.dimension),
            "ordered_feature_names": list(provider.names),
        }
        optional_checks = {
            "profile_name": expected_profile_name,
            "reward_schema_version": expected_reward_schema,
            "target_metric_schema_version": expected_target_metric_schema,
            "certification_schema_version": expected_certification_schema,
        }
        checks.update(
            {key: value for key, value in optional_checks.items() if value is not None}
        )
        for key, expected in checks.items():
            if payload.get(key) != expected:
                raise ValueError(
                    f"policy checkpoint {key} mismatch: expected {expected!r}, "
                    f"found {payload.get(key)!r}"
                )
        weights = np.asarray(payload.get("weights"), dtype=np.float64)
        if weights.shape != (int(provider.dimension),) or not np.isfinite(weights).all():
            raise ValueError("checkpoint weights have an incompatible or invalid shape")
        policy = cls(
            feature_provider=provider,
            lr=float(payload["learning_rate"]),
            gamma=float(payload["discount"]),
            seed=payload.get("training_seed"),
        )
        policy.theta[:] = weights
        if policy.weight_digest() != payload.get("weight_digest"):
            raise ValueError("checkpoint weight digest does not match its bound schema")
        return policy, payload

    def _features(
        self,
        state: CircuitState,
        frontier: Optional[Iterable[object]] = None,
    ) -> np.ndarray:
        features = self.feature_provider.extract(state, frontier)
        if features.shape[0] != self.theta.shape[0]:
            raise ValueError(
                f"feature dimension changed from {self.theta.shape[0]} to {features.shape[0]}"
            )
        return features

    def build_feature_batch(
        self,
        nodes: Sequence[SearchNode],
    ) -> PolicyFeatureBatch:
        """Freeze all candidate features before a frontier transition.

        Article-aware providers may expose a vectorized ``build_batch`` API.
        Legacy providers retain identical behavior through the row-wise
        fallback.  Either path produces one immutable float64 matrix so SARSA
        never recomputes the selected feature vector after archive mutation.
        """

        started = time.perf_counter_ns()
        target_context = getattr(self.feature_provider, "target_context", None)
        cache_metrics = getattr(target_context, "cache_metrics", None)
        target_time_before = 0
        if callable(cache_metrics):
            target_time_before = int(
                cache_metrics().get("target_metric_time_ns", 0)
            )
        frozen_nodes = tuple(nodes)
        build_batch = getattr(self.feature_provider, "build_batch", None)
        if callable(build_batch):
            provider_batch = build_batch(frozen_nodes)
            if isinstance(provider_batch, np.ndarray):
                matrix = provider_batch
            else:
                matrix = getattr(
                    provider_batch,
                    "features",
                    getattr(provider_batch, "matrix", None),
                )
                if matrix is None:
                    features_for = getattr(
                        provider_batch,
                        "features_for_node",
                        getattr(provider_batch, "features_for", None),
                    )
                    if not callable(features_for):
                        raise TypeError(
                            "feature provider build_batch must return a matrix or "
                            "an object with features/features_for"
                        )
                    matrix = np.stack(
                        [np.asarray(features_for(node), dtype=np.float64) for node in frozen_nodes],
                        axis=0,
                    ) if frozen_nodes else np.empty((0, self.feature_dim), dtype=np.float64)
        else:
            matrix = (
                np.stack(
                    [self._features(node.state, frozen_nodes) for node in frozen_nodes],
                    axis=0,
                )
                if frozen_nodes
                else np.empty((0, self.feature_dim), dtype=np.float64)
            )
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (len(frozen_nodes), self.feature_dim):
            raise ValueError(
                "feature provider batch shape does not match frontier and policy dimension"
            )
        result = PolicyFeatureBatch(frozen_nodes, matrix)
        elapsed = time.perf_counter_ns() - started
        target_time_after = target_time_before
        if callable(cache_metrics):
            target_time_after = int(
                cache_metrics().get("target_metric_time_ns", target_time_before)
            )
        # Target reconstruction/dense comparison is reported independently.
        # Keep feature construction exclusive so aggregate timing cannot count
        # the same work in both categories.
        target_delta = max(0, target_time_after - target_time_before)
        self.feature_evaluation_count += len(frozen_nodes)
        self.feature_time_ns += max(0, elapsed - target_delta)
        return result

    def q_value_from_features(self, features: np.ndarray) -> float:
        phi = np.asarray(features, dtype=np.float64)
        if phi.shape != self.theta.shape:
            raise ValueError("frozen feature vector has incompatible dimension")
        return float(np.dot(self.theta, phi))

    def q_value(
        self,
        state: CircuitState,
        frontier: Optional[Iterable[object]] = None,
    ) -> float:
        return float(np.dot(self.theta, self._features(state, frontier)))

    def node_value(
        self,
        node: SearchNode,
        frontier: Optional[Iterable[object]] = None,
    ) -> float:
        return self.q_value(node.state, frontier)

    def evaluate_nodes(self, nodes: Sequence[SearchNode]):
        """Evaluate nodes against one shared, order-invariant frontier context."""

        batch = self.build_feature_batch(nodes)
        return [
            (node, self.q_value_from_features(batch.features[index]))
            for index, node in enumerate(batch.nodes)
        ]

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
        ranking_started = time.perf_counter_ns()
        if self.rng.random() < epsilon:
            selected = nodes[int(self.rng.integers(len(nodes)))]
            self.ranking_time_ns += time.perf_counter_ns() - ranking_started
            return selected
        # Preserve the established fast path for legacy zero-weight policies.
        # The Article V1 zero-weight scheduler is a compute-normalized control
        # for the *same* linear feature/ranking pipeline, however, so it must
        # still materialize its versioned feature batch even though every dot
        # product is zero.  Stable record ID remains only the exact-score tie
        # breaker in both cases.
        article_zero_weight_control = str(self.feature_schema_version).startswith(
            "article-v1-"
        )
        if not np.any(self.theta) and not article_zero_weight_control:
            selected = min(nodes, key=self._stable_id)
            self.ranking_time_ns += time.perf_counter_ns() - ranking_started
            return selected
        # Feature materialization has its own exclusive timer.  Start a fresh
        # ranking interval for scoring and tie-breaking after the batch exists.
        batch = self.build_feature_batch(nodes)
        ranking_started = time.perf_counter_ns()
        selected = self._greedy_from_batch(batch)
        self.ranking_time_ns += time.perf_counter_ns() - ranking_started
        return selected

    def select_from_batch(
        self,
        batch: PolicyFeatureBatch,
        *,
        epsilon: float = 0.1,
    ) -> Optional[SearchNode]:
        """Choose from a previously frozen decision-state feature batch."""

        ranking_started = time.perf_counter_ns()
        nodes = batch.nodes
        if not nodes:
            return None
        if self.rng.random() < epsilon:
            selected = nodes[int(self.rng.integers(len(nodes)))]
            self.ranking_time_ns += time.perf_counter_ns() - ranking_started
            return selected

        # A fresh linear policy gives every record exactly the same score.
        # Resolve that genuine tie directly by stable record ID instead of
        # materialising target-aware features for every open record.  This is
        # semantically identical to the ``max`` below, preserves the public
        # zero-weight baseline, and avoids quadratic dense-context work while
        # a direct-GHZ run is still collecting its first learning signal.
        if not np.any(self.theta):
            selected = min(nodes, key=self._stable_id)
        else:
            selected = self._greedy_from_batch(batch)
        self.ranking_time_ns += time.perf_counter_ns() - ranking_started
        return selected

    def instrumentation(self) -> dict[str, int]:
        return {
            "ranking_time_ns": int(self.ranking_time_ns),
            "feature_time_ns": int(self.feature_time_ns),
            "feature_evaluation_count": int(self.feature_evaluation_count),
        }

    def _greedy_from_batch(self, batch: PolicyFeatureBatch) -> SearchNode:
        nodes = batch.nodes
        values = [
            (
                self.q_value_from_features(batch.features[index]),
                self._stable_id(node),
                node,
            )
            for index, node in enumerate(nodes)
        ]
        # Stable record IDs make ordering changes in a heap irrelevant.  They
        # are a tie-break only and are never part of the learned feature vector.
        return max(values, key=lambda entry: (entry[0], -entry[1]))[2]

    def update_from_features(
        self,
        *,
        current_features: np.ndarray,
        reward: float,
        next_features: Optional[np.ndarray],
        done: bool,
    ) -> float:
        """Apply semi-gradient SARSA using immutable precomputed vectors."""

        phi = np.array(current_features, dtype=np.float64, copy=True)
        if phi.shape != self.theta.shape or not np.isfinite(phi).all():
            raise ValueError("current_features has an incompatible or invalid value")
        q_value = float(np.dot(self.theta, phi))
        if done:
            target = float(reward)
        else:
            if next_features is None:
                raise ValueError("nonterminal SARSA update requires next_features")
            next_phi = np.asarray(next_features, dtype=np.float64)
            if next_phi.shape != self.theta.shape or not np.isfinite(next_phi).all():
                raise ValueError("next_features has an incompatible or invalid value")
            target = float(reward) + self.gamma * float(np.dot(self.theta, next_phi))
        td_error = target - q_value
        self.theta += self.lr * td_error * phi
        return float(td_error)

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
        phi = self._features(state, current_frontier).astype(np.float64)
        next_phi = (
            None
            if done or next_node is None
            else self._features(next_node.state, next_frontier)
        )
        return self.update_from_features(
            current_features=phi,
            reward=reward,
            next_features=next_phi,
            done=bool(done or next_node is None),
        )

    def score_state(
        self,
        state: CircuitState,
        frontier: Optional[Iterable[object]] = None,
    ) -> float:
        """Lower queue priorities correspond to higher learned values."""

        return -self.q_value(state, frontier)


# Explicit alias for readers who prefer the learner's actual algorithm name.
LinearSarsaPolicy = LinearQPolicy


__all__ = ["LinearQPolicy", "LinearSarsaPolicy", "PolicyFeatureBatch"]
