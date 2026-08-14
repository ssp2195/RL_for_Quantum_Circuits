"""A small, transparent semi-gradient SARSA frontier-record scheduler."""

from __future__ import annotations

from hashlib import sha256
from typing import Iterable, Mapping, Optional, Sequence, TYPE_CHECKING

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

        return [(node, self.node_value(node, nodes)) for node in nodes]

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
        if self.rng.random() < epsilon:
            return nodes[int(self.rng.integers(len(nodes)))]

        # A fresh linear policy gives every record exactly the same score.
        # Resolve that genuine tie directly by stable record ID instead of
        # materialising target-aware features for every open record.  This is
        # semantically identical to the ``max`` below, preserves the public
        # zero-weight baseline, and avoids quadratic dense-context work while
        # a direct-GHZ run is still collecting its first learning signal.
        if not np.any(self.theta):
            return min(nodes, key=self._stable_id)

        values = [(self.node_value(node, nodes), self._stable_id(node), node) for node in nodes]
        # Stable record IDs make ordering changes in a heap irrelevant.  They
        # are a tie-break only and are never part of the learned feature vector.
        return max(values, key=lambda entry: (entry[0], -entry[1]))[2]

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
        q_value = float(np.dot(self.theta, phi))

        if done or next_node is None:
            target = float(reward)
        else:
            target = float(reward) + self.gamma * self.node_value(next_node, next_frontier)

        td_error = target - q_value
        self.theta += self.lr * td_error * phi
        return float(td_error)

    def score_state(
        self,
        state: CircuitState,
        frontier: Optional[Iterable[object]] = None,
    ) -> float:
        """Lower queue priorities correspond to higher learned values."""

        return -self.q_value(state, frontier)


# Explicit alias for readers who prefer the learner's actual algorithm name.
LinearSarsaPolicy = LinearQPolicy


__all__ = ["LinearQPolicy", "LinearSarsaPolicy"]
