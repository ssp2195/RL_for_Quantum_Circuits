"""Toffoli-parity observations and potential shaping for frontier scheduling.

This module is deliberately an *adapter* over
``search.problems.toffoli_parity``.  It never reconstructs a phase polynomial,
replays a reference witness, changes legal actions, or decides certification.
The problem analyzer remains the sole authority for a prefix's Toffoli parity
progress; this module turns that stable progress object into deterministic
linear-policy observations and a transition-only potential reward.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
import math
from typing import Any, Optional

import numpy as np

from circuit.circuit_state import CircuitState
from enums import GateType
from rl.features import FeatureProvider
from rl.reward import RewardBreakdown


TOFFOLI_PARITY_FEATURE_SCHEMA_VERSION = "frontier-toffoli-parity-v1"
"""Durable schema identifier for the 66-coordinate provider below."""

TOFFOLI_PARITY_MASK_ORDER = (1, 2, 3, 4, 5, 6, 7)
"""Canonical phase-term/basis-mask order: binary ``001`` through ``111``."""

TOFFOLI_PARITY_STAGE_ORDER = ("PRE_H", "CORE", "POST_H", "DONE")
"""Stable stage one-hot ordering supplied by the Toffoli parity problem."""

_EMITTED_WIDTH = len(TOFFOLI_PARITY_MASK_ORDER)
_FEATURE_NAMES = (
    tuple(f"emitted_term_{mask:03b}" for mask in TOFFOLI_PARITY_MASK_ORDER)
    + tuple(f"remaining_term_{mask:03b}" for mask in TOFFOLI_PARITY_MASK_ORDER)
    + tuple(
        f"basis_row_{row_index}_mask_{mask:03b}"
        for row_index in range(3)
        for mask in TOFFOLI_PARITY_MASK_ORDER
    )
    + tuple(
        f"exposed_remaining_term_{mask:03b}"
        for mask in TOFFOLI_PARITY_MASK_ORDER
    )
    + (
        "emitted_fraction",
        "exposed_count_normalized",
        "exposed_fraction_of_remaining",
        "cnot_fraction",
        "cnot_remaining_fraction",
        "phase_fraction",
        "phase_remaining_fraction",
        "identity_basis_distance_normalized",
        "identity_basis",
        "toffoli_parity_potential",
    )
    + tuple(f"stage_{stage.lower()}" for stage in TOFFOLI_PARITY_STAGE_ORDER)
    + tuple(f"wire_{qubit}_depth_fraction" for qubit in range(3))
    + tuple(
        name
        for metric in (
            "emitted_fraction",
            "identity_basis_distance_normalized",
            "exposed_fraction_of_remaining",
        )
        for name in (f"frontier_{metric}_minus_mean", f"frontier_{metric}_minus_max")
    )
    + ("bias",)
)
TOFFOLI_PARITY_FEATURE_DIMENSION = len(_FEATURE_NAMES)
if TOFFOLI_PARITY_FEATURE_DIMENSION != 66:  # pragma: no cover - schema guard
    raise AssertionError("Toffoli parity feature schema must remain 66-dimensional")


@dataclass(frozen=True, slots=True)
class ToffoliParityMetrics:
    """Validated analyzer data plus derived, bounded potential coordinates."""

    stage: str
    basis_rows: tuple[int, int, int]
    emitted_bitset: int
    emitted_masks: tuple[int, ...]
    remaining_masks: tuple[int, ...]
    exposed_remaining_masks: tuple[int, ...]
    identity_basis_distance: int
    identity_basis_distance_normalized: float
    emitted_fraction: float
    exposed_count_normalized: float
    exposed_fraction: float
    potential: float


@dataclass(frozen=True, slots=True)
class ToffoliParityRewardConfig:
    """Transition-only reward parameters for a Toffoli parity provider.

    The potential weights are intentionally not configurable here: they are
    part of the problem contract and fixed in
    :meth:`ToffoliParityFeatureProvider.metrics` as
    ``.55*p + .20*e + .25*p*r``.
    """

    terminal_bonus: float = 20.0
    step_cost: float = 0.05
    potential_scale: float = 4.0
    dead_end_cost: float = 1.0
    reward_clip: float | None = 20.0

    def __post_init__(self) -> None:
        for name in (
            "terminal_bonus",
            "step_cost",
            "potential_scale",
            "dead_end_cost",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.reward_clip is not None and (
            isinstance(self.reward_clip, bool)
            or not isinstance(self.reward_clip, (int, float))
            or not math.isfinite(self.reward_clip)
            or self.reward_clip <= 0.0
        ):
            raise ValueError("reward_clip must be None or a finite positive number")


def _canonical_mapping_digest(mapping: Mapping[int, object]) -> str:
    """Bind all required phase labels/coefficient payloads in metadata."""

    payload = repr(tuple(sorted((int(mask), repr(value)) for mask, value in mapping.items())))
    return f"sha256:{sha256(payload.encode('utf-8')).hexdigest()}"


def _canonical_distance_digest(distance: Mapping[tuple[int, int, int], int]) -> str:
    payload = repr(tuple(sorted((tuple(rows), int(value)) for rows, value in distance.items())))
    return f"sha256:{sha256(payload.encode('utf-8')).hexdigest()}"


def _stage_name(progress: object) -> str:
    value = getattr(progress, "stage", None)
    name = getattr(value, "name", value)
    if not isinstance(name, str):
        raise TypeError("Toffoli parity progress must expose a named stage")
    normalized = name.upper().split(".")[-1]
    if normalized not in TOFFOLI_PARITY_STAGE_ORDER:
        expected = ", ".join(TOFFOLI_PARITY_STAGE_ORDER)
        raise ValueError(f"unsupported Toffoli parity stage {name!r}; expected {expected}")
    return normalized


def _frontier_states(frontier: Optional[Iterable[object]]) -> list[CircuitState]:
    if frontier is None:
        return []
    states: list[CircuitState] = []
    for item in frontier:
        state = getattr(item, "state", item)
        if isinstance(state, CircuitState):
            states.append(state)
    return states


class ToffoliParityFeatureProvider:
    """A 66-D provider driven only by a Toffoli parity progress analyzer.

    ``problem`` can be a public ``ToffoliParityNetworkProblem`` instance with
    an ``analyze(state)`` method.  For focused tests or external integrations,
    callers can instead inject an ``analyzer`` callable and its immutable
    phase-term/distance metadata.  Calling the constructor with no arguments
    lazily imports the canonical ``search.problems.toffoli_parity`` module.
    """

    def __init__(
        self,
        problem: object | None = None,
        *,
        analyzer: Callable[[CircuitState], object] | None = None,
        required_phase_terms: Mapping[int, object] | None = None,
        cnot_basis_distance_to_identity: Mapping[tuple[int, int, int], int] | None = None,
        target_fingerprint: str | None = None,
        problem_schema_version: str | None = None,
        qubit_convention: str | None = None,
    ) -> None:
        if problem is None and analyzer is None:
            problem = self._canonical_problem_module()

        if analyzer is None:
            analyzer = self._resolve_analyzer(problem)
        if not callable(analyzer):
            raise TypeError("Toffoli parity analyzer must be callable")

        required_phase_terms = required_phase_terms or self._resolve_mapping(
            problem,
            "REQUIRED_PHASE_TERMS",
            "required_phase_terms",
        )
        distance = cnot_basis_distance_to_identity or self._resolve_mapping(
            problem,
            "CNOT_BASIS_DISTANCE_TO_IDENTITY",
            "cnot_basis_distance_to_identity",
        )
        if required_phase_terms is None or distance is None:
            raise TypeError(
                "a custom Toffoli parity analyzer must supply required_phase_terms "
                "and cnot_basis_distance_to_identity"
            )

        normalized_terms = {int(mask): value for mask, value in required_phase_terms.items()}
        if tuple(sorted(normalized_terms)) != TOFFOLI_PARITY_MASK_ORDER:
            raise ValueError(
                "required_phase_terms must use exactly masks 001 through 111"
            )
        normalized_distance: dict[tuple[int, int, int], int] = {}
        for raw_rows, raw_distance in distance.items():
            rows = tuple(raw_rows)
            if len(rows) != 3 or any(mask not in TOFFOLI_PARITY_MASK_ORDER for mask in rows):
                raise ValueError("basis distance keys must be triples of masks 001 through 111")
            if (
                isinstance(raw_distance, bool)
                or not isinstance(raw_distance, (int, np.integer))
                or int(raw_distance) < 0
            ):
                raise ValueError("basis distances must be non-negative integers")
            normalized_distance[(int(rows[0]), int(rows[1]), int(rows[2]))] = int(raw_distance)
        if not normalized_distance:
            raise ValueError("cnot_basis_distance_to_identity must not be empty")
        identity_rows = min(normalized_distance, key=lambda rows: (normalized_distance[rows], rows))
        if normalized_distance[identity_rows] != 0:
            raise ValueError("basis-distance table must contain an identity basis at distance zero")
        distance_max = max(normalized_distance.values())
        if distance_max <= 0:
            # A complete 3-qubit CNOT basis table has non-identity states, but
            # use one in the denominator for a small injected test fixture.
            distance_max = 1

        self._analyzer = analyzer
        self._required_phase_terms = normalized_terms
        self._distance = normalized_distance
        self._identity_rows = identity_rows
        self._distance_max = distance_max
        self._phase_term_digest = _canonical_mapping_digest(normalized_terms)
        self._basis_distance_digest = _canonical_distance_digest(normalized_distance)
        self._problem_schema_version = str(
            problem_schema_version
            or self._resolve_value(problem, "schema_version", "problem_schema_version")
            or "toffoli-parity-problem-v1"
        )
        self._qubit_convention = str(
            qubit_convention
            or self._resolve_value(problem, "qubit_convention", "qubit_order")
            or "q0 is LSB; controls q0,q1; target q2"
        )
        fallback_fingerprint = (
            "toffoli-ccx-q0q1-to-q2:"
            + self._phase_term_digest.split(":", 1)[1][:16]
        )
        self._target_fingerprint = str(
            target_fingerprint
            or self._resolve_value(problem, "target_fingerprint", "fingerprint")
            or fallback_fingerprint
        )
        # Progress calculations are reused heavily while computing the
        # frontier-level potential after an expansion.  Cache only by the
        # complete immutable DAG-operation tuple, never a state/node identity
        # or a hash digest alone, so copied authoritative witnesses share a
        # result without allowing mutable-object cache corruption.
        self._metrics_cache: dict[
            tuple[tuple[str, tuple[int, ...]], ...], ToffoliParityMetrics
        ] = {}

    @staticmethod
    def _canonical_problem_module() -> object:
        try:
            from search.problems import toffoli_parity
        except ImportError as exc:  # pragma: no cover - installation diagnostic
            raise ImportError(
                "ToffoliParityFeatureProvider needs search.problems.toffoli_parity "
                "or injected analyzer/metadata"
            ) from exc
        # The module publishes the analyzer and immutable tables, while its
        # public problem object additionally carries the durable schema,
        # qubit convention, and problem-contract fingerprint.  Prefer that
        # object for the no-argument path so checkpoint metadata is just as
        # fully bound as when callers inject a problem explicitly.
        problem_type = getattr(toffoli_parity, "ToffoliParityNetworkProblem", None)
        return problem_type() if callable(problem_type) else toffoli_parity

    @staticmethod
    def _resolve_value(problem: object | None, *names: str) -> object | None:
        if problem is None:
            return None
        for name in names:
            value = getattr(problem, name, None)
            if value is not None:
                return value
        return None

    @classmethod
    def _resolve_mapping(
        cls,
        problem: object | None,
        *names: str,
    ) -> Mapping[int, object] | None:
        value = cls._resolve_value(problem, *names)
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError(f"{names[0]} must be a mapping")
        return value

    @staticmethod
    def _resolve_analyzer(problem: object | None) -> Callable[[CircuitState], object] | None:
        if problem is None:
            return None
        for name in ("analyze", "analyze_toffoli_prefix"):
            value = getattr(problem, name, None)
            if callable(value):
                return value
        return None

    @property
    def schema_version(self) -> str:
        return TOFFOLI_PARITY_FEATURE_SCHEMA_VERSION

    @property
    def dimension(self) -> int:
        return TOFFOLI_PARITY_FEATURE_DIMENSION

    @property
    def names(self) -> tuple[str, ...]:
        return _FEATURE_NAMES

    @property
    def target_fingerprint(self) -> str:
        return self._target_fingerprint

    def metadata(self) -> Mapping[str, object]:
        """Return deterministic binding metadata for policy checkpoints."""

        return {
            "feature_schema_version": self.schema_version,
            "feature_dim": self.dimension,
            "feature_names": self.names,
            "feature_provider": type(self).__name__,
            "problem_schema_version": self._problem_schema_version,
            "target_fingerprint": self._target_fingerprint,
            "phase_term_digest": self._phase_term_digest,
            "basis_distance_digest": self._basis_distance_digest,
            "required_phase_masks": TOFFOLI_PARITY_MASK_ORDER,
            "qubit_convention": self._qubit_convention,
            "potential_formula": "0.55*p + 0.20*e + 0.25*p*r",
        }

    @staticmethod
    def _bit_vector(bitset: int) -> np.ndarray:
        return np.asarray(
            [float(bool(bitset & (1 << index))) for index in range(_EMITTED_WIDTH)],
            dtype=np.float64,
        )

    def _progress(self, state: CircuitState) -> object:
        if not isinstance(state, CircuitState):
            raise TypeError("Toffoli parity features require a CircuitState")
        if state.dag.num_qubits != 3:
            raise ValueError("Toffoli parity features require exactly three qubits")
        return self._analyzer(state)

    def metrics(self, state: CircuitState) -> ToffoliParityMetrics:
        """Adapt the problem's immutable prefix progress to bounded metrics."""

        if not isinstance(state, CircuitState):
            raise TypeError("Toffoli parity features require a CircuitState")
        if state.dag.num_qubits != 3:
            raise ValueError("Toffoli parity features require exactly three qubits")
        cache_key = tuple(
            (gate.gate_type.name, tuple(int(qubit) for qubit in gate.qubits))
            for gate in state.dag.gates
        )
        cached = self._metrics_cache.get(cache_key)
        if cached is not None:
            return cached

        progress = self._progress(state)
        stage = _stage_name(progress)
        raw_rows = tuple(getattr(progress, "basis_rows", ()))
        if len(raw_rows) != 3 or any(
            isinstance(mask, bool) or not isinstance(mask, (int, np.integer))
            for mask in raw_rows
        ):
            raise TypeError("Toffoli parity progress must expose three integer basis_rows")
        current_rows = tuple(int(mask) for mask in raw_rows)
        if any(mask not in TOFFOLI_PARITY_MASK_ORDER for mask in current_rows):
            raise ValueError("Toffoli parity basis_rows must be masks 001 through 111")
        # Outside the diagonal parity core the representation is interpreted
        # as the identity basis.  This prevents post-H bookkeeping from
        # becoming an accidental learned CNOT-distance signal.
        basis_rows = current_rows if stage in {"PRE_H", "CORE"} else self._identity_rows
        if basis_rows not in self._distance:
            raise ValueError(f"basis rows {basis_rows!r} are absent from the CNOT distance table")

        emitted = getattr(progress, "emitted_terms", None)
        if isinstance(emitted, bool) or not isinstance(emitted, (int, np.integer)):
            raise TypeError("Toffoli parity progress emitted_terms must be an integer bitset")
        emitted_bitset = int(emitted)
        if emitted_bitset < 0 or emitted_bitset & ~((1 << _EMITTED_WIDTH) - 1):
            raise ValueError("Toffoli parity emitted_terms must be a seven-bit term bitset")

        emitted_masks = tuple(
            mask
            for index, mask in enumerate(TOFFOLI_PARITY_MASK_ORDER)
            if emitted_bitset & (1 << index)
        )
        remaining_masks = tuple(
            mask for mask in TOFFOLI_PARITY_MASK_ORDER if mask not in emitted_masks
        )
        exposed_remaining_masks = tuple(
            mask for mask in remaining_masks if mask in set(basis_rows)
        )
        emitted_fraction = len(emitted_masks) / _EMITTED_WIDTH
        remaining_count = len(remaining_masks)
        exposed_count_normalized = len(exposed_remaining_masks) / _EMITTED_WIDTH
        exposed_fraction = len(exposed_remaining_masks) / max(1, remaining_count)
        distance = self._distance[basis_rows]
        distance_normalized = distance / self._distance_max
        restoration = 1.0 - distance_normalized
        potential = (
            0.55 * emitted_fraction
            + 0.20 * exposed_fraction
            + 0.25 * emitted_fraction * restoration
        )
        result = ToffoliParityMetrics(
            stage=stage,
            basis_rows=basis_rows,
            emitted_bitset=emitted_bitset,
            emitted_masks=emitted_masks,
            remaining_masks=remaining_masks,
            exposed_remaining_masks=exposed_remaining_masks,
            identity_basis_distance=distance,
            identity_basis_distance_normalized=float(distance_normalized),
            emitted_fraction=float(emitted_fraction),
            exposed_count_normalized=float(exposed_count_normalized),
            exposed_fraction=float(exposed_fraction),
            potential=float(potential),
        )
        self._metrics_cache[cache_key] = result
        return result

    def potential(self, state: CircuitState) -> float:
        return self.metrics(state).potential

    def frontier_potential(self, frontier: Optional[Iterable[object]]) -> float:
        """Return the prescribed maximum potential over the active frontier."""

        states = _frontier_states(frontier)
        if not states:
            return 0.0
        return float(max(self.potential(state) for state in states))

    @staticmethod
    def _basis_features(rows: tuple[int, int, int]) -> np.ndarray:
        return np.asarray(
            [
                float(row_mask == mask)
                for row_mask in rows
                for mask in TOFFOLI_PARITY_MASK_ORDER
            ],
            dtype=np.float64,
        )

    def _resource_scalars(self, state: CircuitState, metrics: ToffoliParityMetrics) -> np.ndarray:
        cnot_count = sum(
            gate.gate_type is GateType.CNOT for gate in state.dag.gates
        )
        cnot_limit = getattr(state.budget, "max_two_qubit_count", None)
        if cnot_limit is None:
            cnot_limit = state.budget.max_gates
        cnot_denominator = max(1.0, float(cnot_limit))
        phase_denominator = max(1.0, float(state.budget.max_t_count))
        identity_basis = float(metrics.identity_basis_distance == 0)
        return np.asarray(
            [
                metrics.emitted_fraction,
                metrics.exposed_count_normalized,
                metrics.exposed_fraction,
                float(cnot_count) / cnot_denominator,
                max(0.0, float(cnot_limit - cnot_count)) / cnot_denominator,
                float(state.t_count) / phase_denominator,
                max(0.0, float(state.budget.max_t_count - state.t_count))
                / phase_denominator,
                metrics.identity_basis_distance_normalized,
                identity_basis,
                metrics.potential,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _depth_features(state: CircuitState) -> np.ndarray:
        depths = tuple(getattr(state, "wire_depths", ()))
        denominator = max(1.0, float(state.budget.max_depth))
        return np.asarray(
            [float(depths[qubit] if qubit < len(depths) else 0.0) / denominator for qubit in range(3)],
            dtype=np.float64,
        )

    def _frontier_delta_features(
        self,
        metrics: ToffoliParityMetrics,
        frontier_states: list[CircuitState],
    ) -> np.ndarray:
        values = np.asarray(
            [
                metrics.emitted_fraction,
                metrics.identity_basis_distance_normalized,
                metrics.exposed_fraction,
            ],
            dtype=np.float64,
        )
        if not frontier_states:
            return np.zeros(6, dtype=np.float64)
        frontier_values = np.asarray(
            [
                (
                    item.emitted_fraction,
                    item.identity_basis_distance_normalized,
                    item.exposed_fraction,
                )
                for item in (self.metrics(state) for state in frontier_states)
            ],
            dtype=np.float64,
        )
        # Sorting scalar columns makes all reductions invariant to arbitrary
        # frontier storage order at the floating-point bit level.
        sorted_values = np.sort(frontier_values, axis=0)
        mean = np.mean(sorted_values, axis=0)
        maximum = np.max(sorted_values, axis=0)
        return np.asarray(
            [component for pair in zip(values - mean, values - maximum) for component in pair],
            dtype=np.float64,
        )

    def extract(
        self,
        state: CircuitState,
        frontier: Optional[Iterable[object]] = None,
    ) -> np.ndarray:
        metrics = self.metrics(state)
        emitted_vector = self._bit_vector(metrics.emitted_bitset)
        remaining_bitset = ((1 << _EMITTED_WIDTH) - 1) ^ metrics.emitted_bitset
        remaining_vector = self._bit_vector(remaining_bitset)
        exposed_bitset = sum(
            1 << TOFFOLI_PARITY_MASK_ORDER.index(mask)
            for mask in metrics.exposed_remaining_masks
        )
        exposed_vector = self._bit_vector(exposed_bitset)
        stage_vector = np.asarray(
            [float(metrics.stage == stage) for stage in TOFFOLI_PARITY_STAGE_ORDER],
            dtype=np.float64,
        )
        frontier_states = _frontier_states(frontier)
        features = np.concatenate(
            (
                emitted_vector,
                remaining_vector,
                self._basis_features(metrics.basis_rows),
                exposed_vector,
                self._resource_scalars(state, metrics),
                stage_vector,
                self._depth_features(state),
                self._frontier_delta_features(metrics, frontier_states),
                np.asarray([1.0], dtype=np.float64),
            )
        ).astype(np.float32, copy=False)
        if features.shape != (self.dimension,):  # pragma: no cover - schema guard
            raise AssertionError("Toffoli parity feature dimension drifted")
        return features


class ToffoliParityRewardModel:
    """Potential-shaped transition reward backed by a parity feature provider."""

    def __init__(
        self,
        provider: ToffoliParityFeatureProvider,
        config: ToffoliParityRewardConfig | None = None,
    ) -> None:
        if not isinstance(provider, ToffoliParityFeatureProvider):
            raise TypeError("provider must be a ToffoliParityFeatureProvider")
        self.provider = provider
        self.config = config or ToffoliParityRewardConfig()
        if not isinstance(self.config, ToffoliParityRewardConfig):
            raise TypeError("config must be a ToffoliParityRewardConfig")

    def frontier_potential(self, frontier: Optional[Iterable[object]]) -> float:
        return self.provider.frontier_potential(frontier)

    @staticmethod
    def _finite(name: str, value: float) -> float:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    def reward(
        self,
        *,
        potential_before: float,
        potential_after: float,
        certified: bool,
        dead_end: bool,
        selected_node_potential: float | None = None,
        best_generated_child_potential: float | None = None,
    ) -> RewardBreakdown:
        """Return the same auditable transition shape as existing reward models.

        ``selected_node_potential`` and ``best_generated_child_potential`` are
        optional compatibility diagnostics for callers already using the
        generic target-progress environment hook.  Neither creates a hidden
        child-count or archive-pruning reward term.
        """

        before = self._finite("potential_before", potential_before)
        after = self._finite("potential_after", potential_after)
        selected = before if selected_node_potential is None else self._finite(
            "selected_node_potential", selected_node_potential
        )
        generated = after if best_generated_child_potential is None else self._finite(
            "best_generated_child_potential", best_generated_child_potential
        )
        potential_delta = after - before
        terminal_bonus = self.config.terminal_bonus if certified else 0.0
        dead_end_cost = self.config.dead_end_cost if dead_end else 0.0
        raw_reward = (
            terminal_bonus
            - self.config.step_cost
            + self.config.potential_scale * potential_delta
            - dead_end_cost
        )
        clipped_reward = (
            raw_reward
            if self.config.reward_clip is None
            else float(np.clip(raw_reward, -self.config.reward_clip, self.config.reward_clip))
        )
        return RewardBreakdown(
            reward=float(clipped_reward),
            potential_before=before,
            potential_after=after,
            potential_delta=float(potential_delta),
            selected_node_potential=selected,
            best_generated_child_potential=generated,
            terminal_bonus=float(terminal_bonus),
            step_cost=float(self.config.step_cost),
            dead_end_cost=float(dead_end_cost),
            raw_reward=float(raw_reward),
            clipped_reward=float(clipped_reward),
        )

    def metadata(self) -> Mapping[str, object]:
        return {
            "reward_model": type(self).__name__,
            "reward_config": asdict(self.config),
            "potential_formula": "0.55*p + 0.20*e + 0.25*p*r",
            "provider": dict(self.provider.metadata()),
        }


__all__ = [
    "TOFFOLI_PARITY_FEATURE_DIMENSION",
    "TOFFOLI_PARITY_FEATURE_SCHEMA_VERSION",
    "TOFFOLI_PARITY_MASK_ORDER",
    "TOFFOLI_PARITY_STAGE_ORDER",
    "ToffoliParityFeatureProvider",
    "ToffoliParityMetrics",
    "ToffoliParityRewardConfig",
    "ToffoliParityRewardModel",
]
