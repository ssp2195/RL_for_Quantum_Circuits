"""Versioned linear frontier features for article and extended experiments.

``ArticleV1FeatureProvider`` implements the exact feature contract from
``Article_limited_scope.md`` plus the operational definitions in the Article
V1 implementation plan using an exact incremental frontier index. It produces
``[1, x, z, (b/B) x]`` from one frozen frontier snapshot. The preserved
all-pairs oracle is ``ArticleV1ReferenceFeatureProvider``. The older
37-coordinate implementation is retained under
``ExtendedArticleFeatureProvider``; ``ArticleFeatureProvider`` remains its
versioned compatibility wrapper.

Target-relative values are ranking diagnostics only. They reconstruct a
candidate from its authoritative DAG and never decide legality, archive
acceptance, or terminal certification.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import comb, log2, sqrt
from time import perf_counter_ns
from types import MappingProxyType
from typing import Any, Optional
import warnings

import numpy as np

from canonical.canonicalizer import Canonicalizer
from certification.simulator import SynthesisTarget, unitary_from_gates
from certification.unitary_phase_metrics import (
    PROJECTIVE_UNITARY_METRICS_SCHEMA,
    projective_unitary_metrics,
)
from circuit.circuit_state import CircuitState
from enums import GateType


# Article V1 schemas and exact coordinate order.
ARTICLE_V1_TARGET_METRIC_SCHEMA_VERSION = PROJECTIVE_UNITARY_METRICS_SCHEMA
ARTICLE_V1_FEATURE_SCHEMA_VERSION = "article-v1-31d"
ARTICLE_V1_NO_TARGET_FEATURE_SCHEMA_VERSION = "article-v1-no-target-28d"
ARTICLE_V1_NO_Z_FEATURE_SCHEMA_VERSION = "article-v1-no-z-21d"
ARTICLE_V1_REFERENCE_EVALUATOR_SCHEMA_VERSION = "article-v1-reference-all-pairs-v1"
ARTICLE_V1_EXACT_INCREMENTAL_EVALUATOR_SCHEMA_VERSION = (
    "article-v1-exact-incremental-v2"
)
ARTICLE_V1_STANDARDIZATION_ETA = 1e-8
ARTICLE_V1_DTYPE = np.dtype(np.float64)
_TARGET_METRIC_ROUNDOFF_TOLERANCE = 1e-10

ARTICLE_V1_COORDINATE_NAMES = (
    "t_count",
    "two_qubit_count",
    "gate_count",
    "depth",
    "rotation_count",
    "anticommuting_pair_count",
    "mean_pauli_weight",
    "target_process_infidelity",
    "frontier_resource_dominance_fraction",
    "archive_novelty",
)
_ARTICLE_V1_TARGET_COORDINATE = "target_process_infidelity"


def _feature_names(
    coordinates: Sequence[str], *, include_frontier_context: bool
) -> tuple[str, ...]:
    names = ("bias",) + tuple(f"x.{name}" for name in coordinates)
    if include_frontier_context:
        names += tuple(f"z.{name}" for name in coordinates)
    return names + tuple(f"budget_x.{name}" for name in coordinates)


ARTICLE_V1_FEATURE_NAMES = _feature_names(
    ARTICLE_V1_COORDINATE_NAMES, include_frontier_context=True
)
ARTICLE_V1_NO_TARGET_COORDINATE_NAMES = tuple(
    name
    for name in ARTICLE_V1_COORDINATE_NAMES
    if name != _ARTICLE_V1_TARGET_COORDINATE
)
ARTICLE_V1_NO_TARGET_FEATURE_NAMES = _feature_names(
    ARTICLE_V1_NO_TARGET_COORDINATE_NAMES, include_frontier_context=True
)
ARTICLE_V1_NO_Z_FEATURE_NAMES = _feature_names(
    ARTICLE_V1_COORDINATE_NAMES, include_frontier_context=False
)


def _complex_square_matrix(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix, dtype=np.complex128)


def process_infidelity(
    target_unitary: Any,
    candidate_unitary: Any,
    *,
    roundoff_tolerance: float = _TARGET_METRIC_ROUNDOFF_TOLERANCE,
) -> float:
    """Return ``1 - |Tr(V^dagger U)|^2 / d^2`` in ``[0, 1]``.

    Only harmless floating-point overshoot is clipped. A value outside the
    physical interval by more than ``roundoff_tolerance`` is reported.
    """

    target = _complex_square_matrix(target_unitary, name="target_unitary")
    candidate = _complex_square_matrix(candidate_unitary, name="candidate_unitary")
    if candidate.shape != target.shape:
        raise ValueError(
            "candidate_unitary shape does not match target_unitary: "
            f"{candidate.shape!r} != {target.shape!r}"
        )
    tolerance = float(roundoff_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("roundoff_tolerance must be finite and non-negative")

    # ``roundoff_tolerance`` remains in the compatibility signature, but the
    # shared helper owns validation and physical-range clipping for Article V1.
    return projective_unitary_metrics(candidate, target).process_infidelity


WitnessIdentity = tuple[int, tuple[tuple[str, tuple[int, ...]], ...]]
TargetMetricCacheKey = tuple[str, WitnessIdentity]


def _dag_witness_identity(state: CircuitState) -> WitnessIdentity:
    if not isinstance(state, CircuitState):
        raise TypeError("article target metrics require a CircuitState")
    num_qubits = int(state.dag.num_qubits)
    operations: list[tuple[str, tuple[int, ...]]] = []
    for gate in state.dag.gates:
        gate_type = getattr(gate, "gate_type", None)
        gate_name = getattr(gate_type, "name", gate_type)
        if not isinstance(gate_name, str):
            raise TypeError(f"DAG gate has no stable name: {gate!r}")
        qubits = tuple(getattr(gate, "qubits", ()))
        if any(isinstance(qubit, bool) or not isinstance(qubit, int) for qubit in qubits):
            raise TypeError(f"DAG gate has invalid qubit labels: {gate!r}")
        operations.append((gate_name.upper(), qubits))
    return num_qubits, tuple(operations)


class ArticleTargetContext:
    """Dense Article V1 target metric with DAG-witness-safe memoization."""

    schema_version = ARTICLE_V1_TARGET_METRIC_SCHEMA_VERSION

    def __init__(self, target: SynthesisTarget | np.ndarray) -> None:
        value = target.unitary if isinstance(target, SynthesisTarget) else target
        matrix = _complex_square_matrix(value, name="target_unitary")
        dimension = matrix.shape[0]
        num_qubits = int(round(log2(dimension)))
        if 2**num_qubits != dimension:
            raise ValueError("target dimension must be an exact power of two")

        stored = np.array(matrix, dtype=np.complex128, copy=True, order="C")
        stored.setflags(write=False)
        digest = sha256()
        digest.update(self.schema_version.encode("ascii"))
        digest.update(repr(stored.shape).encode("ascii"))
        digest.update(stored.tobytes(order="C"))

        self._target_unitary = stored
        self.num_qubits = num_qubits
        self.fingerprint = f"sha256:{digest.hexdigest()}"
        self._cache: dict[TargetMetricCacheKey, float] = {}
        self._evaluation_count = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._metric_time_ns = 0

    @classmethod
    def from_certification_engine(cls, engine: object) -> "ArticleTargetContext":
        """Construct from a certifier exposing an exact target or target matrix."""

        target = getattr(engine, "target", None)
        if target is None:
            target = getattr(engine, "target_unitary", None)
        if target is None:
            raise ValueError("certification engine exposes no exact target")
        return cls(target)

    @property
    def target_unitary(self) -> np.ndarray:
        return self._target_unitary

    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    @property
    def metric_time_ns(self) -> int:
        """Nanoseconds spent reconstructing/evaluating dense cache misses."""

        return self._metric_time_ns

    def witness_identity(self, state: CircuitState) -> WitnessIdentity:
        identity = _dag_witness_identity(state)
        if identity[0] != self.num_qubits:
            raise ValueError(
                "candidate DAG width does not match target: "
                f"{identity[0]} != {self.num_qubits}"
            )
        return identity

    def cache_key(self, state: CircuitState) -> TargetMetricCacheKey:
        return self.fingerprint, self.witness_identity(state)

    def distance(self, state: CircuitState) -> float:
        """Return Article Eq. (86), reconstructing a cache miss from the DAG."""

        key = self.cache_key(state)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        self._cache_misses += 1
        self._evaluation_count += 1
        started_ns = perf_counter_ns()
        try:
            candidate = unitary_from_gates(self.num_qubits, state.dag.gates)
            value = process_infidelity(self._target_unitary, candidate)
        finally:
            self._metric_time_ns += perf_counter_ns() - started_ns
        self._cache[key] = value
        return value

    def process_infidelity(self, state: CircuitState) -> float:
        """Explicit alias for :meth:`distance`."""

        return self.distance(state)

    def cache_metrics(self) -> dict[str, int | str]:
        return {
            "target_metric_schema_version": self.schema_version,
            "target_metric_evaluation_count": self._evaluation_count,
            "target_metric_cache_hits": self._cache_hits,
            "target_metric_cache_misses": self._cache_misses,
            "target_metric_cache_size": len(self._cache),
            "target_metric_time_ns": self._metric_time_ns,
        }


@dataclass(frozen=True, slots=True)
class FrontierFeatureSnapshot:
    """Read-only feature rows calculated from one complete decision frontier."""

    schema_version: str
    snapshot_id: str
    record_ids: tuple[int, ...]
    frontier_nodes: tuple[object, ...]
    expansions_completed: int
    expansion_budget: int
    archive_generation_counts: Mapping[object, int]
    target_fingerprint: str | None
    feature_vectors: Mapping[int, np.ndarray]
    candidate_vectors: Mapping[int, np.ndarray]

    def features_for_record(self, record_id: int) -> np.ndarray:
        try:
            return self.feature_vectors[int(record_id)]
        except (KeyError, TypeError, ValueError) as exc:
            raise KeyError(f"record ID {record_id!r} is not in this snapshot") from exc

    def features_for_node(self, node: object) -> np.ndarray:
        record_id = getattr(node, "record_id", None)
        if record_id is None:
            raise KeyError("node has no persistent record ID")
        return self.features_for_record(int(record_id))

    def candidate_for_record(self, record_id: int) -> np.ndarray:
        try:
            return self.candidate_vectors[int(record_id)]
        except (KeyError, TypeError, ValueError) as exc:
            raise KeyError(f"record ID {record_id!r} is not in this snapshot") from exc

    def candidate_for_node(self, node: object) -> np.ndarray:
        record_id = getattr(node, "record_id", None)
        if record_id is None:
            raise KeyError("node has no persistent record ID")
        return self.candidate_for_record(int(record_id))


@dataclass(frozen=True, slots=True)
class _FeatureRecord:
    record_id: int
    state: CircuitState


def _readonly_vector(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=ARTICLE_V1_DTYPE).copy()
    result.setflags(write=False)
    return result


def _freeze_generation_counts(values: Mapping[object, int] | None) -> Mapping[object, int]:
    if values is None:
        return MappingProxyType({})
    copied: dict[object, int] = {}
    for key, count in values.items():
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
            raise TypeError("archive generation counts must be integers")
        if int(count) < 0:
            raise ValueError("archive generation counts must be non-negative")
        copied[key] = int(count)
    return MappingProxyType(copied)


def _resource_vector(state: CircuitState) -> tuple[int, ...]:
    return (
        int(state.t_count),
        int(state.two_qubit_count),
        int(state.num_gates),
        *(int(value) for value in state.wire_depths),
    )


def _weakly_dominates(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    if len(left) != len(right):
        raise ValueError("cannot compare resources from different register widths")
    return all(a <= b for a, b in zip(left, right))


def _strictly_dominates(left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return _weakly_dominates(left, right) and any(a < b for a, b in zip(left, right))


class ArticleV1ReferenceFeatureProvider:
    """Reference all-pairs Article V1 evaluator used as a correctness oracle.

    This path intentionally preserves the original nested frontier scan.  It
    is safe for parity tests and bounded microbenchmarks, but should not be
    selected for publication-scale runs.
    """

    schema_version = ARTICLE_V1_FEATURE_SCHEMA_VERSION
    evaluator_schema_version = ARTICLE_V1_REFERENCE_EVALUATOR_SCHEMA_VERSION
    feature_names = ARTICLE_V1_FEATURE_NAMES
    include_target = True
    include_frontier_context = True
    standardization_eta = ARTICLE_V1_STANDARDIZATION_ETA

    def __init__(
        self,
        target_context: ArticleTargetContext,
        *,
        semantic_key: Callable[[CircuitState], object] | None = None,
        search_horizon: int = 1,
        generation_counts: Mapping[object, int] | None = None,
        reference_safe_frontier_size: int = 256,
        warn_above_safe_size: bool = True,
    ) -> None:
        if self.include_target and not isinstance(target_context, ArticleTargetContext):
            raise TypeError("article-v1 target features require an ArticleTargetContext")
        if target_context is not None and not isinstance(target_context, ArticleTargetContext):
            raise TypeError("target_context must be an ArticleTargetContext or None")
        self.target_context = target_context
        if semantic_key is None:
            semantic_key = Canonicalizer().semantic_key
        if not callable(semantic_key):
            raise TypeError("semantic_key must be callable")
        self._semantic_key = semantic_key
        self._search_horizon = 1
        self._search_step = 0
        self._generation_counts: Mapping[object, int] = MappingProxyType({})
        if (
            isinstance(reference_safe_frontier_size, bool)
            or not isinstance(reference_safe_frontier_size, (int, np.integer))
            or int(reference_safe_frontier_size) < 1
        ):
            raise ValueError("reference_safe_frontier_size must be a positive integer")
        if not isinstance(warn_above_safe_size, bool):
            raise TypeError("warn_above_safe_size must be a bool")
        self.reference_safe_frontier_size = int(reference_safe_frontier_size)
        self.warn_above_safe_size = warn_above_safe_size
        self.bind_search_horizon(search_horizon)
        self.bind_generation_counts(generation_counts or {})

    @property
    def names(self) -> tuple[str, ...]:
        return self.feature_names

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    @property
    def candidate_coordinate_names(self) -> tuple[str, ...]:
        if self.include_target:
            return ARTICLE_V1_COORDINATE_NAMES
        return ARTICLE_V1_NO_TARGET_COORDINATE_NAMES

    @property
    def search_horizon(self) -> int:
        return self._search_horizon

    @property
    def search_step(self) -> int:
        return self._search_step

    @property
    def remaining_search_budget_fraction(self) -> float:
        return float((self._search_horizon - self._search_step) / self._search_horizon)

    def bind_search_horizon(self, max_expansions: int) -> None:
        if (
            isinstance(max_expansions, bool)
            or not isinstance(max_expansions, (int, np.integer))
            or int(max_expansions) <= 0
        ):
            raise ValueError("expansion budget must be a positive integer")
        if self._search_step > int(max_expansions):
            raise ValueError("current expansion count exceeds the new expansion budget")
        self._search_horizon = int(max_expansions)

    def set_search_step(self, expanded_records: int) -> None:
        if (
            isinstance(expanded_records, bool)
            or not isinstance(expanded_records, (int, np.integer))
            or int(expanded_records) < 0
            or int(expanded_records) > self._search_horizon
        ):
            raise ValueError("expanded_records must lie in [0, expansion budget]")
        self._search_step = int(expanded_records)

    def bind_generation_counts(self, values: Mapping[object, int]) -> None:
        """Copy deterministic archive-generation counts for ``extract`` calls."""

        if not isinstance(values, Mapping):
            raise TypeError("generation counts must be a mapping")
        self._generation_counts = _freeze_generation_counts(values)

    set_generation_counts = bind_generation_counts

    @staticmethod
    def _safe_div(numerator: float, denominator: int | float) -> float:
        return float(numerator) / float(max(1, denominator))

    def _intrinsic_prefix(self, state: CircuitState) -> tuple[float, ...]:
        budget = state.budget
        maximum_two_qubit = getattr(budget, "max_two_qubit_count", None)
        if maximum_two_qubit is None:
            maximum_two_qubit = budget.max_gates
        maximum_t = int(budget.max_t_count)
        rotations = tuple(state.rotations)
        anticommuting_pairs = sum(
            not left.axis.commutes_with(right.axis)
            for index, left in enumerate(rotations)
            for right in rotations[index + 1 :]
        )
        mean_weight = (
            float(np.mean([rotation.axis.weight for rotation in rotations], dtype=np.float64))
            if rotations
            else 0.0
        )
        target_distance = (
            self.target_context.distance(state) if self.include_target else 0.0
        )
        return (
            self._safe_div(state.t_count, maximum_t),
            self._safe_div(state.two_qubit_count, int(maximum_two_qubit)),
            self._safe_div(state.num_gates, int(budget.max_gates)),
            self._safe_div(state.depth, int(budget.max_depth)),
            self._safe_div(len(rotations), maximum_t),
            self._safe_div(anticommuting_pairs, max(1, comb(maximum_t, 2))),
            self._safe_div(mean_weight, int(state.dag.num_qubits)),
            float(target_distance),
        )

    def _complete_candidate(
        self,
        state: CircuitState,
        *,
        resource_dominance_fraction: float,
        novelty: float,
    ) -> np.ndarray:
        values = self._intrinsic_prefix(state) + (
            float(resource_dominance_fraction),
            float(novelty),
        )
        if not self.include_target:
            values = values[:7] + values[8:]
        return np.asarray(values, dtype=ARTICLE_V1_DTYPE)

    @staticmethod
    def _validated_records(frontier: Sequence[object]) -> tuple[object, ...]:
        records = tuple(frontier)
        if not records:
            raise ValueError("an article feature snapshot requires a nonempty frontier")
        record_ids: set[int] = set()
        widths: set[int] = set()
        for record in records:
            state = getattr(record, "state", None)
            record_id = getattr(record, "record_id", None)
            if not isinstance(state, CircuitState):
                raise TypeError("frontier records must expose a CircuitState as .state")
            if isinstance(record_id, bool) or not isinstance(record_id, (int, np.integer)):
                raise ValueError("frontier records must have persistent integer record IDs")
            normalized = int(record_id)
            if normalized in record_ids:
                raise ValueError(f"duplicate frontier record ID {normalized}")
            record_ids.add(normalized)
            widths.add(int(state.dag.num_qubits))
        if len(widths) != 1:
            raise ValueError("all frontier records must have the same register width")
        return records

    def build_snapshot(
        self,
        frontier: Sequence[object],
        *,
        archive_generation_counts: Mapping[object, int] | None = None,
        expansions_completed: int | None = None,
        expansion_budget: int | None = None,
    ) -> FrontierFeatureSnapshot:
        """Build read-only feature rows from one complete frontier snapshot."""

        records = self._validated_records(frontier)
        if self.warn_above_safe_size and len(records) > self.reference_safe_frontier_size:
            warnings.warn(
                "ArticleV1ReferenceFeatureProvider is executing the O(F^2) "
                f"all-pairs oracle above its safe frontier size "
                f"({len(records)} > {self.reference_safe_frontier_size}); use the "
                "exact incremental evaluator for production ranking",
                RuntimeWarning,
                stacklevel=2,
            )
        budget = self._search_horizon if expansion_budget is None else expansion_budget
        completed = self._search_step if expansions_completed is None else expansions_completed
        if (
            isinstance(budget, bool)
            or not isinstance(budget, (int, np.integer))
            or int(budget) <= 0
        ):
            raise ValueError("expansion_budget must be a positive integer")
        if (
            isinstance(completed, bool)
            or not isinstance(completed, (int, np.integer))
            or int(completed) < 0
            or int(completed) > int(budget)
        ):
            raise ValueError("expansions_completed must lie in [0, expansion_budget]")
        budget = int(budget)
        completed = int(completed)
        counts = _freeze_generation_counts(
            self._generation_counts
            if archive_generation_counts is None
            else archive_generation_counts
        )

        states = tuple(record.state for record in records)
        record_ids = tuple(int(record.record_id) for record in records)
        resources = tuple(_resource_vector(state) for state in states)
        keys = tuple(self._semantic_key(state) for state in states)
        denominator = max(1, len(records) - 1)

        candidates: list[np.ndarray] = []
        for index, state in enumerate(states):
            dominated_by_others = sum(
                other_index != index
                and _weakly_dominates(other_resources, resources[index])
                for other_index, other_resources in enumerate(resources)
            )
            resource_fraction = float(dominated_by_others / denominator)
            generation_count = max(1, int(counts.get(keys[index], 0)))
            novelty = float(1.0 / sqrt(generation_count))
            candidates.append(
                self._complete_candidate(
                    state,
                    resource_dominance_fraction=resource_fraction,
                    novelty=novelty,
                )
            )

        candidate_matrix = np.vstack(candidates).astype(ARTICLE_V1_DTYPE, copy=False)
        # Sorting first makes reductions bitwise stable under storage permutation.
        sorted_matrix = np.sort(candidate_matrix, axis=0)
        mean = np.mean(sorted_matrix, axis=0, dtype=np.float64)
        std = np.std(sorted_matrix, axis=0, ddof=0, dtype=np.float64)
        z_matrix = (candidate_matrix - mean) / (std + ARTICLE_V1_STANDARDIZATION_ETA)
        remaining_fraction = float((budget - completed) / budget)

        feature_vectors: dict[int, np.ndarray] = {}
        candidate_vectors: dict[int, np.ndarray] = {}
        for row, record_id in enumerate(record_ids):
            parts: list[np.ndarray] = [
                np.asarray([1.0], dtype=ARTICLE_V1_DTYPE),
                candidate_matrix[row],
            ]
            if self.include_frontier_context:
                parts.append(z_matrix[row])
            parts.append(remaining_fraction * candidate_matrix[row])
            feature = _readonly_vector(np.concatenate(parts))
            if feature.shape != (self.dimension,):  # pragma: no cover - schema guard
                raise AssertionError("article-v1 feature dimension drifted")
            feature_vectors[record_id] = feature
            candidate_vectors[record_id] = _readonly_vector(candidate_matrix[row])

        target_fingerprint = (
            self.target_context.fingerprint
            if self.include_target and self.target_context is not None
            else None
        )
        digest = sha256()
        digest.update(self.schema_version.encode("ascii"))
        digest.update(str(completed).encode("ascii"))
        digest.update(b"/")
        digest.update(str(budget).encode("ascii"))
        digest.update((target_fingerprint or "").encode("ascii"))
        for record_id in sorted(feature_vectors):
            digest.update(str(record_id).encode("ascii"))
            digest.update(feature_vectors[record_id].tobytes(order="C"))

        return FrontierFeatureSnapshot(
            schema_version=self.schema_version,
            snapshot_id=f"sha256:{digest.hexdigest()}",
            record_ids=record_ids,
            frontier_nodes=records,
            expansions_completed=completed,
            expansion_budget=budget,
            archive_generation_counts=counts,
            target_fingerprint=target_fingerprint,
            feature_vectors=MappingProxyType(feature_vectors),
            candidate_vectors=MappingProxyType(candidate_vectors),
        )

    def build_batch(self, frontier: Sequence[object], **kwargs: Any) -> FrontierFeatureSnapshot:
        """Alias for :meth:`build_snapshot` used by policy integrations."""

        return self.build_snapshot(frontier, **kwargs)

    def extract(
        self,
        state: CircuitState,
        frontier: Optional[Iterable[object]] = None,
    ) -> np.ndarray:
        """Compatibility adapter using the provider's bound horizon/counts.

        Article training should retain a row returned by :meth:`build_snapshot`
        before the transition. This method computes the same row for the
        repository's generic feature-provider protocol.
        """

        if not isinstance(state, CircuitState):
            raise TypeError("article-v1 features require a CircuitState")
        raw = list(frontier or ())
        records: list[object] = []
        selected_id: int | None = None
        used_ids: set[int] = set()
        for index, item in enumerate(raw):
            item_state = getattr(item, "state", item)
            if not isinstance(item_state, CircuitState):
                continue
            record_id = getattr(item, "record_id", None)
            if isinstance(record_id, bool) or not isinstance(record_id, (int, np.integer)):
                record_id = index
                while record_id in used_ids:
                    record_id += len(raw) + 1
                record = _FeatureRecord(int(record_id), item_state)
            else:
                record = item
                record_id = int(record_id)
            used_ids.add(int(record_id))
            records.append(record)
            if item_state is state and selected_id is None:
                selected_id = int(record_id)

        if selected_id is None:
            record_id = 0
            while record_id in used_ids:
                record_id += 1
            records.append(_FeatureRecord(record_id, state))
            selected_id = record_id
        snapshot = self.build_snapshot(records)
        return snapshot.features_for_record(selected_id)

    def metadata(self) -> Mapping[str, object]:
        return {
            "feature_schema_version": self.schema_version,
            "feature_evaluator_schema_version": self.evaluator_schema_version,
            "feature_dim": self.dimension,
            "feature_names": self.names,
            "candidate_feature_dim": len(self.candidate_coordinate_names),
            "candidate_coordinate_names": self.candidate_coordinate_names,
            "article_equations": "81-92",
            "target_metric_schema_version": (
                ARTICLE_V1_TARGET_METRIC_SCHEMA_VERSION if self.include_target else None
            ),
            "target_aware": self.include_target,
            "frontier_context": self.include_frontier_context,
            "target_fingerprint": (
                self.target_context.fingerprint
                if self.include_target and self.target_context is not None
                else None
            ),
            "standardization_eta": ARTICLE_V1_STANDARDIZATION_ETA,
            "dtype": ARTICLE_V1_DTYPE.name,
            "budget_interaction": (
                "remaining external expansion budget fraction times candidate vector"
            ),
            "novelty_source": "externally supplied canonical-key generation counts",
            "evaluator_role": "reference-all-pairs-correctness-oracle",
            "reference_safe_frontier_size": self.reference_safe_frontier_size,
        }


class ArticleV1FeatureProvider(ArticleV1ReferenceFeatureProvider):
    """Production Article V1 provider backed by an exact incremental index.

    ``build_compact_batch`` is the publication-scale path.  ``build_snapshot``
    remains a materializing compatibility adapter for callers that require a
    legacy :class:`FrontierFeatureSnapshot`.
    """

    evaluator_schema_version = ARTICLE_V1_EXACT_INCREMENTAL_EVALUATOR_SCHEMA_VERSION

    def __init__(
        self,
        target_context: ArticleTargetContext,
        *,
        semantic_key: Callable[[CircuitState], object] | None = None,
        search_horizon: int = 1,
        generation_counts: Mapping[object, int] | None = None,
        debug_reconciliation: bool = False,
        initial_index_capacity: int = 16,
        **reference_compatibility: Any,
    ) -> None:
        self._feature_index: object | None = None
        self._debug_reconciliation = bool(debug_reconciliation)
        if (
            isinstance(initial_index_capacity, bool)
            or not isinstance(initial_index_capacity, (int, np.integer))
            or int(initial_index_capacity) < 1
        ):
            raise ValueError("initial_index_capacity must be a positive integer")
        self._initial_index_capacity = int(initial_index_capacity)
        super().__init__(
            target_context,
            semantic_key=semantic_key,
            search_horizon=search_horizon,
            generation_counts=generation_counts,
            **reference_compatibility,
        )

    def _ensure_feature_index(self) -> object:
        if self._feature_index is None:
            from rl.article_frontier_index import ExactArticleFrontierFeatureIndex

            self._feature_index = ExactArticleFrontierFeatureIndex(
                self,
                debug_reconciliation=self._debug_reconciliation,
                initial_capacity=self._initial_index_capacity,
            )
        return self._feature_index

    @property
    def feature_index(self) -> object:
        return self._ensure_feature_index()

    def reset_index(self) -> None:
        """Reset per-episode accelerator and novelty-mirror state."""

        self._feature_index = None
        self._generation_counts = MappingProxyType({})

    def bind_generation_counts(self, values: Mapping[object, int]) -> None:
        # Retain the compatibility copy for ``extract`` and legacy metadata.
        super().bind_generation_counts(values)
        if self._feature_index is not None:
            replace = getattr(self._feature_index, "replace_generation_counts")
            replace(self._generation_counts)

    set_generation_counts = bind_generation_counts

    def update_generation_counts(self, updates: Mapping[object, int]) -> None:
        """Apply absolute changed-key totals without copying the complete map."""

        if not isinstance(updates, Mapping):
            raise TypeError("generation-count updates must be a mapping")
        merged = dict(self._generation_counts)
        for key, count in _freeze_generation_counts(updates).items():
            if count == 0:
                merged.pop(key, None)
            else:
                merged[key] = count
        self._generation_counts = MappingProxyType(merged)
        index = self._ensure_feature_index()
        getattr(index, "update_generation_counts")(updates)

    def increment_generation_counts(self, deltas: Mapping[object, int]) -> None:
        """Apply exact changed-key deltas to the incremental novelty cache."""

        if not isinstance(deltas, Mapping):
            raise TypeError("generation-count deltas must be a mapping")
        merged = dict(self._generation_counts)
        for key, delta in deltas.items():
            if isinstance(delta, bool) or not isinstance(delta, (int, np.integer)):
                raise TypeError("generation-count deltas must be integers")
            count = int(merged.get(key, 0)) + int(delta)
            if count < 0:
                raise ValueError("generation-count delta would make a count negative")
            if count == 0:
                merged.pop(key, None)
            else:
                merged[key] = count
        self._generation_counts = MappingProxyType(merged)
        index = self._ensure_feature_index()
        getattr(index, "increment_generation_counts")(deltas)

    def synchronize_frontier(
        self,
        records: Sequence[object],
        *,
        generation_count_updates: Mapping[object, int] | None = None,
    ) -> object:
        """Mirror an authoritative ArchiveRecord/SearchNode sequence."""

        index = self._ensure_feature_index()
        if not bool(getattr(index, "initialized")):
            counts = dict(self._generation_counts)
            if generation_count_updates:
                counts.update(generation_count_updates)
            getattr(index, "initialize")(records, generation_counts=counts)
            return None
        return getattr(index, "synchronize")(
            records,
            generation_count_updates=generation_count_updates or {},
        )

    def build_compact_batch(
        self,
        records: Sequence[object],
        *,
        theta: np.ndarray | None = None,
        archive_generation_counts: Mapping[object, int] | None = None,
        generation_count_updates: Mapping[object, int] | None = None,
        expansions_completed: int | None = None,
        expansion_budget: int | None = None,
    ) -> object:
        """Return an exact compact decision batch from authoritative records."""

        if archive_generation_counts is not None:
            self.bind_generation_counts(archive_generation_counts)
        self.synchronize_frontier(
            records, generation_count_updates=generation_count_updates
        )
        budget = self._search_horizon if expansion_budget is None else expansion_budget
        completed = self._search_step if expansions_completed is None else expansions_completed
        return getattr(self._ensure_feature_index(), "build_decision_batch")(
            theta=theta,
            expansions_completed=completed,
            expansion_budget=budget,
        )

    def build_batch(self, frontier: Sequence[object], **kwargs: Any) -> object:
        """Compatibility alias for the compact production batch."""

        return self.build_compact_batch(frontier, **kwargs)

    def build_snapshot(
        self,
        frontier: Sequence[object],
        *,
        archive_generation_counts: Mapping[object, int] | None = None,
        expansions_completed: int | None = None,
        expansion_budget: int | None = None,
    ) -> FrontierFeatureSnapshot:
        """Materialize a legacy snapshot from one compact exact batch."""

        batch = self.build_compact_batch(
            frontier,
            archive_generation_counts=archive_generation_counts,
            expansions_completed=expansions_completed,
            expansion_budget=expansion_budget,
        )
        record_ids = tuple(int(value) for value in batch.record_ids)
        feature_vectors = {
            record_id: batch.features_for_record(record_id) for record_id in record_ids
        }
        candidate_vectors = {
            record_id: batch.candidate_for_record(record_id) for record_id in record_ids
        }
        index = self._ensure_feature_index()
        counts = getattr(index, "generation_counts_snapshot")()
        return FrontierFeatureSnapshot(
            schema_version=self.schema_version,
            snapshot_id=batch.snapshot_id,
            record_ids=record_ids,
            frontier_nodes=tuple(batch.frontier_nodes),
            expansions_completed=int(batch.expansions_completed),
            expansion_budget=int(batch.expansion_budget),
            archive_generation_counts=counts,
            target_fingerprint=batch.target_fingerprint,
            feature_vectors=MappingProxyType(feature_vectors),
            candidate_vectors=MappingProxyType(candidate_vectors),
        )

    def minimum_target_distance(self) -> float:
        return float(getattr(self._ensure_feature_index(), "minimum_target_distance")())

    def select_target_distance_node(self) -> object:
        return getattr(self._ensure_feature_index(), "select_target_distance_node")()

    def reconcile_index(self, records: Sequence[object] | None = None) -> None:
        getattr(self._ensure_feature_index(), "reconcile")(records)

    def recent_compact_batch_times_ns(self) -> tuple[int, ...]:
        """Return exact completed-batch timings used by progress telemetry."""

        if self._feature_index is None:
            return ()
        values = getattr(
            self._feature_index, "recent_compact_batch_times_ns"
        )()
        return tuple(int(value) for value in values)

    def instrumentation(self) -> Mapping[str, int | str]:
        if self._feature_index is None:
            return MappingProxyType(
                {
                    "feature_evaluator_schema_version": self.evaluator_schema_version,
                    "feature_static_cache_hits": 0,
                    "feature_static_cache_misses": 0,
                    "frontier_index_additions": 0,
                    "frontier_index_removals": 0,
                    "frontier_index_rebuilds": 0,
                    "unique_resource_group_count": 0,
                    "resource_group_peak": 0,
                    "dominance_update_time_ns": 0,
                    "compact_batch_time_ns": 0,
                    "compact_batch_count": 0,
                    "last_compact_batch_time_ns": 0,
                    "candidate_gather_time_ns": 0,
                    "standardization_time_ns": 0,
                    "score_time_ns": 0,
                    "selected_row_materialization_time_ns": 0,
                    "feature_index_memory_bytes": 0,
                    "frontier_revision": 0,
                    "generation_count_revision": 0,
                }
            )
        return MappingProxyType(dict(getattr(self._feature_index, "instrumentation")()))

    def metadata(self) -> Mapping[str, object]:
        metadata = dict(super().metadata())
        metadata.update(
            {
                "feature_evaluator_schema_version": self.evaluator_schema_version,
                "evaluator_role": "production-exact-incremental",
                "dominance_evaluation": "exact dynamic resource-group counts",
                "compact_linear_scoring": True,
            }
        )
        metadata.pop("reference_safe_frontier_size", None)
        return metadata


class ArticleV1NoTargetFeatureProvider(ArticleV1FeatureProvider):
    """28-D ablation removing only the target-infidelity coordinate family."""

    schema_version = ARTICLE_V1_NO_TARGET_FEATURE_SCHEMA_VERSION
    feature_names = ARTICLE_V1_NO_TARGET_FEATURE_NAMES
    include_target = False

    def __init__(
        self,
        target_context: ArticleTargetContext | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(target_context, **kwargs)


class ArticleV1NoZFeatureProvider(ArticleV1FeatureProvider):
    """21-D ablation removing the frontier-standardization block."""

    schema_version = ARTICLE_V1_NO_Z_FEATURE_SCHEMA_VERSION
    feature_names = ARTICLE_V1_NO_Z_FEATURE_NAMES
    include_frontier_context = False


# Preserved 37-D extended provider.
EXTENDED_ARTICLE_CANDIDATE_FEATURE_NAMES = (
    "t_count_fraction",
    "two_qubit_count_fraction",
    "gate_count_fraction",
    "depth_fraction",
    "peak_ancilla_fraction",
    "gate_block_fraction",
    "rotation_count_fraction",
    "anticommutation_dependency_density",
    "mean_pauli_weight_fraction",
    "target_discrepancy",
    "pareto_rank_fraction",
    "semantic_novelty",
)
EXTENDED_ARTICLE_SEARCH_BUDGET_FEATURE_NAME = "remaining_expansion_budget_fraction"
EXTENDED_ARTICLE_FEATURE_SCHEMA_VERSION = "extended-target-aware-37d-v1"
_EXTENDED_STABILIZER = 1e-6

# Backward-compatible names retain the pre-V1 checkpoint contract.
ARTICLE_CANDIDATE_FEATURE_NAMES = EXTENDED_ARTICLE_CANDIDATE_FEATURE_NAMES
ARTICLE_SEARCH_BUDGET_FEATURE_NAME = EXTENDED_ARTICLE_SEARCH_BUDGET_FEATURE_NAME
ARTICLE_FEATURE_SCHEMA_VERSION = "article-frontier-eq19-v2"


def _safe_fraction(value: float, maximum: float) -> float:
    return float(value) / float(maximum) if maximum > 0 else 0.0


def _frontier_states(frontier: Optional[Iterable[object]]) -> list[CircuitState]:
    states: list[CircuitState] = []
    if frontier is None:
        return states
    for item in frontier:
        state = getattr(item, "state", item)
        if isinstance(state, CircuitState):
            states.append(state)
    return states


def _gate_blocks(state: CircuitState) -> int:
    classes = [
        "rotation" if gate.gate_type in {GateType.T, GateType.TDG} else "clifford"
        for gate in state.dag.gates
    ]
    return sum(index == 0 or value != classes[index - 1] for index, value in enumerate(classes))


class ExtendedArticleFeatureProvider:
    """Preserved order-invariant 37-D extended target-aware provider."""

    schema_version = EXTENDED_ARTICLE_FEATURE_SCHEMA_VERSION

    def __init__(
        self,
        target_context: object | None = None,
        *,
        semantic_key: Callable[[CircuitState], object] | None = None,
        search_horizon: int = 1,
    ) -> None:
        self.target_context = target_context
        if semantic_key is None:
            semantic_key = Canonicalizer().semantic_key
        if not callable(semantic_key):
            raise TypeError("semantic_key must be callable")
        self._search_horizon = 1
        self._search_step = 0
        self.bind_search_horizon(search_horizon)
        self._semantic_key = semantic_key
        interaction_names = tuple(
            f"{EXTENDED_ARTICLE_SEARCH_BUDGET_FEATURE_NAME}_x_{candidate}"
            for candidate in EXTENDED_ARTICLE_CANDIDATE_FEATURE_NAMES
        )
        self._names = (
            ("bias",)
            + EXTENDED_ARTICLE_CANDIDATE_FEATURE_NAMES
            + tuple(
                f"frontier_z_{name}"
                for name in EXTENDED_ARTICLE_CANDIDATE_FEATURE_NAMES
            )
            + interaction_names
        )

    @property
    def dimension(self) -> int:
        return len(self._names)

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    def bind_search_horizon(self, max_expansions: int) -> None:
        if (
            isinstance(max_expansions, bool)
            or not isinstance(max_expansions, (int, np.integer))
            or int(max_expansions) <= 0
        ):
            raise ValueError("search horizon must be a positive integer")
        self._search_horizon = int(max_expansions)
        self._search_step = min(self._search_step, self._search_horizon)

    def set_search_step(self, expanded_records: int) -> None:
        if (
            isinstance(expanded_records, bool)
            or not isinstance(expanded_records, (int, np.integer))
            or int(expanded_records) < 0
        ):
            raise ValueError("expanded_records must be a non-negative integer")
        self._search_step = int(expanded_records)

    @property
    def remaining_search_budget_fraction(self) -> float:
        remaining = max(0, self._search_horizon - self._search_step)
        return float(remaining / self._search_horizon)

    @property
    def search_horizon(self) -> int:
        return self._search_horizon

    def _target_discrepancy(self, state: CircuitState) -> float:
        if self.target_context is None:
            return 0.0
        metrics = getattr(self.target_context, "metrics", None)
        if not callable(metrics):
            raise TypeError("target_context must expose metrics(state)")
        value = float(getattr(metrics(state), "phase_aligned_frobenius_distance"))
        if not np.isfinite(value):
            raise ValueError("target discrepancy must be finite")
        return float(np.clip(value, 0.0, 1.0))

    @staticmethod
    def _intrinsic(state: CircuitState, target_discrepancy: float) -> tuple[float, ...]:
        budget = state.budget
        maximum_two_qubit = getattr(budget, "max_two_qubit_count", None)
        if maximum_two_qubit is None:
            maximum_two_qubit = budget.max_gates
        rotations = tuple(state.rotations)
        dependencies = sum(
            not left.axis.commutes_with(right.axis)
            for index, left in enumerate(rotations)
            for right in rotations[index + 1 :]
        )
        possible_dependencies = len(rotations) * (len(rotations) - 1) // 2
        mean_weight = (
            float(np.mean([rotation.axis.weight for rotation in rotations]))
            if rotations
            else 0.0
        )
        peak_ancillas = float(getattr(state, "peak_ancilla_count", 0))
        maximum_ancillas = float(getattr(budget, "max_ancilla_count", 0) or 0)
        return (
            _safe_fraction(state.t_count, budget.max_t_count),
            _safe_fraction(state.two_qubit_count, maximum_two_qubit),
            _safe_fraction(state.num_gates, budget.max_gates),
            _safe_fraction(state.depth, budget.max_depth),
            _safe_fraction(peak_ancillas, maximum_ancillas),
            _safe_fraction(_gate_blocks(state), budget.max_gates),
            _safe_fraction(len(rotations), max(1, budget.max_t_count)),
            _safe_fraction(dependencies, possible_dependencies),
            _safe_fraction(mean_weight, max(1, state.dag.num_qubits)),
            target_discrepancy,
        )

    def _candidate_vector(
        self,
        state: CircuitState,
        frontier_states: list[CircuitState],
        signatures: list[object],
    ) -> np.ndarray:
        resources = _resource_vector(state)
        denominator = max(1, len(frontier_states) - 1)
        rank = sum(
            _strictly_dominates(_resource_vector(other), resources)
            for other in frontier_states
            if other is not state
        ) / denominator
        signature = self._semantic_key(state)
        frequency = max(1, sum(other == signature for other in signatures))
        novelty = 1.0 / float(frequency)
        return np.asarray(
            self._intrinsic(state, self._target_discrepancy(state)) + (rank, novelty),
            dtype=np.float64,
        )

    def extract(
        self,
        state: CircuitState,
        frontier: Optional[Iterable[object]] = None,
    ) -> np.ndarray:
        states = _frontier_states(frontier)
        if not states:
            states = [state]
        signatures = [self._semantic_key(candidate) for candidate in states]
        candidate = self._candidate_vector(state, states, signatures)
        matrix = np.vstack(
            [self._candidate_vector(other, states, signatures) for other in states]
        )
        sorted_matrix = np.sort(matrix, axis=0)
        mean = np.mean(sorted_matrix, axis=0)
        std = np.std(sorted_matrix, axis=0)
        z_score = (candidate - mean) / (std + _EXTENDED_STABILIZER)
        interactions = self.remaining_search_budget_fraction * candidate
        result = np.concatenate(([1.0], candidate, z_score, interactions)).astype(
            np.float32, copy=False
        )
        if result.shape != (self.dimension,):  # pragma: no cover - schema guard
            raise AssertionError("extended feature dimension drifted")
        return result

    def metadata(self) -> Mapping[str, object]:
        return {
            "feature_schema_version": self.schema_version,
            "feature_dim": self.dimension,
            "feature_names": self.names,
            "profile": "extended-target-aware-37d",
            "frontier_order_invariant": True,
            "budget_interaction": "remaining expansion budget fraction times candidate vector",
            "target_aware": self.target_context is not None,
            "target_fingerprint": (
                None
                if self.target_context is None
                else str(getattr(self.target_context, "fingerprint", ""))
            ),
        }


class ArticleFeatureProvider(ExtendedArticleFeatureProvider):
    """Compatibility wrapper for the pre-Article-V1 37-D checkpoint schema."""

    schema_version = ARTICLE_FEATURE_SCHEMA_VERSION

    def metadata(self) -> Mapping[str, object]:
        metadata = dict(super().metadata())
        metadata.update(
            {
                "article_equation": 19,
                "compatibility_profile": "pre-article-v1-extended-37d",
            }
        )
        return metadata


__all__ = [
    "ARTICLE_CANDIDATE_FEATURE_NAMES",
    "ARTICLE_FEATURE_SCHEMA_VERSION",
    "ARTICLE_SEARCH_BUDGET_FEATURE_NAME",
    "ARTICLE_V1_COORDINATE_NAMES",
    "ARTICLE_V1_DTYPE",
    "ARTICLE_V1_EXACT_INCREMENTAL_EVALUATOR_SCHEMA_VERSION",
    "ARTICLE_V1_FEATURE_NAMES",
    "ARTICLE_V1_FEATURE_SCHEMA_VERSION",
    "ARTICLE_V1_NO_TARGET_COORDINATE_NAMES",
    "ARTICLE_V1_NO_TARGET_FEATURE_NAMES",
    "ARTICLE_V1_NO_TARGET_FEATURE_SCHEMA_VERSION",
    "ARTICLE_V1_NO_Z_FEATURE_NAMES",
    "ARTICLE_V1_NO_Z_FEATURE_SCHEMA_VERSION",
    "ARTICLE_V1_REFERENCE_EVALUATOR_SCHEMA_VERSION",
    "ARTICLE_V1_STANDARDIZATION_ETA",
    "ARTICLE_V1_TARGET_METRIC_SCHEMA_VERSION",
    "EXTENDED_ARTICLE_CANDIDATE_FEATURE_NAMES",
    "EXTENDED_ARTICLE_FEATURE_SCHEMA_VERSION",
    "EXTENDED_ARTICLE_SEARCH_BUDGET_FEATURE_NAME",
    "ArticleFeatureProvider",
    "ArticleTargetContext",
    "ArticleV1FeatureProvider",
    "ArticleV1ReferenceFeatureProvider",
    "ArticleV1NoTargetFeatureProvider",
    "ArticleV1NoZFeatureProvider",
    "ExtendedArticleFeatureProvider",
    "FrontierFeatureSnapshot",
    "process_infidelity",
]
