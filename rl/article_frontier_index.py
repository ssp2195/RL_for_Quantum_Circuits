"""Exact incremental Article V1 frontier-feature evaluation.

The classes in this module are accelerators only.  Frontier membership and
archive acceptance remain authoritative in :mod:`search.frontier` and
:mod:`search.archive`.  The index mirrors an authoritative record sequence,
caches immutable per-record coordinates, and maintains exact resource-group
dominance counts under additions and removals.

No approximation is used.  If ``f(r)`` is the number of open records with
resource tuple ``r`` and ``D(r) = sum(f(s) for s <= r)``, the ninth Article V1
candidate coordinate is exactly ``(D(r) - 1) / max(1, F - 1)``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
import heapq
from math import sqrt
import sys
from time import perf_counter_ns
from types import MappingProxyType
from typing import Any

import numpy as np

from circuit.circuit_state import CircuitState


ARTICLE_V1_EXACT_INCREMENTAL_EVALUATOR_SCHEMA_VERSION = (
    "article-v1-exact-incremental-v2"
)


def _readonly_array(values: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy(order="C")
    result.setflags(write=False)
    return result


def _record_id(record: object) -> int:
    value = getattr(record, "record_id", None)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("frontier records must have persistent integer record IDs")
    return int(value)


def _record_node(record: object) -> object:
    node = getattr(record, "node", None)
    return record if node is None else node


def _record_state(record: object) -> CircuitState:
    node = _record_node(record)
    state = getattr(node, "state", None)
    if not isinstance(state, CircuitState):
        raise TypeError("frontier records must expose a CircuitState through .state or .node.state")
    return state


def _resource_tuple(record: object, state: CircuitState) -> tuple[int, ...]:
    resources = getattr(record, "resources", None)
    as_tuple = getattr(resources, "as_tuple", None)
    if callable(as_tuple):
        values = tuple(as_tuple())
    else:
        values = (
            int(state.t_count),
            int(state.two_qubit_count),
            int(state.num_gates),
            *(int(value) for value in state.wire_depths),
        )
    if len(values) < 4:
        raise ValueError("resource tuples must contain counts and per-wire depths")
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values):
        raise TypeError("resource tuple coordinates must be integers")
    normalized = tuple(int(value) for value in values)
    if any(value < 0 for value in normalized):
        raise ValueError("resource tuple coordinates must be non-negative")
    return normalized


def _record_key(
    record: object,
    state: CircuitState,
    semantic_key: Callable[[CircuitState], object],
) -> object:
    sentinel = object()
    key = getattr(record, "key", sentinel)
    if key is sentinel:
        key = semantic_key(state)
    try:
        hash(key)
    except TypeError as exc:
        raise TypeError("semantic keys must be hashable full canonical payloads") from exc
    return key


def _validated_nonnegative_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("generation counts must be integers")
    normalized = int(value)
    if normalized < 0:
        raise ValueError("generation counts must be non-negative")
    return normalized


@dataclass(frozen=True, slots=True)
class StaticArticleCandidate:
    """Immutable cached coordinates for one persistent frontier record."""

    record_id: int
    semantic_key: object
    resource_tuple: tuple[int, ...]
    intrinsic_coordinates: np.ndarray


@dataclass(frozen=True, slots=True)
class FrontierMutationBatch:
    """Exact set-difference applied while mirroring an authoritative frontier."""

    revision_before: int
    revision_after: int
    removed_record_ids: tuple[int, ...]
    added_records: tuple[object, ...]
    generation_count_updates: tuple[tuple[object, int], ...]

    @property
    def added_record_ids(self) -> tuple[int, ...]:
        return tuple(_record_id(record) for record in self.added_records)


@dataclass(frozen=True, slots=True)
class CompactArticleDecisionBatch:
    """Compact exact candidate matrix and deterministic frontier statistics.

    Full mathematical feature rows are materialized only on request.  Linear
    scores use the exact effective-weight identity, avoiding an ``F x 31``
    allocation in the normal ranking path.
    """

    schema_version: str
    evaluator_schema_version: str
    frontier_revision: int
    generation_count_revision: int
    records: tuple[object, ...]
    frontier_nodes: tuple[object, ...]
    record_ids: np.ndarray
    candidate_matrix: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    remaining_budget_fraction: float
    target_fingerprint: str | None
    expansions_completed: int
    expansion_budget: int
    include_frontier_context: bool
    standardization_eta: float
    snapshot_id: str
    _row_by_record_id: Mapping[int, int] = field(repr=False, compare=False)
    _score_observer: Callable[[int], None] | None = field(
        default=None, repr=False, compare=False
    )
    _row_observer: Callable[[int], None] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def candidate_dimension(self) -> int:
        return int(self.candidate_matrix.shape[1])

    @property
    def feature_dimension(self) -> int:
        multiplier = 3 if self.include_frontier_context else 2
        return 1 + multiplier * self.candidate_dimension

    @property
    def dimension(self) -> int:
        return self.feature_dimension

    def _validated_theta(self, theta: np.ndarray) -> np.ndarray:
        weights = np.asarray(theta, dtype=np.float64)
        if weights.shape != (self.feature_dimension,):
            raise ValueError(
                "theta shape does not match compact Article feature schema: "
                f"{weights.shape!r} != {(self.feature_dimension,)!r}"
            )
        if not np.isfinite(weights).all():
            raise ValueError("theta must contain only finite values")
        return weights

    def effective_linear_terms(self, theta: np.ndarray) -> tuple[np.ndarray, float]:
        """Return exact candidate weight and candidate-independent constant."""

        weights = self._validated_theta(theta)
        width = self.candidate_dimension
        theta_x = weights[1 : 1 + width]
        if self.include_frontier_context:
            theta_z = weights[1 + width : 1 + 2 * width]
            theta_budget = weights[1 + 2 * width : 1 + 3 * width]
            scale = self.std + self.standardization_eta
            effective = (
                theta_x
                + self.remaining_budget_fraction * theta_budget
                + theta_z / scale
            )
            constant = float(weights[0] - np.dot(theta_z, self.mean / scale))
        else:
            theta_budget = weights[1 + width : 1 + 2 * width]
            effective = theta_x + self.remaining_budget_fraction * theta_budget
            constant = float(weights[0])
        return np.asarray(effective, dtype=np.float64), constant

    def scores(self, theta: np.ndarray) -> np.ndarray:
        started_ns = perf_counter_ns()
        try:
            weights = self._validated_theta(theta)
            width = self.candidate_dimension
            theta_x = weights[1 : 1 + width]
            if self.include_frontier_context:
                theta_z = weights[1 + width : 1 + 2 * width]
                theta_budget = weights[1 + 2 * width : 1 + 3 * width]
                base = theta_x + self.remaining_budget_fraction * theta_budget
                effective = base + theta_z / (
                    self.std + self.standardization_eta
                )
            else:
                theta_budget = weights[1 + width : 1 + 2 * width]
                base = theta_x + self.remaining_budget_fraction * theta_budget
                effective = base
            # This is the same effective-weight identity expressed around the
            # frontier mean.  Centering before the one matrix multiply avoids
            # catastrophic cancellation when a column has zero variance and
            # ``theta_z / eta`` is large:
            # Q = (x - mu) @ w + theta_0 + mu @ base.
            constant = float(weights[0] + np.dot(self.mean, base))
            result = (self.candidate_matrix - self.mean) @ effective + constant
            return np.asarray(result, dtype=np.float64)
        finally:
            if self._score_observer is not None:
                self._score_observer(perf_counter_ns() - started_ns)

    def greedy_row(self, theta: np.ndarray) -> int:
        values = self.scores(theta)
        if values.size == 0:
            raise ValueError("cannot select from an empty frontier")
        maximum = float(np.max(values))
        tied_rows = np.flatnonzero(values == maximum)
        if tied_rows.size == 0:  # pragma: no cover - finite schema guard
            raise AssertionError("no greedy row found")
        tied_ids = self.record_ids[tied_rows]
        return int(tied_rows[int(np.argmin(tied_ids))])

    def select_greedy_record_id(self, theta: np.ndarray) -> int:
        return int(self.record_ids[self.greedy_row(theta)])

    def select_greedy(self, theta: np.ndarray) -> object:
        """Return the authoritative SearchNode selected by exact linear rank."""

        return self.frontier_nodes[self.greedy_row(theta)]

    def row_for_record_id(self, record_id: int) -> int:
        try:
            return int(self._row_by_record_id[int(record_id)])
        except (KeyError, TypeError, ValueError) as exc:
            raise KeyError(f"record ID {record_id!r} is not in this batch") from exc

    def features_for_row(self, row: int) -> np.ndarray:
        started_ns = perf_counter_ns()
        try:
            if isinstance(row, bool) or not isinstance(row, (int, np.integer)):
                raise TypeError("row must be an integer")
            normalized = int(row)
            if normalized < 0 or normalized >= len(self.records):
                raise IndexError("compact batch row is out of range")
            candidate = self.candidate_matrix[normalized]
            parts: list[np.ndarray] = [
                np.asarray([1.0], dtype=np.float64),
                candidate,
            ]
            if self.include_frontier_context:
                parts.append(
                    (candidate - self.mean)
                    / (self.std + self.standardization_eta)
                )
            parts.append(self.remaining_budget_fraction * candidate)
            return _readonly_array(np.concatenate(parts), dtype=np.dtype(np.float64))
        finally:
            if self._row_observer is not None:
                self._row_observer(perf_counter_ns() - started_ns)

    def features_for_record(self, record_id: int) -> np.ndarray:
        return self.features_for_row(self.row_for_record_id(record_id))

    def features_for_node(self, node: object) -> np.ndarray:
        return self.features_for_record(_record_id(node))

    # Compatibility name used by the generic policy adapter.
    features_for = features_for_node

    def candidate_for_record(self, record_id: int) -> np.ndarray:
        row = self.row_for_record_id(record_id)
        return _readonly_array(self.candidate_matrix[row], dtype=np.dtype(np.float64))

    def candidate_for_node(self, node: object) -> np.ndarray:
        return self.candidate_for_record(_record_id(node))

    def node_for_record_id(self, record_id: int) -> object:
        return self.frontier_nodes[self.row_for_record_id(record_id)]

    def record_for_record_id(self, record_id: int) -> object:
        return self.records[self.row_for_record_id(record_id)]

    def materialize_feature_matrix(self) -> np.ndarray:
        if not self.records:
            result = np.empty((0, self.feature_dimension), dtype=np.float64)
            result.setflags(write=False)
            return result
        matrix = np.stack(
            [self.features_for_row(row) for row in range(len(self.records))], axis=0
        )
        matrix.setflags(write=False)
        return matrix

    def full_dot_scores(self, theta: np.ndarray) -> np.ndarray:
        """Debug/reference helper that explicitly materializes all full rows."""

        weights = self._validated_theta(theta)
        return self.materialize_feature_matrix() @ weights


class ExactArticleFrontierFeatureIndex:
    """Mirror an authoritative frontier with exact incremental feature indices."""

    evaluator_schema_version = ARTICLE_V1_EXACT_INCREMENTAL_EVALUATOR_SCHEMA_VERSION

    def __init__(
        self,
        feature_provider: object,
        *,
        debug_reconciliation: bool = False,
        initial_capacity: int = 16,
    ) -> None:
        semantic_key = getattr(feature_provider, "_semantic_key", None)
        intrinsic_prefix = getattr(feature_provider, "_intrinsic_prefix", None)
        if not callable(semantic_key) or not callable(intrinsic_prefix):
            raise TypeError(
                "feature_provider must expose Article V1 semantic-key and intrinsic callbacks"
            )
        if isinstance(initial_capacity, bool) or int(initial_capacity) < 1:
            raise ValueError("initial_capacity must be a positive integer")
        self.feature_provider = feature_provider
        self._semantic_key = semantic_key
        self._intrinsic_prefix = intrinsic_prefix
        self.schema_version = str(getattr(feature_provider, "schema_version"))
        self.include_target = bool(getattr(feature_provider, "include_target", True))
        self.include_frontier_context = bool(
            getattr(feature_provider, "include_frontier_context", True)
        )
        self.standardization_eta = float(
            getattr(feature_provider, "standardization_eta", 1e-8)
        )
        if not np.isfinite(self.standardization_eta) or self.standardization_eta <= 0.0:
            raise ValueError("standardization eta must be finite and positive")
        self.debug_reconciliation = bool(debug_reconciliation)
        self._initial_capacity = int(initial_capacity)
        self._initialized = False
        self._reset_storage()

    def _reset_storage(self) -> None:
        capacity = self._initial_capacity
        self._capacity = capacity
        self._record_ids = np.full(capacity, -1, dtype=np.int64)
        self._static_intrinsic = np.zeros((capacity, 8), dtype=np.float64)
        self._active_mask = np.zeros(capacity, dtype=np.bool_)
        self._slot_group = np.full(capacity, -1, dtype=np.int64)
        self._novelty_by_slot = np.ones(capacity, dtype=np.float64)
        self._keys_by_slot: list[object | None] = [None] * capacity
        self._resources_by_slot: list[tuple[int, ...] | None] = [None] * capacity
        self._free_slots = list(reversed(range(capacity)))
        self._slot_by_record_id: dict[int, int] = {}
        self._record_by_id: dict[int, object] = {}
        self._ordered_record_ids: tuple[int, ...] = ()
        self._key_to_active_slots: dict[object, set[int]] = {}
        self._generation_counts: dict[object, int] = {}

        self._resource_dimension: int | None = None
        self._group_capacity = max(8, capacity // 2)
        self._resource_group_matrix = np.empty((self._group_capacity, 0), dtype=np.int64)
        self._group_frequency = np.zeros(self._group_capacity, dtype=np.int64)
        self._group_dominator_count = np.zeros(self._group_capacity, dtype=np.int64)
        self._group_active = np.zeros(self._group_capacity, dtype=np.bool_)
        self._group_resources: list[tuple[int, ...] | None] = [None] * self._group_capacity
        self._group_to_active_slots: list[set[int]] = [
            set() for _ in range(self._group_capacity)
        ]
        self._free_groups = list(reversed(range(self._group_capacity)))
        self._group_by_resource: dict[tuple[int, ...], int] = {}

        self._target_heap: list[tuple[float, int]] = []
        self._frontier_revision = 0
        self._generation_count_revision = 0
        self._static_cache_hits = 0
        self._static_cache_misses = 0
        self._frontier_index_additions = 0
        self._frontier_index_removals = 0
        self._frontier_index_rebuilds = 0
        self._resource_group_peak = 0
        self._dominance_update_time_ns = 0
        self._compact_batch_time_ns = 0
        self._candidate_gather_time_ns = 0
        self._standardization_time_ns = 0
        self._score_time_ns = 0
        self._selected_row_materialization_time_ns = 0

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def frontier_revision(self) -> int:
        return self._frontier_revision

    @property
    def generation_count_revision(self) -> int:
        return self._generation_count_revision

    @property
    def active_record_ids(self) -> tuple[int, ...]:
        """Persistent IDs in the authoritative sequence from the last sync."""

        return self._ordered_record_ids

    @property
    def active_record_id_set(self) -> frozenset[int]:
        return frozenset(self._slot_by_record_id)

    @property
    def active_count(self) -> int:
        return len(self._slot_by_record_id)

    @property
    def unique_resource_group_count(self) -> int:
        return len(self._group_by_resource)

    @property
    def resource_group_peak(self) -> int:
        return self._resource_group_peak

    def _grow_slots(self) -> None:
        old = self._capacity
        new = max(old + 1, old * 2)
        self._record_ids = np.pad(self._record_ids, (0, new - old), constant_values=-1)
        self._static_intrinsic = np.pad(
            self._static_intrinsic, ((0, new - old), (0, 0)), constant_values=0.0
        )
        self._active_mask = np.pad(
            self._active_mask, (0, new - old), constant_values=False
        )
        self._slot_group = np.pad(self._slot_group, (0, new - old), constant_values=-1)
        self._novelty_by_slot = np.pad(
            self._novelty_by_slot, (0, new - old), constant_values=1.0
        )
        self._keys_by_slot.extend([None] * (new - old))
        self._resources_by_slot.extend([None] * (new - old))
        self._free_slots.extend(reversed(range(old, new)))
        self._capacity = new

    def _ensure_group_resource_matrix(self, width: int) -> None:
        if self._resource_dimension is None:
            self._resource_dimension = width
            self._resource_group_matrix = np.zeros(
                (self._group_capacity, width), dtype=np.int64
            )
        elif self._resource_dimension != width:
            raise ValueError("cannot mix resource tuples from different register widths")

    def _grow_groups(self) -> None:
        old = self._group_capacity
        new = max(old + 1, old * 2)
        width = 0 if self._resource_dimension is None else self._resource_dimension
        self._resource_group_matrix = np.pad(
            self._resource_group_matrix,
            ((0, new - old), (0, 0)),
            constant_values=0,
        ).reshape(new, width)
        self._group_frequency = np.pad(
            self._group_frequency, (0, new - old), constant_values=0
        )
        self._group_dominator_count = np.pad(
            self._group_dominator_count, (0, new - old), constant_values=0
        )
        self._group_active = np.pad(
            self._group_active, (0, new - old), constant_values=False
        )
        self._group_resources.extend([None] * (new - old))
        self._group_to_active_slots.extend(set() for _ in range(new - old))
        self._free_groups.extend(reversed(range(old, new)))
        self._group_capacity = new

    def _active_group_indices(self) -> np.ndarray:
        return np.flatnonzero(self._group_active)

    def _insert_resource(self, resources: tuple[int, ...]) -> int:
        started_ns = perf_counter_ns()
        try:
            self._ensure_group_resource_matrix(len(resources))
            active = self._active_group_indices()
            incoming = np.asarray(resources, dtype=np.int64)
            existing = self._group_by_resource.get(resources)
            if existing is not None:
                dominated_groups = np.all(
                    incoming <= self._resource_group_matrix[active], axis=1
                )
                self._group_dominator_count[active[dominated_groups]] += 1
                self._group_frequency[existing] += 1
                return existing

            dominator_count = 1
            if active.size:
                active_resources = self._resource_group_matrix[active]
                dominates_existing = np.all(incoming <= active_resources, axis=1)
                self._group_dominator_count[active[dominates_existing]] += 1
                existing_dominates = np.all(active_resources <= incoming, axis=1)
                dominator_count += int(
                    np.sum(self._group_frequency[active[existing_dominates]], dtype=np.int64)
                )
            if not self._free_groups:
                self._grow_groups()
            group = self._free_groups.pop()
            self._resource_group_matrix[group] = incoming
            self._group_frequency[group] = 1
            self._group_dominator_count[group] = dominator_count
            self._group_active[group] = True
            self._group_resources[group] = resources
            self._group_to_active_slots[group].clear()
            self._group_by_resource[resources] = group
            self._resource_group_peak = max(
                self._resource_group_peak, self.unique_resource_group_count
            )
            return group
        finally:
            self._dominance_update_time_ns += perf_counter_ns() - started_ns

    def _remove_resource(self, group: int, resources: tuple[int, ...]) -> None:
        started_ns = perf_counter_ns()
        try:
            if not self._group_active[group] or self._group_resources[group] != resources:
                raise AssertionError("record resource group is not active")
            active = self._active_group_indices()
            removed = np.asarray(resources, dtype=np.int64)
            dominated_groups = np.all(
                removed <= self._resource_group_matrix[active], axis=1
            )
            self._group_dominator_count[active[dominated_groups]] -= 1
            self._group_frequency[group] -= 1
            if self._group_frequency[group] < 0:  # pragma: no cover - invariant guard
                raise AssertionError("resource-group frequency became negative")
            if self._group_frequency[group] == 0:
                if self._group_to_active_slots[group]:  # pragma: no cover
                    raise AssertionError("empty resource group still owns active slots")
                del self._group_by_resource[resources]
                self._group_active[group] = False
                self._group_resources[group] = None
                self._group_dominator_count[group] = 0
                self._free_groups.append(group)
        finally:
            self._dominance_update_time_ns += perf_counter_ns() - started_ns

    def _allocate_slot(self) -> int:
        if not self._free_slots:
            self._grow_slots()
        return self._free_slots.pop()

    def _add_record(self, record: object) -> None:
        record_id = _record_id(record)
        if record_id in self._slot_by_record_id:
            raise ValueError(f"duplicate active record ID {record_id}")
        state = _record_state(record)
        resources = _resource_tuple(record, state)
        key = _record_key(record, state, self._semantic_key)
        intrinsic = np.asarray(self._intrinsic_prefix(state), dtype=np.float64)
        if intrinsic.shape != (8,) or not np.isfinite(intrinsic).all():
            raise ValueError("Article V1 intrinsic callback must return eight finite values")
        slot = self._allocate_slot()
        group = self._insert_resource(resources)
        self._record_ids[slot] = record_id
        self._static_intrinsic[slot] = intrinsic
        self._active_mask[slot] = True
        self._slot_group[slot] = group
        self._keys_by_slot[slot] = key
        self._resources_by_slot[slot] = resources
        self._slot_by_record_id[record_id] = slot
        self._record_by_id[record_id] = record
        self._key_to_active_slots.setdefault(key, set()).add(slot)
        self._group_to_active_slots[group].add(slot)
        count = max(1, self._generation_counts.get(key, 0))
        self._novelty_by_slot[slot] = 1.0 / sqrt(count)
        heapq.heappush(self._target_heap, (float(intrinsic[7]), record_id))
        self._static_cache_misses += 1
        self._frontier_index_additions += 1

    def _remove_record(self, record_id: int) -> None:
        slot = self._slot_by_record_id.pop(record_id)
        key = self._keys_by_slot[slot]
        resources = self._resources_by_slot[slot]
        group = int(self._slot_group[slot])
        if key is None or resources is None:  # pragma: no cover - invariant guard
            raise AssertionError("active slot has no cached key/resources")
        key_slots = self._key_to_active_slots[key]
        key_slots.remove(slot)
        if not key_slots:
            del self._key_to_active_slots[key]
        self._group_to_active_slots[group].remove(slot)
        self._remove_resource(group, resources)
        self._record_by_id.pop(record_id, None)
        self._record_ids[slot] = -1
        self._active_mask[slot] = False
        self._slot_group[slot] = -1
        self._novelty_by_slot[slot] = 1.0
        self._keys_by_slot[slot] = None
        self._resources_by_slot[slot] = None
        self._free_slots.append(slot)
        self._frontier_index_removals += 1

    def initialize(
        self,
        open_records: Sequence[object],
        *,
        generation_counts: Mapping[object, int] | None = None,
    ) -> None:
        """Reset and build from one authoritative frontier snapshot."""

        self._reset_storage()
        self._initialized = True
        self._frontier_index_rebuilds = 1
        if generation_counts is not None:
            self.replace_generation_counts(generation_counts)
        records = tuple(open_records)
        ids = tuple(_record_id(record) for record in records)
        if len(set(ids)) != len(ids):
            raise ValueError("authoritative frontier contains duplicate record IDs")
        by_id = {record_id: record for record_id, record in zip(ids, records)}
        for record_id in sorted(by_id):
            self._add_record(by_id[record_id])
        self._ordered_record_ids = ids
        if records:
            self._frontier_revision = 1
        if self.debug_reconciliation:
            self.reconcile(records)

    def _apply_generation_counts(
        self, updates: Mapping[object, int]
    ) -> tuple[tuple[object, int], ...]:
        if not isinstance(updates, Mapping):
            raise TypeError("generation-count updates must be a mapping")
        changed: list[tuple[object, int]] = []
        for key, raw_count in updates.items():
            try:
                hash(key)
            except TypeError as exc:
                raise TypeError("generation-count keys must be hashable") from exc
            count = _validated_nonnegative_count(raw_count)
            if self._generation_counts.get(key, 0) == count:
                continue
            if count == 0:
                self._generation_counts.pop(key, None)
            else:
                self._generation_counts[key] = count
            novelty = 1.0 / sqrt(max(1, count))
            for slot in self._key_to_active_slots.get(key, ()):
                self._novelty_by_slot[slot] = novelty
            changed.append((key, count))
        if changed:
            self._generation_count_revision += 1
        return tuple(sorted(changed, key=lambda item: repr(item[0])))

    def update_generation_counts(
        self, updates: Mapping[object, int]
    ) -> tuple[tuple[object, int], ...]:
        """Apply absolute counts only for keys whose totals changed."""

        return self._apply_generation_counts(updates)

    def increment_generation_counts(
        self, deltas: Mapping[object, int]
    ) -> tuple[tuple[object, int], ...]:
        """Apply signed count deltas without copying the complete count map."""

        absolute: dict[object, int] = {}
        for key, raw_delta in deltas.items():
            if isinstance(raw_delta, bool) or not isinstance(raw_delta, (int, np.integer)):
                raise TypeError("generation-count deltas must be integers")
            value = self._generation_counts.get(key, 0) + int(raw_delta)
            if value < 0:
                raise ValueError("generation-count delta would make a count negative")
            absolute[key] = value
        return self._apply_generation_counts(absolute)

    def replace_generation_counts(
        self, values: Mapping[object, int]
    ) -> tuple[tuple[object, int], ...]:
        """Compatibility adapter for a complete absolute count mapping."""

        if not isinstance(values, Mapping):
            raise TypeError("generation counts must be a mapping")
        normalized = {key: _validated_nonnegative_count(value) for key, value in values.items()}
        updates: dict[object, int] = dict(normalized)
        for key in self._generation_counts.keys() - normalized.keys():
            updates[key] = 0
        return self._apply_generation_counts(updates)

    def generation_counts_snapshot(self) -> Mapping[object, int]:
        return MappingProxyType(dict(self._generation_counts))

    def synchronize(
        self,
        open_records: Sequence[object],
        *,
        generation_count_updates: Mapping[object, int] | None = None,
    ) -> FrontierMutationBatch:
        """Mirror an authoritative record sequence using an exact set difference."""

        if not self._initialized:
            self.initialize(open_records, generation_counts=generation_count_updates or {})
            records = tuple(open_records)
            return FrontierMutationBatch(
                revision_before=0,
                revision_after=self._frontier_revision,
                removed_record_ids=(),
                added_records=tuple(sorted(records, key=_record_id)),
                generation_count_updates=tuple(
                    sorted(
                        ((key, int(value)) for key, value in (generation_count_updates or {}).items()),
                        key=lambda item: repr(item[0]),
                    )
                ),
            )

        records = tuple(open_records)
        ids = tuple(_record_id(record) for record in records)
        if len(set(ids)) != len(ids):
            raise ValueError("authoritative frontier contains duplicate record IDs")
        by_id = {record_id: record for record_id, record in zip(ids, records)}
        previous_ids = set(self._slot_by_record_id)
        next_ids = set(ids)
        common_ids = previous_ids & next_ids
        replacement_ids: set[int] = set()
        for record_id in common_ids:
            incoming = by_id[record_id]
            previous = self._record_by_id[record_id]
            if previous is incoming:
                continue
            slot = self._slot_by_record_id[record_id]
            if hasattr(incoming, "resources") and hasattr(incoming, "key"):
                state = _record_state(incoming)
                if _resource_tuple(incoming, state) != self._resources_by_slot[slot]:
                    raise AssertionError(
                        f"authoritative resources changed for persistent record {record_id}"
                    )
                if getattr(incoming, "key") != self._keys_by_slot[slot]:
                    raise AssertionError(
                        f"authoritative semantic key changed for persistent record {record_id}"
                    )
            else:
                # Compatibility-only synthetic records do not promise stable
                # wrapper identity.  Refresh their static row rather than
                # trusting an ID that may have been reassigned by ``extract``.
                replacement_ids.add(record_id)
        removed = tuple(sorted((previous_ids - next_ids) | replacement_ids))
        added_ids = tuple(sorted((next_ids - previous_ids) | replacement_ids))
        revision_before = self._frontier_revision
        for record_id in removed:
            self._remove_record(record_id)
        for record_id in added_ids:
            self._add_record(by_id[record_id])
        common = common_ids - replacement_ids
        self._static_cache_hits += len(common)
        for record_id in common:
            incoming = by_id[record_id]
            # A replacement authoritative ArchiveRecord with equal immutable
            # payload may safely refresh only the object association.
            self._record_by_id[record_id] = incoming
        self._ordered_record_ids = ids
        if removed or added_ids:
            self._frontier_revision += 1
        changed_counts = self._apply_generation_counts(generation_count_updates or {})
        if self.debug_reconciliation:
            self.reconcile(records)
        return FrontierMutationBatch(
            revision_before=revision_before,
            revision_after=self._frontier_revision,
            removed_record_ids=removed,
            added_records=tuple(by_id[record_id] for record_id in added_ids),
            generation_count_updates=changed_counts,
        )

    def _candidate_coordinates_for_slots(self, slots: np.ndarray) -> np.ndarray:
        intrinsic = self._static_intrinsic[slots]
        groups = self._slot_group[slots]
        denominator = max(1, len(slots) - 1)
        dominance = (
            self._group_dominator_count[groups].astype(np.float64) - 1.0
        ) / float(denominator)
        novelty = self._novelty_by_slot[slots]
        if self.include_target:
            return np.column_stack((intrinsic, dominance, novelty))
        return np.column_stack((intrinsic[:, :7], dominance, novelty))

    def _record_score_time(self, elapsed_ns: int) -> None:
        self._score_time_ns += int(elapsed_ns)

    def _record_row_time(self, elapsed_ns: int) -> None:
        self._selected_row_materialization_time_ns += int(elapsed_ns)

    def build_decision_batch(
        self,
        *,
        theta: np.ndarray | None = None,
        expansions_completed: int,
        expansion_budget: int,
    ) -> CompactArticleDecisionBatch:
        """Gather one exact compact batch in authoritative frontier order."""

        del theta  # accepted for integration symmetry; scoring stays explicit
        if not self._initialized:
            raise RuntimeError("feature index must be initialized before building a batch")
        if not self._ordered_record_ids:
            raise ValueError("an Article feature batch requires a nonempty frontier")
        if (
            isinstance(expansion_budget, bool)
            or not isinstance(expansion_budget, (int, np.integer))
            or int(expansion_budget) <= 0
        ):
            raise ValueError("expansion_budget must be a positive integer")
        if (
            isinstance(expansions_completed, bool)
            or not isinstance(expansions_completed, (int, np.integer))
            or int(expansions_completed) < 0
            or int(expansions_completed) > int(expansion_budget)
        ):
            raise ValueError("expansions_completed must lie in [0, expansion_budget]")
        started_ns = perf_counter_ns()
        ids = self._ordered_record_ids
        slots = np.fromiter(
            (self._slot_by_record_id[record_id] for record_id in ids),
            dtype=np.int64,
            count=len(ids),
        )
        gather_started_ns = perf_counter_ns()
        candidate_matrix = self._candidate_coordinates_for_slots(slots)
        records = tuple(self._record_by_id[record_id] for record_id in ids)
        nodes = tuple(_record_node(record) for record in records)
        record_ids = np.asarray(ids, dtype=np.int64)
        self._candidate_gather_time_ns += perf_counter_ns() - gather_started_ns

        standardization_started_ns = perf_counter_ns()
        sorted_matrix = np.sort(candidate_matrix, axis=0)
        mean = np.mean(sorted_matrix, axis=0, dtype=np.float64)
        std = np.std(sorted_matrix, axis=0, ddof=0, dtype=np.float64)
        self._standardization_time_ns += perf_counter_ns() - standardization_started_ns
        budget = int(expansion_budget)
        completed = int(expansions_completed)
        remaining = float((budget - completed) / budget)
        target_context = getattr(self.feature_provider, "target_context", None)
        fingerprint = (
            str(getattr(target_context, "fingerprint"))
            if self.include_target and target_context is not None
            else None
        )
        digest = sha256()
        digest.update(self.schema_version.encode("ascii"))
        digest.update(self.evaluator_schema_version.encode("ascii"))
        digest.update(str(self._frontier_revision).encode("ascii"))
        digest.update(str(self._generation_count_revision).encode("ascii"))
        digest.update(str(completed).encode("ascii"))
        digest.update(b"/")
        digest.update(str(budget).encode("ascii"))
        digest.update((fingerprint or "").encode("ascii"))
        for record_id in sorted(ids):
            digest.update(str(record_id).encode("ascii"))
        frozen_candidates = _readonly_array(candidate_matrix, dtype=np.dtype(np.float64))
        frozen_ids = _readonly_array(record_ids, dtype=np.dtype(np.int64))
        frozen_mean = _readonly_array(mean, dtype=np.dtype(np.float64))
        frozen_std = _readonly_array(std, dtype=np.dtype(np.float64))
        row_map = MappingProxyType(
            {record_id: row for row, record_id in enumerate(ids)}
        )
        result = CompactArticleDecisionBatch(
            schema_version=self.schema_version,
            evaluator_schema_version=self.evaluator_schema_version,
            frontier_revision=self._frontier_revision,
            generation_count_revision=self._generation_count_revision,
            records=records,
            frontier_nodes=nodes,
            record_ids=frozen_ids,
            candidate_matrix=frozen_candidates,
            mean=frozen_mean,
            std=frozen_std,
            remaining_budget_fraction=remaining,
            target_fingerprint=fingerprint,
            expansions_completed=completed,
            expansion_budget=budget,
            include_frontier_context=self.include_frontier_context,
            standardization_eta=self.standardization_eta,
            snapshot_id=f"sha256:{digest.hexdigest()}",
            _row_by_record_id=row_map,
            _score_observer=self._record_score_time,
            _row_observer=self._record_row_time,
        )
        self._compact_batch_time_ns += perf_counter_ns() - started_ns
        return result

    def minimum_target_distance(self) -> float:
        if not self._slot_by_record_id:
            raise ValueError("an empty frontier has no target-distance minimum")
        while self._target_heap:
            distance, record_id = self._target_heap[0]
            slot = self._slot_by_record_id.get(record_id)
            if slot is not None and float(self._static_intrinsic[slot, 7]) == distance:
                return float(distance)
            heapq.heappop(self._target_heap)
        raise AssertionError("active frontier has no target-distance heap entry")

    def minimum_target_record_id(self) -> int:
        minimum = self.minimum_target_distance()
        tied = [
            record_id
            for record_id, slot in self._slot_by_record_id.items()
            if float(self._static_intrinsic[slot, 7]) == minimum
        ]
        return min(tied)

    def select_target_distance_node(self) -> object:
        return _record_node(self._record_by_id[self.minimum_target_record_id()])

    def static_candidate(self, record_id: int) -> StaticArticleCandidate:
        try:
            slot = self._slot_by_record_id[int(record_id)]
        except (KeyError, TypeError, ValueError) as exc:
            raise KeyError(f"record ID {record_id!r} is not active") from exc
        key = self._keys_by_slot[slot]
        resources = self._resources_by_slot[slot]
        if key is None or resources is None:  # pragma: no cover
            raise AssertionError("active slot has incomplete static cache")
        return StaticArticleCandidate(
            record_id=int(record_id),
            semantic_key=key,
            resource_tuple=resources,
            intrinsic_coordinates=_readonly_array(
                self._static_intrinsic[slot], dtype=np.dtype(np.float64)
            ),
        )

    def reconcile(self, open_records: Sequence[object] | None = None) -> None:
        """Fail if cached membership or exact group counts have drifted."""

        if open_records is not None:
            authoritative = tuple(_record_id(record) for record in open_records)
            if authoritative != self._ordered_record_ids:
                raise AssertionError(
                    "feature-index record order differs from the authoritative frontier"
                )
            if set(authoritative) != set(self._slot_by_record_id):
                raise AssertionError(
                    "feature-index membership differs from the authoritative frontier"
                )
        active_groups = self._active_group_indices()
        if int(np.sum(self._group_frequency[active_groups], dtype=np.int64)) != self.active_count:
            raise AssertionError("resource-group frequencies do not sum to frontier size")
        for group in active_groups:
            resources = self._resource_group_matrix[group]
            expected = int(
                np.sum(
                    self._group_frequency[
                        active_groups[
                            np.all(
                                self._resource_group_matrix[active_groups] <= resources,
                                axis=1,
                            )
                        ]
                    ],
                    dtype=np.int64,
                )
            )
            actual = int(self._group_dominator_count[group])
            if actual != expected:
                raise AssertionError(
                    "resource-group dominator count drifted: "
                    f"group={int(group)}, actual={actual}, expected={expected}"
                )
            slots = self._group_to_active_slots[group]
            if len(slots) != int(self._group_frequency[group]):
                raise AssertionError("resource-group slot membership drifted")
        denominator = max(1, self.active_count - 1)
        for record_id, slot in self._slot_by_record_id.items():
            resources = self._resources_by_slot[slot]
            if resources is None:
                raise AssertionError(f"active record {record_id} has no resources")
            reference_count = sum(
                other_id != record_id
                and all(left <= right for left, right in zip(other_resources, resources))
                for other_id, other_slot in self._slot_by_record_id.items()
                if (other_resources := self._resources_by_slot[other_slot]) is not None
            )
            group = int(self._slot_group[slot])
            indexed_count = int(self._group_dominator_count[group]) - 1
            if indexed_count != reference_count:
                raise AssertionError(
                    "record dominance count differs from the all-pairs oracle: "
                    f"record_id={record_id}, indexed={indexed_count}, "
                    f"reference={reference_count}, denominator={denominator}"
                )

    def feature_index_memory_bytes(self) -> int:
        arrays = (
            self._record_ids,
            self._static_intrinsic,
            self._active_mask,
            self._slot_group,
            self._novelty_by_slot,
            self._resource_group_matrix,
            self._group_frequency,
            self._group_dominator_count,
            self._group_active,
        )
        total = sum(int(array.nbytes) for array in arrays)
        containers = (
            self._keys_by_slot,
            self._resources_by_slot,
            self._free_slots,
            self._slot_by_record_id,
            self._record_by_id,
            self._ordered_record_ids,
            self._key_to_active_slots,
            self._generation_counts,
            self._group_resources,
            self._group_to_active_slots,
            self._free_groups,
            self._group_by_resource,
            self._target_heap,
        )
        total += sum(sys.getsizeof(value) for value in containers)
        total += sum(sys.getsizeof(value) for value in self._key_to_active_slots.values())
        total += sum(sys.getsizeof(value) for value in self._group_to_active_slots)
        return int(total)

    def instrumentation(self) -> dict[str, int | str]:
        return {
            "feature_evaluator_schema_version": self.evaluator_schema_version,
            "feature_static_cache_hits": self._static_cache_hits,
            "feature_static_cache_misses": self._static_cache_misses,
            "frontier_index_additions": self._frontier_index_additions,
            "frontier_index_removals": self._frontier_index_removals,
            "frontier_index_rebuilds": self._frontier_index_rebuilds,
            "unique_resource_group_count": self.unique_resource_group_count,
            "resource_group_peak": self._resource_group_peak,
            "dominance_update_time_ns": self._dominance_update_time_ns,
            "compact_batch_time_ns": self._compact_batch_time_ns,
            "candidate_gather_time_ns": self._candidate_gather_time_ns,
            "standardization_time_ns": self._standardization_time_ns,
            "score_time_ns": self._score_time_ns,
            "selected_row_materialization_time_ns": (
                self._selected_row_materialization_time_ns
            ),
            "feature_index_memory_bytes": self.feature_index_memory_bytes(),
            "frontier_revision": self._frontier_revision,
            "generation_count_revision": self._generation_count_revision,
        }


__all__ = [
    "ARTICLE_V1_EXACT_INCREMENTAL_EVALUATOR_SCHEMA_VERSION",
    "CompactArticleDecisionBatch",
    "ExactArticleFrontierFeatureIndex",
    "FrontierMutationBatch",
    "StaticArticleCandidate",
]
