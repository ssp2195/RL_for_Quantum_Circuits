"""Priority frontier backed by a semantic Pareto archive.

``Frontier`` remains node-oriented for the existing Gymnasium environment,
while internally queueing stable archive-record IDs.  This lets a semantic
state be reopened with an incomparable resource witness and makes stale heap
entries safe after a dominance replacement.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import replace
from typing import TYPE_CHECKING, Optional, Union

from search.archive import (
    ArchiveRecord,
    InsertResult,
    ParetoArchive,
    PreparedArchiveCandidate,
    ResourceVector,
)
from search.node import SearchNode

if TYPE_CHECKING:
    from canonical.canonicalizer import Canonicalizer


RecordReference = Union[SearchNode, ArchiveRecord, int]


class Frontier:
    """An active-record priority queue coordinated with ``ParetoArchive``."""

    def __init__(
        self,
        canonicalizer: Optional[Canonicalizer] = None,
        *,
        archive: Optional[ParetoArchive] = None,
        canonicalization_enabled: Optional[bool] = None,
        pareto_dominance_enabled: Optional[bool] = None,
    ):
        if archive is not None and canonicalizer is not None:
            if archive.canonicalizer is not canonicalizer:
                raise ValueError("pass either canonicalizer or archive, not mismatched both")

        if archive is not None:
            if (
                canonicalization_enabled is not None
                and archive.canonicalization_enabled != canonicalization_enabled
            ):
                raise ValueError("archive canonicalization setting does not match")
            if (
                pareto_dominance_enabled is not None
                and archive.pareto_dominance_enabled != pareto_dominance_enabled
            ):
                raise ValueError("archive Pareto setting does not match")
            self.archive = archive
        else:
            self.archive = ParetoArchive(
                canonicalizer=canonicalizer,
                canonicalization_enabled=(
                    True
                    if canonicalization_enabled is None
                    else canonicalization_enabled
                ),
                pareto_dominance_enabled=(
                    True
                    if pareto_dominance_enabled is None
                    else pareto_dominance_enabled
                ),
            )
        self.canonicalizer = self.archive.canonicalizer

        # (priority, insertion_order, record_id).  Entries are deliberately
        # not deleted on dominance; ``pop`` skips tombstones lazily.
        self._queue: list[tuple[float, int, int]] = []
        self._counter = itertools.count()
        self._queued_ids: set[int] = set()
        # Exact record-ID index for O(1) membership/update and an optional
        # allocation-order engineering view.  Scientific selection continues
        # to use the reviewed priority-then-record-ID ordering below.
        self._open_records_by_id: dict[int, ArchiveRecord] = {}
        self._ordered_cache_revision = -1
        self._ordered_active_records: tuple[ArchiveRecord, ...] = ()
        # Monotone engineering revision for caches that mirror the authoritative
        # open-record set.  Record identity and archive membership remain the
        # scientific source of truth; the revision is only an invalidation key.
        self._revision = 0

    @property
    def revision(self) -> int:
        """Return the monotone open-frontier membership revision."""

        return int(self._revision)

    def active_record_ids(self) -> tuple[int, ...]:
        """Return authoritative open record IDs in deterministic frontier order."""

        return tuple(record.record_id for record in self.active_records())

    @property
    def heap(self) -> list[SearchNode]:
        """Deterministic active-node snapshot for the legacy environment API."""
        return self.nodes()

    def insert(self, node: SearchNode) -> InsertResult:
        """Archive then queue ``node``; expose detailed dominance information."""
        return self._finish_insert(node, self.archive.insert(node))

    def insert_prepared(
        self,
        node: SearchNode,
        prepared: PreparedArchiveCandidate,
        *,
        canonicalization_time_ns: int = 0,
        debug_recompute: bool = False,
    ) -> InsertResult:
        """Archive and queue a node using its already-computed exact payload."""

        return self._finish_insert(
            node,
            self.archive.insert_prepared(
                node,
                prepared,
                canonicalization_time_ns=canonicalization_time_ns,
                debug_recompute=debug_recompute,
            ),
        )

    def _finish_insert(
        self,
        node: SearchNode,
        result: InsertResult,
    ) -> InsertResult:
        open_before = set(self._queued_ids)
        dominated_open_ids = {
            record.record_id
            for record in result.dominated
            if record.record_id in open_before
        }
        if dominated_open_ids:
            self._queued_ids.difference_update(dominated_open_ids)
            for record_id in dominated_open_ids:
                self._open_records_by_id.pop(record_id, None)
            self._revision += len(dominated_open_ids)
        if result.accepted:
            assert result.record is not None
            queued = self.add(result.record)
            if not queued:  # pragma: no cover - archive/frontier invariant
                raise AssertionError("accepted archive record was not selectable")
            result = replace(
                result,
                reopened=bool(result.previously_expanded and queued),
            )
        return result

    def push(self, node: SearchNode) -> bool:
        """Legacy boolean insertion API used by ``CircuitSynthesisEnv``."""
        return self.insert(node).accepted

    def add(self, record: ArchiveRecord) -> bool:
        """Queue an already-archived unexpanded record exactly once."""
        known = self.archive.record(record.record_id)
        if known is not record:
            raise ValueError("record does not belong to this frontier archive")
        if (
            not record.active
            or record.tombstoned
            or record.expanded
            or record.record_id in self._queued_ids
        ):
            return False

        record.queued = True
        heapq.heappush(
            self._queue,
            (float(record.node.priority), next(self._counter), record.record_id),
        )
        self._queued_ids.add(record.record_id)
        self._open_records_by_id[record.record_id] = record
        self._revision += 1
        return True

    def pop(self) -> SearchNode:
        """Remove and mark the best active record expanded, skipping tombstones."""
        while self._queue:
            _, _, record_id = heapq.heappop(self._queue)
            self._queued_ids.discard(record_id)
            self._open_records_by_id.pop(record_id, None)
            record = self.archive.record(record_id)
            if record is None or not self._is_open(record):
                continue

            self.archive.mark_expanded(record)
            self._revision += 1
            return record.node
        raise IndexError("pop from an empty frontier")

    def remove(self, node: RecordReference) -> bool:
        """Select a particular open record and mark it expanded.

        This only removes the selected record from the *frontier*.  The record
        remains active in the archive so it can still dominate later, weaker
        witnesses of the same semantic state.
        """
        record = self._resolve_record(node)
        if record is None or not self._is_open(record):
            return False

        self._queued_ids.discard(record.record_id)
        self._open_records_by_id.pop(record.record_id, None)
        removed = self.archive.mark_expanded(record)
        if removed:
            self._revision += 1
        return removed

    def active_records(self) -> list[ArchiveRecord]:
        """Return currently selectable records in a deterministic order."""
        # Preserve the reviewed priority/record-ID enumeration exactly.  The
        # cached tuple avoids repeated sorts when feature synchronization and
        # policy ranking inspect the same immutable frontier revision.
        self._remove_stale_open_records()
        if self._ordered_cache_revision != self._revision:
            self._ordered_active_records = tuple(
                sorted(
                    self._open_records_by_id.values(),
                    key=lambda record: (record.node.priority, record.record_id),
                )
            )
            self._ordered_cache_revision = self._revision
        return list(self._ordered_active_records)

    def _remove_stale_open_records(self) -> None:
        stale_ids = [
            record_id
            for record_id, record in self._open_records_by_id.items()
            if not self._is_open(record)
        ]
        if stale_ids:
            for record_id in stale_ids:
                self._open_records_by_id.pop(record_id, None)
                self._queued_ids.discard(record_id)
            self._revision += len(stale_ids)

    def active_records_by_id(self) -> list[ArchiveRecord]:
        """Return the open set in persistent-record-ID order without sorting."""

        self._remove_stale_open_records()
        return list(self._open_records_by_id.values())

    def active_nodes_by_id(self) -> list[SearchNode]:
        """Return selectable nodes in persistent-record-ID order."""

        return [record.node for record in self.active_records_by_id()]

    def nodes(self) -> list[SearchNode]:
        """Return currently selectable nodes ordered by priority then record ID."""
        return [record.node for record in self.active_records()]

    def active_nodes(self) -> list[SearchNode]:
        """Explicit alias for callers that do not rely on ``heap`` compatibility."""
        return self.nodes()

    def contains(self, node: RecordReference) -> bool:
        record = self._resolve_record(node)
        return record is not None and self._is_open(record)

    def is_empty(self) -> bool:
        return not self.active_records()

    def __len__(self) -> int:
        return len(self.active_records())

    @staticmethod
    def _dominates(left: ResourceVector, right: ResourceVector) -> bool:
        """Compatibility helper for callers of the former frontier API."""
        if isinstance(left, ResourceVector) and isinstance(right, ResourceVector):
            return left.strictly_dominates(right)

        left_tuple = tuple(left)
        right_tuple = tuple(right)
        if len(left_tuple) != len(right_tuple):
            raise ValueError("cannot compare resource vectors of different lengths")
        return all(a <= b for a, b in zip(left_tuple, right_tuple)) and any(
            a < b for a, b in zip(left_tuple, right_tuple)
        )

    @staticmethod
    def _is_open(record: ArchiveRecord) -> bool:
        return (
            record.active
            and not record.tombstoned
            and record.queued
            and not record.expanded
        )

    def _resolve_record(self, reference: RecordReference) -> Optional[ArchiveRecord]:
        if isinstance(reference, ArchiveRecord):
            return reference if self.archive.record(reference.record_id) is reference else None
        if isinstance(reference, int):
            return self.archive.record(reference)

        if reference.record_id is not None:
            record = self.archive.record(reference.record_id)
            if record is not None and record.node is reference:
                return record

        # Defensive support for a node whose ID was stripped by an external
        # caller.  Identity, not semantic equality, is intentional here.
        for record in self.archive.all_records():
            if record.node is reference:
                return record
        return None
