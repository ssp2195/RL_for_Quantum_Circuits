"""Priority frontier backed by a semantic Pareto archive.

``Frontier`` remains node-oriented for the existing Gymnasium environment,
while internally queueing stable archive-record IDs.  This lets a semantic
state be reopened with an incomparable resource witness and makes stale heap
entries safe after a dominance replacement.
"""

from __future__ import annotations

import heapq
import itertools
from typing import TYPE_CHECKING, Optional, Union

from search.archive import ArchiveRecord, InsertResult, ParetoArchive, ResourceVector
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
    ):
        if archive is not None and canonicalizer is not None:
            if archive.canonicalizer is not canonicalizer:
                raise ValueError("pass either canonicalizer or archive, not mismatched both")

        self.archive = archive or ParetoArchive(canonicalizer=canonicalizer)
        self.canonicalizer = self.archive.canonicalizer

        # (priority, insertion_order, record_id).  Entries are deliberately
        # not deleted on dominance; ``pop`` skips tombstones lazily.
        self._queue: list[tuple[float, int, int]] = []
        self._counter = itertools.count()
        self._queued_ids: set[int] = set()

    @property
    def heap(self) -> list[SearchNode]:
        """Deterministic active-node snapshot for the legacy environment API."""
        return self.nodes()

    def insert(self, node: SearchNode) -> InsertResult:
        """Archive then queue ``node``; expose detailed dominance information."""
        result = self.archive.insert(node)
        if result.accepted:
            assert result.record is not None
            self.add(result.record)
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
        return True

    def pop(self) -> SearchNode:
        """Remove and mark the best active record expanded, skipping tombstones."""
        while self._queue:
            _, _, record_id = heapq.heappop(self._queue)
            self._queued_ids.discard(record_id)
            record = self.archive.record(record_id)
            if record is None or not self._is_open(record):
                continue

            self.archive.mark_expanded(record)
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
        return self.archive.mark_expanded(record)

    def active_records(self) -> list[ArchiveRecord]:
        """Return currently selectable records in a deterministic order."""
        # The frontier owns ``_queued_ids``.  Looking up only those record
        # IDs avoids a full archive scan for every RL observation; a
        # constrained normal-form search can retain many historical expanded
        # witnesses while only a comparatively small open subset is eligible
        # for selection.  Tombstoned stale IDs are filtered lazily exactly as
        # heap entries are in ``pop``.
        records = [
            record
            for record_id in tuple(self._queued_ids)
            if (record := self.archive.record(record_id)) is not None
            and self._is_open(record)
        ]
        self._queued_ids.intersection_update(record.record_id for record in records)
        return sorted(records, key=lambda record: (record.node.priority, record.record_id))

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
