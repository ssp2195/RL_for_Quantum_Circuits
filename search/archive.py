"""Pareto archive for concrete witnesses of canonical semantic states.

The canonical key answers *what circuit transformation is represented*;
``ResourceVector`` answers *how cheaply a concrete witness reached it*.
Keeping them separate is important: equal semantic states can have multiple
incomparable, continuation-safe resource records.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import time
from typing import TYPE_CHECKING, DefaultDict, Dict, Hashable, Optional, Tuple

from search.node import SearchNode

if TYPE_CHECKING:
    from canonical.canonicalizer import Canonicalizer


CanonicalKey = Hashable


@dataclass(frozen=True, slots=True)
class ResourceVector:
    """Continuation-monotone cost vector used for Pareto dominance.

    The per-wire depths are intentionally retained componentwise.  A scalar
    circuit depth is insufficient because a common suffix may run in parallel
    with work on another wire.
    """

    t_count: int
    two_qubit_count: int
    num_gates: int
    wire_depths: Tuple[int, ...]

    def __post_init__(self) -> None:
        values = (self.t_count, self.two_qubit_count, self.num_gates)
        if any(int(value) < 0 for value in values):
            raise ValueError("resource counts must be non-negative")

        depths = tuple(int(depth) for depth in self.wire_depths)
        if any(depth < 0 for depth in depths):
            raise ValueError("wire depths must be non-negative")
        object.__setattr__(self, "wire_depths", depths)

    @classmethod
    def from_state(cls, state: object) -> "ResourceVector":
        """Read the consumed resources from a ``CircuitState``.

        The primary path uses the post-migration CircuitState fields.  The
        small DAG-derived fallbacks retain compatibility with an old state
        while the semantic/resource migration is being integrated.
        """

        dag = getattr(state, "dag", None)

        two_qubit_count = getattr(state, "two_qubit_count", None)
        if two_qubit_count is None:
            two_qubit_count = sum(
                1
                for gate in getattr(dag, "gates", ())
                if _is_two_qubit(gate)
            )

        wire_depths = getattr(state, "wire_depths", None)
        if wire_depths is None:
            wire_depths = _wire_depths_from_dag(dag)

        return cls(
            t_count=int(getattr(state, "t_count", 0)),
            two_qubit_count=int(two_qubit_count),
            num_gates=int(getattr(state, "num_gates", 0)),
            wire_depths=tuple(wire_depths),
        )

    def weakly_dominates(self, other: "ResourceVector") -> bool:
        """Return whether this record is no worse in every resource."""
        self._validate_wire_arity(other)
        return (
            self.t_count <= other.t_count
            and self.two_qubit_count <= other.two_qubit_count
            and self.num_gates <= other.num_gates
            and all(
                left <= right
                for left, right in zip(self.wire_depths, other.wire_depths)
            )
        )

    def strictly_dominates(self, other: "ResourceVector") -> bool:
        """Return whether this record is no worse and better somewhere."""
        return self.weakly_dominates(other) and self != other

    def as_tuple(self) -> Tuple[int, ...]:
        """Flatten the vector for diagnostics and stable test assertions."""
        return (
            self.t_count,
            self.two_qubit_count,
            self.num_gates,
            *self.wire_depths,
        )

    def _validate_wire_arity(self, other: "ResourceVector") -> None:
        if len(self.wire_depths) != len(other.wire_depths):
            raise ValueError(
                "cannot compare resource vectors for different numbers of wires"
            )


def _is_two_qubit(gate: object) -> bool:
    predicate = getattr(gate, "is_two_qubit", None)
    return bool(predicate()) if callable(predicate) else len(gate.qubits) == 2


def _wire_depths_from_dag(dag: object) -> Tuple[int, ...]:
    if dag is None:
        return ()

    depths = []
    for node_id in getattr(dag, "last_gate_on_qubit", ()):
        if node_id is None:
            depths.append(0)
        else:
            depths.append(int(dag.nodes[node_id].level))
    return tuple(depths)


@dataclass(slots=True)
class ArchiveRecord:
    """One resource-bearing witness in a semantic state's Pareto antichain."""

    record_id: int
    node: SearchNode
    key: CanonicalKey
    resources: ResourceVector
    expanded: bool = False
    active: bool = True
    # Queue ownership belongs to Frontier; direct archive insertions are not
    # selectable until its record is explicitly added to a frontier.
    queued: bool = False
    tombstoned: bool = False


@dataclass(frozen=True, slots=True)
class InsertResult:
    """Outcome of one Pareto insertion attempt."""

    accepted: bool
    record: Optional[ArchiveRecord] = None
    dominated: Tuple[ArchiveRecord, ...] = ()
    rejected_by: Optional[ArchiveRecord] = None
    semantic_key_existed: bool = False
    pareto_incomparable_accepted: bool = False
    previously_expanded: bool = False
    reopened: bool = False
    rejection_kind: Optional[str] = None

    @property
    def duplicate_rejected(self) -> bool:
        """Whether weak Pareto dominance rejected this semantic duplicate."""

        return not self.accepted and self.rejected_by is not None

    @property
    def dominated_retired(self) -> int:
        """Number of active records strictly dominated by this insertion."""

        return len(self.dominated)

    @property
    def exact_duplicate_rejected(self) -> bool:
        return self.rejection_kind == "exact_duplicate"

    @property
    def dominance_rejected(self) -> bool:
        return self.rejection_kind == "strict_weak_dominance"


@dataclass(frozen=True, slots=True)
class PreparedArchiveCandidate:
    """Complete exact insertion payload computed once per child."""

    semantic_key: CanonicalKey
    resources: ResourceVector


class ParetoArchive:
    """Map each exact canonical key to its active resource antichain."""

    def __init__(
        self,
        canonicalizer: Optional[Canonicalizer] = None,
        *,
        canonicalization_enabled: bool = True,
        pareto_dominance_enabled: bool = True,
    ):
        if canonicalizer is None:
            # Keep resource-archive tests and utilities importable without
            # forcing the full quantum semantic implementation to load.
            from canonical.canonicalizer import Canonicalizer

            canonicalizer = Canonicalizer()
        key_fn = getattr(canonicalizer, "semantic_key", None)
        if not callable(key_fn):
            raise TypeError("canonicalizer must expose semantic_key(state)")
        if not isinstance(canonicalization_enabled, bool):
            raise TypeError("canonicalization_enabled must be a bool")
        if not isinstance(pareto_dominance_enabled, bool):
            raise TypeError("pareto_dominance_enabled must be a bool")

        self.canonicalizer = canonicalizer
        self.canonicalization_enabled = canonicalization_enabled
        self.pareto_dominance_enabled = pareto_dominance_enabled
        self._records_by_key: DefaultDict[CanonicalKey, list[ArchiveRecord]] = (
            defaultdict(list)
        )
        self._records_by_id: Dict[int, ArchiveRecord] = {}
        self._next_record_id = 0
        self._next_uncanonicalized_key = 0
        self._pareto_width_peak = 0
        self._active_record_peak = 0
        self._active_record_count = 0
        self._active_width_by_key: DefaultDict[CanonicalKey, int] = defaultdict(int)
        self.last_canonicalization_time_ns = 0
        self.last_archive_time_ns = 0

    def semantic_key(self, state: object) -> CanonicalKey:
        """Use a full canonical payload; never merge based on a digest alone."""
        key_fn = getattr(self.canonicalizer, "semantic_key", None)
        if not callable(key_fn):  # pragma: no cover - constructor guards this
            raise TypeError("canonicalizer must expose semantic_key(state)")

        key = key_fn(state)
        try:
            hash(key)
        except TypeError as error:  # pragma: no cover - canonicalizer contract
            raise TypeError("semantic_key(state) must return a hashable payload") from error
        return key

    def insert(self, node: SearchNode) -> InsertResult:
        """Insert a node unless an active record weakly dominates it.

        Strictly dominated records are tombstoned in-place instead of removed
        from the archive.  Lazy tombstones make stale priority-queue entries
        harmless and preserve a useful diagnostic history.
        """
        key_started = time.perf_counter_ns()
        prepared = self.prepare(node)
        self.last_canonicalization_time_ns = time.perf_counter_ns() - key_started
        return self.insert_prepared(
            node,
            prepared,
            canonicalization_time_ns=self.last_canonicalization_time_ns,
        )

    def prepare(self, node: SearchNode) -> PreparedArchiveCandidate:
        """Compute the exact semantic key and resource vector once."""

        return PreparedArchiveCandidate(
            semantic_key=self.semantic_key(node.state),
            resources=ResourceVector.from_state(node.state),
        )

    def insert_prepared(
        self,
        node: SearchNode,
        prepared: PreparedArchiveCandidate,
        *,
        canonicalization_time_ns: int = 0,
        debug_recompute: bool = False,
    ) -> InsertResult:
        """Insert using a caller-prepared complete payload, never a digest."""

        if not isinstance(prepared, PreparedArchiveCandidate):
            raise TypeError("prepared must be a PreparedArchiveCandidate")
        if debug_recompute and prepared != self.prepare(node):
            raise AssertionError("prepared archive candidate disagrees with node")
        insert_started = time.perf_counter_ns()
        self.last_canonicalization_time_ns = int(canonicalization_time_ns)
        semantic_key = prepared.semantic_key
        if self.canonicalization_enabled:
            key = semantic_key
        else:
            # A unique wrapper disables all semantic merging without changing
            # expansion, certification, or the canonicalizer itself.  This is
            # deliberately an explicit tiny-instance experiment mode.
            key = (
                "uncanonicalized-record",
                self._next_uncanonicalized_key,
                semantic_key,
            )
            self._next_uncanonicalized_key += 1
        resources = prepared.resources
        semantic_key_existed = key in self._records_by_key
        records = self._records_by_key[key]
        active_records = [record for record in records if record.active]
        previously_expanded = any(record.expanded for record in records)

        for old in active_records:
            # Canonical duplicate elimination remains active in the
            # Pareto-off ablation: an identical semantic/resource record has
            # neither a new continuation nor a new trade-off.  Disabling
            # Pareto dominance only retains *different* comparable resource
            # profiles, matching the article's duplicate-only control.
            exact_duplicate = old.resources == resources
            pareto_rejected = (
                self.pareto_dominance_enabled
                and old.resources.weakly_dominates(resources)
            )
            if exact_duplicate or pareto_rejected:
                result = InsertResult(
                    accepted=False,
                    rejected_by=old,
                    semantic_key_existed=semantic_key_existed,
                    previously_expanded=previously_expanded,
                    rejection_kind=(
                        "exact_duplicate"
                        if exact_duplicate
                        else "strict_weak_dominance"
                    ),
                )
                self.last_archive_time_ns = (
                    time.perf_counter_ns() - insert_started
                )
                return result

        dominated = (
            tuple(
                old
                for old in active_records
                if resources.strictly_dominates(old.resources)
            )
            if self.pareto_dominance_enabled
            else ()
        )
        for old in dominated:
            self.tombstone(old)

        record = ArchiveRecord(
            record_id=self._next_record_id,
            node=node,
            key=key,
            resources=resources,
        )
        self._next_record_id += 1
        records.append(record)
        self._records_by_id[record.record_id] = record
        self._active_record_count += 1
        self._active_width_by_key[key] += 1

        node.record_id = record.record_id
        node.expanded = False
        active_width = self._active_width_by_key[key]
        self._pareto_width_peak = max(self._pareto_width_peak, active_width)
        self._active_record_peak = max(
            self._active_record_peak,
            self._active_record_count,
        )

        # Since weakly dominated records were rejected above, every surviving
        # old active record is incomparable with the accepted new one.  A
        # pure dominating replacement therefore does not count as an
        # additional incomparable Pareto record.
        incomparable_accepted = any(
            old.active
            and not old.resources.weakly_dominates(resources)
            and not resources.weakly_dominates(old.resources)
            for old in active_records
        )
        result = InsertResult(
            accepted=True,
            record=record,
            dominated=dominated,
            semantic_key_existed=semantic_key_existed,
            pareto_incomparable_accepted=incomparable_accepted,
            previously_expanded=previously_expanded,
        )
        self.last_archive_time_ns = (
            time.perf_counter_ns() - insert_started
        )
        return result

    def tombstone(self, record: ArchiveRecord) -> None:
        """Retire a strictly dominated archive record without deleting it."""
        if not record.active:
            return
        record.active = False
        record.queued = False
        record.tombstoned = True
        self._active_record_count -= 1
        self._active_width_by_key[record.key] -= 1

    def mark_expanded(self, record: ArchiveRecord) -> bool:
        """Remove a selected record from the open frontier, not the archive."""
        if not record.active or record.tombstoned:
            return False
        record.expanded = True
        record.queued = False
        record.node.expanded = True
        return True

    def record(self, record_id: int) -> Optional[ArchiveRecord]:
        return self._records_by_id.get(record_id)

    def records_for(
        self,
        key: CanonicalKey,
        *,
        active_only: bool = False,
    ) -> Tuple[ArchiveRecord, ...]:
        records = tuple(self._records_by_key.get(key, ()))
        if active_only:
            return tuple(record for record in records if record.active)
        return records

    def all_records(self) -> Tuple[ArchiveRecord, ...]:
        """Return records in stable allocation order, including tombstones."""
        return tuple(
            self._records_by_id[record_id]
            for record_id in sorted(self._records_by_id)
        )

    @property
    def archive_size(self) -> int:
        """Number of distinct semantic identities ever admitted."""

        return len(self._records_by_key)

    @property
    def archive_record_count(self) -> int:
        """Number of admitted records, including expanded/tombstoned history."""

        return len(self._records_by_id)

    @property
    def active_record_count(self) -> int:
        """Number of non-tombstoned resource witnesses in the archive."""

        return self._active_record_count

    @property
    def active_record_peak(self) -> int:
        return self._active_record_peak

    @property
    def pareto_width_peak(self) -> int:
        """Largest active antichain width observed at any semantic key."""

        return self._pareto_width_peak

    def pareto_width(self, key: CanonicalKey) -> int:
        """Current active Pareto-antichain width for ``key``."""

        return self._active_width_by_key.get(key, 0)
