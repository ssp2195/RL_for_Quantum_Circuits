from dataclasses import dataclass

import pytest

from search.archive import ResourceVector
from search.frontier import Frontier
from search.node import SearchNode


class StubCanonicalizer:
    """Keep archive tests independent of the quantum semantic layer."""

    def semantic_key(self, state):
        return state.key


@dataclass
class StubState:
    key: str
    t_count: int
    two_qubit_count: int
    num_gates: int
    wire_depths: tuple[int, ...]


def make_node(
    *,
    key="same",
    resources=(4, 2, 5, (3, 2)),
    priority=0.0,
):
    t_count, two_qubit_count, num_gates, wire_depths = resources
    return SearchNode(
        priority=priority,
        state=StubState(
            key=key,
            t_count=t_count,
            two_qubit_count=two_qubit_count,
            num_gates=num_gates,
            wire_depths=wire_depths,
        ),
    )


def make_frontier():
    return Frontier(canonicalizer=StubCanonicalizer())


def test_resource_vector_uses_componentwise_wire_depth_dominance():
    better = ResourceVector(2, 1, 4, (2, 3))
    worse = ResourceVector(2, 1, 4, (2, 4))

    assert better.weakly_dominates(worse)
    assert better.strictly_dominates(worse)
    assert not worse.weakly_dominates(better)

    with pytest.raises(ValueError, match="different numbers of wires"):
        better.weakly_dominates(ResourceVector(2, 1, 4, (2,)))


def test_reopens_same_semantic_key_with_incomparable_resources():
    frontier = make_frontier()
    first = make_node(resources=(4, 2, 5, (3, 2)), priority=3.0)

    assert frontier.push(first)
    assert frontier.remove(first)
    first_record = frontier.archive.record(first.record_id)
    assert first_record is not None
    assert first_record.active
    assert first_record.expanded

    # Better T count but worse two-qubit count: neither record dominates.
    reopened = make_node(resources=(3, 3, 5, (3, 2)), priority=1.0)
    result = frontier.insert(reopened)

    assert result.accepted
    assert result.record is not None
    assert not result.record.expanded
    assert result.record.active
    assert frontier.contains(result.record.record_id)
    assert len(frontier.archive.records_for("same")) == 2
    assert frontier.active_nodes() == [reopened]


def test_weakly_dominated_and_equal_records_are_rejected():
    frontier = make_frontier()
    best = make_node(resources=(2, 1, 3, (2, 2)))
    assert frontier.push(best)

    equal = make_node(resources=(2, 1, 3, (2, 2)))
    weaker = make_node(resources=(3, 1, 4, (2, 3)))

    equal_result = frontier.insert(equal)
    weak_result = frontier.insert(weaker)

    assert not equal_result.accepted
    assert equal_result.rejected_by is not None
    assert not weak_result.accepted
    assert weak_result.rejected_by is not None
    assert frontier.active_nodes() == [best]


def test_dominating_record_tombstones_old_heap_entry_and_pop_skips_it():
    frontier = make_frontier()
    old = make_node(resources=(5, 2, 7, (4, 4)), priority=-10.0)
    assert frontier.push(old)

    replacement = make_node(resources=(4, 2, 6, (3, 4)), priority=10.0)
    result = frontier.insert(replacement)

    assert result.accepted
    assert result.dominated
    old_record = frontier.archive.record(old.record_id)
    assert old_record is not None
    assert old_record.tombstoned
    assert not old_record.active
    assert frontier.active_nodes() == [replacement]

    # The old record has the better raw heap priority but is stale; pop must
    # lazily discard it and expand the still-active replacement.
    assert frontier.pop() is replacement
    assert replacement.expanded
    assert frontier.is_empty()


def test_active_nodes_and_heap_snapshot_have_stable_priority_record_order():
    frontier = make_frontier()
    later_tie = make_node(key="a", priority=2.0)
    first_tie = make_node(key="b", priority=2.0)
    first = make_node(key="c", priority=1.0)

    assert frontier.push(later_tie)
    assert frontier.push(first_tie)
    assert frontier.push(first)

    expected = [first, later_tie, first_tie]
    assert frontier.nodes() == expected
    assert frontier.active_nodes() == expected
    assert frontier.heap == expected


def test_remove_marks_only_selected_record_expanded_and_keeps_archive_active():
    frontier = make_frontier()
    node = make_node()
    assert frontier.push(node)

    assert frontier.remove(node)
    record = frontier.archive.record(node.record_id)
    assert record is not None
    assert record.active
    assert record.expanded
    assert not record.queued
    assert not record.tombstoned
    assert not frontier.contains(node)
