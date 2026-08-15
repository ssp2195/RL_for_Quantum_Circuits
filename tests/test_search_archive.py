from dataclasses import dataclass

import pytest

from search.archive import ParetoArchive, ResourceVector
from search.frontier import Frontier
from search.node import SearchNode


class StubCanonicalizer:
    """Keep archive tests independent of the quantum semantic layer."""

    def semantic_key(self, state):
        return state.key


class CollidingDiagnosticHashCanonicalizer(StubCanonicalizer):
    def identity_hash(self, state):
        return "forced-collision"


class HashOnlyCanonicalizer:
    def identity_hash(self, state):
        return "forced-collision"


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


def test_archive_requires_a_full_semantic_key_contract():
    with pytest.raises(TypeError, match=r"semantic_key\(state\)"):
        ParetoArchive(canonicalizer=HashOnlyCanonicalizer())


def test_colliding_diagnostic_hashes_cannot_merge_distinct_semantic_keys():
    frontier = Frontier(canonicalizer=CollidingDiagnosticHashCanonicalizer())

    left = make_node(key="left")
    right = make_node(key="right")
    assert frontier.insert(left).accepted
    assert frontier.insert(right).accepted

    assert frontier.archive.archive_size == 2
    assert len(frontier.active_nodes()) == 2


def test_canonicalization_ablation_retains_every_concrete_record():
    frontier = Frontier(
        canonicalizer=StubCanonicalizer(),
        canonicalization_enabled=False,
    )

    assert frontier.insert(make_node()).accepted
    result = frontier.insert(make_node())

    assert result.accepted
    assert not result.semantic_key_existed
    assert not result.pareto_incomparable_accepted
    assert frontier.archive.archive_size == 2
    assert len(frontier.active_nodes()) == 2


def test_pareto_ablation_retains_comparable_records_without_mislabeling_them():
    frontier = Frontier(
        canonicalizer=StubCanonicalizer(),
        pareto_dominance_enabled=False,
    )

    assert frontier.insert(make_node(resources=(2, 1, 3, (2, 2)))).accepted
    result = frontier.insert(make_node(resources=(3, 1, 4, (2, 3))))

    assert result.accepted
    assert result.semantic_key_existed
    assert not result.duplicate_rejected
    assert not result.pareto_incomparable_accepted
    assert result.dominated_retired == 0
    assert frontier.archive.pareto_width("same") == 2

    exact_duplicate = frontier.insert(
        make_node(resources=(3, 1, 4, (2, 3)))
    )
    assert not exact_duplicate.accepted
    assert exact_duplicate.duplicate_rejected
    assert frontier.archive.pareto_width("same") == 2


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
    assert result.semantic_key_existed
    assert result.previously_expanded
    assert result.pareto_incomparable_accepted
    assert result.reopened
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
    assert equal_result.duplicate_rejected
    assert not weak_result.accepted
    assert weak_result.rejected_by is not None
    assert weak_result.duplicate_rejected
    assert frontier.active_nodes() == [best]


def test_dominating_record_tombstones_old_heap_entry_and_pop_skips_it():
    frontier = make_frontier()
    old = make_node(resources=(5, 2, 7, (4, 4)), priority=-10.0)
    assert frontier.push(old)

    replacement = make_node(resources=(4, 2, 6, (3, 4)), priority=10.0)
    result = frontier.insert(replacement)

    assert result.accepted
    assert result.dominated
    assert result.dominated_retired == 1
    assert not result.reopened
    assert not result.pareto_incomparable_accepted
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


def test_reopening_depends_on_prior_expansion_not_dominance_replacement():
    frontier = make_frontier()
    old = make_node(resources=(5, 2, 7, (4, 4)))
    assert frontier.push(old)

    # Replacing a never-expanded record is retirement, not reopening.
    first_replacement = make_node(resources=(4, 2, 6, (3, 4)))
    result = frontier.insert(first_replacement)
    assert result.dominated_retired == 1
    assert not result.previously_expanded
    assert not result.reopened

    assert frontier.remove(first_replacement)
    second_replacement = make_node(resources=(3, 2, 5, (3, 3)))
    result = frontier.insert(second_replacement)
    assert result.dominated_retired == 1
    assert result.previously_expanded
    assert result.reopened


def test_pareto_width_tracks_incomparable_records_at_one_key():
    frontier = make_frontier()
    assert frontier.push(make_node(resources=(4, 2, 5, (3, 2))))
    result = frontier.insert(make_node(resources=(3, 3, 5, (3, 2))))

    assert result.pareto_incomparable_accepted
    assert frontier.archive.pareto_width("same") == 2
    assert frontier.archive.pareto_width_peak == 2


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
