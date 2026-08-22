import random
from dataclasses import dataclass

from search.archive import ParetoArchive
from search.frontier import Frontier
from search.node import SearchNode


class StubCanonicalizer:
    def semantic_key(self, state):
        return state.key


@dataclass
class StubState:
    key: str
    t_count: int
    two_qubit_count: int
    num_gates: int
    wire_depths: tuple[int, ...]


def make_node(*, key="same", resources=(4, 2, 5, (3, 2)), priority=0.0):
    t_count, two_qubit_count, num_gates, wire_depths = resources
    return SearchNode(
        priority=priority,
        state=StubState(
            key,
            t_count,
            two_qubit_count,
            num_gates,
            tuple(wire_depths),
        ),
    )


def test_by_id_view_has_same_open_set_without_priority_sorting():
    frontier = Frontier(canonicalizer=StubCanonicalizer())
    nodes = [
        make_node(key="a", priority=3.0),
        make_node(key="b", priority=1.0),
        make_node(key="c", priority=2.0),
    ]
    for node in nodes:
        assert frontier.push(node)

    assert {r.record_id for r in frontier.active_records()} == {
        r.record_id for r in frontier.active_records_by_id()
    }
    assert [r.record_id for r in frontier.active_records_by_id()] == sorted(
        r.record_id for r in frontier.active_records_by_id()
    )
    assert frontier.active_nodes_by_id() == nodes


def test_incremental_archive_counts_match_slow_oracle_randomized():
    rng = random.Random(20260821)
    archive = ParetoArchive(canonicalizer=StubCanonicalizer())
    keys = tuple(f"k{i}" for i in range(5))

    for _ in range(250):
        values = tuple(rng.randrange(0, 8) for _ in range(5))
        node = make_node(
            key=rng.choice(keys),
            resources=(values[0], values[1], values[2], values[3:]),
        )
        prepared = archive.prepare(node)
        archive.insert_prepared(node, prepared, debug_recompute=True)

        slow_active = [record for record in archive.all_records() if record.active]
        assert archive.active_record_count == len(slow_active)
        for key in keys:
            assert archive.pareto_width(key) == sum(
                record.active for record in archive.records_for(key)
            )


def test_prepared_insert_matches_normal_insert_outcome():
    left = ParetoArchive(canonicalizer=StubCanonicalizer())
    right = ParetoArchive(canonicalizer=StubCanonicalizer())
    first_left = make_node(resources=(4, 2, 5, (3, 2)))
    first_right = make_node(resources=(4, 2, 5, (3, 2)))
    second_left = make_node(resources=(3, 3, 5, (3, 2)))
    second_right = make_node(resources=(3, 3, 5, (3, 2)))

    assert left.insert(first_left).accepted
    assert right.insert_prepared(first_right, right.prepare(first_right)).accepted
    normal = left.insert(second_left)
    prepared = right.insert_prepared(second_right, right.prepare(second_right))

    assert normal.accepted == prepared.accepted
    assert normal.pareto_incomparable_accepted == prepared.pareto_incomparable_accepted
    assert left.active_record_count == right.active_record_count
    assert left.pareto_width("same") == right.pareto_width("same")
