from __future__ import annotations

from dataclasses import dataclass

from evaluate import select_article_target_distance


@dataclass
class _Node:
    record_id: int
    state: str


class _Metric:
    def __init__(self, distances):
        self.distances = distances

    def distance(self, state):
        return self.distances[state]


def test_article_target_distance_selects_minimum_and_stable_id_tie():
    nodes = [_Node(9, "far"), _Node(7, "near-a"), _Node(3, "near-b")]
    metric = _Metric({"far": 0.8, "near-a": 0.2, "near-b": 0.2})
    assert select_article_target_distance(nodes, metric).record_id == 3


def test_article_target_distance_is_invariant_to_frontier_storage_order():
    nodes = [_Node(2, "a"), _Node(1, "b"), _Node(3, "c")]
    metric = _Metric({"a": 0.4, "b": 0.4, "c": 0.6})
    assert select_article_target_distance(nodes, metric).record_id == 1
    assert select_article_target_distance(tuple(reversed(nodes)), metric).record_id == 1
