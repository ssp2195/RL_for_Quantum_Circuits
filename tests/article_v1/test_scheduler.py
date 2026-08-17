from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evaluate import select_article_target_distance
from certification.unitary_phase_metrics import projective_unitary_metrics
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType
from rl.article_features import ArticleTargetContext


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


def test_direct_scheduler_uses_article_context_shared_metric(monkeypatch):
    calls = 0

    def recording_metric(candidate, target):
        nonlocal calls
        calls += 1
        return projective_unitary_metrics(candidate, target)

    monkeypatch.setattr("rl.article_features.projective_unitary_metrics", recording_metric)
    budget = ResourceBudget(4, 4, 4, 4)
    root = CircuitState(CircuitDAG(1), budget)
    t_state = CircuitState(
        CircuitDAG.from_gates(1, (Gate(GateType.T, (0,)),)), budget
    )
    nodes = [_Node(2, t_state), _Node(1, root)]
    context = ArticleTargetContext(np.eye(2, dtype=np.complex128))

    assert select_article_target_distance(nodes, context).record_id == 1
    assert calls == 2
