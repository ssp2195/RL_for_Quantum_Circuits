from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ckt_types import ResourceBudget
from config import Config
from env.rl_env import CircuitSynthesisEnv
from search.node import SearchNode


@dataclass
class MetricState:
    key: str
    t_count: int
    two_qubit_count: int
    num_gates: int
    wire_depths: tuple[int, ...]
    depth: int


class MetricCanonicalizer:
    def semantic_key(self, state: MetricState):
        return state.key


class ConstantFeatureProvider:
    dimension = 1

    def extract(self, state, frontier):
        return np.zeros(1, dtype=np.float32)


class NeverCertifier:
    def certify(self, state):  # pragma: no cover - structural prefixes skip it
        raise AssertionError("nonterminal prefixes must not invoke certification")


class MetricScenarioProblem:
    """One deterministic expansion containing every Stage-1 archive event."""

    name = "metric-scenario"
    schema_version = "metric-scenario-v1"
    reject_failed_terminal = False

    def initial_state(self, config):
        return MetricState("root", 4, 2, 5, (3, 2), 3)

    def analyze(self, state):
        return ()

    def canonicalizer(self, *, phase_sensitive=False):
        return MetricCanonicalizer()

    def is_terminal_candidate(self, node):
        return False

    def metadata(self):
        return {"name": self.name, "schema_version": self.schema_version}

    def expand(self, node):
        if node.parent is not None:
            return []
        states = (
            # Incomparable with the already-expanded root: accepted reopening.
            MetricState("root", 3, 3, 5, (3, 2), 3),
            # Weakly dominated at root: duplicate rejection.
            MetricState("root", 5, 2, 6, (4, 2), 4),
            # New key, followed by a strict replacement before expansion.
            MetricState("branch", 5, 2, 7, (4, 4), 4),
            MetricState("branch", 4, 2, 6, (3, 4), 4),
            # Incomparable with the replacement, but branch was never expanded.
            MetricState("branch", 3, 3, 6, (3, 3), 3),
        )
        return [
            SearchNode(priority=float(index), state=state, parent=node, action=index)
            for index, state in enumerate(states)
        ]


def test_search_metrics_have_exact_event_and_size_definitions():
    config = Config(
        num_qubits=2,
        budget=ResourceBudget(10, 10, 10, 10),
        max_steps=1,
        max_frontier=8,
    )
    env = CircuitSynthesisEnv(
        config,
        NeverCertifier(),
        problem=MetricScenarioProblem(),
        feature_provider=ConstantFeatureProvider(),
    )
    env.reset(seed=1)

    _, _, terminated, truncated, info = env.step(0)

    assert not terminated
    assert truncated
    assert info["num_children"] == 5
    assert info["num_certification_nonmatches"] == 5
    assert info["frontier_size"] == 3
    expected_legacy_metrics = {
        "generated": 5,
        "certification_nonmatch": 5,
        "duplicate_rejected": 1,
        "dominated_retired": 1,
        "pareto_incomparable_accepted": 2,
        "reopened": 1,
        "expanded": 1,
        "frontier_peak": 3,
        "frontier_mean": 2.0,
        "archive_size": 2,
        "pareto_width_peak": 2,
        "accepted": 5,
        "canonical_pruned": 1,
        "dominated": 1,
        "peak_frontier": 3,
        "terminal_candidates": 0,
        "terminal_certification_failures": 0,
    }
    metrics = info["search_metrics"]
    assert {name: metrics[name] for name in expected_legacy_metrics} == expected_legacy_metrics
    assert metrics["num_expanded"] == 1
    assert metrics["num_generated"] == 5
    # The rejected root record has strictly worse resources, so the precise
    # Article V1 taxonomy calls it dominance rather than exact equality.
    assert metrics["num_exact_duplicate_rejections"] == 0
    assert metrics["num_dominance_rejections"] == 1
    assert metrics["num_dominance_replacements"] == 1
    assert metrics["num_pareto_incomparable_acceptances"] == 2
    assert metrics["num_reopenings"] == 1
    assert metrics["archive_record_count"] == 5
    assert metrics["maximum_pareto_antichain_width"] == 2


def test_invalid_positional_action_does_not_count_as_an_expansion_sample():
    config = Config(
        num_qubits=2,
        budget=ResourceBudget(10, 10, 10, 10),
        max_steps=2,
        max_frontier=8,
    )
    env = CircuitSynthesisEnv(
        config,
        NeverCertifier(),
        problem=MetricScenarioProblem(),
        feature_provider=ConstantFeatureProvider(),
    )
    env.reset(seed=1)

    _, _, _, _, info = env.step(7)

    assert info["invalid_action"]
    assert env.search_metrics["expanded"] == 0
    assert env.search_metrics["frontier_mean"] == 1.0
    assert env.search_metrics["frontier_peak"] == 1
