from __future__ import annotations

import numpy as np
import pytest

from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType
from rl.article_features import (
    ARTICLE_V1_STANDARDIZATION_ETA,
    ArticleTargetContext,
    ArticleV1FeatureProvider,
    ArticleV1NoTargetFeatureProvider,
    ArticleV1NoZFeatureProvider,
    ArticleV1ReferenceFeatureProvider,
)
from search.archive import ArchiveRecord, ResourceVector
from search.node import SearchNode


BUDGET = ResourceBudget(
    max_t_count=8,
    max_two_qubit_count=8,
    max_gates=16,
    max_depth=16,
)


def state(*gates: Gate) -> CircuitState:
    result = CircuitState(CircuitDAG(2), BUDGET)
    for gate in gates:
        assert result.apply_gate(gate)
    return result


def witness_key(value: CircuitState):
    return tuple((gate.gate_type.name, gate.qubits) for gate in value.dag.gates)


def context() -> ArticleTargetContext:
    return ArticleTargetContext(np.eye(4, dtype=np.complex128))


def record(record_id: int, value: CircuitState, *, priority: float = 0.0) -> ArchiveRecord:
    node = SearchNode(priority=priority, state=value, record_id=record_id)
    return ArchiveRecord(
        record_id=record_id,
        node=node,
        key=witness_key(value),
        resources=ResourceVector.from_state(value),
        queued=True,
    )


def records() -> list[ArchiveRecord]:
    return [
        record(17, state()),
        record(3, state(Gate(GateType.H, (0,)))),
        record(41, state(Gate(GateType.T, (0,)))),
        record(9, state(Gate(GateType.CNOT, (0, 1)))),
        record(28, state(Gate(GateType.H, (1,)), Gate(GateType.T, (1,)))),
    ]


@pytest.mark.parametrize(
    "provider_factory",
    [
        lambda: ArticleV1FeatureProvider(context(), semantic_key=witness_key, search_horizon=13),
        lambda: ArticleV1NoTargetFeatureProvider(semantic_key=witness_key, search_horizon=13),
        lambda: ArticleV1NoZFeatureProvider(context(), semantic_key=witness_key, search_horizon=13),
    ],
)
def test_compact_effective_weight_scores_equal_explicit_full_feature_dot_products(
    provider_factory,
):
    provider = provider_factory()
    batch = provider.build_compact_batch(
        records(), expansions_completed=5, expansion_budget=13
    )
    rng = np.random.default_rng(1729)

    for _ in range(32):
        theta = rng.normal(size=provider.dimension)
        compact = batch.scores(theta)
        full = batch.full_dot_scores(theta)
        assert np.allclose(compact, full, rtol=0.0, atol=2e-15)
        expected_row = min(
            range(len(batch.records)),
            key=lambda row: (-full[row], int(batch.record_ids[row])),
        )
        assert batch.greedy_row(theta) == expected_row
        assert batch.select_greedy_record_id(theta) == int(
            batch.record_ids[expected_row]
        )
        assert batch.select_greedy(theta) is batch.frontier_nodes[expected_row]


def test_effective_linear_terms_are_the_article_identity():
    provider = ArticleV1FeatureProvider(
        context(), semantic_key=witness_key, search_horizon=10
    )
    batch = provider.build_compact_batch(records(), expansions_completed=4)
    theta = np.linspace(-1.5, 2.5, provider.dimension)
    effective, constant = batch.effective_linear_terms(theta)
    width = 10
    expected_effective = (
        theta[1:11]
        + 0.6 * theta[21:31]
        + theta[11:21] / (batch.std + ARTICLE_V1_STANDARDIZATION_ETA)
    )
    expected_constant = theta[0] - np.dot(
        theta[11:21],
        batch.mean / (batch.std + ARTICLE_V1_STANDARDIZATION_ETA),
    )

    assert np.array_equal(effective, expected_effective)
    assert constant == expected_constant
    assert effective.shape == (width,)


def test_exact_ties_use_lowest_persistent_record_id_not_storage_order():
    provider = ArticleV1FeatureProvider(
        context(), semantic_key=witness_key, search_horizon=4
    )
    forward_records = records()
    forward = provider.build_compact_batch(forward_records)
    zero = np.zeros(provider.dimension, dtype=np.float64)
    assert forward.select_greedy_record_id(zero) == 3
    assert forward.select_greedy(zero).record_id == 3

    reverse = provider.build_compact_batch(list(reversed(forward_records)))
    assert reverse.select_greedy_record_id(zero) == 3
    assert reverse.select_greedy(zero).record_id == 3


def test_selected_row_materialization_matches_reference_31d_exactly():
    active = records()
    counts = {item.key: index + 1 for index, item in enumerate(active)}
    production = ArticleV1FeatureProvider(
        context(), semantic_key=witness_key, search_horizon=9
    )
    reference = ArticleV1ReferenceFeatureProvider(
        context(), semantic_key=witness_key, search_horizon=9
    )
    compact = production.build_compact_batch(
        active,
        archive_generation_counts=counts,
        expansions_completed=2,
    )
    oracle = reference.build_snapshot(
        [item.node for item in active],
        archive_generation_counts=counts,
        expansions_completed=2,
    )

    assert compact.feature_dimension == 31
    for item in active:
        assert np.array_equal(
            compact.features_for_record(item.record_id),
            oracle.features_for_record(item.record_id),
        )
    with pytest.raises(ValueError):
        compact.features_for_record(active[0].record_id)[0] = 2.0


def test_compact_batch_arrays_are_frozen_and_invalid_weights_fail_closed():
    provider = ArticleV1FeatureProvider(
        context(), semantic_key=witness_key, search_horizon=4
    )
    batch = provider.build_compact_batch(records())

    assert not batch.record_ids.flags.writeable
    assert not batch.candidate_matrix.flags.writeable
    assert not batch.mean.flags.writeable
    assert not batch.std.flags.writeable
    with pytest.raises(ValueError, match="theta shape"):
        batch.scores(np.zeros(provider.dimension - 1))
    invalid = np.zeros(provider.dimension)
    invalid[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        batch.scores(invalid)


def test_compact_batch_keeps_authoritative_records_and_search_nodes():
    active = records()
    provider = ArticleV1FeatureProvider(
        context(), semantic_key=witness_key, search_horizon=4
    )
    batch = provider.build_compact_batch(active)

    assert batch.records == tuple(active)
    assert batch.frontier_nodes == tuple(item.node for item in active)
    for item in active:
        assert batch.record_for_record_id(item.record_id) is item
        assert batch.node_for_record_id(item.record_id) is item.node


def test_score_and_selected_row_instrumentation_are_separate():
    provider = ArticleV1FeatureProvider(
        context(), semantic_key=witness_key, search_horizon=4
    )
    batch = provider.build_compact_batch(records())
    before = dict(provider.instrumentation())
    batch.scores(np.ones(provider.dimension))
    batch.features_for_record(3)
    after = provider.instrumentation()

    assert after["score_time_ns"] > before["score_time_ns"]
    assert (
        after["selected_row_materialization_time_ns"]
        > before["selected_row_materialization_time_ns"]
    )
    assert after["feature_index_memory_bytes"] > 0
