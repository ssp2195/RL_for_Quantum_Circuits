from __future__ import annotations

import numpy as np
import pytest

from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType
from rl.article_features import (
    ARTICLE_V1_EXACT_INCREMENTAL_EVALUATOR_SCHEMA_VERSION,
    ARTICLE_V1_REFERENCE_EVALUATOR_SCHEMA_VERSION,
    ArticleTargetContext,
    ArticleV1FeatureProvider,
    ArticleV1ReferenceFeatureProvider,
)
from rl.article_frontier_index import ExactArticleFrontierFeatureIndex
from search.archive import ArchiveRecord, ResourceVector
from search.node import SearchNode


BUDGET = ResourceBudget(
    max_t_count=16,
    max_two_qubit_count=16,
    max_gates=32,
    max_depth=32,
)


def make_state(*gates: Gate, num_qubits: int = 2) -> CircuitState:
    result = CircuitState(CircuitDAG(num_qubits), BUDGET)
    for gate in gates:
        assert result.apply_gate(gate)
    return result


def witness_key(value: CircuitState):
    return tuple((gate.gate_type.name, gate.qubits) for gate in value.dag.gates)


def target_context(num_qubits: int = 2) -> ArticleTargetContext:
    return ArticleTargetContext(np.eye(2**num_qubits, dtype=np.complex128))


def archive_record(record_id: int, value: CircuitState) -> ArchiveRecord:
    node = SearchNode(priority=float(record_id % 3), state=value, record_id=record_id)
    return ArchiveRecord(
        record_id=record_id,
        node=node,
        key=witness_key(value),
        resources=ResourceVector.from_state(value),
        queued=True,
    )


def states() -> list[CircuitState]:
    return [
        make_state(),
        make_state(Gate(GateType.H, (0,))),
        make_state(Gate(GateType.H, (1,))),
        make_state(Gate(GateType.T, (0,))),
        make_state(Gate(GateType.S, (0,))),
        make_state(Gate(GateType.CNOT, (0, 1))),
        make_state(Gate(GateType.CNOT, (1, 0))),
        make_state(Gate(GateType.H, (0,)), Gate(GateType.T, (0,))),
    ]


def reference_matrix(
    records: list[ArchiveRecord],
    counts: dict[object, int],
    *,
    completed: int = 3,
    budget: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    provider = ArticleV1ReferenceFeatureProvider(
        target_context(), semantic_key=witness_key, search_horizon=budget
    )
    snapshot = provider.build_snapshot(
        [record.node for record in records],
        archive_generation_counts=counts,
        expansions_completed=completed,
    )
    candidates = np.stack(
        [snapshot.candidate_for_record(record.record_id) for record in records]
    )
    features = np.stack(
        [snapshot.features_for_record(record.record_id) for record in records]
    )
    return candidates, features


@pytest.mark.parametrize("size", [1, 2, 3, 8, 16, 32, 64, 128])
def test_incremental_snapshot_is_exactly_equal_to_all_pairs_oracle(size: int):
    templates = states()
    records = [archive_record(index + 10, templates[index % len(templates)].copy()) for index in range(size)]
    counts: dict[object, int] = {}
    for index, record in enumerate(records):
        counts[record.key] = max(counts.get(record.key, 0), (index % 11) + 1)
    production = ArticleV1FeatureProvider(
        target_context(), semantic_key=witness_key, search_horizon=16
    )

    batch = production.build_compact_batch(
        records,
        archive_generation_counts=counts,
        expansions_completed=3,
    )
    expected_candidates, expected_features = reference_matrix(records, counts)

    assert batch.evaluator_schema_version == (
        ARTICLE_V1_EXACT_INCREMENTAL_EVALUATOR_SCHEMA_VERSION
    )
    assert np.array_equal(batch.record_ids, [record.record_id for record in records])
    assert np.array_equal(batch.candidate_matrix, expected_candidates)
    for row, record in enumerate(records):
        assert np.array_equal(batch.features_for_row(row), expected_features[row])
        assert batch.node_for_record_id(record.record_id) is record.node


def test_exact_resource_groups_handle_frequency_zero_readd_and_batch_mutations():
    values = states()
    records = [archive_record(index, value) for index, value in enumerate(values)]
    counts = {record.key: index + 1 for index, record in enumerate(records)}
    production = ArticleV1FeatureProvider(
        target_context(), semantic_key=witness_key, search_horizon=20,
        debug_reconciliation=True,
    )

    sequences = [
        records[:4],
        records[1:7],
        [records[6], records[2], records[7]],
        [records[0], records[6], records[2], records[7]],
        [records[0]],
        records,
    ]
    for step, active in enumerate(sequences):
        updates = {record.key: counts[record.key] + step for record in active}
        batch = production.build_compact_batch(
            active,
            generation_count_updates=updates,
            expansions_completed=step,
            expansion_budget=20,
        )
        expected_candidates, expected_features = reference_matrix(
            active,
            dict(production.feature_index.generation_counts_snapshot()),
            completed=step,
            budget=20,
        )
        assert np.array_equal(batch.candidate_matrix, expected_candidates)
        assert np.array_equal(batch.materialize_feature_matrix(), expected_features)
        production.reconcile_index(active)

    metrics = production.instrumentation()
    assert metrics["frontier_index_additions"] > len(records)
    assert metrics["frontier_index_removals"] > 0
    assert metrics["resource_group_peak"] >= metrics["unique_resource_group_count"]
    assert metrics["dominance_update_time_ns"] > 0


def test_novelty_changed_key_delta_updates_only_exact_formula():
    root = archive_record(3, make_state())
    other = archive_record(4, make_state(Gate(GateType.H, (0,))))
    provider = ArticleV1FeatureProvider(
        target_context(), semantic_key=witness_key, search_horizon=4
    )
    first = provider.build_compact_batch(
        [root, other], archive_generation_counts={root.key: 1, other.key: 4}
    )
    assert first.candidate_for_record(3)[-1] == 1.0
    assert first.candidate_for_record(4)[-1] == 0.5

    provider.increment_generation_counts({root.key: 8})
    second = provider.build_compact_batch([root, other])
    assert second.candidate_for_record(3)[-1] == pytest.approx(1 / 3)
    assert second.candidate_for_record(4)[-1] == 0.5
    assert second.generation_count_revision > first.generation_count_revision


def test_static_intrinsics_and_target_distance_are_cached_once_per_admission():
    context = target_context()
    records = [archive_record(index, value) for index, value in enumerate(states())]
    provider = ArticleV1FeatureProvider(
        context, semantic_key=witness_key, search_horizon=8
    )

    provider.build_compact_batch(records)
    misses = context.cache_misses
    evaluations = context.evaluation_count
    minimum = provider.minimum_target_distance()
    selected = provider.select_target_distance_node()
    for step in range(1, 5):
        provider.build_compact_batch(records, expansions_completed=step)

    assert context.cache_misses == misses
    assert context.evaluation_count == evaluations
    assert minimum >= 0.0
    assert selected in [record.node for record in records]
    metrics = provider.instrumentation()
    assert metrics["feature_static_cache_misses"] == len(records)
    assert metrics["feature_static_cache_hits"] == len(records) * 4


def test_authoritative_order_only_permutes_rows_without_revision_or_value_change():
    records = [archive_record(index + 1, value) for index, value in enumerate(states())]
    counts = {record.key: index + 1 for index, record in enumerate(records)}
    provider = ArticleV1FeatureProvider(
        target_context(), semantic_key=witness_key, search_horizon=8
    )
    first = provider.build_compact_batch(records, archive_generation_counts=counts)
    reverse = provider.build_compact_batch(list(reversed(records)))

    assert reverse.frontier_revision == first.frontier_revision
    for record in records:
        assert np.array_equal(
            first.candidate_for_record(record.record_id),
            reverse.candidate_for_record(record.record_id),
        )


def test_corruption_reconciliation_fails_without_mutating_authoritative_records():
    records = [archive_record(index, value) for index, value in enumerate(states()[:4])]
    provider = ArticleV1FeatureProvider(
        target_context(), semantic_key=witness_key, search_horizon=8
    )
    provider.build_compact_batch(records)
    index: ExactArticleFrontierFeatureIndex = provider.feature_index
    group = int(index._slot_group[index._slot_by_record_id[0]])
    index._group_dominator_count[group] += 1

    with pytest.raises(AssertionError, match="dominator count drifted"):
        index.reconcile(records)
    assert [record.record_id for record in records] == [0, 1, 2, 3]
    assert all(record.active and not record.tombstoned for record in records)


def test_reference_provider_warns_above_safe_size_and_identifies_schema():
    records = [archive_record(index, states()[index % 8]) for index in range(3)]
    provider = ArticleV1ReferenceFeatureProvider(
        target_context(),
        semantic_key=witness_key,
        reference_safe_frontier_size=2,
    )

    with pytest.warns(RuntimeWarning, match=r"O\(F\^2\)"):
        provider.build_snapshot([record.node for record in records])
    assert provider.evaluator_schema_version == (
        ARTICLE_V1_REFERENCE_EVALUATOR_SCHEMA_VERSION
    )
    assert provider.metadata()["feature_evaluator_schema_version"] == (
        ARTICLE_V1_REFERENCE_EVALUATOR_SCHEMA_VERSION
    )


def test_same_persistent_archive_record_id_cannot_silently_change_resources():
    record = archive_record(11, make_state())
    provider = ArticleV1FeatureProvider(
        target_context(), semantic_key=witness_key, search_horizon=4
    )
    provider.build_compact_batch([record])
    changed = archive_record(11, make_state(Gate(GateType.H, (0,))))

    with pytest.raises(AssertionError, match="resources changed"):
        provider.build_compact_batch([changed])


def test_three_qubit_resource_dimension_and_wire_depth_antichain_are_exact():
    three_budget = ResourceBudget(
        max_t_count=8,
        max_two_qubit_count=8,
        max_gates=16,
        max_depth=16,
    )

    def three_state(*gates: Gate) -> CircuitState:
        value = CircuitState(CircuitDAG(3), three_budget)
        for gate in gates:
            assert value.apply_gate(gate)
        return value

    active = [
        archive_record(10, three_state()),
        archive_record(11, three_state(Gate(GateType.H, (0,)))),
        archive_record(12, three_state(Gate(GateType.H, (1,)))),
        archive_record(13, three_state(Gate(GateType.H, (2,)))),
    ]
    provider = ArticleV1FeatureProvider(
        target_context(3), semantic_key=witness_key, search_horizon=4
    )
    batch = provider.build_compact_batch(active)

    # Root weakly dominates every one-gate record.  The one-gate records form
    # a complete antichain because their per-wire depth coordinates differ.
    assert batch.candidate_for_record(10)[-2] == 0.0
    assert batch.candidate_for_record(11)[-2] == pytest.approx(1 / 3)
    assert batch.candidate_for_record(12)[-2] == pytest.approx(1 / 3)
    assert batch.candidate_for_record(13)[-2] == pytest.approx(1 / 3)
    assert provider.feature_index.static_candidate(11).resource_tuple == (
        0,
        0,
        1,
        1,
        0,
        0,
    )
    provider.reconcile_index(active)
