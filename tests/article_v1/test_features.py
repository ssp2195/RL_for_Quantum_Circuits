from __future__ import annotations

from math import sqrt

import numpy as np
import pytest

from certification.simulator import SynthesisTarget, unitary_from_gates
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType
from rl.article_features import (
    ARTICLE_FEATURE_SCHEMA_VERSION,
    ARTICLE_V1_COORDINATE_NAMES,
    ARTICLE_V1_FEATURE_NAMES,
    ARTICLE_V1_FEATURE_SCHEMA_VERSION,
    ARTICLE_V1_NO_TARGET_FEATURE_NAMES,
    ARTICLE_V1_NO_TARGET_FEATURE_SCHEMA_VERSION,
    ARTICLE_V1_NO_Z_FEATURE_NAMES,
    ARTICLE_V1_NO_Z_FEATURE_SCHEMA_VERSION,
    ARTICLE_V1_STANDARDIZATION_ETA,
    EXTENDED_ARTICLE_FEATURE_SCHEMA_VERSION,
    ArticleFeatureProvider,
    ArticleTargetContext,
    ArticleV1FeatureProvider,
    ArticleV1NoTargetFeatureProvider,
    ArticleV1NoZFeatureProvider,
    ExtendedArticleFeatureProvider,
    FrontierFeatureSnapshot,
)
from search.node import SearchNode


BUDGET = ResourceBudget(
    max_t_count=4,
    max_two_qubit_count=4,
    max_gates=8,
    max_depth=8,
)


def state(*gates: Gate, budget: ResourceBudget = BUDGET, num_qubits: int = 2):
    result = CircuitState(CircuitDAG(num_qubits), budget)
    for gate in gates:
        assert result.apply_gate(gate)
    return result


def node(record_id: int, value: CircuitState) -> SearchNode:
    return SearchNode(priority=0.0, state=value, record_id=record_id)


def witness_key(value: CircuitState):
    return tuple((gate.gate_type.name, gate.qubits) for gate in value.dag.gates)


def identity_context(num_qubits: int = 2) -> ArticleTargetContext:
    dimension = 2**num_qubits
    return ArticleTargetContext(np.eye(dimension, dtype=np.complex128))


def candidate_map(snapshot: FrontierFeatureSnapshot, record_id: int):
    return dict(
        zip(ARTICLE_V1_COORDINATE_NAMES, snapshot.candidate_for_record(record_id))
    )


def test_article_v1_schema_is_exactly_31d_and_excludes_extended_coordinates():
    provider = ArticleV1FeatureProvider(identity_context())

    assert provider.schema_version == ARTICLE_V1_FEATURE_SCHEMA_VERSION
    assert provider.dimension == 31
    assert provider.names == ARTICLE_V1_FEATURE_NAMES
    assert provider.names[0] == "bias"
    assert provider.names[1:11] == tuple(
        f"x.{name}" for name in ARTICLE_V1_COORDINATE_NAMES
    )
    assert not any("ancilla" in name or "gate_block" in name for name in provider.names)
    assert provider.metadata()["dtype"] == "float64"
    assert provider.metadata()["standardization_eta"] == ARTICLE_V1_STANDARDIZATION_ETA


def test_root_vector_matches_hand_calculation_and_has_live_root_budget():
    root = state()
    record = node(41, root)
    provider = ArticleV1FeatureProvider(
        identity_context(),
        semantic_key=witness_key,
        search_horizon=8,
        generation_counts={witness_key(root): 4},
    )

    snapshot = provider.build_snapshot([record])
    expected_x = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5], dtype=float)
    expected = np.concatenate(([1.0], expected_x, np.zeros(10), expected_x))

    assert snapshot.record_ids == (41,)
    assert snapshot.expansions_completed == 0
    assert snapshot.expansion_budget == 8
    assert snapshot.features_for_node(record).dtype == np.float64
    assert np.array_equal(snapshot.features_for_record(41), expected)
    assert not snapshot.features_for_record(41).flags.writeable


@pytest.mark.parametrize(
    ("gates", "expected"),
    [
        (
            (Gate(GateType.H, (0,)),),
            {
                "t_count": 0.0,
                "two_qubit_count": 0.0,
                "gate_count": 1 / 8,
                "depth": 1 / 8,
                "rotation_count": 0.0,
                "anticommuting_pair_count": 0.0,
                "mean_pauli_weight": 0.0,
                "target_process_infidelity": 1.0,
            },
        ),
        (
            (Gate(GateType.T, (0,)),),
            {
                "t_count": 1 / 4,
                "two_qubit_count": 0.0,
                "gate_count": 1 / 8,
                "depth": 1 / 8,
                "rotation_count": 1 / 4,
                "anticommuting_pair_count": 0.0,
                "mean_pauli_weight": 1 / 2,
                "target_process_infidelity": (2 - sqrt(2)) / 4,
            },
        ),
        (
            (Gate(GateType.CNOT, (0, 1)),),
            {
                "t_count": 0.0,
                "two_qubit_count": 1 / 4,
                "gate_count": 1 / 8,
                "depth": 1 / 8,
                "rotation_count": 0.0,
                "anticommuting_pair_count": 0.0,
                "mean_pauli_weight": 0.0,
                "target_process_infidelity": 3 / 4,
            },
        ),
    ],
)
def test_one_gate_candidate_coordinates_match_hand_calculations(gates, expected):
    value = state(*gates)
    record = node(7, value)
    provider = ArticleV1FeatureProvider(
        identity_context(), semantic_key=witness_key, search_horizon=8
    )

    actual = candidate_map(provider.build_snapshot([record]), 7)

    for name, expected_value in expected.items():
        assert actual[name] == pytest.approx(expected_value, abs=1e-12)
    assert actual["frontier_resource_dominance_fraction"] == 0.0
    assert actual["archive_novelty"] == 1.0


def test_mixed_prefix_has_exact_anticommuting_pair_and_pauli_normalization():
    # T, H, T transports the new T axis to X while retaining a Z rotation.
    value = state(
        Gate(GateType.T, (0,)),
        Gate(GateType.H, (0,)),
        Gate(GateType.T, (0,)),
    )
    assert len(value.rotations) == 2
    assert not value.rotations[0].axis.commutes_with(value.rotations[1].axis)
    provider = ArticleV1FeatureProvider(
        identity_context(), semantic_key=witness_key, search_horizon=8
    )

    actual = candidate_map(provider.build_snapshot([node(3, value)]), 3)

    assert actual["t_count"] == 2 / 4
    assert actual["gate_count"] == 3 / 8
    assert actual["depth"] == 3 / 8
    assert actual["rotation_count"] == 2 / 4
    assert actual["anticommuting_pair_count"] == 1 / 6
    assert actual["mean_pauli_weight"] == 1 / 2


def test_zero_resource_budgets_are_safe_and_empty_rotation_statistics_are_zero():
    zero = ResourceBudget(
        max_t_count=0,
        max_two_qubit_count=0,
        max_gates=0,
        max_depth=0,
    )
    root = state(budget=zero)
    provider = ArticleV1FeatureProvider(
        identity_context(), semantic_key=witness_key, search_horizon=1
    )

    values = candidate_map(provider.build_snapshot([node(1, root)]), 1)

    assert all(np.isfinite(tuple(values.values())))
    assert values["rotation_count"] == 0.0
    assert values["anticommuting_pair_count"] == 0.0
    assert values["mean_pauli_weight"] == 0.0


def test_resource_position_uses_weak_dominance_of_other_complete_records():
    root = state()
    h = state(Gate(GateType.H, (0,)))
    t = state(Gate(GateType.T, (0,)))
    records = [node(10, root), node(20, h), node(30, t)]
    provider = ArticleV1FeatureProvider(
        identity_context(), semantic_key=witness_key, search_horizon=8
    )

    snapshot = provider.build_snapshot(records)

    assert candidate_map(snapshot, 10)["frontier_resource_dominance_fraction"] == 0.0
    assert candidate_map(snapshot, 20)["frontier_resource_dominance_fraction"] == 0.5
    assert candidate_map(snapshot, 30)["frontier_resource_dominance_fraction"] == 1.0


def test_novelty_uses_supplied_generation_counts_not_record_or_policy_visits():
    root = state()
    key = witness_key(root)
    provider = ArticleV1FeatureProvider(
        identity_context(), semantic_key=witness_key, search_horizon=4
    )

    first = provider.build_snapshot(
        [node(5, root)], archive_generation_counts={key: 1}
    )
    repeated = provider.build_snapshot(
        [node(99, root.copy())], archive_generation_counts={key: 9}
    )

    assert candidate_map(first, 5)["archive_novelty"] == 1.0
    assert candidate_map(repeated, 99)["archive_novelty"] == 1 / 3


def test_frontier_permutation_only_permutes_rows_and_snapshot_rows_are_frozen():
    records = [
        node(8, state()),
        node(2, state(Gate(GateType.H, (0,)))),
        node(5, state(Gate(GateType.T, (0,)))),
    ]
    counts = {witness_key(record.state): index + 1 for index, record in enumerate(records)}
    provider = ArticleV1FeatureProvider(
        identity_context(), semantic_key=witness_key, search_horizon=10
    )

    forward = provider.build_batch(
        records,
        archive_generation_counts=counts,
        expansions_completed=3,
        expansion_budget=10,
    )
    reverse = provider.build_snapshot(
        list(reversed(records)),
        archive_generation_counts=counts,
        expansions_completed=3,
        expansion_budget=10,
    )

    assert forward.snapshot_id == reverse.snapshot_id
    for record in records:
        assert np.array_equal(
            forward.features_for_node(record), reverse.features_for_node(record)
        )
    with pytest.raises(ValueError):
        forward.features_for_node(records[0])[0] = 2.0


def test_budget_interaction_uses_live_external_expansion_horizon():
    record = node(3, state(Gate(GateType.T, (0,))))
    provider = ArticleV1FeatureProvider(
        identity_context(), semantic_key=witness_key, search_horizon=8
    )

    snapshot = provider.build_snapshot([record], expansions_completed=2)
    features = snapshot.features_for_node(record)

    assert np.allclose(features[21:31], 0.75 * features[1:11])
    provider.set_search_step(8)
    exhausted = provider.build_snapshot([record]).features_for_node(record)
    assert np.array_equal(exhausted[21:31], np.zeros(10))


def test_record_id_and_frontier_position_are_not_feature_coordinates():
    value = state(Gate(GateType.H, (0,)))
    provider = ArticleV1FeatureProvider(
        identity_context(), semantic_key=witness_key, search_horizon=4
    )

    first = provider.build_snapshot([node(1, value)])
    second = provider.build_snapshot([node(987, value.copy())])

    assert np.array_equal(first.features_for_record(1), second.features_for_record(987))


def test_frozen_pretransition_feature_does_not_change_when_live_state_mutates():
    live = state()
    record = node(4, live)
    provider = ArticleV1FeatureProvider(
        identity_context(), semantic_key=witness_key, search_horizon=4
    )
    snapshot = provider.build_snapshot([record])
    before = snapshot.features_for_record(4).copy()

    assert live.apply_gate(Gate(GateType.H, (0,)))

    assert np.array_equal(snapshot.features_for_record(4), before)


def test_article_ablations_have_distinct_exact_schemas_and_dimensions():
    no_target = ArticleV1NoTargetFeatureProvider(
        semantic_key=witness_key, search_horizon=4
    )
    no_z = ArticleV1NoZFeatureProvider(
        identity_context(), semantic_key=witness_key, search_horizon=4
    )

    assert no_target.schema_version == ARTICLE_V1_NO_TARGET_FEATURE_SCHEMA_VERSION
    assert no_target.dimension == 28
    assert no_target.names == ARTICLE_V1_NO_TARGET_FEATURE_NAMES
    assert not any("target_process_infidelity" in name for name in no_target.names)
    assert no_z.schema_version == ARTICLE_V1_NO_Z_FEATURE_SCHEMA_VERSION
    assert no_z.dimension == 21
    assert no_z.names == ARTICLE_V1_NO_Z_FEATURE_NAMES
    assert not any(name.startswith("z.") for name in no_z.names)
    assert no_target.extract(state()).dtype == np.float64
    assert no_z.extract(state()).dtype == np.float64


def test_extended_37d_provider_and_legacy_schema_remain_explicitly_available():
    extended = ExtendedArticleFeatureProvider()
    legacy = ArticleFeatureProvider()

    assert extended.dimension == legacy.dimension == 37
    assert extended.schema_version == EXTENDED_ARTICLE_FEATURE_SCHEMA_VERSION
    assert legacy.schema_version == ARTICLE_FEATURE_SCHEMA_VERSION
    assert extended.extract(state()).dtype == np.float32
    assert legacy.extract(state()).dtype == np.float32
