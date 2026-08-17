from __future__ import annotations

from math import pi, sqrt

import numpy as np
import pytest

from certification.simulator import SimulatorCertificationEngine, SynthesisTarget
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType
from rl.article_features import (
    ARTICLE_V1_TARGET_METRIC_SCHEMA_VERSION,
    ArticleTargetContext,
    process_infidelity,
)
from certification.unitary_phase_metrics import projective_unitary_metrics
from search.node import SearchNode


def state(num_qubits: int, *gates: Gate) -> CircuitState:
    result = CircuitState(
        CircuitDAG(num_qubits),
        ResourceBudget(8, 16, 16, 8),
    )
    for gate in gates:
        assert result.apply_gate(gate)
    return result


def test_process_infidelity_is_global_phase_invariant():
    identity = np.eye(2, dtype=np.complex128)

    assert process_infidelity(identity, np.exp(0.731j) * identity) == pytest.approx(
        0.0, abs=1e-12
    )


def test_process_infidelity_matches_hand_values_for_h_and_t():
    identity = np.eye(2, dtype=np.complex128)
    h = np.asarray([[1, 1], [1, -1]], dtype=np.complex128) / sqrt(2)
    t = np.diag([1.0, np.exp(1j * pi / 4)]).astype(np.complex128)

    assert process_infidelity(identity, h) == pytest.approx(1.0, abs=1e-12)
    assert process_infidelity(identity, t) == pytest.approx(
        (2 - sqrt(2)) / 4, abs=1e-12
    )


def test_process_infidelity_validates_shape_finiteness_and_roundoff():
    identity = np.eye(2, dtype=np.complex128)

    with pytest.raises(ValueError, match="shape does not match"):
        process_infidelity(identity, np.eye(4))
    with pytest.raises(ValueError, match="finite"):
        process_infidelity(identity, np.asarray([[np.nan, 0], [0, 1]]))
    with pytest.raises(ValueError, match="finite"):
        ArticleTargetContext(np.asarray([[1, 0], [0, np.inf]]))

    # Tiny nonunitary scaling exercises only the specified roundoff clamp.
    almost_identity = (1.0 + 1e-12) * identity
    assert process_infidelity(identity, almost_identity) == 0.0
    with pytest.raises(ValueError, match="candidate must be unitary"):
        process_infidelity(identity, 2.0 * identity)


def test_context_reconstructs_from_full_dag_and_counts_cache_events():
    context = ArticleTargetContext(np.eye(2, dtype=np.complex128))
    root_a = state(1)
    root_b = state(1)
    h = state(1, Gate(GateType.H, (0,)))

    assert context.distance(root_a) == 0.0
    # A different state/record object with the same complete DAG is one cache hit.
    assert context.process_infidelity(root_b) == 0.0
    assert context.distance(h) == pytest.approx(1.0, abs=1e-12)

    assert context.schema_version == ARTICLE_V1_TARGET_METRIC_SCHEMA_VERSION
    assert context.evaluation_count == 2
    assert context.cache_misses == 2
    assert context.cache_hits == 1
    assert context.cache_size == 2
    assert context.metric_time_ns > 0
    assert context.cache_metrics() == {
        "target_metric_schema_version": "projective-unitary-metrics-v2",
        "target_metric_evaluation_count": 2,
        "target_metric_cache_hits": 1,
        "target_metric_cache_misses": 2,
        "target_metric_cache_size": 2,
        "target_metric_time_ns": context.metric_time_ns,
    }


def test_cache_identity_never_uses_frontier_record_id():
    context = ArticleTargetContext(np.eye(2, dtype=np.complex128))
    first = SearchNode(0.0, state(1), record_id=1)
    second = SearchNode(0.0, state(1), record_id=999)

    assert context.cache_key(first.state) == context.cache_key(second.state)
    context.distance(first.state)
    context.distance(second.state)
    assert context.evaluation_count == 1
    assert context.cache_hits == 1


def test_context_rejects_candidate_register_dimension_mismatch():
    context = ArticleTargetContext(np.eye(2, dtype=np.complex128))

    with pytest.raises(ValueError, match="width does not match"):
        context.distance(state(2))


def test_context_can_be_constructed_from_the_certification_engine_target():
    target = SynthesisTarget(np.eye(2, dtype=np.complex128))
    engine = SimulatorCertificationEngine(target)

    context = ArticleTargetContext.from_certification_engine(engine)

    assert context.fingerprint.startswith("sha256:")
    assert context.distance(state(1)) == 0.0


def test_article_d_tar_calls_shared_projective_metric(monkeypatch) -> None:
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def recording_metric(candidate, target):
        calls.append((np.asarray(candidate), np.asarray(target)))
        return projective_unitary_metrics(candidate, target)

    monkeypatch.setattr("rl.article_features.projective_unitary_metrics", recording_metric)
    context = ArticleTargetContext(np.eye(2, dtype=np.complex128))

    assert context.distance(state(1, Gate(GateType.T, (0,)))) > 0.0
    assert len(calls) == 1
    assert context.schema_version == "projective-unitary-metrics-v2"
