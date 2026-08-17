from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from certification.article_v1 import (
    ARTICLE_V1_CERTIFICATION_SCHEMA,
    DEFAULT_TAU_CERT,
    ArticleV1CertificationEngine,
    article_v1_certification_diagnostics,
    phase_frobenius_discrepancy,
)
from certification.base import CertStatus
from certification.simulator import SynthesisTarget, unitary_from_gates
from certification.unitary_phase_metrics import projective_unitary_metrics
from circuit.dag import CircuitDAG
from circuit.circuit_state import CircuitState
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType
from rl.article_features import ArticleTargetContext


def _state(num_qubits: int, gates: tuple[Gate, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(dag=CircuitDAG.from_gates(num_qubits, gates))


def _relative_phase_unitary(delta: float) -> np.ndarray:
    """Return diag(1, exp(i theta)) with the requested Eq. (129) delta."""

    c_phi = 1.0 - float(delta) ** 2
    theta = 2.0 * math.acos(c_phi)
    return np.diag([1.0, np.exp(1.0j * theta)]).astype(np.complex128)


def test_discrepancy_matches_direct_phase_aligned_frobenius_formula() -> None:
    target = unitary_from_gates(1, (Gate(GateType.H, (0,)),))
    candidate = unitary_from_gates(
        1,
        (Gate(GateType.T, (0,)), Gate(GateType.H, (0,))),
    )
    overlap = np.trace(target.conj().T @ candidate)
    phase = overlap / abs(overlap)
    direct = np.linalg.norm(candidate - phase * target, ord="fro") / math.sqrt(4.0)

    diagnostics = article_v1_certification_diagnostics(candidate, target)

    assert diagnostics.phase_frobenius_discrepancy == pytest.approx(direct)
    assert diagnostics.phase_aligned_matrix_error == pytest.approx(direct)
    assert diagnostics.c_phi == diagnostics.normalized_trace_magnitude
    assert diagnostics.delta_phi == diagnostics.phase_frobenius_discrepancy


def test_global_phase_passes_and_relative_phase_fails() -> None:
    target = unitary_from_gates(1, (Gate(GateType.H, (0,)),))
    globally_phased = np.exp(0.371j) * target
    global_result = article_v1_certification_diagnostics(globally_phased, target)
    relative_result = article_v1_certification_diagnostics(
        np.diag([1.0, 1.0j]),
        np.eye(2),
    )

    assert global_result.passed
    assert global_result.delta_phi <= DEFAULT_TAU_CERT
    assert not relative_result.passed
    assert relative_result.process_infidelity != pytest.approx(
        relative_result.phase_frobenius_discrepancy
    )


def test_threshold_is_inclusive_and_distinguishes_just_below_and_above() -> None:
    tau = 0.1
    below = article_v1_certification_diagnostics(
        _relative_phase_unitary(tau * (1.0 - 1e-6)),
        np.eye(2),
        tau_cert=tau,
    )
    above = article_v1_certification_diagnostics(
        _relative_phase_unitary(tau * (1.0 + 1e-6)),
        np.eye(2),
        tau_cert=tau,
    )

    assert below.delta_phi < tau
    assert below.passed
    assert above.delta_phi > tau
    assert not above.passed


def test_frozen_production_threshold_brackets_just_below_and_above() -> None:
    below = article_v1_certification_diagnostics(
        _relative_phase_unitary(DEFAULT_TAU_CERT * 0.999),
        np.eye(2),
        tau_cert=DEFAULT_TAU_CERT,
    )
    above = article_v1_certification_diagnostics(
        _relative_phase_unitary(DEFAULT_TAU_CERT * 1.001),
        np.eye(2),
        tau_cert=DEFAULT_TAU_CERT,
    )

    assert below.delta_phi < DEFAULT_TAU_CERT and below.passed
    assert above.delta_phi > DEFAULT_TAU_CERT and not above.passed


def test_diagnostics_are_frozen_and_json_ready() -> None:
    diagnostics = article_v1_certification_diagnostics(np.eye(4), np.eye(4))

    with pytest.raises(FrozenInstanceError):
        diagnostics.passed = False
    payload = diagnostics.to_dict()
    json.dumps(payload, allow_nan=False)

    assert payload["schema_version"] == ARTICLE_V1_CERTIFICATION_SCHEMA
    assert payload["c_phi"] == 1.0
    assert payload["delta_phi"] == 0.0
    assert payload["tau_cert"] == DEFAULT_TAU_CERT
    assert payload["passed"] is True
    assert payload["candidate_dimension"] == payload["target_dimension"] == 4
    assert payload["candidate_num_qubits"] == payload["target_num_qubits"] == 2
    assert payload["candidate_finite"] and payload["target_finite"]
    assert payload["candidate_unitary"] and payload["target_unitary"]


@pytest.mark.parametrize(
    ("candidate", "target", "message"),
    (
        (np.ones((2, 3)), np.eye(2), "candidate must be a square matrix"),
        (np.full((2, 2), np.nan), np.eye(2), "candidate must contain only finite"),
        (np.diag([1.0, 2.0]), np.eye(2), "candidate must be unitary"),
        (np.eye(2), np.full((2, 2), np.inf), "target must contain only finite"),
        (np.eye(2), np.diag([1.0, 2.0]), "target must be unitary"),
        (np.eye(2), np.eye(4), "dimensions must match"),
    ),
)
def test_matrix_validation_rejects_malformed_nonfinite_and_nonunitary_inputs(
    candidate: np.ndarray,
    target: np.ndarray,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        article_v1_certification_diagnostics(candidate, target)


def test_engine_reconstructs_only_from_authoritative_dag() -> None:
    gates = (Gate(GateType.H, (0,)), Gate(GateType.T, (0,)))
    target = unitary_from_gates(1, gates)
    state = _state(1, gates)
    state.features = np.full(31, np.nan)
    state.cached_unitary = np.eye(2)
    state.canonical_key = ("forged-target",)

    result = ArticleV1CertificationEngine(target).certify(state)

    assert result.status is CertStatus.SUCCESS
    assert result.info["reason"] == "equivalent_phase_frobenius"
    assert result.info["passed"] is True


def test_engine_returns_inconclusive_for_a_valid_prefix_nonmatch() -> None:
    engine = ArticleV1CertificationEngine(
        unitary_from_gates(1, (Gate(GateType.T, (0,)),))
    )

    result = engine.certify(_state(1, (Gate(GateType.H, (0,)),)))

    assert result.status is CertStatus.INCONCLUSIVE
    assert result.score == 0.0
    assert result.info["reason"] == "not_target"
    assert result.info["passed"] is False


def test_engine_is_phase_quotient_only_and_rejects_bad_targets() -> None:
    literal = SynthesisTarget(np.eye(2), quotient_global_phase=False)
    with pytest.raises(ValueError, match="global-phase quotienting only"):
        ArticleV1CertificationEngine(literal)
    with pytest.raises(ValueError, match="target must contain only finite"):
        ArticleV1CertificationEngine(np.full((2, 2), np.nan))
    with pytest.raises(ValueError, match="target must be unitary"):
        ArticleV1CertificationEngine(np.diag([1.0, 2.0]))


def test_engine_rejects_malformed_candidate_witness_clearly() -> None:
    malformed = SimpleNamespace(dag=SimpleNamespace(num_qubits=1, gates=(object(),)))

    result = ArticleV1CertificationEngine(np.eye(2)).certify(malformed)

    assert result.status is CertStatus.FAILURE
    assert result.info["reason"] == "candidate_dag_reconstruction_rejected"
    assert result.info["passed"] is False


def test_standalone_metric_has_the_frozen_default_formula() -> None:
    candidate = np.diag([1.0, 1.0j])
    expected_c = abs(np.trace(candidate)) / 2.0

    assert phase_frobenius_discrepancy(candidate, np.eye(2)) == pytest.approx(
        math.sqrt(1.0 - expected_c)
    )


def test_shared_metric_never_hides_scaling_by_frobenius_normalization() -> None:
    epsilon = 1e-8
    metrics = projective_unitary_metrics(
        (1.0 - epsilon) * np.eye(2),
        np.eye(2),
        unitarity_tolerance=1e-6,
    )

    assert metrics.normalized_trace_magnitude_raw == pytest.approx(1.0 - epsilon)
    assert metrics.phase_frobenius_discrepancy > 0.0


def test_shared_metric_rejects_slight_nonunitarity_under_production_tolerance() -> None:
    with pytest.raises(ValueError, match="candidate must be unitary"):
        projective_unitary_metrics((1.0 - 1e-8) * np.eye(2), np.eye(2))


def test_shared_metric_rejects_non_power_of_two_dimensions() -> None:
    with pytest.raises(ValueError, match="positive power of two"):
        projective_unitary_metrics(np.eye(3), np.eye(3))


def test_certifier_calls_the_shared_metric_implementation(monkeypatch) -> None:
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def recording_metric(candidate, target, *, unitarity_tolerance):
        calls.append((np.asarray(candidate), np.asarray(target)))
        return projective_unitary_metrics(
            candidate, target, unitarity_tolerance=unitarity_tolerance
        )

    monkeypatch.setattr(
        "certification.article_v1.projective_unitary_metrics", recording_metric
    )
    result = ArticleV1CertificationEngine(np.eye(2)).certify(_state(1))

    assert result.status is CertStatus.SUCCESS
    assert len(calls) == 1


def test_corrupt_target_metric_cache_cannot_influence_final_certification() -> None:
    gates = (Gate(GateType.H, (0,)),)
    target = unitary_from_gates(1, gates)
    state = CircuitState(
        CircuitDAG.from_gates(1, gates),
        ResourceBudget(4, 4, 4, 4),
    )
    context = ArticleTargetContext(target)
    context._cache[context.cache_key(state)] = 1.0

    assert context.distance(state) == 1.0
    assert ArticleV1CertificationEngine(target).certify(state).status is CertStatus.SUCCESS
