from __future__ import annotations

import numpy as np

from certification.article_v1 import ArticleV1CertificationEngine
from certification.base import CertStatus
from certification.simulator import SynthesisTarget, unitary_from_gates
from circuit.circuit_state import CircuitState
from circuit.dag import CircuitDAG
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType
from evaluate import evaluate
from rl.article_features import (
    ArticleTargetContext,
    ArticleV1FeatureProvider,
    ArticleV1ReferenceFeatureProvider,
)


def _successful_zero_weight_run(provider_type):
    target = SynthesisTarget(
        unitary_from_gates(2, (Gate(GateType.H, (0,)),))
    )
    context = ArticleTargetContext(target)
    provider = provider_type(context, search_horizon=2)
    budget = ResourceBudget(
        max_t_count=1,
        max_two_qubit_count=1,
        max_gates=1,
        max_depth=1,
    )
    result = evaluate(
        num_qubits=2,
        target_gates=(),
        target_unitary=target.unitary,
        budget=budget,
        max_steps=2,
        seed=1729,
        scheduler="zero_weight_linear",
        collect_trace=True,
        reward_mode="article_v1_expansion_potential",
        article_v1_beta=0.0,
        feature_provider=provider,
        target_metric=context,
        certification_engine=ArticleV1CertificationEngine(target),
        instrumentation_enabled=False,
        observation_features=False,
    )
    return target, budget, result


def test_reference_and_optimized_success_witnesses_certify_identically() -> None:
    reference_target, reference_budget, reference = _successful_zero_weight_run(
        ArticleV1ReferenceFeatureProvider
    )
    optimized_target, optimized_budget, optimized = _successful_zero_weight_run(
        ArticleV1FeatureProvider
    )

    assert np.array_equal(reference_target.unitary, optimized_target.unitary)
    for name in (
        "certified",
        "terminated",
        "truncated",
        "expansions",
        "trace",
        "witness_operations",
        "solution_resource_vector",
    ):
        assert reference[name] == optimized[name]
    assert optimized["certified"] is True
    assert optimized["witness_operations"] == [{"gate": "H", "qubits": [0]}]

    witness = tuple(
        Gate(GateType[item["gate"]], tuple(item["qubits"]))
        for item in optimized["witness_operations"]
    )
    state = CircuitState(CircuitDAG.from_gates(2, witness), optimized_budget)
    assert (
        ArticleV1CertificationEngine(optimized_target).certify(state).status
        is CertStatus.SUCCESS
    )
    reference_state = CircuitState(
        CircuitDAG.from_gates(2, witness), reference_budget
    )
    assert (
        ArticleV1CertificationEngine(reference_target).certify(reference_state).status
        is CertStatus.SUCCESS
    )
