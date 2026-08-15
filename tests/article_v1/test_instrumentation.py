from __future__ import annotations

import pytest

from certification.article_v1 import ArticleV1CertificationEngine
from certification.simulator import SynthesisTarget, unitary_from_gates
from circuit.gate import Gate
from ckt_types import ResourceBudget
from enums import GateType
from evaluate import evaluate
from rl.article_features import ArticleTargetContext, ArticleV1FeatureProvider


def _h_target() -> SynthesisTarget:
    return SynthesisTarget(
        unitary_from_gates(2, (Gate(GateType.H, (0,)),))
    )


def _run(*, instrumentation_enabled: bool) -> dict:
    target = _h_target()
    return evaluate(
        num_qubits=2,
        target_gates=(),
        target_unitary=target.unitary,
        budget=ResourceBudget(
            max_t_count=1,
            max_two_qubit_count=1,
            max_gates=2,
            max_depth=2,
        ),
        max_steps=3,
        seed=41,
        scheduler="fifo",
        collect_trace=True,
        reward_mode="expansion_cost",
        certification_engine=ArticleV1CertificationEngine(target),
        instrumentation_enabled=instrumentation_enabled,
        observation_features=False,
    )


def test_instrumentation_toggle_preserves_deterministic_search_trace() -> None:
    enabled = _run(instrumentation_enabled=True)
    disabled = _run(instrumentation_enabled=False)

    assert enabled["trace"] == disabled["trace"]
    for key in (
        "certified",
        "terminated",
        "truncated",
        "expansions",
        "frontier_size",
        "witness",
        "witness_operations",
        "scheduler_semantics",
        "action_semantics",
        "solution_resource_vector",
    ):
        assert enabled[key] == disabled[key]

    deterministic_enabled = {
        key: value
        for key, value in enabled["search_metrics"].items()
        if not key.endswith("_time_ns")
    }
    deterministic_disabled = {
        key: value
        for key, value in disabled["search_metrics"].items()
        if not key.endswith("_time_ns")
    }
    assert deterministic_enabled == deterministic_disabled
    assert disabled["search_metrics"]["wall_time_ns"] == 0
    assert disabled["search_metrics"]["certification_time_ns"] == 0
    assert enabled["search_metrics"]["wall_time_ns"] > 0
    assert enabled["search_metrics"]["certification_time_ns"] > 0
    assert enabled["search_metrics"]["environment_step_time_ns"] <= enabled[
        "search_metrics"
    ]["wall_time_ns"]
    assert enabled["runtime_seconds"] == pytest.approx(
        enabled["search_metrics"]["wall_time_ns"] / 1e9
    )


def test_zero_weight_article_control_materializes_the_same_feature_pipeline() -> None:
    target = _h_target()
    context = ArticleTargetContext(target)
    provider = ArticleV1FeatureProvider(context, search_horizon=2)
    result = evaluate(
        num_qubits=2,
        target_gates=(),
        target_unitary=target.unitary,
        budget=ResourceBudget(
            max_t_count=1,
            max_two_qubit_count=1,
            max_gates=2,
            max_depth=2,
        ),
        max_steps=2,
        seed=43,
        scheduler="zero_weight_linear",
        reward_mode="article_v1_expansion_potential",
        feature_provider=provider,
        target_metric=context,
        certification_engine=ArticleV1CertificationEngine(target),
        instrumentation_enabled=True,
        observation_features=False,
    )

    assert result["search_metrics"]["feature_evaluation_count"] > 0
    assert result["search_metrics"]["target_metric_evaluation_count"] > 0


def test_amended_reward_end_to_end_success_and_early_frontier_exhaustion() -> None:
    target = _h_target()
    expansion_budget = 5
    common = {
        "num_qubits": 2,
        "target_gates": (),
        "target_unitary": target.unitary,
        "max_steps": expansion_budget,
        "seed": 7,
        "scheduler": "fifo",
        "collect_trace": True,
        "reward_mode": "article_v1_expansion_potential",
        "article_v1_beta": 0.0,
        "instrumentation_enabled": False,
        "observation_features": False,
    }

    success = evaluate(
        **common,
        budget=ResourceBudget(
            max_t_count=1,
            max_two_qubit_count=1,
            max_gates=1,
            max_depth=1,
        ),
        certification_engine=ArticleV1CertificationEngine(target),
    )
    assert success["certified"] is True
    assert success["expansions"] == 1
    assert sum(row["reward"] for row in success["trace"]) == pytest.approx(
        expansion_budget - success["expansions"]
    )
    assert success["reward_coefficients"]["schema_version"] == (
        "article-v1-expansion-potential-amended"
    )
    assert success["trace"][-1]["potential_after"] == 0.0

    exhausted = evaluate(
        **common,
        budget=ResourceBudget(
            max_t_count=0,
            max_two_qubit_count=0,
            max_gates=0,
            max_depth=0,
        ),
        certification_engine=ArticleV1CertificationEngine(target),
    )
    assert exhausted["certified"] is False
    assert exhausted["terminated"] is True
    assert exhausted["truncated"] is False
    assert exhausted["expansions"] == 1 < expansion_budget
    assert sum(row["reward"] for row in exhausted["trace"]) == pytest.approx(
        -expansion_budget
    )
    assert exhausted["trace"][-1]["reward"] == -expansion_budget
    assert exhausted["trace"][-1]["potential_after"] == 0.0
