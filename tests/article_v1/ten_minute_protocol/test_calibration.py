from pathlib import Path

import pytest

from experiments.article_v1_ten_minute_runner import (
    plan_ten_minute_campaign,
    select_feasible_hard_cap,
    select_training_interaction_budget,
)


ROOT = Path(__file__).resolve().parents[3]


def _row(cap, cpu, *, timeout=False, memory=1024):
    return {
        "executed_max_horizon": cap,
        "process_cpu_seconds": cpu,
        "terminal_reason": "OPERABILITY_TIMEOUT" if timeout else "UNSOLVED_WITHIN_EXPANSION_BUDGET",
        "search_metrics": {"feature_index_memory_bytes": memory},
    }


def test_selector_chooses_largest_cap_passing_all_gates():
    rows = [
        _row(1024, 100),
        _row(1024, 120),
        _row(1280, 500),
        _row(1280, 530),
        _row(1536, 601, timeout=True),
    ]
    result = select_feasible_hard_cap(
        rows,
        candidate_caps=(1024, 1280, 1536),
        target_cpu_seconds=540,
        hard_cpu_limit_seconds=600,
        maximum_feature_index_memory_bytes=100 * 1024 * 1024,
        correctness_parity_passed=True,
    )
    assert result["selected_hard_expansion_cap"] == 1280
    assert result["candidate_decisions"][-1]["passed"] is False


def test_selector_fails_closed_without_correctness_or_samples():
    result = select_feasible_hard_cap(
        [_row(1024, 10)],
        candidate_caps=(1024, 1280),
        target_cpu_seconds=540,
        hard_cpu_limit_seconds=600,
        maximum_feature_index_memory_bytes=100 * 1024 * 1024,
        correctness_parity_passed=False,
    )
    assert result["selected_hard_expansion_cap"] is None
    assert all(not row["passed"] for row in result["candidate_decisions"])


def test_campaign_plan_separates_physical_and_derived_counts():
    plan = plan_ten_minute_campaign(
        ROOT / "configs/article_v1_10min_pilot.json",
        training_cpu_seconds_per_expansion=0.1,
    )
    assert plan["executes_search"] is False
    assert plan["physical_search_execution_count"] > 0
    assert plan["derived_budget_threshold_observation_count"] > plan["physical_search_execution_count"]
    assert set(plan["by_evaluation_family"]) == {"in_distribution", "hard_generalization", "length_ood"}
    assert plan["total_training_interactions_all_seeds"] == 24000
    assert plan["projected_training_cpu_seconds_all_seeds"] == 2400


def test_training_budget_selection_is_validation_only_and_deterministic():
    rows = [
        {"split": "validation", "total_training_expansions_per_seed": 20, "certified": True, "expansions": 8},
        {"split": "validation", "total_training_expansions_per_seed": 40, "certified": True, "expansions": 8},
        {"split": "validation", "total_training_expansions_per_seed": 60, "certified": False, "expansions": 10},
    ]
    result = select_training_interaction_budget(
        rows, candidate_totals=(20, 40, 60)
    )
    assert result["selected_total_expansions_per_seed"] == 20
    assert result["no_test_access"] is True

    with pytest.raises(ValueError, match="non-validation"):
        select_training_interaction_budget(
            [{**rows[0], "split": "test"}], candidate_totals=(20,)
        )
