from __future__ import annotations

import pytest

from rl.article_v1_reward import ArticleV1RewardModel


class _Metric:
    def distance(self, state):
        return float(state)


def _model(*, beta=0.5, budget=10):
    return ArticleV1RewardModel(_Metric(), expansion_budget=budget, beta=beta)


def test_article_v1_success_return_is_budget_minus_hit_time():
    model = _model(beta=0.0, budget=10)
    rewards = [
        model.transition(
            expansion_index=step,
            potential_before=0.0,
            potential_after=0.0,
            certified_success=step == 4,
            terminal_failure=False,
        ).reward
        for step in range(1, 5)
    ]
    assert sum(rewards) == pytest.approx(6.0)


@pytest.mark.parametrize("failure_step", [1, 4, 10])
def test_article_v1_every_failure_returns_negative_full_budget(failure_step):
    model = _model(beta=0.0, budget=10)
    rewards = []
    for step in range(1, failure_step + 1):
        rewards.append(
            model.transition(
                expansion_index=step,
                potential_before=0.0,
                potential_after=0.0,
                certified_success=False,
                terminal_failure=step == failure_step,
            ).reward
        )
    assert sum(rewards) == pytest.approx(-10.0)


def test_article_v1_potential_is_only_negative_minimum_target_distance():
    model = _model(beta=2.0)
    assert model.frontier_potential([0.8, 0.2, 0.5]) == pytest.approx(-0.2)
    assert model.frontier_potential([0.2], terminal=True) == 0.0
    breakdown = model.transition(
        expansion_index=1,
        potential_before=-0.8,
        potential_after=-0.2,
        certified_success=False,
        terminal_failure=False,
    )
    assert breakdown.base_reward == -1.0
    assert breakdown.potential_delta == pytest.approx(0.6)
    assert breakdown.reward == pytest.approx(0.2)
    assert "best_generated_child" not in breakdown.info()
    assert "support" not in repr(breakdown.info()).lower()
    assert "entanglement" not in repr(breakdown.info()).lower()


def test_article_v1_shaping_telescopes_to_terminal_zero():
    model = _model(beta=0.75, budget=5)
    potentials = [-0.9, -0.6, -0.1, 0.0]
    shaped = []
    for step, (before, after) in enumerate(zip(potentials, potentials[1:]), 1):
        shaped.append(
            model.transition(
                expansion_index=step,
                potential_before=before,
                potential_after=after,
                certified_success=step == 3,
                terminal_failure=False,
            ).reward
        )
    base_return = 5 - 3
    assert sum(shaped) == pytest.approx(base_return + 0.75 * (0.0 - potentials[0]))
