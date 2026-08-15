"""Article V1 expansion-count reward and direct target-distance potential."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Protocol


class ArticleTargetMetric(Protocol):
    """Minimal target metric required by the Article V1 reward."""

    def distance(self, state: object) -> float: ...


@dataclass(frozen=True, slots=True)
class ArticleV1RewardBreakdown:
    reward: float
    base_reward: float
    expansion_cost: float
    certified_success_correction: float
    terminal_failure_correction: float
    potential_before: float
    potential_after: float
    potential_delta: float
    shaping_beta: float

    def info(self) -> dict[str, float | str]:
        return {
            "reward_mode": "article_v1_expansion_potential",
            "reward_schema": "article-v1-expansion-potential-amended",
            **asdict(self),
        }


class ArticleV1RewardModel:
    """Implement the amended finite-budget return and Eq. (105--107) shaping.

    The base return is ``B - T`` for certification after ``T`` expansions and
    exactly ``-B`` for every unsuccessful episode, including early frontier
    exhaustion.  Potential shaping is target-distance-only and telescopes
    because every terminal state's potential is defined as zero.
    """

    schema_version = "article-v1-expansion-potential-amended"

    def __init__(
        self,
        target_metric: ArticleTargetMetric,
        *,
        expansion_budget: int,
        beta: float,
    ) -> None:
        if isinstance(expansion_budget, bool) or int(expansion_budget) < 1:
            raise ValueError("expansion_budget must be a positive integer")
        if isinstance(beta, bool) or not isinstance(beta, (int, float)):
            raise TypeError("beta must be a finite real number")
        if not math.isfinite(float(beta)) or float(beta) < 0.0:
            raise ValueError("beta must be finite and non-negative")
        distance = getattr(target_metric, "distance", None)
        if not callable(distance):
            raise TypeError("target_metric must expose distance(state)")
        self.target_metric = target_metric
        self.expansion_budget = int(expansion_budget)
        self.beta = float(beta)

    @staticmethod
    def _state(node_or_state: Any) -> Any:
        return getattr(node_or_state, "state", node_or_state)

    def frontier_potential(
        self,
        frontier: Iterable[object],
        *,
        terminal: bool = False,
    ) -> float:
        """Return ``-min d_tar`` for a nonterminal frontier, else zero."""

        if terminal:
            return 0.0
        nodes = tuple(frontier)
        if not nodes:
            return 0.0
        distances = [
            float(self.target_metric.distance(self._state(candidate)))
            for candidate in nodes
        ]
        if any(not math.isfinite(value) or value < 0.0 for value in distances):
            raise ValueError("target metric returned an invalid distance")
        return -min(distances)

    def transition(
        self,
        *,
        expansion_index: int,
        potential_before: float,
        potential_after: float,
        certified_success: bool,
        terminal_failure: bool,
    ) -> ArticleV1RewardBreakdown:
        """Score one completed selected-record expansion.

        ``expansion_index`` is one-based.  A budget-limit failure at exactly
        ``B`` has zero padding; an early failure is charged the unused horizon.
        """

        if isinstance(expansion_index, bool) or not isinstance(expansion_index, int):
            raise TypeError("expansion_index must be an integer")
        if expansion_index < 1 or expansion_index > self.expansion_budget:
            raise ValueError("expansion_index must lie in [1, expansion_budget]")
        if certified_success and terminal_failure:
            raise ValueError("success and terminal failure are mutually exclusive")
        if not all(math.isfinite(float(value)) for value in (potential_before, potential_after)):
            raise ValueError("potentials must be finite")

        success_correction = (
            float(self.expansion_budget) if certified_success else 0.0
        )
        failure_correction = (
            -float(self.expansion_budget - expansion_index)
            if terminal_failure
            else 0.0
        )
        base_reward = -1.0 + success_correction + failure_correction
        potential_delta = float(potential_after) - float(potential_before)
        reward = base_reward + self.beta * potential_delta
        return ArticleV1RewardBreakdown(
            reward=float(reward),
            base_reward=float(base_reward),
            expansion_cost=1.0,
            certified_success_correction=success_correction,
            terminal_failure_correction=failure_correction,
            potential_before=float(potential_before),
            potential_after=float(potential_after),
            potential_delta=float(potential_delta),
            shaping_beta=self.beta,
        )

    def metadata(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "expansion_budget": self.expansion_budget,
            "beta": self.beta,
            "discount": 1.0,
            "reward_clip": None,
            "exploration_bonus": 0.0,
            "terminal_potential": 0.0,
            "uses_pruning_bonus": False,
            "uses_dead_end_penalty": False,
            "uses_best_generated_child": False,
            "uses_support_or_entanglement": False,
        }


__all__ = [
    "ArticleTargetMetric",
    "ArticleV1RewardBreakdown",
    "ArticleV1RewardModel",
]
