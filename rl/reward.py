"""Opt-in reward models for frontier-record scheduling.

Reward code deliberately sees only transition diagnostics.  It never feeds
back into gate legality, canonicalisation, archive dominance, or dense
certification, which keeps learned search order separate from correctness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Protocol

import numpy as np

from config import TargetProgressRewardConfig


@dataclass(frozen=True)
class RewardBreakdown:
    """Auditable terms for one reward transition.

    ``reward`` is the value returned by the environment.  The individual
    costs are stored as positive magnitudes so a report can reproduce the
    formula without depending on sign conventions.
    """

    reward: float
    potential_before: float
    potential_after: float
    potential_delta: float
    selected_node_potential: float
    best_generated_child_potential: float
    terminal_bonus: float
    step_cost: float
    dead_end_cost: float
    raw_reward: float
    clipped_reward: float

    def info(self) -> dict[str, float]:
        """Return JSON-ready diagnostics for Gymnasium ``info``."""

        return {name: float(value) for name, value in asdict(self).items() if name != "reward"}


class RewardModel(Protocol):
    """Small transition-only interface used by selectable environment rewards."""

    def reward(
        self,
        *,
        potential_before: float,
        potential_after: float,
        selected_node_potential: float,
        best_generated_child_potential: float,
        certified: bool,
        dead_end: bool,
    ) -> RewardBreakdown:
        """Score a completed frontier expansion."""


class TargetProgressRewardModel:
    """Potential-shaped reward for a dense small-instance target context.

    The returned value is

    ``B * terminal - c_step + eta * (Omega_after - Omega_before)
    - c_dead * dead_end``.

    ``Omega`` is computed by the environment from the active frontier plus a
    terminal child when appropriate; this class intentionally does not inspect
    or mutate search records.
    """

    def __init__(self, config: TargetProgressRewardConfig) -> None:
        self.config = config

    @staticmethod
    def _finite(name: str, value: float) -> float:
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError(f"{name} must be finite")
        return normalized

    def reward(
        self,
        *,
        potential_before: float,
        potential_after: float,
        selected_node_potential: float,
        best_generated_child_potential: float,
        certified: bool,
        dead_end: bool,
    ) -> RewardBreakdown:
        before = self._finite("potential_before", potential_before)
        after = self._finite("potential_after", potential_after)
        selected = self._finite("selected_node_potential", selected_node_potential)
        generated = self._finite(
            "best_generated_child_potential", best_generated_child_potential
        )
        potential_delta = after - before
        terminal_bonus = self.config.terminal_bonus if certified else 0.0
        dead_end_cost = self.config.dead_end_cost if dead_end else 0.0
        raw_reward = (
            terminal_bonus
            - self.config.step_cost
            + self.config.potential_scale * potential_delta
            - dead_end_cost
        )
        if self.config.reward_clip is None:
            clipped_reward = raw_reward
        else:
            clipped_reward = float(
                np.clip(raw_reward, -self.config.reward_clip, self.config.reward_clip)
            )
        return RewardBreakdown(
            reward=float(clipped_reward),
            potential_before=before,
            potential_after=after,
            potential_delta=potential_delta,
            selected_node_potential=selected,
            best_generated_child_potential=generated,
            terminal_bonus=float(terminal_bonus),
            step_cost=float(self.config.step_cost),
            dead_end_cost=float(dead_end_cost),
            raw_reward=float(raw_reward),
            clipped_reward=float(clipped_reward),
        )


__all__ = [
    "RewardBreakdown",
    "RewardModel",
    "TargetProgressRewardModel",
]
