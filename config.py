from dataclasses import dataclass, field
import math
from typing import Optional

from ckt_types import ResourceBudget


@dataclass(frozen=True)
class TargetProgressRewardConfig:
    """Parameters for the opt-in target-progress reward model.

    The potential coefficients intentionally live next to the shaping
    parameters.  A target context consumes the former and the environment's
    reward model consumes the latter, so a GHZ experiment has one explicit,
    serialisable configuration instead of scattered constants.
    """

    process_fidelity_weight: float = 0.60
    support_match_weight: float = 0.15
    entanglement_match_weight: float = 0.25
    terminal_bonus: float = 10.0
    step_cost: float = 0.1
    potential_scale: float = 4.0
    dead_end_cost: float = 1.0
    reward_clip: Optional[float] = 10.0

    def __post_init__(self) -> None:
        non_negative = (
            "process_fidelity_weight",
            "support_match_weight",
            "entanglement_match_weight",
            "terminal_bonus",
            "step_cost",
            "potential_scale",
            "dead_end_cost",
        )
        for name in non_negative:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")
        potential_weight_sum = (
            self.process_fidelity_weight
            + self.support_match_weight
            + self.entanglement_match_weight
        )
        if not math.isclose(
            potential_weight_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "target-progress potential weights must sum to 1.0"
            )
        if self.reward_clip is not None and (
            isinstance(self.reward_clip, bool)
            or not isinstance(self.reward_clip, (int, float))
            or not math.isfinite(self.reward_clip)
            or self.reward_clip <= 0
        ):
            raise ValueError("reward_clip must be None or a finite positive number")

    @property
    def potential_weights(self) -> tuple[float, float, float]:
        """Return ``(process, support, entanglement)`` potential weights."""

        return (
            float(self.process_fidelity_weight),
            float(self.support_match_weight),
            float(self.entanglement_match_weight),
        )


@dataclass
class Config:
    num_qubits: int
    budget: ResourceBudget

    # RL-related placeholders (used later)
    max_steps: int = 100
    discount: float = 1.0
    max_frontier: int = 100
    # Every Kth selection can be forced to the oldest open record.  Zero
    # disables the fairness interleave for purely learned experiments.
    fairness_interval: int = 0
    seed: Optional[int] = None
    # The legacy feature/reward behavior remains the default for existing
    # searches.  GHZ target-aware experiments opt in explicitly.
    target_aware_features: bool = False
    reward_mode: str = "legacy"
    target_progress_reward: TargetProgressRewardConfig = field(
        default_factory=TargetProgressRewardConfig
    )

    def __post_init__(self) -> None:
        if self.reward_mode not in {"legacy", "target_progress"}:
            raise ValueError(
                "reward_mode must be either 'legacy' or 'target_progress'"
            )
        if not isinstance(self.target_aware_features, bool):
            raise TypeError("target_aware_features must be a bool")
        if not isinstance(self.target_progress_reward, TargetProgressRewardConfig):
            raise TypeError(
                "target_progress_reward must be a TargetProgressRewardConfig"
            )
