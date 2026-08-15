from dataclasses import dataclass, field
import math
from typing import Optional

from ckt_types import ResourceBudget


REWARD_MODE_ALIASES = {
    "legacy": "legacy_archive_shaping",
    "target_progress": "target_progress_shaping",
}

REWARD_MODES = frozenset(
    {
        "legacy_archive_shaping",
        "target_progress_shaping",
        "expansion_cost",
        "expansion_cost_plus_visit_bonus",
        "article_v1_expansion_potential",
    }
)


def normalize_reward_mode(value: str) -> str:
    """Return the durable reward-mode name used in reports and checkpoints."""

    if not isinstance(value, str):
        raise TypeError("reward_mode must be a string")
    normalized = REWARD_MODE_ALIASES.get(value, value)
    if normalized not in REWARD_MODES:
        choices = ", ".join(sorted(REWARD_MODES))
        raise ValueError(f"unsupported reward_mode {value!r}; choose one of {choices}")
    return normalized


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
    # Explicit experiment switches.  Both remain enabled for production
    # search; disabling either is supported only to run declared tiny-instance
    # ablations with the same expansion/certification engine.
    canonicalization_enabled: bool = True
    pareto_dominance_enabled: bool = True
    absorb_clifford_angles: bool = True
    canonicalization_mode: str = "enhanced"
    seed: Optional[int] = None
    # The legacy feature/reward behavior remains the default for existing
    # searches.  GHZ target-aware experiments opt in explicitly.
    target_aware_features: bool = False
    reward_mode: str = "legacy_archive_shaping"
    # Article V1 potential-shaping coefficient.  It is consumed only by the
    # versioned Article V1 reward strategy; legacy modes ignore it.
    article_v1_beta: float = 1.0
    target_progress_reward: TargetProgressRewardConfig = field(
        default_factory=TargetProgressRewardConfig
    )

    def __post_init__(self) -> None:
        # Accept the two historical spellings for checkpoint/config migration,
        # but expose one unambiguous name to every new report.
        object.__setattr__(self, "reward_mode", normalize_reward_mode(self.reward_mode))
        if self.canonicalization_mode not in {"enhanced", "raw_witness"}:
            raise ValueError(
                "canonicalization_mode must be 'enhanced' or 'raw_witness'"
            )
        if not isinstance(self.target_aware_features, bool):
            raise TypeError("target_aware_features must be a bool")
        if not isinstance(self.canonicalization_enabled, bool):
            raise TypeError("canonicalization_enabled must be a bool")
        if not isinstance(self.pareto_dominance_enabled, bool):
            raise TypeError("pareto_dominance_enabled must be a bool")
        if not isinstance(self.absorb_clifford_angles, bool):
            raise TypeError("absorb_clifford_angles must be a bool")
        if not isinstance(self.target_progress_reward, TargetProgressRewardConfig):
            raise TypeError(
                "target_progress_reward must be a TargetProgressRewardConfig"
            )
        if (
            isinstance(self.article_v1_beta, bool)
            or not isinstance(self.article_v1_beta, (int, float))
            or not math.isfinite(float(self.article_v1_beta))
            or float(self.article_v1_beta) < 0.0
        ):
            raise ValueError("article_v1_beta must be finite and non-negative")
