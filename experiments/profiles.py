"""Versioned scientific profiles for synthesis experiments.

Profiles name a coherent collection of schemas.  They are intentionally
separate from :class:`config.Config`, whose fields describe one concrete
environment.  This prevents a publication configuration from being inferred
from a loose collection of legacy Boolean switches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ExperimentProfile:
    """Immutable schema contract serialized into runs and checkpoints."""

    name: str
    feature_schema: str
    feature_evaluator_schema: str
    reward_schema: str
    target_metric_schema: str
    certification_schema: str
    gamma: float
    reward_clip: float | None
    exploration_bonus: float
    frozen_evaluation_fairness_interval: int

    def metadata(self) -> dict[str, object]:
        return asdict(self)


ARTICLE_V1_PROFILE = ExperimentProfile(
    name="article_v1_raw_metric_v2",
    feature_schema="article-v1-31d",
    feature_evaluator_schema="article-v1-exact-incremental-v2",
    reward_schema="article-v1-expansion-potential-amended",
    target_metric_schema="projective-unitary-metrics-v2",
    certification_schema="phase-frobenius-raw-v2",
    gamma=1.0,
    reward_clip=None,
    exploration_bonus=0.0,
    frozen_evaluation_fairness_interval=0,
)

LEGACY_RESOURCE_PROFILE = ExperimentProfile(
    name="legacy-resource-v1",
    feature_schema="frontier-resource-v1",
    feature_evaluator_schema="legacy-rowwise-v1",
    reward_schema="legacy-archive-shaping-v1",
    target_metric_schema="none",
    certification_schema="legacy-dense-phase-quotient-v1",
    gamma=1.0,
    reward_clip=10.0,
    exploration_bonus=0.1,
    frozen_evaluation_fairness_interval=0,
)

EXTENDED_TARGET_AWARE_PROFILE = ExperimentProfile(
    name="extended-target-aware-37d-v1",
    feature_schema="extended-target-aware-37d-v1",
    feature_evaluator_schema="legacy-rowwise-v1",
    reward_schema="article-expansion-cost-legacy-v1",
    target_metric_schema="phase-aligned-frobenius-v1",
    certification_schema="legacy-dense-phase-quotient-v1",
    gamma=1.0,
    reward_clip=None,
    exploration_bonus=0.0,
    frozen_evaluation_fairness_interval=0,
)

COMPOSITE_TARGET_PROGRESS_PROFILE = ExperimentProfile(
    name="composite-target-progress-v1",
    feature_schema="frontier-target-aware-v1",
    feature_evaluator_schema="legacy-rowwise-v1",
    reward_schema="composite-target-progress-v1",
    target_metric_schema="process-support-entanglement-composite-v1",
    certification_schema="legacy-dense-phase-quotient-v1",
    gamma=1.0,
    reward_clip=10.0,
    exploration_bonus=0.0,
    frozen_evaluation_fairness_interval=0,
)

GHZ3_DIRECT_PROFILE = ExperimentProfile(
    name="ghz3-direct-v1",
    feature_schema="frontier-target-aware-v1",
    feature_evaluator_schema="legacy-rowwise-v1",
    reward_schema="ghz3-terminal-direct-v1",
    target_metric_schema="ghz3-labelled-frame-v1",
    certification_schema="legacy-dense-phase-quotient-v1",
    gamma=1.0,
    reward_clip=None,
    exploration_bonus=0.0,
    frozen_evaluation_fairness_interval=0,
)

TOFFOLI_PARITY_PROFILE = ExperimentProfile(
    name="toffoli-parity-v1",
    feature_schema="toffoli-parity-frontier-v1",
    feature_evaluator_schema="toffoli-rowwise-v1",
    reward_schema="toffoli-parity-potential-v1",
    target_metric_schema="toffoli-parity-progress-v1",
    certification_schema="toffoli-exact-dense-v1",
    gamma=1.0,
    reward_clip=None,
    exploration_bonus=0.0,
    frozen_evaluation_fairness_interval=0,
)


EXPERIMENT_PROFILES: Mapping[str, ExperimentProfile] = {
    profile.name: profile
    for profile in (
        ARTICLE_V1_PROFILE,
        LEGACY_RESOURCE_PROFILE,
        EXTENDED_TARGET_AWARE_PROFILE,
        COMPOSITE_TARGET_PROGRESS_PROFILE,
        GHZ3_DIRECT_PROFILE,
        TOFFOLI_PARITY_PROFILE,
    )
}
# Compatibility lookup only; serialized metadata retains the V2 profile name.
EXPERIMENT_PROFILES = {**EXPERIMENT_PROFILES, "article_v1": ARTICLE_V1_PROFILE}


def experiment_profile(name: str) -> ExperimentProfile:
    """Resolve a declared profile or fail instead of guessing a migration."""

    try:
        return EXPERIMENT_PROFILES[name]
    except KeyError as error:
        choices = ", ".join(sorted(EXPERIMENT_PROFILES))
        raise ValueError(f"unknown experiment profile {name!r}; choose one of {choices}") from error


__all__ = [
    "ARTICLE_V1_PROFILE",
    "COMPOSITE_TARGET_PROGRESS_PROFILE",
    "EXPERIMENT_PROFILES",
    "EXTENDED_TARGET_AWARE_PROFILE",
    "ExperimentProfile",
    "GHZ3_DIRECT_PROFILE",
    "LEGACY_RESOURCE_PROFILE",
    "TOFFOLI_PARITY_PROFILE",
    "experiment_profile",
]
