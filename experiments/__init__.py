"""Reproducible experiment harnesses built on the shared symbolic engine.

Imports are lazy so profile/config consumers do not initialize the dense
benchmark stack or create circular imports.
"""

from __future__ import annotations

from experiments.profiles import (
    ARTICLE_V1_PROFILE,
    EXPERIMENT_PROFILES,
    ExperimentProfile,
    experiment_profile,
)

_LEGACY_EXPORTS = {
    "TrainedArticlePolicy",
    "evaluate_native_corpus",
    "run_tiny_ablations",
    "summarize_runs",
    "train_article_policy",
}


def __getattr__(name: str):
    if name in _LEGACY_EXPORTS:
        from experiments import article_benchmark

        return getattr(article_benchmark, name)
    raise AttributeError(name)

__all__ = [
    "TrainedArticlePolicy",
    "evaluate_native_corpus",
    "run_tiny_ablations",
    "summarize_runs",
    "train_article_policy",
    "ARTICLE_V1_PROFILE",
    "EXPERIMENT_PROFILES",
    "ExperimentProfile",
    "experiment_profile",
]
