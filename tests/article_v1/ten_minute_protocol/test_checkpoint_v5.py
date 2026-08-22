from pathlib import Path
from dataclasses import replace

import pytest

from experiments.article_v1_ten_minute_protocol import TenMinuteCheckpoint


def _checkpoint():
    return TenMinuteCheckpoint(
        training_seed=19,
        weights=(0.1, -0.2),
        feature_schema_version="article-v1-31d",
        feature_evaluator_schema_version="article-v1-feature-evaluator-v2",
        frontier_enumeration_schema_version="article-v1-priority-then-record-id-order-v1",
        ordered_feature_names=("a", "b"),
        learning_rate=0.001,
        corpus_config_digest="sha256:" + "a" * 64,
        runtime_protocol_schema_version="article-v1-runtime-protocol-v1",
        training_protocol_schema_version="article-v1-training-protocol-v1",
        total_expansion_budget=5,
        total_completed_expansions=5,
        eligible_splits=("train",),
        eligible_difficulties=("easy", "medium"),
        ordered_target_ids=("easy-id", "medium-id"),
        executed_target_schedule=("medium-id", "easy-id"),
        target_schedule_seed=19,
        episode_caps_by_difficulty=(("easy", 2), ("medium", 4)),
        effective_episode_caps=(4, 1),
    )


def test_v5_checkpoint_round_trip_and_digest(tmp_path: Path):
    checkpoint = _checkpoint()
    path = tmp_path / "checkpoint.json"
    checkpoint.save(path)
    loaded = TenMinuteCheckpoint.load(path)
    assert loaded == checkpoint
    assert loaded.digest == checkpoint.digest


def test_v5_checkpoint_rejects_legacy_schema(tmp_path: Path):
    path = tmp_path / "legacy.json"
    path.write_text('{"checkpoint_schema":"article-v1-transferable-linear-checkpoint-v4"}', encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        TenMinuteCheckpoint.load(path)


def test_incomplete_v5_checkpoint_is_recovery_only():
    with pytest.raises(ValueError, match="not transferable"):
        replace(_checkpoint(), total_completed_expansions=4).require_evaluation_eligible()
