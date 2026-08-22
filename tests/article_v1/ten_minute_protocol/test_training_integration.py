from pathlib import Path

import pytest

from experiments.article_v1_ten_minute_protocol import TenMinuteCheckpoint
from experiments.article_v1_ten_minute_runner import train_ten_minute
from experiments.article_v1_ten_minute_protocol import load_ten_minute_config, load_ten_minute_corpus_config
from benchmarks.article_native_corpus import build_article_v1_corpus
from experiments.article_v1_runner import evaluate_article_v1_run


ROOT = Path(__file__).resolve().parents[3]


def test_real_one_expansion_curriculum_writes_v5_checkpoint(tmp_path: Path):
    result = train_ten_minute(
        ROOT / "configs/article_v1_10min_pilot.json",
        tmp_path,
        training_seed=19,
        total_expansions_override=2,
    )
    assert result["complete"] is True
    assert result["total_training_expansions_completed"] == 2
    assert result["total_training_expansions_remaining"] == 0
    assert result["hard_targets_used_for_training"] is False
    assert set(result["expansions_by_difficulty"]) == {"easy", "medium"}
    checkpoint = TenMinuteCheckpoint.load(result["checkpoint_path"])
    assert checkpoint.total_completed_expansions == 2
    assert checkpoint.eligible_difficulties == ("easy", "medium")
    validation_case = build_article_v1_corpus(
        load_ten_minute_corpus_config(ROOT / "configs/article_v1_10min_pilot.json")
    ).cases(split="validation", difficulty="easy")[0]
    row = evaluate_article_v1_run(
        validation_case,
        scheduler="article_sarsa",
        expansion_budget=1,
        evaluation_seed=7,
        checkpoint=checkpoint,
        config_digest=load_ten_minute_config(
            ROOT / "configs/article_v1_10min_pilot.json"
        ).digest,
        budget_mode="fixed-max-horizon-anytime-v1",
        budget_thresholds=(1,),
    )
    assert row["schema_version"] == "article-v1-10min-raw-run-v1"


def test_mid_episode_resume_reproduces_schedule_trace_weights_and_digest(
    tmp_path: Path,
):
    config = ROOT / "configs/article_v1_10min_pilot.json"
    uninterrupted_dir = tmp_path / "uninterrupted"
    resumed_dir = tmp_path / "resumed"

    uninterrupted = train_ten_minute(
        config,
        uninterrupted_dir,
        training_seed=29,
        total_expansions_override=4,
        checkpoint_every_expansions=1,
    )
    with pytest.raises(KeyboardInterrupt):
        train_ten_minute(
            config,
            resumed_dir,
            training_seed=29,
            total_expansions_override=4,
            checkpoint_every_expansions=1,
            interrupt_after_expansions=1,
        )
    resumed = train_ten_minute(
        config,
        resumed_dir,
        training_seed=29,
        total_expansions_override=4,
        checkpoint_every_expansions=1,
    )

    uninterrupted_checkpoint = TenMinuteCheckpoint.load(
        uninterrupted["checkpoint_path"]
    )
    resumed_checkpoint = TenMinuteCheckpoint.load(resumed["checkpoint_path"])
    assert resumed_checkpoint.executed_target_schedule == (
        uninterrupted_checkpoint.executed_target_schedule
    )
    assert resumed_checkpoint.weights == uninterrupted_checkpoint.weights
    assert resumed_checkpoint.digest == uninterrupted_checkpoint.digest

    def selected_trace(directory: Path) -> list[int]:
        return [
            int(__import__("json").loads(line)["selected_record_id"])
            for line in (directory / "progress.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]

    assert selected_trace(resumed_dir) == selected_trace(uninterrupted_dir)
