from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

import experiments.article_v1_replay_timing as replay_timing
from experiments.article_v1_replay_timing import (
    ARTICLE_V1_REPLAY_TIMING_SCHEMA,
    REPLAY_TIMING_EXPECTED_EXPANSIONS,
    ArticleV1ReplayTimingEvidence,
    ReplayTimingFormatError,
    ReplayValidationResult,
    checkpoint_file_sha256,
    load_replay_timing,
    measure_replay_timing,
    write_replay_timing,
)


def _digest(label: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _validation_result(*, measured_expansions: int = 1024) -> ReplayValidationResult:
    return ReplayValidationResult(
        measured_expansions=measured_expansions,
        frontier_active_ids_digest=_digest("frontier"),
        archive_digest=_digest("archive"),
        generation_count_digest=_digest("generation"),
        policy_weight_digest=_digest("weights"),
        pending_feature_digest=_digest("pending"),
    )


def _evidence(
    tmp_path: Path,
    *,
    elapsed_seconds: float = 5.0,
    projected_full_episode_seconds: float = 100.0,
    source_committed_and_clean: bool = True,
    measured_expansions: int = 1024,
    journal_entry_count: int = 1024,
    target_id: str = "hard-3q-target",
) -> ArticleV1ReplayTimingEvidence:
    checkpoint = tmp_path / "latest.json"
    checkpoint.write_bytes(b'{"checkpoint":"portable"}\n')
    ticks = iter((1_000_000_000, 1_000_000_000 + int(elapsed_seconds * 1e9)))
    return measure_replay_timing(
        lambda: _validation_result(measured_expansions=measured_expansions),
        source_commit_sha="bd251b9",
        source_worktree_digest=_digest("worktree"),
        source_committed_and_clean=source_committed_and_clean,
        config_digest=_digest("config"),
        target_id=target_id,
        target_fingerprint=_digest("target"),
        feature_evaluator_schema_version="article-v1-exact-incremental-v2",
        checkpoint_path=checkpoint,
        checkpoint_file_sha256=checkpoint_file_sha256(checkpoint),
        checkpoint_schema_version="article-v1-mid-episode-replay-checkpoint-v2",
        journal_digest=_digest("journal"),
        journal_entry_count=journal_entry_count,
        expected_expansions=REPLAY_TIMING_EXPECTED_EXPANSIONS,
        projected_full_episode_seconds=projected_full_episode_seconds,
        clock_ns=lambda: next(ticks),
    )


@pytest.mark.parametrize(
    (
        "elapsed_seconds",
        "projected_seconds",
        "exceeds_seconds",
        "exceeds_fraction",
    ),
    (
        (5.0, 100.0, False, False),
        (60.0, 1000.0, False, False),
        (61.0, 1000.0, True, False),
        (11.0, 100.0, False, True),
    ),
)
def test_preregistered_thresholds_are_exact_and_compaction_is_their_or(
    tmp_path: Path,
    elapsed_seconds: float,
    projected_seconds: float,
    exceeds_seconds: bool,
    exceeds_fraction: bool,
) -> None:
    evidence = _evidence(
        tmp_path,
        elapsed_seconds=elapsed_seconds,
        projected_full_episode_seconds=projected_seconds,
    )

    assert evidence.replay_time_threshold_seconds == 60.0
    assert evidence.replay_time_threshold_fraction == 0.10
    assert evidence.fraction_threshold_seconds == projected_seconds * 0.10
    assert evidence.exceeds_seconds_threshold is exceeds_seconds
    assert evidence.exceeds_fraction_threshold is exceeds_fraction
    assert evidence.compaction_required is (exceeds_seconds or exceeds_fraction)
    assert evidence.engineering_timing_valid is True
    assert evidence.pilot_relaunch_ready is not evidence.compaction_required


def test_dirty_or_incomplete_measurement_is_labelled_and_never_relaunch_ready(
    tmp_path: Path,
) -> None:
    dirty = _evidence(tmp_path, source_committed_and_clean=False)
    assert dirty.engineering_timing_valid is True
    assert dirty.source_committed_and_clean is False
    assert dirty.pilot_relaunch_ready is False

    incomplete = _evidence(tmp_path, measured_expansions=1023)
    assert incomplete.engineering_timing_valid is False
    assert incomplete.pilot_relaunch_ready is False

    journal_mismatch = _evidence(tmp_path, journal_entry_count=1023)
    assert journal_mismatch.engineering_timing_valid is False
    assert journal_mismatch.pilot_relaunch_ready is False


def test_measurement_binds_exact_checkpoint_bytes_and_validated_digests(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_bytes(b"original\n")
    expected_digest = checkpoint_file_sha256(checkpoint)
    checkpoint.write_bytes(b"changed\n")

    with pytest.raises(ReplayTimingFormatError, match="checkpoint file sha256 mismatch"):
        measure_replay_timing(
            lambda: _validation_result(),
            source_commit_sha="bd251b9",
            source_worktree_digest=_digest("worktree"),
            source_committed_and_clean=True,
            config_digest=_digest("config"),
            target_id="target",
            target_fingerprint=_digest("target"),
            feature_evaluator_schema_version="article-v1-exact-incremental-v2",
            checkpoint_path=checkpoint,
            checkpoint_file_sha256=expected_digest,
            checkpoint_schema_version="checkpoint-v1",
            journal_digest=_digest("journal"),
            journal_entry_count=1024,
            projected_full_episode_seconds=100.0,
            clock_ns=lambda: 0,
        )

    evidence = _evidence(tmp_path)
    assert evidence.validated_final_frontier_active_ids_digest == _digest("frontier")
    assert evidence.validated_final_archive_digest == _digest("archive")
    assert evidence.validated_final_generation_count_digest == _digest("generation")
    assert evidence.validated_final_policy_weight_digest == _digest("weights")
    assert evidence.validated_final_pending_feature_digest == _digest("pending")
    with pytest.raises(FrozenInstanceError):
        evidence.elapsed_seconds = 1.0  # type: ignore[misc]


def test_round_trip_is_canonical_atomic_and_exact_member_strict(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    path = tmp_path / "profiles" / "replay_timing.json"

    assert write_replay_timing(path, evidence) == path
    assert path.read_bytes().endswith(b"\n")
    assert load_replay_timing(path) == evidence
    assert not tuple(path.parent.glob(f".{path.name}.*.tmp"))

    payload = evidence.to_payload()
    assert payload["replay_timing_schema"] == ARTICLE_V1_REPLAY_TIMING_SCHEMA
    payload["unexpected"] = True
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ReplayTimingFormatError, match="members mismatch"):
        load_replay_timing(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda payload: payload.__setitem__("replay_timing_schema", "old-v0"),
            "unsupported",
        ),
        (
            lambda payload: payload.__setitem__("journal_entry_count", True),
            "integer",
        ),
        (
            lambda payload: payload.__setitem__("journal_digest", "not-a-digest"),
            "canonical sha256",
        ),
        (
            lambda payload: payload.__setitem__("compaction_required", True),
            "compaction-required result is incoherent",
        ),
    ),
)
def test_loader_rejects_schema_types_digests_and_derived_tampering(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload = _evidence(tmp_path).to_payload()
    mutation(payload)
    path = tmp_path / "forged.json"
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ReplayTimingFormatError, match=message):
        load_replay_timing(path)


def test_loader_rejects_duplicate_nonfinite_and_incomplete_json(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)
    canonical = json.dumps(evidence.to_payload(), separators=(",", ":"))
    path = tmp_path / "invalid.json"

    path.write_text('{"target_id":"duplicate",' + canonical[1:] + "\n", encoding="utf-8")
    with pytest.raises(ReplayTimingFormatError, match="duplicate JSON object member"):
        load_replay_timing(path)

    nonfinite = evidence.to_payload()
    nonfinite["elapsed_seconds"] = float("nan")
    path.write_text(json.dumps(nonfinite) + "\n", encoding="utf-8")
    with pytest.raises(ReplayTimingFormatError, match="non-finite JSON constant"):
        load_replay_timing(path)

    path.write_text(canonical, encoding="utf-8")
    with pytest.raises(ReplayTimingFormatError, match="incomplete final record"):
        load_replay_timing(path)


def test_failed_atomic_replace_preserves_previous_complete_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "replay_timing.json"
    original = _evidence(tmp_path, target_id="original")
    replacement = _evidence(tmp_path, target_id="replacement")
    write_replay_timing(path, original)
    original_bytes = path.read_bytes()

    def fail_replace(source: str, destination: str | Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(replay_timing.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        write_replay_timing(path, replacement)

    assert path.read_bytes() == original_bytes
    assert load_replay_timing(path) == original
    assert not tuple(path.parent.glob(f".{path.name}.*.tmp"))


def test_clock_must_be_monotonic_integer_nanoseconds(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_bytes(b"checkpoint\n")
    common = dict(
        source_commit_sha="bd251b9",
        source_worktree_digest=_digest("worktree"),
        source_committed_and_clean=True,
        config_digest=_digest("config"),
        target_id="target",
        target_fingerprint=_digest("target"),
        feature_evaluator_schema_version="article-v1-exact-incremental-v2",
        checkpoint_path=checkpoint,
        checkpoint_file_sha256=checkpoint_file_sha256(checkpoint),
        checkpoint_schema_version="checkpoint-v1",
        journal_digest=_digest("journal"),
        journal_entry_count=1024,
        projected_full_episode_seconds=100.0,
    )

    ticks = iter((2, 1))
    with pytest.raises(ReplayTimingFormatError, match="clock regressed"):
        measure_replay_timing(
            _validation_result, clock_ns=lambda: next(ticks), **common
        )

    with pytest.raises(ReplayTimingFormatError, match="integer nanoseconds"):
        measure_replay_timing(
            _validation_result, clock_ns=lambda: 1.5, **common
        )
