"""Strict portable engineering evidence for bounded checkpoint replay timing.

This module is deliberately independent of the Article V1 runner and recovery
implementation.  A caller supplies one callable that performs and validates a
bounded deterministic replay.  The elapsed time is engineering evidence only;
it never enters scientific replay state or scheduler comparisons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Callable, Mapping


ARTICLE_V1_REPLAY_TIMING_SCHEMA = "article-v1-replay-timing-v2"
REPLAY_TIMING_EXPECTED_EXPANSIONS = 1024
REPLAY_TIMING_THRESHOLD_SECONDS = 60.0
REPLAY_TIMING_THRESHOLD_FRACTION = 0.10

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


class ReplayTimingFormatError(ValueError):
    """Raised when replay-timing evidence is incomplete or non-portable."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ReplayTimingFormatError(f"{name} must be a nonempty string")
    return value


def _require_commit(name: str, value: object) -> str:
    result = _require_nonempty_string(name, value)
    if _GIT_COMMIT_RE.fullmatch(result) is None:
        raise ReplayTimingFormatError(
            f"{name} must be a lowercase hexadecimal Git commit ID"
        )
    return result


def _require_digest(name: str, value: object) -> str:
    result = _require_nonempty_string(name, value)
    if _SHA256_RE.fullmatch(result) is None:
        raise ReplayTimingFormatError(f"{name} must be a canonical sha256 digest")
    return result


def _require_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ReplayTimingFormatError(f"{name} must be a bool")
    return value


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if not _is_int(value) or int(value) < minimum:
        raise ReplayTimingFormatError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _require_finite(
    name: str,
    value: object,
    *,
    minimum: float = 0.0,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayTimingFormatError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum or (positive and result <= 0.0):
        qualifier = "positive" if positive else f">= {minimum}"
        raise ReplayTimingFormatError(f"{name} must be finite and {qualifier}")
    return result


@dataclass(frozen=True, slots=True)
class ReplayValidationResult:
    """Validated deterministic state observed when the bounded replay ends."""

    measured_expansions: int
    frontier_active_ids_digest: str
    archive_digest: str
    generation_count_digest: str
    policy_weight_digest: str
    pending_feature_digest: str
    replay_mode: str = "portable-root-journal"
    runtime_snapshot_schema_version: str | None = None
    runtime_snapshot_base_expansion: int = 0
    delta_journal_entry_count: int | None = None
    runtime_snapshot_payload_sha256: str | None = None
    portable_replay_fallback_retained: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "measured_expansions",
            _require_int("measured_expansions", self.measured_expansions),
        )
        for name in (
            "frontier_active_ids_digest",
            "archive_digest",
            "generation_count_digest",
            "policy_weight_digest",
            "pending_feature_digest",
        ):
            _require_digest(name, getattr(self, name))
        if self.replay_mode not in {
            "portable-root-journal",
            "trusted-runtime-snapshot-plus-delta",
        }:
            raise ReplayTimingFormatError("unsupported replay mode")
        base = _require_int(
            "runtime_snapshot_base_expansion",
            self.runtime_snapshot_base_expansion,
        )
        delta_value = self.delta_journal_entry_count
        if delta_value is None and self.replay_mode == "portable-root-journal":
            delta_value = self.measured_expansions
            object.__setattr__(self, "delta_journal_entry_count", delta_value)
        delta = _require_int("delta_journal_entry_count", delta_value)
        _require_bool(
            "portable_replay_fallback_retained",
            self.portable_replay_fallback_retained,
        )
        if self.portable_replay_fallback_retained is not True:
            raise ReplayTimingFormatError(
                "portable JSON replay fallback must remain available"
            )
        if self.replay_mode == "portable-root-journal":
            if (
                self.runtime_snapshot_schema_version is not None
                or self.runtime_snapshot_payload_sha256 is not None
                or base != 0
                or delta != self.measured_expansions
            ):
                raise ReplayTimingFormatError("root replay metadata is incoherent")
        else:
            _require_nonempty_string(
                "runtime_snapshot_schema_version",
                self.runtime_snapshot_schema_version,
            )
            _require_digest(
                "runtime_snapshot_payload_sha256",
                self.runtime_snapshot_payload_sha256,
            )
            if base < 1 or base > self.measured_expansions:
                raise ReplayTimingFormatError("runtime snapshot base is incoherent")
            if delta != self.measured_expansions - base:
                raise ReplayTimingFormatError("delta replay count is incoherent")


@dataclass(frozen=True, slots=True)
class ArticleV1ReplayTimingEvidence:
    """One immutable, source-bound Section 15.10 replay measurement."""

    source_commit_sha: str
    source_worktree_digest: str
    source_committed_and_clean: bool
    config_digest: str
    target_id: str
    target_fingerprint: str
    feature_evaluator_schema_version: str
    checkpoint_path: str
    checkpoint_file_sha256: str
    checkpoint_schema_version: str
    journal_digest: str
    journal_entry_count: int
    expected_expansions: int
    measured_expansions: int
    elapsed_seconds: float
    projected_full_episode_seconds: float
    replay_time_threshold_seconds: float
    replay_time_threshold_fraction: float
    fraction_threshold_seconds: float
    exceeds_seconds_threshold: bool
    exceeds_fraction_threshold: bool
    compaction_required: bool
    engineering_timing_valid: bool
    pilot_relaunch_ready: bool
    validated_final_frontier_active_ids_digest: str
    validated_final_archive_digest: str
    validated_final_generation_count_digest: str
    validated_final_policy_weight_digest: str
    validated_final_pending_feature_digest: str
    replay_mode: str
    runtime_snapshot_schema_version: str | None
    runtime_snapshot_base_expansion: int
    delta_journal_entry_count: int
    runtime_snapshot_payload_sha256: str | None
    portable_replay_fallback_retained: bool
    schema_version: str = ARTICLE_V1_REPLAY_TIMING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ARTICLE_V1_REPLAY_TIMING_SCHEMA:
            raise ReplayTimingFormatError("unsupported Article V1 replay-timing schema")
        _require_commit("source_commit_sha", self.source_commit_sha)
        for name in (
            "source_worktree_digest",
            "config_digest",
            "target_fingerprint",
            "checkpoint_file_sha256",
            "journal_digest",
            "validated_final_frontier_active_ids_digest",
            "validated_final_archive_digest",
            "validated_final_generation_count_digest",
            "validated_final_policy_weight_digest",
            "validated_final_pending_feature_digest",
        ):
            _require_digest(name, getattr(self, name))
        for name in (
            "target_id",
            "feature_evaluator_schema_version",
            "checkpoint_path",
            "checkpoint_schema_version",
        ):
            _require_nonempty_string(name, getattr(self, name))
        _require_bool("source_committed_and_clean", self.source_committed_and_clean)
        journal_count = _require_int("journal_entry_count", self.journal_entry_count)
        expected = _require_int(
            "expected_expansions", self.expected_expansions, minimum=1
        )
        if expected != REPLAY_TIMING_EXPECTED_EXPANSIONS:
            raise ReplayTimingFormatError(
                "expected_expansions must be 1024 for Article V1 replay timing"
            )
        measured = _require_int("measured_expansions", self.measured_expansions)
        elapsed = _require_finite("elapsed_seconds", self.elapsed_seconds)
        projected = _require_finite(
            "projected_full_episode_seconds",
            self.projected_full_episode_seconds,
            positive=True,
        )
        seconds_threshold = _require_finite(
            "replay_time_threshold_seconds",
            self.replay_time_threshold_seconds,
            positive=True,
        )
        fraction = _require_finite(
            "replay_time_threshold_fraction",
            self.replay_time_threshold_fraction,
            positive=True,
        )
        fraction_seconds = _require_finite(
            "fraction_threshold_seconds", self.fraction_threshold_seconds, positive=True
        )
        for name in (
            "exceeds_seconds_threshold",
            "exceeds_fraction_threshold",
            "compaction_required",
            "engineering_timing_valid",
            "pilot_relaunch_ready",
        ):
            _require_bool(name, getattr(self, name))
        validation = ReplayValidationResult(
            measured_expansions=self.measured_expansions,
            frontier_active_ids_digest=(
                self.validated_final_frontier_active_ids_digest
            ),
            archive_digest=self.validated_final_archive_digest,
            generation_count_digest=(
                self.validated_final_generation_count_digest
            ),
            policy_weight_digest=self.validated_final_policy_weight_digest,
            pending_feature_digest=self.validated_final_pending_feature_digest,
            replay_mode=self.replay_mode,
            runtime_snapshot_schema_version=(
                self.runtime_snapshot_schema_version
            ),
            runtime_snapshot_base_expansion=(
                self.runtime_snapshot_base_expansion
            ),
            delta_journal_entry_count=self.delta_journal_entry_count,
            runtime_snapshot_payload_sha256=(
                self.runtime_snapshot_payload_sha256
            ),
            portable_replay_fallback_retained=(
                self.portable_replay_fallback_retained
            ),
        )
        if validation.measured_expansions != measured:
            raise ReplayTimingFormatError("replay validation metadata is incoherent")

        if seconds_threshold != REPLAY_TIMING_THRESHOLD_SECONDS:
            raise ReplayTimingFormatError("replay-time seconds threshold must be 60")
        if fraction != REPLAY_TIMING_THRESHOLD_FRACTION:
            raise ReplayTimingFormatError("replay-time fraction threshold must be 0.10")
        expected_fraction_seconds = projected * fraction
        if fraction_seconds != expected_fraction_seconds:
            raise ReplayTimingFormatError("fraction threshold seconds are incoherent")
        exceeds_seconds = elapsed > seconds_threshold
        exceeds_fraction = elapsed > fraction_seconds
        if self.exceeds_seconds_threshold is not exceeds_seconds:
            raise ReplayTimingFormatError("seconds-threshold result is incoherent")
        if self.exceeds_fraction_threshold is not exceeds_fraction:
            raise ReplayTimingFormatError("fraction-threshold result is incoherent")
        compaction_required = exceeds_seconds or exceeds_fraction
        if self.compaction_required is not compaction_required:
            raise ReplayTimingFormatError("compaction-required result is incoherent")
        timing_valid = (
            elapsed > 0.0
            and measured == expected
            and journal_count == expected
        )
        if self.engineering_timing_valid is not timing_valid:
            raise ReplayTimingFormatError("engineering-timing validity is incoherent")
        relaunch_ready = (
            self.source_committed_and_clean
            and timing_valid
            and not compaction_required
        )
        if self.pilot_relaunch_ready is not relaunch_ready:
            raise ReplayTimingFormatError("pilot-relaunch readiness is incoherent")

        # Normalize numeric values so direct construction and strict loading have
        # identical immutable in-memory representations.
        object.__setattr__(self, "journal_entry_count", journal_count)
        object.__setattr__(self, "expected_expansions", expected)
        object.__setattr__(self, "measured_expansions", measured)
        object.__setattr__(self, "elapsed_seconds", elapsed)
        object.__setattr__(self, "projected_full_episode_seconds", projected)
        object.__setattr__(self, "replay_time_threshold_seconds", seconds_threshold)
        object.__setattr__(self, "replay_time_threshold_fraction", fraction)
        object.__setattr__(self, "fraction_threshold_seconds", fraction_seconds)

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["replay_timing_schema"] = payload.pop("schema_version")
        return payload

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> "ArticleV1ReplayTimingEvidence":
        if not isinstance(payload, Mapping):
            raise ReplayTimingFormatError("replay-timing evidence must be an object")
        expected = {field.name for field in fields(cls) if field.name != "schema_version"}
        expected.add("replay_timing_schema")
        observed = set(payload)
        if observed != expected:
            raise ReplayTimingFormatError(
                "replay-timing members mismatch; "
                f"missing={sorted(expected - observed)}, "
                f"extra={sorted(observed - expected)}"
            )
        values = dict(payload)
        values["schema_version"] = values.pop("replay_timing_schema")
        return cls(**values)  # type: ignore[arg-type]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except (OSError, ValueError) as error:
        raise ReplayTimingFormatError(f"could not hash checkpoint file: {error}") from error
    return f"sha256:{digest.hexdigest()}"


def checkpoint_file_sha256(path: str | Path) -> str:
    """Return the canonical SHA-256 binding for the exact checkpoint bytes."""

    source = Path(path)
    if not source.is_file():
        raise ReplayTimingFormatError("checkpoint_path must identify a regular file")
    return _sha256_file(source)


def measure_replay_timing(
    replay_callable: Callable[[], ReplayValidationResult],
    *,
    source_commit_sha: str,
    source_worktree_digest: str,
    source_committed_and_clean: bool,
    config_digest: str,
    target_id: str,
    target_fingerprint: str,
    feature_evaluator_schema_version: str,
    checkpoint_path: str | Path,
    checkpoint_file_sha256: str,
    checkpoint_schema_version: str,
    journal_digest: str,
    journal_entry_count: int,
    expected_expansions: int = REPLAY_TIMING_EXPECTED_EXPANSIONS,
    projected_full_episode_seconds: float,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> ArticleV1ReplayTimingEvidence:
    """Time one validated bounded replay and derive the preregistered gates.

    Checkpoint hashing is performed before the clock starts, so filesystem I/O
    is not accidentally included in the replay duration.  Exceptions from the
    replay callable propagate and therefore cannot produce success evidence.
    """

    if not callable(replay_callable):
        raise TypeError("replay_callable must be callable")
    checkpoint = Path(checkpoint_path)
    supplied_checkpoint_digest = _require_digest(
        "checkpoint_file_sha256", checkpoint_file_sha256
    )
    if not checkpoint.is_file():
        raise ReplayTimingFormatError("checkpoint_path must identify a regular file")
    observed_checkpoint_digest = _sha256_file(checkpoint)
    if observed_checkpoint_digest != supplied_checkpoint_digest:
        raise ReplayTimingFormatError("checkpoint file sha256 mismatch")

    started = clock_ns()
    if not _is_int(started):
        raise ReplayTimingFormatError("clock_ns must return integer nanoseconds")
    result = replay_callable()
    stopped = clock_ns()
    if not _is_int(stopped):
        raise ReplayTimingFormatError("clock_ns must return integer nanoseconds")
    if stopped < started:
        raise ReplayTimingFormatError("replay timing clock regressed")
    if not isinstance(result, ReplayValidationResult):
        raise ReplayTimingFormatError(
            "replay_callable must return ReplayValidationResult"
        )

    elapsed = (stopped - started) / 1_000_000_000.0
    projected = _require_finite(
        "projected_full_episode_seconds",
        projected_full_episode_seconds,
        positive=True,
    )
    fraction_threshold_seconds = projected * REPLAY_TIMING_THRESHOLD_FRACTION
    exceeds_seconds = elapsed > REPLAY_TIMING_THRESHOLD_SECONDS
    exceeds_fraction = elapsed > fraction_threshold_seconds
    compaction_required = exceeds_seconds or exceeds_fraction
    expected = _require_int("expected_expansions", expected_expansions, minimum=1)
    journal_count = _require_int("journal_entry_count", journal_entry_count)
    timing_valid = (
        elapsed > 0.0
        and result.measured_expansions == expected
        and journal_count == expected
    )
    clean = _require_bool(
        "source_committed_and_clean", source_committed_and_clean
    )

    return ArticleV1ReplayTimingEvidence(
        source_commit_sha=source_commit_sha,
        source_worktree_digest=source_worktree_digest,
        source_committed_and_clean=clean,
        config_digest=config_digest,
        target_id=target_id,
        target_fingerprint=target_fingerprint,
        feature_evaluator_schema_version=feature_evaluator_schema_version,
        checkpoint_path=str(checkpoint),
        checkpoint_file_sha256=supplied_checkpoint_digest,
        checkpoint_schema_version=checkpoint_schema_version,
        journal_digest=journal_digest,
        journal_entry_count=journal_count,
        expected_expansions=expected,
        measured_expansions=result.measured_expansions,
        elapsed_seconds=elapsed,
        projected_full_episode_seconds=projected,
        replay_time_threshold_seconds=REPLAY_TIMING_THRESHOLD_SECONDS,
        replay_time_threshold_fraction=REPLAY_TIMING_THRESHOLD_FRACTION,
        fraction_threshold_seconds=fraction_threshold_seconds,
        exceeds_seconds_threshold=exceeds_seconds,
        exceeds_fraction_threshold=exceeds_fraction,
        compaction_required=compaction_required,
        engineering_timing_valid=timing_valid,
        pilot_relaunch_ready=clean and timing_valid and not compaction_required,
        validated_final_frontier_active_ids_digest=(
            result.frontier_active_ids_digest
        ),
        validated_final_archive_digest=result.archive_digest,
        validated_final_generation_count_digest=result.generation_count_digest,
        validated_final_policy_weight_digest=result.policy_weight_digest,
        validated_final_pending_feature_digest=result.pending_feature_digest,
        replay_mode=result.replay_mode,
        runtime_snapshot_schema_version=result.runtime_snapshot_schema_version,
        runtime_snapshot_base_expansion=result.runtime_snapshot_base_expansion,
        delta_journal_entry_count=result.delta_journal_entry_count,
        runtime_snapshot_payload_sha256=result.runtime_snapshot_payload_sha256,
        portable_replay_fallback_retained=(
            result.portable_replay_fallback_retained
        ),
    )


def _reject_constant(value: str) -> None:
    raise ReplayTimingFormatError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayTimingFormatError(f"duplicate JSON object member: {key}")
        result[key] = value
    return result


def _strict_json_loads(encoded: bytes) -> object:
    try:
        return json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplayTimingFormatError("invalid UTF-8 JSON replay-timing artifact") from error


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ReplayTimingFormatError(
            "replay-timing payload is not strict portable JSON"
        ) from error


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_replay_timing(
    path: str | Path, evidence: ArticleV1ReplayTimingEvidence
) -> Path:
    """Atomically write one complete canonical replay-timing JSON artifact."""

    if not isinstance(evidence, ArticleV1ReplayTimingEvidence):
        raise TypeError("evidence must be ArticleV1ReplayTimingEvidence")
    destination = Path(path)
    _atomic_write_bytes(destination, _canonical_json_bytes(evidence.to_payload()))
    return destination


def load_replay_timing(path: str | Path) -> ArticleV1ReplayTimingEvidence:
    """Strictly load one complete replay-timing artifact without repair."""

    source = Path(path)
    raw = source.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ReplayTimingFormatError(
            "replay-timing JSON has an incomplete final record"
        )
    payload = _strict_json_loads(raw)
    if not isinstance(payload, dict):
        raise ReplayTimingFormatError("replay-timing evidence must be one JSON object")
    return ArticleV1ReplayTimingEvidence.from_payload(payload)


__all__ = [
    "ARTICLE_V1_REPLAY_TIMING_SCHEMA",
    "ArticleV1ReplayTimingEvidence",
    "REPLAY_TIMING_EXPECTED_EXPANSIONS",
    "REPLAY_TIMING_THRESHOLD_FRACTION",
    "REPLAY_TIMING_THRESHOLD_SECONDS",
    "ReplayTimingFormatError",
    "ReplayValidationResult",
    "checkpoint_file_sha256",
    "load_replay_timing",
    "measure_replay_timing",
    "write_replay_timing",
]
