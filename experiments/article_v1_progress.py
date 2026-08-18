"""Crash-conscious structured progress reporting for Article V1 campaigns.

The objects in this module deliberately know nothing about the trainer or the
search environment.  A caller supplies an immutable snapshot at a safe point;
the reporter applies an engineering-only cadence and durably publishes it.
No timing value emitted here is part of deterministic scientific replay.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, TextIO


ARTICLE_V1_PROGRESS_EVENT_SCHEMA = "article-v1-progress-event-v2"
ARTICLE_V1_PROGRESS_STATUS_SCHEMA = "article-v1-progress-status-v2"


class ProgressFormatError(ValueError):
    """Raised when a progress artifact is not strict, complete, or valid."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ProgressFormatError(f"{name} must be a nonempty string")
    return value


def _require_int(name: str, value: object, *, minimum: int = 0) -> int:
    if not _is_int(value) or value < minimum:
        raise ProgressFormatError(f"{name} must be an integer >= {minimum}")
    return value


def _require_finite(name: str, value: object, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProgressFormatError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ProgressFormatError(f"{name} must be finite and >= {minimum}")
    return result


def _parse_utc_timestamp(value: object) -> str:
    timestamp = _require_nonempty_string("timestamp_utc", value)
    if not timestamp.endswith("Z"):
        raise ProgressFormatError("timestamp_utc must use a trailing Z UTC designator")
    try:
        parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as error:
        raise ProgressFormatError("timestamp_utc is not valid ISO-8601") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ProgressFormatError("timestamp_utc must represent UTC")
    return timestamp


def utc_timestamp() -> str:
    """Return a compact ISO-8601 UTC timestamp for a nondeterministic event field."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


@dataclass(frozen=True, slots=True)
class ArticleV1ProgressEvent:
    """One immutable, schema-versioned engineering progress observation."""

    timestamp_utc: str
    run_id: str
    phase: str
    feature_evaluator_schema_version: str
    training_seed: int | None
    target_index: int
    target_count: int
    target_id: str
    split: str
    stratum: str
    num_qubits: int
    episode_index: int
    episode_count: int
    expansion: int
    expansion_cap: int
    frontier_size: int
    frontier_peak: int
    archive_records: int
    active_archive_records: int
    unique_resource_groups: int
    last_feature_batch_seconds: float
    rolling_feature_batch_seconds: float
    elapsed_seconds: float
    expansions_per_second: float
    checkpoint_path: str | None = None
    schema_version: str = ARTICLE_V1_PROGRESS_EVENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ARTICLE_V1_PROGRESS_EVENT_SCHEMA:
            raise ProgressFormatError("unsupported Article V1 progress-event schema")
        _parse_utc_timestamp(self.timestamp_utc)
        for name in (
            "run_id",
            "phase",
            "feature_evaluator_schema_version",
            "target_id",
            "split",
            "stratum",
        ):
            _require_nonempty_string(name, getattr(self, name))
        if self.training_seed is not None:
            _require_int("training_seed", self.training_seed)
        target_index = _require_int("target_index", self.target_index)
        target_count = _require_int("target_count", self.target_count, minimum=1)
        if target_index >= target_count:
            raise ProgressFormatError("target_index must be smaller than target_count")
        episode_index = _require_int("episode_index", self.episode_index)
        episode_count = _require_int("episode_count", self.episode_count, minimum=1)
        if episode_index >= episode_count:
            raise ProgressFormatError("episode_index must be smaller than episode_count")
        _require_int("num_qubits", self.num_qubits, minimum=1)
        expansion = _require_int("expansion", self.expansion)
        expansion_cap = _require_int("expansion_cap", self.expansion_cap, minimum=1)
        if expansion > expansion_cap:
            raise ProgressFormatError("expansion must not exceed expansion_cap")
        frontier_size = _require_int("frontier_size", self.frontier_size)
        frontier_peak = _require_int("frontier_peak", self.frontier_peak)
        if frontier_size > frontier_peak:
            raise ProgressFormatError("frontier_size must not exceed frontier_peak")
        archive_records = _require_int("archive_records", self.archive_records)
        active_records = _require_int(
            "active_archive_records", self.active_archive_records
        )
        if active_records > archive_records:
            raise ProgressFormatError(
                "active_archive_records must not exceed archive_records"
            )
        _require_int("unique_resource_groups", self.unique_resource_groups)
        for name in (
            "last_feature_batch_seconds",
            "rolling_feature_batch_seconds",
            "elapsed_seconds",
            "expansions_per_second",
        ):
            _require_finite(name, getattr(self, name))
        if self.checkpoint_path is not None:
            _require_nonempty_string("checkpoint_path", self.checkpoint_path)

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        # Use the field name expected by every other portable Article V1 artifact.
        payload["progress_event_schema"] = payload.pop("schema_version")
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ArticleV1ProgressEvent":
        if not isinstance(payload, Mapping):
            raise ProgressFormatError("progress event must be a JSON object")
        expected = {field for field in cls.__dataclass_fields__ if field != "schema_version"}
        expected.add("progress_event_schema")
        observed = set(payload)
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ProgressFormatError(
                f"progress event members mismatch; missing={missing}, extra={extra}"
            )
        values = dict(payload)
        values["schema_version"] = values.pop("progress_event_schema")
        return cls(**values)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class ProgressCadence:
    """Engineering-only emission cadence; either trigger can be disabled."""

    every_expansions: int | None = 25
    every_seconds: float | None = 10.0

    def __post_init__(self) -> None:
        if self.every_expansions is not None:
            _require_int("every_expansions", self.every_expansions, minimum=1)
        if self.every_seconds is not None:
            _require_finite("every_seconds", self.every_seconds, minimum=0.0)
            if float(self.every_seconds) <= 0.0:
                raise ProgressFormatError("every_seconds must be > 0")
        if self.every_expansions is None and self.every_seconds is None:
            raise ProgressFormatError("at least one progress cadence trigger is required")


class ProgressEmissionGate:
    """Stateful `whichever occurs first` cadence gate using a monotonic clock."""

    def __init__(
        self,
        cadence: ProgressCadence | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        initial_expansion: int = 0,
    ) -> None:
        self.cadence = cadence or ProgressCadence()
        self._clock = clock
        self._last_expansion = _require_int("initial_expansion", initial_expansion)
        self._last_time = float(clock())

    def due(
        self,
        expansion: int,
        *,
        force: bool = False,
        now: float | None = None,
    ) -> bool:
        current_expansion = _require_int("expansion", expansion)
        if current_expansion < self._last_expansion:
            raise ProgressFormatError(
                "progress expansion regressed; reset the gate at an episode boundary"
            )
        if force:
            return True
        current_time = float(self._clock() if now is None else now)
        if not math.isfinite(current_time) or current_time < self._last_time:
            raise ProgressFormatError("progress monotonic clock regressed or is non-finite")
        expansion_due = (
            self.cadence.every_expansions is not None
            and current_expansion - self._last_expansion
            >= self.cadence.every_expansions
        )
        time_due = (
            self.cadence.every_seconds is not None
            and current_time - self._last_time >= self.cadence.every_seconds
        )
        return bool(expansion_due or time_due)

    def mark_emitted(self, expansion: int, *, now: float | None = None) -> None:
        current_expansion = _require_int("expansion", expansion)
        current_time = float(self._clock() if now is None else now)
        if current_expansion < self._last_expansion or current_time < self._last_time:
            raise ProgressFormatError("cannot mark a regressed progress event")
        self._last_expansion = current_expansion
        self._last_time = current_time

    def reset(self, *, expansion: int = 0, now: float | None = None) -> None:
        self._last_expansion = _require_int("expansion", expansion)
        self._last_time = float(self._clock() if now is None else now)


def _reject_constant(value: str) -> None:
    raise ProgressFormatError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProgressFormatError(f"duplicate JSON object member: {key}")
        result[key] = value
    return result


def _strict_json_loads(encoded: bytes) -> object:
    try:
        text = encoded.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProgressFormatError("invalid UTF-8 JSON progress artifact") from error


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
        raise ProgressFormatError("progress payload is not strict portable JSON") from error


def progress_event_digest(event: ArticleV1ProgressEvent) -> str:
    encoded = _canonical_json_bytes(event.to_payload()).rstrip(b"\n")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


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


def load_progress_events(path: str | Path) -> tuple[ArticleV1ProgressEvent, ...]:
    """Strictly load complete progress JSONL; never repair or skip bad records."""

    source = Path(path)
    if not source.exists():
        return ()
    raw = source.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ProgressFormatError("progress JSONL has an incomplete final record")
    events: list[ArticleV1ProgressEvent] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise ProgressFormatError(f"blank progress JSONL line {line_number}")
        payload = _strict_json_loads(line)
        if not isinstance(payload, dict):
            raise ProgressFormatError(
                f"progress JSONL line {line_number} is not an object"
            )
        events.append(ArticleV1ProgressEvent.from_payload(payload))
    return tuple(events)


def _status_payload(event: ArticleV1ProgressEvent) -> dict[str, object]:
    return {
        "progress_status_schema": ARTICLE_V1_PROGRESS_STATUS_SCHEMA,
        "latest_event_digest": progress_event_digest(event),
        "latest_event": event.to_payload(),
    }


def load_progress_status(path: str | Path) -> ArticleV1ProgressEvent:
    source = Path(path)
    raw = source.read_bytes()
    if not raw.endswith(b"\n"):
        raise ProgressFormatError("status JSON has an incomplete final record")
    payload = _strict_json_loads(raw)
    if not isinstance(payload, dict) or set(payload) != {
        "progress_status_schema",
        "latest_event_digest",
        "latest_event",
    }:
        raise ProgressFormatError("status JSON has an invalid member set")
    if payload["progress_status_schema"] != ARTICLE_V1_PROGRESS_STATUS_SCHEMA:
        raise ProgressFormatError("unsupported Article V1 progress-status schema")
    latest = payload["latest_event"]
    if not isinstance(latest, dict):
        raise ProgressFormatError("status latest_event must be an object")
    event = ArticleV1ProgressEvent.from_payload(latest)
    if payload["latest_event_digest"] != progress_event_digest(event):
        raise ProgressFormatError("status latest-event digest mismatch")
    return event


class ArticleV1ProgressReporter:
    """Cadenced callback that appends JSONL, atomically updates status, and prints."""

    def __init__(
        self,
        output_directory: str | Path,
        *,
        cadence: ProgressCadence | None = None,
        quiet: bool = False,
        stdout: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
        timing_clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.output_directory / "progress.jsonl"
        self.status_path = self.output_directory / "status.json"
        # Existing progress must be complete before we append to it.
        existing = load_progress_events(self.progress_path)
        initial_expansion = existing[-1].expansion if existing else 0
        self._clock = clock
        self._gate = ProgressEmissionGate(
            cadence, clock=clock, initial_expansion=initial_expansion
        )
        self._quiet = bool(quiet)
        self._stdout = stdout if stdout is not None else sys.stdout
        self._timing_clock_ns = timing_clock_ns
        self.progress_reporting_time_ns = 0

    @staticmethod
    def concise_line(event: ArticleV1ProgressEvent) -> str:
        seed = "-" if event.training_seed is None else str(event.training_seed)
        return (
            f"[{event.phase}] seed={seed} target="
            f"{event.target_index + 1}/{event.target_count}:{event.target_id} "
            f"episode={event.episode_index + 1}/{event.episode_count} "
            f"expansion={event.expansion}/{event.expansion_cap} "
            f"frontier={event.frontier_size} peak={event.frontier_peak} "
            f"rate={event.expansions_per_second:.3f}/s"
        )

    def _emit(self, event: ArticleV1ProgressEvent) -> None:
        encoded = _canonical_json_bytes(event.to_payload())
        with self.progress_path.open("ab") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_write_bytes(
            self.status_path, _canonical_json_bytes(_status_payload(event))
        )
        if not self._quiet:
            print(self.concise_line(event), file=self._stdout, flush=True)

    def maybe_emit(
        self,
        event: ArticleV1ProgressEvent,
        *,
        force: bool = False,
        now: float | None = None,
    ) -> bool:
        """Emit when the expansion/time cadence is due; return whether emitted."""

        current_time = float(self._clock() if now is None else now)
        if not self._gate.due(event.expansion, force=force, now=current_time):
            return False
        started = self._timing_clock_ns()
        self._emit(event)
        self.progress_reporting_time_ns += self._timing_clock_ns() - started
        self._gate.mark_emitted(event.expansion, now=current_time)
        return True

    def reset_cadence(self, *, expansion: int = 0, now: float | None = None) -> None:
        """Reset at a target/episode boundary where expansion counters restart."""

        self._gate.reset(expansion=expansion, now=now)


__all__ = [
    "ARTICLE_V1_PROGRESS_EVENT_SCHEMA",
    "ARTICLE_V1_PROGRESS_STATUS_SCHEMA",
    "ArticleV1ProgressEvent",
    "ArticleV1ProgressReporter",
    "ProgressCadence",
    "ProgressEmissionGate",
    "ProgressFormatError",
    "load_progress_events",
    "load_progress_status",
    "progress_event_digest",
    "utc_timestamp",
]
