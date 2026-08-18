from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import io
import json
from pathlib import Path

import pytest

from experiments.article_v1_progress import (
    ARTICLE_V1_PROGRESS_EVENT_SCHEMA,
    ARTICLE_V1_PROGRESS_STATUS_SCHEMA,
    ArticleV1ProgressEvent,
    ArticleV1ProgressReporter,
    ProgressCadence,
    ProgressEmissionGate,
    ProgressFormatError,
    load_progress_events,
    load_progress_status,
)


def _event(expansion: int = 0, *, checkpoint_path: str | None = None):
    return ArticleV1ProgressEvent(
        timestamp_utc="2026-08-17T12:34:56.789Z",
        run_id="bounded-progress-test",
        phase="training",
        feature_evaluator_schema_version="article-v1-exact-incremental-v2",
        training_seed=7,
        target_index=0,
        target_count=2,
        target_id="train-2q-000",
        split="train",
        stratum="smoke",
        num_qubits=2,
        episode_index=0,
        episode_count=2,
        expansion=expansion,
        expansion_cap=100,
        frontier_size=3 + expansion,
        frontier_peak=5 + expansion,
        archive_records=9 + expansion,
        active_archive_records=4 + expansion,
        unique_resource_groups=2 + expansion,
        last_feature_batch_seconds=0.01,
        rolling_feature_batch_seconds=0.02,
        elapsed_seconds=float(expansion),
        expansions_per_second=1.5,
        checkpoint_path=checkpoint_path,
    )


def test_progress_event_is_immutable_schema_strict_and_round_trips() -> None:
    event = _event(4)
    with pytest.raises(FrozenInstanceError):
        event.expansion = 5  # type: ignore[misc]
    payload = event.to_payload()
    assert payload["progress_event_schema"] == ARTICLE_V1_PROGRESS_EVENT_SCHEMA
    assert ArticleV1ProgressEvent.from_payload(payload) == event

    with pytest.raises(ProgressFormatError, match="unsupported"):
        ArticleV1ProgressEvent.from_payload(
            {**payload, "progress_event_schema": "article-v1-progress-event-v1"}
        )

    with pytest.raises(ProgressFormatError, match="members mismatch"):
        ArticleV1ProgressEvent.from_payload({**payload, "timing_is_scientific": False})
    with pytest.raises(ProgressFormatError, match="integer"):
        replace(event, expansion=True)  # type: ignore[arg-type]
    with pytest.raises(ProgressFormatError, match="finite"):
        replace(event, elapsed_seconds=float("nan"))
    with pytest.raises(ProgressFormatError, match="trailing Z"):
        replace(event, timestamp_utc="2026-08-17T12:34:56+05:30")


def test_progress_cadence_emits_on_expansion_or_time_whichever_first() -> None:
    gate = ProgressEmissionGate(
        ProgressCadence(every_expansions=25, every_seconds=10.0),
        clock=lambda: 100.0,
    )
    assert not gate.due(24, now=109.9)
    assert gate.due(25, now=101.0)
    gate.mark_emitted(25, now=101.0)
    assert gate.due(26, now=111.0)
    gate.mark_emitted(26, now=111.0)
    assert gate.due(26, force=True, now=111.0)
    with pytest.raises(ProgressFormatError, match="regressed"):
        gate.due(25, now=112.0)
    gate.reset(expansion=0, now=200.0)
    assert not gate.due(1, now=200.1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"every_expansions": 0, "every_seconds": 1.0},
        {"every_expansions": None, "every_seconds": 0.0},
        {"every_expansions": None, "every_seconds": None},
    ],
)
def test_invalid_progress_cadence_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ProgressFormatError):
        ProgressCadence(**kwargs)  # type: ignore[arg-type]


def test_reporter_writes_flushed_jsonl_atomic_status_and_concise_stdout(
    tmp_path: Path,
) -> None:
    output = io.StringIO()
    reporter = ArticleV1ProgressReporter(
        tmp_path,
        cadence=ProgressCadence(every_expansions=2, every_seconds=None),
        stdout=output,
        clock=lambda: 10.0,
    )
    assert not reporter.maybe_emit(_event(1), now=10.0)
    assert reporter.maybe_emit(_event(2, checkpoint_path="checkpoints/latest.json"), now=10.0)
    assert reporter.progress_path.read_bytes().endswith(b"\n")
    assert load_progress_events(reporter.progress_path) == (
        _event(2, checkpoint_path="checkpoints/latest.json"),
    )
    assert load_progress_status(reporter.status_path) == _event(
        2, checkpoint_path="checkpoints/latest.json"
    )
    status = json.loads(reporter.status_path.read_text(encoding="utf-8"))
    assert status["progress_status_schema"] == ARTICLE_V1_PROGRESS_STATUS_SCHEMA
    line = output.getvalue()
    assert "target=1/2:train-2q-000" in line
    assert "expansion=2/100" in line
    assert "frontier=5" in line
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_quiet_suppresses_stdout_not_durable_progress_and_timing_is_separate(
    tmp_path: Path,
) -> None:
    timing_values = iter((100, 145))
    output = io.StringIO()
    reporter = ArticleV1ProgressReporter(
        tmp_path,
        quiet=True,
        stdout=output,
        clock=lambda: 1.0,
        timing_clock_ns=lambda: next(timing_values),
    )
    assert reporter.maybe_emit(_event(0), force=True, now=1.0)
    assert output.getvalue() == ""
    assert reporter.progress_reporting_time_ns == 45
    payload = _event(0).to_payload()
    assert "progress_reporting_time_ns" not in payload
    assert "checkpoint_io_time_ns" not in payload


@pytest.mark.parametrize(
    "raw, message",
    [
        (b'{"progress_event_schema":"x"}', "incomplete"),
        (b"\n", "blank"),
        (b'{"x":NaN}\n', "non-finite"),
        (b'{"x":1,"x":2}\n', "duplicate"),
        (b"[]\n", "not an object"),
    ],
)
def test_progress_jsonl_loader_never_repairs_or_skips_invalid_records(
    tmp_path: Path, raw: bytes, message: str
) -> None:
    path = tmp_path / "progress.jsonl"
    path.write_bytes(raw)
    before = path.read_bytes()
    with pytest.raises(ProgressFormatError, match=message):
        load_progress_events(path)
    assert path.read_bytes() == before


def test_status_digest_tampering_is_detected(tmp_path: Path) -> None:
    reporter = ArticleV1ProgressReporter(tmp_path, quiet=True, clock=lambda: 0.0)
    reporter.maybe_emit(_event(0), force=True, now=0.0)
    payload = json.loads(reporter.status_path.read_text(encoding="utf-8"))
    payload["latest_event"]["frontier_size"] += 1
    reporter.status_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ProgressFormatError, match="digest mismatch"):
        load_progress_status(reporter.status_path)


def test_failed_atomic_status_replace_leaves_previous_status_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reporter = ArticleV1ProgressReporter(tmp_path, quiet=True, clock=lambda: 0.0)
    reporter.maybe_emit(_event(0), force=True, now=0.0)
    previous = reporter.status_path.read_bytes()

    import experiments.article_v1_progress as progress_module

    real_replace = progress_module.os.replace

    def fail_status(source: str, destination: str | Path) -> None:
        if Path(destination) == reporter.status_path:
            raise OSError("simulated replace failure")
        real_replace(source, destination)

    monkeypatch.setattr(progress_module.os, "replace", fail_status)
    with pytest.raises(OSError, match="simulated"):
        reporter.maybe_emit(_event(1), force=True, now=1.0)
    assert reporter.status_path.read_bytes() == previous
    assert load_progress_status(reporter.status_path) == _event(0)
    assert not tuple(tmp_path.glob(".*.tmp"))
