# Article V1 progress-reporting contract

## Purpose and research boundary

Long exact-search episodes must visibly demonstrate forward progress. Progress
reporting is an operability mechanism, not a scientific stopping rule or
hyperparameter. Changing its cadence must not change the frontier, selected
record IDs, generated children, SARSA updates, termination, certification, or
raw scientific results.

The portable event schema is:

```text
article-v1-progress-event-v2
```

The atomic latest-status schema is:

```text
article-v1-progress-status-v2
```

The implementation is in `experiments/article_v1_progress.py`.

## Immutable event

`ArticleV1ProgressEvent` contains:

```text
timestamp_utc
run_id
phase
feature_evaluator_schema_version
training_seed
target_index, target_count, target_id
split, stratum, num_qubits
episode_index, episode_count
expansion, expansion_cap
frontier_size, frontier_peak
archive_records, active_archive_records
unique_resource_groups
last_feature_batch_seconds
rolling_feature_batch_seconds
elapsed_seconds
expansions_per_second
checkpoint_path
```

On the wire, the schema field is named `progress_event_schema`, not
`schema_version`. The event object therefore has exactly these 26 JSON members:

```text
progress_event_schema
timestamp_utc, run_id, phase, feature_evaluator_schema_version, training_seed
target_index, target_count, target_id, split, stratum, num_qubits
episode_index, episode_count, expansion, expansion_cap
frontier_size, frontier_peak, archive_records, active_archive_records
unique_resource_groups
last_feature_batch_seconds, rolling_feature_batch_seconds
elapsed_seconds, expansions_per_second, checkpoint_path
```

`status.json` has exactly:

```text
progress_status_schema, latest_event_digest, latest_event
```

Target and episode indexes are zero-based in the JSON event. The concise console
line presents them as one-based counts. Integers reject booleans, all timing
values must be finite and nonnegative, the timestamp must be ISO-8601 UTC with a
trailing `Z`, and unknown or missing JSON members are rejected.

The timestamp and all timing/rate fields are nondeterministic engineering
observations. They must never be included in replay equality checks, policy
features, rewards, run keys, target fingerprints, or scientific termination.
The structural search counters and identities reflect deterministic state but
are informational here; the recovery journal separately binds the authoritative
deterministic replay fields.

## Cadence

The default is whichever happens first:

```text
25 completed expansions
10 monotonic seconds
```

Configure it with `ProgressCadence`. Either trigger may be disabled by setting it
to `None`, but at least one trigger is required. Use `force=True` for useful
boundaries. The current runner forces episode end and a handled interrupt. At a
handled interrupt it saves the latest safe checkpoint first and then emits the
forced event, so `checkpoint_path` names the recoverable artifact. Call
`reset_cadence(expansion=0)` when a new episode resets its expansion counter.
Cadence decisions use a monotonic clock and must not inspect or alter scientific
state.

Suggested CLI mapping:

```text
--progress-every-expansions -> ProgressCadence.every_expansions
--progress-every-seconds    -> ProgressCadence.every_seconds
--quiet                     -> suppress concise stdout only
```

`--quiet` does not weaken the durable file contract. A caller that intentionally
wants no progress artifacts should omit the callback rather than reinterpret
quiet as a scientific mode.

## Outputs and durability

`ArticleV1ProgressReporter(run_directory, ...)` writes:

```text
run_directory/
├── progress.jsonl
└── status.json
```

Each JSONL event is canonical strict JSON, ends in a newline, is flushed, and is
`fsync`ed. `status.json` contains the latest event and its SHA-256 digest. It is
written to a same-directory temporary file, flushed, `fsync`ed, and atomically
replaced. On platforms supporting directory `fsync`, the parent directory is
also synchronized.

The JSONL stream is the history; `status.json` is an atomic latest-value
projection. If the process fails after committing a JSONL event but before the
status replacement, the history can be one event ahead of status. No completed
event is silently removed. The next successful emission catches status up.

`load_progress_events` and `load_progress_status` fail closed on:

- missing final newline;
- blank JSONL records;
- invalid UTF-8 or JSON;
- duplicate object members;
- `NaN` or infinity;
- wrong schema;
- missing or extra members;
- invalid field types/ranges;
- status-event digest mismatch.

They do not repair or skip malformed records.

## Callback integration

`Trainer` accepts keyword-only `progress_callback` and `checkpoint_callback`
hooks. It emits an immutable `TrainerBoundaryEvent` after every completed
transition, once the next behavior record/feature is frozen and the SARSA update
is complete. It also emits a forced `episode_end` boundary after advancing
epsilon for the next episode. The checkpoint callback runs first so a progress
adapter may report the newly committed checkpoint path. Callback exceptions
abort training rather than claiming that required output succeeded.

At each callback, the runner adapts the immutable trainer boundary and run/target
context into `ArticleV1ProgressEvent`, then invokes:

```python
emitted = reporter.maybe_emit(event, force=force_boundary)
```

Progress construction must not trigger another feature batch, policy ranking,
environment transition, canonicalization, or certification. The optimized
feature index should expose the frontier/group counters needed by the event.

`TrainerBoundaryEvent` includes the selected and pending record IDs, frozen
feature rows, transition reward/flags/TD error, post-update weights/digest,
portable RNG snapshots, frontier revision/active IDs, and scalar search metrics.
The runner should use `boundary == "episode_end"` to force progress and reset the
cadence before the next episode.

`ArticleV1ProgressReporter.progress_reporting_time_ns` is a separate cumulative
engineering timer. It includes JSONL/status I/O and optional console output. It
must be reported separately from:

```text
feature_time_ns
ranking_time_ns
environment_step_time_ns
checkpoint_io_time_ns
```

End-to-end wall time may include all of them.

The trainer additionally exposes `progress_callback_time_ns` and
`checkpoint_callback_time_ns`. These measure adapter callback wall time and must
not be added to the reporter/store counters as if they were disjoint; report the
layer being measured explicitly.

## Current runner integration

The committed starting point `bd251b9` provided the durable reporter and CLI
cadence. The current uncommitted qualification worktree advances the portable
event/status contract to v2 and keeps the same `pilot` and `train` flags:

```text
--progress-every-expansions (default 25)
--progress-every-seconds    (default 10.0)
--quiet
```

The runner emits after completed trainer boundaries and forces an episode-end
event. It records current frontier/archive/group counters, evaluator identity,
episode-local elapsed time/rate, and the last committed checkpoint path. The
feature index records every completed compact-batch duration and retains the
exact latest 25. `last_feature_batch_seconds` is the most recent complete
compact-batch duration; `rolling_feature_batch_seconds` is the arithmetic mean
of the latest `min(25, completed batches)` durations in the current episode.
Re-observing the same provider count at episode end does not double-count the
last batch. If a callback first observes a gap larger than the retained history,
the retained 25 values are still the exact requested rolling-25 suffix.

`elapsed_seconds` starts at the current episode boundary and
`expansions_per_second` is the current episode expansion count divided by that
elapsed time. Both reset only after the forced final event for an episode has
been published. A handled interrupt writes a latest safe checkpoint and then
forces one progress event even when neither ordinary cadence trigger is due.

The focused v2 progress/resume/compact/runner command passed 95 tests in 17.65
seconds in the current uncommitted worktree. It covers cadence, durable JSONL,
atomic status, strict v1 rejection, evaluator provenance, quiet mode, separate
reporter timing, exact rolling timing without duplicate episode-end samples,
episode clock reset, mini-CI event/status evaluator provenance, and a real
bounded forced-interrupt event with positive measured feature timings and a
valid checkpoint path. Benchmark,
clean-schema mini-CI, pilot, and publication campaign sample artifacts remain
pending. The runner does not emit a second target-specific event in addition to
the forced final episode event.

## Example console line

```text
[training] seed=17 target=2/6:train-3q-hard-000 episode=1/2 expansion=256/8192 frontier=3875 peak=3875 rate=4.812/s
```

The line is intentionally concise and derived from the same durable event. No
trainer-internal `print()` output is required.
