# Article V1 intermediate-checkpoint and deterministic-resume contract

## Purpose and boundary

Article V1 training may contain long exact-search episodes. Waiting until all
targets for a learner seed finish before saving state is operationally unsafe.
The recovery system therefore has two internal layers while preserving normal
exhaustive search and SARSA semantics.

These files are recovery artifacts, not transferable evaluation checkpoints.
They never authorize evaluation. After all training targets finish, the runner
must still construct and validate the existing final scope-bound Article V1
checkpoint.

The implementation is in
`experiments/article_v1_training_checkpoint.py`. It uses portable strict JSON;
raw pickle is neither written nor accepted.

## Schemas

```text
episode/target progress: article-v1-training-progress-v1
mid-episode recovery:   article-v1-mid-episode-replay-checkpoint-v2
event journal:          article-v1-training-event-journal-v2
journal entry:          article-v1-training-expansion-event-v2
slot manifest:          article-v1-training-checkpoint-manifest-v1
```

Both checkpoint schemas explicitly serialize:

```text
checkpoint_kind = internal-...
safe_sarsa_boundary = true
transferable_for_evaluation = false
```

`TrainingProgressCheckpoint` remains nontransferable even when its target cursor
shows that every training target completed. `require_evaluation_eligible()` and
`reject_internal_checkpoint_for_evaluation()` fail before an internal checkpoint
can be accepted as a transferable checkpoint. The campaign evaluator continues
to load only `article-v1-transferable-linear-checkpoint-v4` artifacts from the
separate `checkpoints/` directory; internal recovery artifacts live under
`training_state/`.

## Exact wire members

The event journal object has exactly:

```text
event_journal_schema, base_expansion, entries
```

Each journal entry has exactly:

```text
journal_entry_schema
expansion_index, selected_record_id, selected_feature_digest
reward, terminated, truncated
frontier_revision, state_digest_verified
frontier_active_ids_digest
archive_digest, generation_count_digest
policy_weight_digest_after_update, pending_next_record_id
```

The mid-episode payload has exactly:

```text
checkpoint_schema, checkpoint_kind
safe_sarsa_boundary, training_complete, transferable_for_evaluation
provenance
training_seed, episode_index, episode_count
expansion_count, expansion_cap
journal, journal_digest
episode_initial_theta, episode_initial_weight_digest
theta, policy_weight_digest, epsilon
policy_rng_state, environment_rng_state
pending_next_record_id, pending_next_feature_row, pending_next_feature_digest
total_reward, training_aggregates, search_metrics
frontier_revision, frontier_active_ids_digest
archive_digest, generation_count_digest
```

The episode/target progress payload has exactly:

```text
checkpoint_schema, checkpoint_kind
safe_sarsa_boundary, training_complete, transferable_for_evaluation
provenance
training_seed, target_cursor, target_count
episode_cursor, episodes_per_target
theta, policy_weight_digest, epsilon, policy_rng_state
training_history, completed_target_ids, effective_budgets
```

`provenance` has exactly the eleven members listed in the next section. A slot
manifest has exactly:

```text
checkpoint_manifest_schema, slot, checkpoint_schema
checkpoint_sha256, checkpoint_byte_length
```

## Required provenance

`ArticleV1CheckpointProvenance` binds:

```text
source commit SHA
source worktree digest
config digest
corpus digest
profile digest
target ID and target fingerprint
mathematical feature schema
feature-evaluator schema
reward schema
certifier schema
```

Digest fields use canonical `sha256:<64 lowercase hex>` values. Resume also binds
the training seed, episode index/count, expansion cap, feature/weight dimension,
checkpoint schema, and event-journal digest. A mismatch is an error; there is no
best-effort migration, truncation, resizing, or schema reinterpretation.

## Layer 1: episode/target progress

`TrainingProgressCheckpoint` stores:

```text
learner seed
target cursor/count
next episode cursor and episodes per target
policy weights and digest
epsilon
portable policy RNG state
training history
completed target IDs
effective per-target expansion budgets
full provenance
```

The target cursor equals the number of fully completed target IDs. The episode
cursor identifies the next episode and resets to zero when a target (including
the last target) completes.

Write this layer after every episode and target and before clean process exit.
Episode-final writes use the dedicated `episode-final` slot as well as the
rotating latest slot. A handled safe-boundary interrupt instead writes the
mid-episode layer below to `latest`, because it must retain the journal and
pending behavior action needed for exact continuation.

## Layer 2: safe mid-episode boundary

Construct `MidEpisodeCheckpoint` only when all four conditions hold:

1. the previous environment transition and SARSA update completed;
2. the next behavior record was selected;
3. its exact frozen feature row is available;
4. no environment/frontier/archive mutation is in progress.

It stores:

```text
training seed, target identity, episode and expansion cursor/cap
deterministic selected-record event journal and digest
current theta and digest
epsilon
portable policy and environment RNG states
pending next persistent record ID and frozen feature row/digest
total reward and TD-error/training aggregates
deterministic search metrics
frontier revision and active-ID digest
archive digest
generation-count digest
full provenance
```

A mid-episode checkpoint must be nonterminal and strictly below its expansion
cap. Its last journal entry must agree with the checkpoint's pending record,
frontier/archive/generation digests, revision, and current policy-weight digest.
That final entry must set `state_digest_verified=true` and contain all three
full-state digests; a partial or unmarked final state fails closed.
The checkpoint seals a copy of its journal so subsequent trainer mutations cannot
alter already-bound bytes.

`Trainer(checkpoint_callback=...)` supplies the integration boundary. Its
immutable `TrainerBoundaryEvent` carries the exact selected and pending record
IDs/features, reward/flags/TD error, post-update weights and RNG state, frontier
revision/active IDs, and deterministic scalar search metrics. For
`boundary == "expansion"`, an adapter appends the journal entry and writes a
mid-episode checkpoint only when the cadence is due and the event is
nonterminal. For `boundary == "episode_end"`, it writes
`TrainingProgressCheckpoint` through `save_episode_final`. The runner writes a
target boundary through `save_latest` after `Trainer.train` returns.

## Deterministic event journal

Append one `ArticleV1JournalEntry` after each completed expansion/update. Entries
are sequential and contain no timing:

```text
1-based expansion index
selected persistent record ID
selected frozen-feature digest
reward
terminated and truncated flags
frontier revision
state_digest_verified
frontier active-ID digest or null
archive digest or null
generation-count digest or null
policy-weight digest after update
pending next persistent record ID
```

Every entry binds the deterministic transition core: selected record/feature,
reward and terminal flags, frontier revision, updated policy digest, and pending
next record. Full frontier/archive/generation serialization is intentionally
interval-based: the three state digests are either all present with
`state_digest_verified=true` or all null with it false. They are mandatory at
each published mid-episode checkpoint boundary, including a handled interrupt.
This avoids serializing the complete growing archive after every expansion while
preserving exact replay checks at configured intervals and at the final pending
state.

A nonterminal entry requires a pending next record. A terminal/truncated entry
requires it to be null and is the last possible journal entry. Journal and
feature/weight digests use domain-separated SHA-256 over canonical portable JSON.

## Cadence

The default mid-episode checkpoint cadence is whichever happens first:

```text
64 completed expansions
60 monotonic seconds
```

`CheckpointCadenceGate` implements this engineering-only cadence. Callers also
force checkpoints at episode end, target end, handled interrupt, and clean exit.
Cadence must not influence search termination or policy behavior.

The runner does not attach a recovery callback or construct journal entries when
no checkpoint store and no controlled interrupt are requested. With recovery
enabled, it computes full-state digests only when a nonterminal cadence or
interrupt boundary is due. A resume initializes the cadence gate from the
recovered expansion, so a checkpoint restored at expansion 64 is next due at
128, not 65.

## Atomic slots and crash behavior

`ArticleV1TrainingCheckpointStore(directory)` maintains:

```text
latest.json                  latest.manifest.json
previous.json                previous.manifest.json
episode-final.json           episode-final.manifest.json
```

Before replacing latest, a valid old latest is atomically copied to previous.
For each slot:

1. canonical checkpoint bytes are written to a same-directory temporary file;
2. the file is flushed and `fsync`ed;
3. it atomically replaces the slot payload;
4. the manifest containing schema, byte length, and whole-file SHA-256 is written
   through the same sequence **last**.

A payload/manifest pair must both exist and agree. Partial files, missing final
newlines, duplicate JSON members, non-finite numbers, unknown schemas, extra or
missing members, invalid types, digest mismatches, and incoherent cursors fail
closed. `load_latest_or_previous()` may fall back to an independently valid
previous slot after an interrupted latest publication; it never weakens
validation of either slot.

`checkpoint_callback_time_ns`, `checkpoint_state_digest_time_ns`,
`checkpoint_io_time_ns`, and each `CheckpointWriteReceipt.elapsed_ns` are
separate engineering timings. They must not be added to feature, ranking, or
environment step timers, although wall time includes them.

## Resume by deterministic replay

Resume follows this order:

1. Build a `ResumeExpectation` from the current frozen run and call
   `store.load_for_resume(expectation)`.
2. Construct a fresh environment and reset it to the root.
3. For every journal entry, apply its recorded selected persistent record ID.
   Do not call policy ranking or epsilon exploration.
4. Keep normal exhaustive child generation, symbolic updates, canonicalization,
   archive insertion/pruning, exact feature-index deltas, SARSA update, and
   certification active.
5. Construct a `ReplayObservation` after each transition and pass it to
   `validate_replay_observation`, or use `replay_and_validate(journal,
   replay_step)`. Validate the deterministic transition core on every entry and
   recompute/compare the three full-state digests exactly on entries marked
   `state_digest_verified=true`.
6. After the journal, call `validate_pending_resume_state` with current open IDs,
   the recomputed pending feature row, revision, and frontier/archive/generation
   digests.
7. Only after every replay check passes, restore theta, epsilon, policy RNG,
   environment RNG, total reward, and training aggregates from the checkpoint.
8. Continue using the stored pending record and stored frozen feature row. Do not
   rank again and do not consume another random draw.

Replay mode may suppress progress printing, redundant observation materialization,
and nonessential timing. It must never skip certification, canonicalization,
archive logic, exhaustive expansion, or exact feature-index synchronization.

## Acceptance evidence

`tests/article_v1/test_training_resume.py` includes a bounded deterministic
interruption fixture. It compares uninterrupted training with an interrupt after
four expansions, strict save/load, replay from root without policy selection,
state validation, and continuation. It requires equality of:

```text
selected persistent-record trace
rewards
TD errors
final weights and weight digest
terminal result/witness surrogate
deterministic counters
frontier/archive/generation digests
```

Timing is intentionally excluded. Full environment integration must retain this
test.

The same test module now also contains the real runner/environment acceptance
test `test_real_article_training_interrupt_replays_and_resumes_identically`.
It trains a bounded two-qubit X target for one episode at expansion cap 4,
interrupts after expansion 1, verifies that `latest` is a
`MidEpisodeCheckpoint`, resumes through `train_article_v1_checkpoint`, and
requires equality with the uninterrupted run for final weights, transferable
weight digest, serialized episode history, and deterministic search metrics.
This test passed as part of the focused 84-test progress/resume/runner run at
commit `bd251b9`. In the current uncommitted progress-v2 worktree it passed
again inside a 95-test subset and additionally verified that the interrupt
forces exactly one v2 progress event below ordinary cadence, after the latest
safe checkpoint path is known. The current interval-digest/replay-timing
follow-up also requires the interrupt checkpoint's final journal entry to carry
all three full-state digests, checks target-final and clean-exit progress slots,
and keeps the exact uninterrupted/resumed equality assertion. Focused final
checkpoint/replay/runner/campaign-audit/progress validation passed 195 tests in
67.87 seconds in the current uncommitted worktree. The clean authoritative
suite is still pending.

## Measured replay and optional trusted-local compaction

The initial portable replay of 1,024 expansions took 360.3569 seconds. That
exceeded both the 60-second limit and 10% of the projected 3,515.338-second
full episode (351.5338 seconds), so the preregistered OR gate required a compact
recovery cache.

The implementation therefore writes an optional, hashed, two-slot
`article-v1-trusted-runtime-snapshot-v1` cache at verified checkpoint
boundaries. Its strict JSON manifest binds the payload bytes, portable journal
prefix, provenance, policy/RNG state, pending action, visit counts, and full
frontier/archive/generation digests. Pickle is confined to this explicitly
trusted local cache; corrupt, torn, incompatible, or absent cache slots fall
back to the authoritative portable JSON journal. The portable checkpoint and
complete deterministic root-replay path are retained.

The measurement path is now explicit:

```text
python article_benchmark.py capture-replay-checkpoint \
  --output-root outputs/article_v1 \
  --run-id article-v1-replay-capture-1024 --quiet

python article_benchmark.py measure-replay-timing \
  --checkpoint outputs/article_v1/article-v1-replay-capture-1024/training_state/latest.json \
  --output outputs/article_v1/article-v1-replay-capture-1024/replay_timing.json \
  --projected-full-episode-seconds 3515.337979379539
```

The capture command resolves only the checked-in pilot config and its fixed
train/hard/3q target, preserves the 8,192 scientific horizon, and stops only at
the safe expansion-1,024 boundary. It treats the expected interrupt as success
only after strict slot-manifest, checkpoint schema, provenance, journal length,
and final-digest validation. `measure-replay-timing` reloads that exact slot,
loads the newest compatible runtime snapshot and replays only its journal
suffix (zero entries for the measured expansion-1,024 boundary), validates the
final pending state and five scientific digests, and atomically writes
`article-v1-replay-timing-v2` evidence. The evidence
binds source commit/worktree, config, target/fingerprint, evaluator schema,
checkpoint bytes/schema, journal digest/count, elapsed time, both thresholds,
and five validated final digests. Dirty-source measurements may be valid
engineering timing but always set `pilot_relaunch_ready=false`.

The validated local compact replay took **11.3827 seconds**, versus 360.3569
seconds for portable root replay (about 31.7x faster). It is below both limits,
so `compaction_required=false`; no further environment-state compaction is
required. The measured snapshot was 18,308,485 bytes, based at expansion 1,024,
with zero delta-journal entries. This is dirty-worktree engineering evidence,
not pilot authorization.
