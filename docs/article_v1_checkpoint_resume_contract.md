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
mid-episode recovery:   article-v1-mid-episode-replay-checkpoint-v1
event journal:          article-v1-training-event-journal-v1
journal entry:          article-v1-training-expansion-event-v1
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
can be dispatched to evaluation code.

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

Write this layer after every episode, after every target, on handled interrupt,
and before clean process exit. Episode-final writes use the dedicated
`episode-final` slot as well as the rotating latest slot.

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
frontier active-ID digest
archive digest
generation-count digest
policy-weight digest after update
pending next persistent record ID
```

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

`checkpoint_io_time_ns` and each `CheckpointWriteReceipt.elapsed_ns` are separate
engineering timings. They must not be added to feature, ranking, or environment
step timers, although wall time includes them.

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
   replay_step)`.
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
test and add a bounded real Article V1 target recovery test when the trainer and
environment replay hooks are connected.

## Optional compaction

Measure replay time before adding binary caches. If replaying 1,024 expansions
exceeds 60 seconds or 10% of projected full-episode time, a portable compact
environment snapshot plus shorter journal may be added. Any trusted local binary
cache remains optional; a portable manifest and deterministic replay fallback are
mandatory.
