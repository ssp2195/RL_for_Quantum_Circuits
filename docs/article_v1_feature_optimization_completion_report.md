# Article V1 frontier-feature optimization completion report

## Evidence boundary

This report inventories the implementation committed as `bd251b9` on
`frontier-rl-exper` plus the uncommitted qualification and progress-v2
operability follow-up visible in the shared worktree at final capture. Follow-up
files are identified separately and are not misreported as part of `bd251b9`.
It now includes the authoritative local engineering benchmark and mini-CI
evidence, but it is not a pilot or scientific-results report. Historical
measurements from `f653193` are retained only as the pre-optimization baseline.

Current pilot decision:

```text
engineering_qualification_passed: true
pilot_relaunch_ready: false
```

The post-optimization feature benchmark and a twice-invoked local mini-CI were
completed. Both bind an uncommitted, dirty source snapshot and therefore count
only as engineering evidence. No pilot, publication run, or held-out scheduler
comparison was launched.

## A. Starting state

| Field | Pre-optimization evidence | Optimization completion evidence |
|---|---|---|
| Branch | `frontier-rl` | `frontier-rl-exper` |
| Commit | `f653193ec1fd15b17b948a476a0e89a343cbf062` | `bd251b9dcd7e9e70cb22a292774e5592c59acd09` |
| Worktree | clean at preflight | clean at `bd251b9`; uncommitted qualification/progress-v2 changes at final capture |
| Focused tests | `tests/article_v1`: 240 passed | `tests/article_v1`: **340 passed in 160.65 s** |
| Full tests | 408 passed | **509 passed in 188.84 s** |
| Compileall | passed | **exit 0** |
| Baseline performance | cap 32/64 and isolated batches recorded below | engineering qualification **passed**; complete bundle recorded below |
| New-schema mini-CI | not applicable | local dirty-worktree run **passed twice with byte-stable resume**; two clean committed invocations pending |
| Pilot/publication | not run | **not run** |

The verified post-optimization focused command was:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q `
  tests\article_v1\test_progress_reporting.py `
  tests\article_v1\test_training_resume.py `
  tests\article_v1\test_compact_linear_scoring.py `
  tests\article_v1\test_runner.py `
  --basetemp .pytest-progress-v2
```

Current uncommitted progress-v2 result: `95 passed` in 17.65 seconds. The prior
committed-state progress/resume/runner result was `84 passed` in 18.23 seconds,
and the real runner recovery test was also run alone and passed (`1 passed` in
2.01 seconds). These timings describe tests, not feature-evaluator or pilot
performance. The final broader validation also completed: `compileall` exited
0, the Article V1 directory passed 340 tests in 160.65 seconds, and the full
repository passed 509 tests in 188.84 seconds. These results are bound to the
same uncommitted qualification worktree.

## B. Root-cause confirmation

The algorithm was terminating by certification, frontier exhaustion, or the
external expansion cap. It was not waiting for SARSA convergence. The blocker
was repeated construction of exact frontier-wide policy features. The reference
implementation recomputed resource dominance for each candidate against every
other candidate, making one decision batch `O(F^2 d_r)`. When the frontier grew
with expansion count, the accumulated episode path approached cubic growth.

Historical baseline evidence from `f653193`:

| Expansion cap | Runtime | Peak frontier | Aggregate feature-time share |
|---:|---:|---:|---:|
| 32 | 8.589 s representative | 543 | 84.7% |
| 64 | 61.837 s representative | 1,039 | 94.5% |

| Isolated reference frontier | Batch time |
|---:|---:|
| 534 | 0.790 s |
| 1,021 | 2.330 s |

The requested function-level cProfile percentages were not present in the
supplied baseline and were not regenerated during this documentation task:

| Component | cProfile percentage at `bd251b9` |
|---|---:|
| all-pairs dominance | **pending**; only the aggregate feature share above is available |
| key/resource recomputation | **pending** |
| target metric | **pending** |
| matrix construction | **pending** |
| standardization | **pending** |
| score computation | **pending** |

No component percentage should be inferred by subdividing the aggregate 84.7%
or 94.5% values.

## C. Exact optimization

### Preserved mathematical feature

For open persistent record `v`, the ten candidate coordinates, the exact
frontier statistics, and the 31-D feature remain:

```text
x(v)   = [seven normalized structural coordinates,
          target process infidelity, r_P(v), novelty(v)]
z(v)   = (x(v) - mean) / (std + 1e-8)
b_hat  = (expansion_budget - expansions_completed) / expansion_budget
phi(v) = [1, x(v), z(v), b_hat * x(v)]
```

The production provider keeps the deterministic reference reduction order:

```python
sorted_matrix = np.sort(candidate_matrix, axis=0)
mean = np.mean(sorted_matrix, axis=0, dtype=np.float64)
std = np.std(sorted_matrix, axis=0, ddof=0, dtype=np.float64)
```

### Static row cache

On admission, `ExactArticleFrontierFeatureIndex` caches the persistent ID,
complete canonical semantic key, complete resource tuple, and the first eight
candidate coordinates. Target distance is therefore computed once per admitted
record rather than once per decision. The authoritative archive record remains
the source of truth.

### Resource-group index and exact dominance updates

For resource tuple

```text
rho(v) = (T count, two-qubit count, gate count, wire depths...)
```

the index maintains:

```text
f(r) = number of open records with tuple r
D(r) = sum(f(s) for active s such that s <= r componentwise)
r_P(v; F) = (D(rho(v)) - 1) / max(1, F - 1)
```

Insertion/removal updates `D` by vectorized comparison over active unique
resource groups. Equal-resource multiplicity is included exactly; the candidate
itself is removed by subtracting one. This is not an archive same-key rank and
does not approximate or sample the frontier.

### Novelty deltas

For complete canonical key `k`:

```text
novelty(v) = 1 / sqrt(max(1, generation_count[k]))
```

Generation-count changes are applied only to the active slots for changed full
keys. Resource-feasible generated records continue to count even when the
archive rejects them.

### Compact decision batch and effective linear weights

The production path gathers one contiguous `F x 10` candidate matrix rather
than allocating one 31-D array per open record. Partitioning

```text
theta = [theta_0, theta_x, theta_z, theta_b]
```

gives the exact identity:

```text
w = theta_x + b_hat * theta_b + theta_z / (std + eta)
c = theta_0 - dot(theta_z, mean / (std + eta))
Q(v) = c + dot(x(v), w) = dot(theta, phi(v))
```

Only the selected current/next full rows are materialized for SARSA. Greedy ties
still resolve by lowest persistent record ID, and epsilon exploration retains
the existing random-number consumption order.

### Exact target minimum

The index maintains a lazy heap keyed by cached target distance and persistent
record ID. Stale/removed records are discarded when queried. The minimum and
selected target-distance node therefore remain exact without rescanning or
recomputing target metrics across the complete frontier.

### Frontier synchronization

`Frontier` exposes a monotone engineering revision and deterministic active
persistent IDs. The index applies the exact set difference of removed/added IDs,
refreshes authoritative object associations, and can reconcile against the real
frontier in debug mode. It does not own insertion, Pareto pruning, expansion, or
certification decisions.

For `Delta` membership changes, `G` unique resource groups, resource width
`d_r`, and frontier size `F`, dominance synchronization is approximately
`O(Delta G d_r)`, candidate gathering and vectorized scoring are `O(F d)`, and
the retained deterministic column sorting is `O(d F log F)`. Thus the production
path removes the Python all-record-pairs `O(F^2)` batch, but the complete exact
decision is not claimed to be strict `O(F)` while sorted standardization remains.

### Exact schemas and principal APIs

| Role | Schema |
|---|---|
| Primary mathematical feature | `article-v1-31d` |
| Target-free ablation | `article-v1-no-target-28d` |
| No-standardization ablation | `article-v1-no-z-21d` |
| Reference evaluator | `article-v1-reference-all-pairs-v1` |
| Production evaluator | `article-v1-exact-incremental-v2` |
| Transferable learner checkpoint | `article-v1-transferable-linear-checkpoint-v4` |
| Training progress checkpoint | `article-v1-training-progress-v1` |
| Mid-episode recovery checkpoint | `article-v1-mid-episode-replay-checkpoint-v2` |
| Recovery event journal | `article-v1-training-event-journal-v2` |
| Recovery journal entry | `article-v1-training-expansion-event-v2` |
| Recovery slot manifest | `article-v1-training-checkpoint-manifest-v1` |
| Progress event | `article-v1-progress-event-v2` |
| Progress latest status | `article-v1-progress-status-v2` |
| Feature benchmark bundle | `article-v1-feature-benchmark-v1` |
| Feature benchmark baseline | `article-v1-feature-baseline-v1` |
| Feature cost projection | `article-v1-feature-cost-projection-v1` |
| Focused correctness evidence | `article-v1-feature-correctness-pytest-v1` |
| Production dominance source check | `article-v1-dominance-source-check-v1` |
| Feature benchmark command summary | `article-v1-feature-benchmark-command-v2` |
| Feature benchmark final status | `article-v1-feature-benchmark-status-v1` |
| Replay checkpoint capture | `article-v1-replay-checkpoint-capture-v1` |
| Replay timing evidence | `article-v1-replay-timing-v2` |
| Trusted runtime snapshot | `article-v1-trusted-runtime-snapshot-v1` |
| Raw run/report | `article-v1-raw-run-v4` / `article-v1-publication-report-v4` |

| Component | Current public integration surface |
|---|---|
| Production feature provider | `synchronize_frontier`, `build_compact_batch`, `minimum_target_distance`, `select_target_distance_node`, `instrumentation`, `recent_compact_batch_times_ns` |
| Compact batch | `scores`, `effective_linear_terms`, `features_for_record`, `candidate_for_record`, `materialize_feature_matrix` (debug/reference) |
| Linear policy | `select_from_compact_batch`, `update_from_features` |
| Progress | `ArticleV1ProgressEvent`, `ProgressCadence`, `ArticleV1ProgressReporter.maybe_emit`, strict loaders |
| Recovery | checkpoint dataclasses, `ArticleV1TrainingCheckpointStore`, `ResumeExpectation`, interval/full-state replay and pending-state validators |
| Trainer | immutable `TrainerBoundaryEvent`, `TrainerEpisodeResume`, progress/checkpoint callbacks |
| Runner | `train_article_v1_checkpoint(..., training_checkpoint_dir, progress_reporter, checkpoint_cadence, resume_training, ...)`, canonical replay capture/timing APIs and CLI commands |
| Benchmark | fixed JUnit gate, production AST check, generic/repository adapters, and fail-before-timing `benchmark-features` runner command |

## D. Files changed in optimization commit `bd251b9`

“Preserved” means the scientific object is unchanged. The test-status column in
this commit inventory records the evidence available when `bd251b9` was
prepared; the authoritative later qualification superseding its pending cells
is reported in sections E through J.

| Path | Purpose | Scientific behavior | Test mapping/status |
|---|---|---|---|
| `README.md` | Bind V4 checkpoint/raw schemas and evaluator identity in user documentation. | Preserved; provenance contract tightened. | Documentation; current full suite pending. |
| `docs/article_v1_alignment_matrix.md` | Map compact scoring, evaluator identity, and internal recovery to Article requirements. | Preserved; qualification wording updated. | Documentation review. |
| `docs/article_v1_checkpoint_resume_contract.md` | Define portable internal recovery, replay, and fail-closed store. | Preserved; engineering recovery only. | `test_training_resume.py` passed in the 84-test subset. |
| `docs/article_v1_experiment_protocol.md` | Bind V4 schemas and new-evaluator campaign boundary. | Preserved; old artifacts made incompatible by schema. | Protocol/audit tests mapped; full rerun pending. |
| `docs/article_v1_feature_evaluator_contract.md` | Freeze equations, index algorithm, compact score identity, and benchmark gate boundary. | Preserved by contract. | Feature-index/compact tests exist; clean gate pending. |
| `docs/article_v1_feature_scaling_blocker.md` | Record the nontermination diagnosis and permitted exact correction. | No algorithm change. | Historical baseline only. |
| `docs/article_v1_feature_scaling_preflight.md` | Preserve starting revision, environment, tests, and baseline timing. | No algorithm change. | Historical evidence only. |
| `docs/article_v1_manuscript_patch.md` | Reconcile evaluator complexity and V4 provenance without editing the external manuscript. | Mathematical claims preserved; implementation blocker disclosed. | Documentation review. |
| `docs/article_v1_progress_contract.md` | Define progress event/status durability and cadence. | Engineering-only; no stopping rule. | `test_progress_reporting.py` passed in the 84-test subset. |
| `env/rl_env.py` | Feed authoritative frontier/generation deltas to the provider and expose feature metrics. | Exhaustive expansion, reward, archive, and certification intended preserved. | Runner/environment coverage passed in focused subset; full suite pending. |
| `evaluate.py` | Use compact evaluator/target-minimum paths and record evaluator instrumentation. | Scheduler semantics intended preserved. | Runner tests passed; trace parity gate pending. |
| `experiments/article_v1_feature_benchmark.py` | Repository adapter, staged measurements, qualification, artifacts, and projection. | Engineering diagnostics only; no scientific ledger writes. | `test_feature_benchmark.py` exists; clean rerun and real benchmark pending. |
| `experiments/article_v1_progress.py` | Immutable strict progress schema, cadence, durable JSONL/status, timing. | Engineering-only. | Progress tests passed. |
| `experiments/article_v1_runner.py` | Bind evaluator schema, progress, checkpoint/replay continuation, and CLI cadence flags. | Training/evaluation semantics intended preserved; operability added. | Runner + real recovery test passed; campaign tests pending. |
| `experiments/article_v1_training_checkpoint.py` | Portable JSON recovery schemas, journal, atomic slots, and replay validation. | Engineering-only; explicitly nontransferable. | Recovery tests passed. |
| `experiments/profiles.py` | Add evaluator schema to every experiment profile. | Provenance identity changed; mathematical features unchanged. | Profile/policy clean rerun pending. |
| `reporting/article_v1.py` | Advance raw/report schema to V4 and group by evaluator schema. | Aggregation semantics preserved; cross-evaluator mixing rejected. | Reporting clean rerun pending. |
| `rl/article_features.py` | Split reference all-pairs oracle from exact incremental production provider and keep ablations. | 10-D/31-D formulas intended unchanged. | Feature-index and compact parity tests exist; clean gate pending. |
| `rl/article_frontier_index.py` | Static cache, exact resource groups, novelty deltas, compact batch, target heap, instrumentation. | Accelerator only; frontier/archive remain authoritative. | Incremental/compact tests exist; clean gate pending. |
| `rl/policy.py` | Score/select compact batches and materialize only selected features. | Exact linear score/tie/exploration identity intended preserved. | Compact scoring tests exist; clean gate pending. |
| `search/frontier.py` | Add revision/active-ID hooks and immediately drop dominated open IDs. | Persistent record and archive semantics intended preserved. | `test_search_archive.py` updated; full rerun pending. |
| `tests/article_v1/test_ablations.py` | Update ablation expectations for production evaluator. | Test only. | Current full Article suite pending. |
| `tests/article_v1/test_campaign_audit.py` | Require evaluator schema in campaign identity/audit records. | Test only; prevents mixing. | Current full Article suite pending. |
| `tests/article_v1/test_campaign_projection.py` | Bind evaluator schema in projection fixture. | Test only. | Current full Article suite pending. |
| `tests/article_v1/test_compact_linear_scoring.py` | Exact full-dot, tie, selected-row, immutability, instrumentation tests. | Test only. | Present; clean focused gate pending. |
| `tests/article_v1/test_feature_benchmark.py` | Gate, artifact, projection, adapter, and fail-before-timing tests. | Test only. | Present; clean focused gate pending. |
| `tests/article_v1/test_incremental_feature_index.py` | Exact oracle parity, group lifecycle, novelty, cache, order, reconciliation tests. | Test only. | Present; clean focused gate pending. |
| `tests/article_v1/test_profiles_and_policy.py` | Bind evaluator schema through policy/profile metadata. | Test only. | Current full Article suite pending. |
| `tests/article_v1/test_progress_reporting.py` | Cadence, strict schema, durability, atomic failure, and timing tests. | Test only. | Passed in 84-test subset. |
| `tests/article_v1/test_reporting.py` | Require evaluator identity in raw/report fixtures. | Test only. | Current full Article suite pending. |
| `tests/article_v1/test_training_resume.py` | Store/replay/callback/unit and real runner recovery acceptance. | Test only. | Passed in 84-test subset. |
| `tests/test_search_archive.py` | Verify selected/dominated open-record lifecycle and revisions. | Test only. | Full repository rerun pending. |
| `train.py` | Immutable safe-boundary callbacks and mid-episode continuation state. | SARSA update/action semantics intended preserved. | Callback and recovery tests passed; full regressions pending. |

Post-commit qualification follow-up visible in the shared worktree is not part
of `bd251b9` and remains uncommitted at this capture:

| Path | Purpose | Current evidence |
|---|---|---|
| `experiments/article_v1_feature_benchmark.py` | Execute/parse fixed pytest JUnit evidence, AST-check production dominance, require current-host reference/end-to-end gates, and label the historical environment unknown. | Engineering qualification passed; bundle and profiles complete. |
| `experiments/article_v1_runner.py` | Add config/source-bound, nonempty-destination-refusing, status-manifested `benchmark-features` command wrapper; integrate progress-v2 telemetry, interval recovery digests, and canonical expansion-1,024 capture/replay-timing commands. | Short benchmark/gate/index/compact set: 49 passed; current focused replay/resume/runner result recorded below; real 1,024 run pending. |
| `article_benchmark.py` | Dispatch feature qualification plus replay capture/timing commands from the root CLI. | Root replay dispatch tests and CLI help passed; long capture was not started. |
| `experiments/article_v1_progress.py` | Advance strict event/status schemas to v2 and require evaluator identity. | Included in the 95-test progress-v2 set. |
| `experiments/article_v1_training_checkpoint.py` | Advance the mid-checkpoint/journal/entry schemas to v2 with all-or-none interval state digests and a mandatory verified final boundary. | Focused interval, fail-closed, atomic-store, and exact replay tests passed locally; authoritative combined rerun pending. |
| `experiments/article_v1_replay_timing.py` | Define strict atomic `article-v1-replay-timing-v2` evidence, exact checkpoint/snapshot-byte binding, replay mode, five final digests, and the preregistered 60 s/10% OR gate. | Compact expansion-1,024 replay passed both timing limits. |
| `experiments/article_v1_runtime_snapshot.py` | Add an optional hashed two-slot trusted-local runtime cache bound to the portable journal and exact state digests. | Real interrupt/resume equivalence passed; portable root replay remains the fallback. |
| `rl/article_frontier_index.py` | Retain exact compact-batch count, latest duration, and latest 25 durations. | Included in the 95-test progress-v2 set. |
| `rl/article_features.py` | Expose retained compact-batch duration suffix without another evaluation. | Included in the 95-test progress-v2 set. |
| `tests/article_v1/test_feature_benchmark_gate.py` | Compare successful reference/optimized witness and independent certification. | Passed in the captured combined JUnit gate. |
| `tests/article_v1/test_search_trace_equivalence.py` | Pin real hard-target reference/optimized SARSA state at caps 8/16/32/64. | Passed in the captured combined JUnit gate. |
| `tests/article_v1/test_feature_benchmark.py` | Exercise JUnit/source evidence, config/source integrity, nonempty destination, current-host gates, final status, and abort-before-timing behavior. | Passed within the 49-test focused set. |
| `tests/article_v1/test_progress_reporting.py` | Require v2/status-v2, evaluator provenance, and fail-closed v1 rejection. | Included in the 95-test progress-v2 set. |
| `tests/article_v1/test_training_resume.py` | Verify a sub-cadence interrupt forces one event after the latest safe checkpoint and reports measured timing. | Included in the 95-test progress-v2 set. |
| `tests/article_v1/test_replay_timing.py` | Exercise threshold boundaries, dirty/readiness labeling, checkpoint-byte binding, strict/duplicate/nonfinite rejection, injected clocks, and atomic replace failure. | 14 passed locally. |
| `tests/article_v1/test_compact_linear_scoring.py` | Verify the provider timing count/latest/suffix instrumentation. | Included in the 95-test progress-v2 set. |
| `tests/article_v1/test_runner.py` | Verify exact rolling-25 accounting, cadence-gap attachment, duplicate suppression, episode clock reset, and mini-CI event/status evaluator provenance. | Included in the 95-test progress-v2 set. |
| `docs/article_v1_progress_contract.md`, `docs/article_v1_feature_evaluator_contract.md`, and this report | Reconcile the v2 wire schema and exact telemetry semantics. | Documentation inspection plus focused implementation evidence above. |

## E. Semantic parity

The benchmark runner executed its fixed pytest node IDs, mapped JUnit cases to
each semantic check, and AST-inspected the bounded same-class production
dominance call graph before timing. It required the canonical pilot config,
refused a nonempty destination, bound fresh source snapshots/config digests to
the evidence, rechecked both around timing, and wrote the final artifact
manifest. The authoritative local combined gate passed, including the real
hard-target caps 8, 16, 32, and 64 trace test and an independently certified
success witness. The source and config stayed unchanged throughout the command,
but the bound source snapshot was already dirty; this is engineering parity
evidence, not a clean campaign qualification.

| Required field | Implemented evidence path | Qualification status |
|---|---|---|
| snapshot parity | reference provider vs incremental batch in `test_incremental_feature_index.py` | **passed** in captured JUnit gate |
| score parity | compact scores vs explicit 31-D dot in `test_compact_linear_scoring.py` | **passed** in captured JUnit gate |
| selected-record parity | pinned IDs in real hard-target traces | **passed** at caps 8/16/32/64 |
| SARSA trace parity | exact selected/next features, rewards, TD errors, per-update theta/digests | **passed** at caps 8/16/32/64 |
| final-weight parity | exact final theta and digest in real hard-target traces | **passed** at caps 8/16/32/64 |
| terminal-status parity | exact staged/final status and deterministic counters | **passed** at caps 8/16/32/64 |
| witness/certification parity | successful reference/optimized witness and independent certification test | **passed** in captured JUnit gate |

The successful interrupted/resumed equality in section H is recovery evidence;
it is not a substitute for reference-vs-optimized evaluator parity.

Source inspection shows the intended scientific boundaries remain represented:
the frontier has no scientific size cap, actions are persistent records,
expansion remains exhaustive, the native grammar and certifier are shared, and
the mathematical schemas remain 31/28/21-D. A repeat from a clean committed
revision and the campaign audit remain required before the pilot.

## F. Performance

The authoritative local engineering bundle is
`outputs/article_v1/article-v1-feature-index-v2-local-20260818-a`. The canonical
pilot configuration and source digest remained unchanged before timing and
after artifact generation. The final manifest records
`engineering_qualification_passed: true`, config/source integrity true, and
`pilot_relaunch_ready: false` because its bound worktree is dirty.

| Frontier size | Reference seconds | Optimized sync | Compact batch | Score | Selected row | Speedup | Memory | Groups |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.0040109 | 0.0000425 | 0.0001352 | 0.0000262 | 0.0000139 | 18.4155x | 92,728 B | 5 |
| 64 | 0.0097297 | 0.0000603 | 0.0001454 | 0.0000269 | 0.0000134 | 39.5516x | 111,248 B | 13 |
| 128 | 0.0285425 | 0.0001739 | 0.0002393 | 0.0000318 | 0.0000143 | 62.1435x | 147,848 B | 21 |
| 256 | 0.0946873 | 0.0001997 | 0.0003271 | 0.0000392 | 0.0000158 | 162.749x | 221,144 B | 49 |
| 512 | 0.7762657 | 0.0004558 | 0.0009079 | 0.0000687 | 0.0000200 | 534.471x | 367,608 B | 89 |
| 1,024 | 2.0747946 | 0.0008328 | 0.0012094 | 0.0000643 | 0.0000190 | **976.144x** | 659,808 B | 173 |
| 2,048 | not run by safe-reference policy | 0.0044033 | 0.0052554 | 0.0002893 | 0.0000470 | n/a | 1,377,344 B | 539 |

The complete optimized decision at F=1,024, including synchronization, compact
batch, score, and selected-row materialization, was 0.0021255 seconds. The
compact-batch 512-to-1,024 ratio was 1.3042 and the fitted memory exponent was
0.64119. These pass the respective 2.5x and 1.25 limits.

| End-to-end cap | Optimized runtime | Reference runtime | Speedup | Optimized feature share | Reference feature share | Peak F | Peak groups | Reference parity |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1.4496152 s | 6.7472089 s | **4.65448x** | 0.03397 | 0.86755 | 543 | 205 | passed |
| 64 | 2.4842977 s | 44.6477131 s | **17.97197x** | 0.04982 | 0.94525 | 1,039 | 347 | passed |
| 128 | 5.47783 s | not required | n/a | 0.06733 | n/a | 1,873 | 507 | not run |
| 256 | 16.47306 s | not required | n/a | 0.08299 | n/a | 3,667 | 821 | not run |
| 512 | 58.68196 s | not required | n/a | 0.09608 | n/a | 6,763 | 1,237 | not run |
| 1,024 | 223.12976 s | not required | n/a | 0.10560 | n/a | 13,142 | 2,044 | not run |

The engineering cap-runtime fit has exponent 1.4744. Extrapolating eightfold
beyond the largest measured cap projects 3,515.33798 seconds and peak frontier
F=87,002.64 at cap 8,192. This is a workload-specific engineering
extrapolation, not a confidence interval, feasibility decision, or scientific
scheduler result.

All required benchmark artifacts and profiles are complete: `baseline.json`,
`microbenchmarks.csv`, `end_to_end_scaling.csv`, `profiles/`,
`scaling_report.md`, and `projected_pilot_cost.json`, followed by the
SHA-256-bearing `benchmark_status.json` written last. The historical F=1,021
timing retains explicitly unknown environment metadata and was not substituted
for the current-host F=1,024 gate.

A post-run audit found that this bundle's Markdown report labels parity as
`fail` at caps where no reference was expected, and its file named
`frontier-F1024-optimized` includes the reference all-pairs calculation. The
implementation now emits not-applicable parity for those caps and excludes
reference work from an optimized-named profile. These are artifact semantics,
not timing-gate failures: the CSV measurements and engineering qualification
remain valid, but a corrected final bundle has not yet been rerun.

## G. Progress

Schemas:

```text
event:  article-v1-progress-event-v2
status: article-v1-progress-status-v2
```

Default cadence is the first of 25 completed expansions or 10 monotonic
seconds. Episode end is forced. `--quiet` suppresses stdout only, not durable
files.

Illustrative console line (contract example, not a campaign measurement):

```text
[training:standard] seed=17 target=2/6:train-3q-hard-000 episode=1/2 expansion=256/8192 frontier=3875 peak=3875 rate=4.812/s
```

Output files:

```text
progress.jsonl
status.json
```

The event payload has exactly 26 members:

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

`status.json` has exactly
`progress_status_schema`, `latest_event_digest`, and `latest_event`. JSONL is
flushed and fsynced; status is fsynced and atomically replaced. Loaders reject
partial lines, blanks, duplicate members, non-finite numbers, wrong schemas,
member mismatch, and digest mismatch.

`last_feature_batch_seconds` is the actual latest complete compact-batch
duration. `rolling_feature_batch_seconds` is the exact arithmetic mean of the
latest at most 25 durations retained by the current episode's provider; a
duplicate episode-end observation does not add a second copy. `elapsed_seconds`
and `expansions_per_second` are episode-local and reset only after the forced
episode-end event. A handled interrupt first writes the latest safe checkpoint
and then forces one event, even below the configured ordinary cadence.

Verified in the current uncommitted worktree: cadence, durable/atomic output,
strict loading and v1 rejection, evaluator provenance, quiet mode, separate
reporter timing, exact rolling-25 accounting (including cadence gaps and
duplicate boundaries), episode clock reset, and a real bounded forced interrupt
with positive measured timings plus mini-CI event/status evaluator provenance
passed in the 95-test progress-v2 subset.

The twice-invoked local mini-CI emitted
`article-v1-progress-event-v2`/`article-v1-progress-status-v2`, bound evaluator
`article-v1-exact-incremental-v2`, and reported positive measured timing. A
distinct target-end event in addition to the final episode event and clean pilot
campaign samples remain pending.

## H. Checkpoint and recovery

Default mid-episode cadence is the first of 64 completed expansions or 60
monotonic seconds. The runner also writes an episode-final progress checkpoint
after every episode, a latest progress checkpoint after every target, and a
latest mid-episode checkpoint at a handled safe-boundary interrupt.

Schemas:

```text
training progress: article-v1-training-progress-v1
mid episode:       article-v1-mid-episode-replay-checkpoint-v2
event journal:     article-v1-training-event-journal-v2
journal entry:     article-v1-training-expansion-event-v2
slot manifest:     article-v1-training-checkpoint-manifest-v1
transferable:      article-v1-transferable-linear-checkpoint-v4
replay capture:    article-v1-replay-checkpoint-capture-v1
replay timing:     article-v1-replay-timing-v2
runtime snapshot:  article-v1-trusted-runtime-snapshot-v1
```

Internal artifacts serialize
`checkpoint_kind`, `safe_sarsa_boundary=true`, and
`transferable_for_evaluation=false`. `TrainingProgressCheckpoint` remains
nontransferable even with `training_complete=true`.

`ArticleV1CheckpointProvenance` has exactly:

```text
source_commit_sha, source_worktree_digest
config_digest, corpus_digest, profile_digest
target_id, target_fingerprint
feature_schema_version, feature_evaluator_schema_version
reward_schema_version, certifier_schema_version
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

The target/episode progress payload has exactly:

```text
checkpoint_schema, checkpoint_kind
safe_sarsa_boundary, training_complete, transferable_for_evaluation
provenance
training_seed, target_cursor, target_count
episode_cursor, episodes_per_target
theta, policy_weight_digest, epsilon, policy_rng_state
training_history, completed_target_ids, effective_budgets
```

The journal has `event_journal_schema`, `base_expansion`, and `entries`. Each
entry has `journal_entry_schema`, expansion/selected IDs, selected-feature
digest, reward/terminal flags, frontier revision,
`state_digest_verified`, nullable frontier/archive/generation digests,
post-update weight digest, and pending next ID. Timing is forbidden. Every
entry validates the deterministic transition core. The three full-state digests
are all present only at configured checkpoint/interrupt boundaries and are all
null otherwise; the final entry of every published mid-episode checkpoint must
be a verified full-state boundary.

The store uses manifest-last fsync/replace slots:

```text
latest.json / latest.manifest.json
previous.json / previous.manifest.json
episode-final.json / episode-final.manifest.json
```

Resume first validates every provenance/schema/seed/episode/budget/dimension
binding, then replays recorded selected persistent IDs from a fresh root through
normal exhaustive transitions without ranking or epsilon draws. Every observed
transition core is compared by exact value/type; full-state digests are
recomputed at every marked interval and at the final pending state.
Only after validation are theta, epsilon, RNG states, reward aggregates, and the
stored pending record/feature restored.

Verified recovery evidence:

- A deterministic fixture interrupted after four expansions matched the
  uninterrupted selected trace, rewards, TD errors, final weights/digest,
  terminal-state surrogate, deterministic counters, and state digests.
- The real runner test used one bounded two-qubit X target, one episode, cap 4,
  and an interrupt after expansion 1. The stored latest artifact was a
  mid-episode checkpoint. Resumption matched uninterrupted final weights,
  transferable weight digest, episode history, and deterministic search metrics.
- The committed-state recovery, progress, and runner subset passed 84 tests;
  the current progress-v2/resume/compact/runner subset passed 95 tests.
- `article-v1-feature-index-v2-mini-ci-local-20260818-a` passed twice. The first
  invocation trained and appended nine records; the second reported training
  false, appended zero, and skipped all nine existing records. The checkpoint
  bytes remained SHA-256
  `8bfaa8a22ac4c54faccc85cb1fa2fa19aa6499d71a3de0bd64325579250abf49`
  and the raw ledger bytes remained SHA-256
  `f8d49cb1d67e2b3c40f3a5ececaf861bddda755f83e03566c038af4a6346214d`.
- A post-benchmark operability profile exposed a distinct recovery-path cost:
  the v1 callback serialized active IDs, the complete archive, and all
  generation counts on every expansion. With real checkpoint callbacks, cap 32
  took 6.8281 s with 32 digest calls consuming 5.1456 s (75.36% of wall), and
  cap 64 took 19.8361 s with 64 calls consuming 17.7113 s (89.29%). The
  feature-only cap-1,024 benchmark did not include this callback and therefore
  did not qualify the operable recovery path.
- The v2 interval journal removes that cumulative per-expansion serialization.
  A current uncommitted short check with a controlled safe-boundary interrupt
  and the 8,192 scientific horizon took 1.9735 s at expansion 32 and 2.5474 s
  at expansion 64. Each run made exactly one full-state digest call, exactly one
  atomic latest save, retained 32/64 journal entries, and marked exactly one
  full-state boundary. Digest/save components were respectively 0.3881/0.0106 s
  and 0.8140/0.0299 s. These are provisional engineering diagnostics, not the
  authoritative cap-1,024 operable qualification.
- With no recovery store and no controlled interrupt, the runner now attaches
  no checkpoint callback and builds no recovery journal/state digest. Returned
  training history separates `checkpoint_callback_time_ns`,
  `checkpoint_state_digest_time_ns`, and `checkpoint_io_time_ns`.
- The canonical `capture-replay-checkpoint` command now captures only the
  frozen pilot train/hard/3q target at expansion 1,024, and
  `measure-replay-timing` atomically emits strict source/config/target/evaluator/
  checkpoint/journal/snapshot-bound `article-v1-replay-timing-v2` evidence. An expected
  interrupt is accepted only after the exact valid mid checkpoint is loaded.
- Portable root replay of the real 1,024-expansion checkpoint took **360.3569
  s**, exceeding both the 60 s limit and the 351.5338 s fractional limit. The
  required trusted-local snapshot cache was therefore implemented while the
  full portable journal remained authoritative. The compact replay took
  **11.3827 s** from an 18,308,485-byte expansion-1,024 snapshot with zero
  delta entries, passed both limits, and reported `compaction_required=false`.

## I. Pilot decision

```text
engineering_qualification_passed: true
pilot_relaunch_ready: false
```

The local correctness, implementation, scaling, projection, progress, recovery,
and byte-stable mini-CI evidence is sufficient for engineering qualification.
It is not sufficient for campaign relaunch because the benchmark and mini-CI
were produced from an uncommitted dirty worktree. The next evidence must be two
mini-CI invocations from the reviewed clean committed revision, followed by the
pilot. No cap/stratum amendment has been authorized or preregistered by this
report.

## J. Tests

| Required test evidence | Current local engineering status |
|---|---|
| `compileall` | **passed**, exit 0 |
| focused qualification-module `py_compile` | **passed** in uncommitted follow-up |
| focused Article V1 full directory | **362 passed in 118.61 s** |
| full repository | **509 passed in 188.84 s** |
| committed-state progress/resume/runner subset | **84 passed** |
| current progress-v2/resume/compact/runner subset | **95 passed in 17.65 s** |
| current interval-recovery/replay/runner/campaign-audit/progress subset | **195 passed in 67.87 s** |
| real runner interrupt/resume test | **passed**; current version also verifies forced v2 progress, evaluator identity, positive actual timing, and latest safe checkpoint path |
| new feature-index and compact linear-score tests | **passed** in captured combined correctness gate |
| short benchmark/gate/index/compact focused set | **49 passed** before authoritative run |
| cap-8/16/32/64 reference/optimized trace test | **passed** in authoritative captured JUnit gate |
| authoritative combined correctness gate | **passed**; evidence retained under benchmark `profiles/` |
| local new-schema mini-CI twice | **passed with byte-stable checkpoint/raw ledger**, dirty-worktree engineering evidence only |
| clean committed new-schema mini-CI twice | **pending** |
| pilot/publication | **not run** |
| `git diff --check` | **passed** at documentation capture (line-ending warnings only) |

## K. Remaining limitations and relaunch gaps

- Full-frontier exact ranking remains at least linear in `F`; deterministic
  sorted standardization currently adds an `O(F log F)` component per column.
- Exact circuit synthesis remains exponential in gate-count/resource budget.
- The optimization removes the unintended all-pairs/cumulative cubic
  implementation path; it does not make exact synthesis polynomial.
- A lower equal cap may still be necessary, but only after measured projection
  and a dated preregistration made without held-out scheduler inspection.
- The low-level benchmark harness still accepts caller-supplied correctness and
  source-check data. The high-level `benchmark-features` command now executes
  fixed tests, parses JUnit evidence, AST-inspects the production methods,
  enforces canonical config/fresh source integrity and an empty destination,
  stops before timing on failure, and writes a final status manifest. It has
  not yet completed a qualification run on a clean committed revision.
- The required `F=32...2048` and cap `32...1024` engineering bundle is
  complete, including profiles, but must be repeated after commit if clean
  provenance is required for relaunch review.
- Canonical expansion-1,024 capture and compact replay are complete as local
  dirty-worktree engineering evidence: 11.3827 s, below both limits. A clean
  committed evidence rerun remains part of relaunch review.
- The progress-v2 runner now reports actual compact-batch timing, an exact
  rolling-25 suffix, episode-local elapsed/rate, evaluator identity, and a
  checkpoint-first forced interrupt event. It does not emit a second distinct
  target-end phase in addition to the forced final episode boundary.
- The Article suite, full repository suite, and `compileall` passed locally, but
  the worktree was uncommitted and dirty.
- The new-schema mini-CI passed twice with byte-stable resume locally; two clean
  committed invocations remain mandatory.
- The cap-8,192 estimate is an engineering extrapolation from cap 1,024 and does
  not itself establish campaign feasibility or authorize a lower equal cap.
- The pilot and publication campaigns remain unrun.

## L. Git state

Committed starting state before the uncommitted qualification/operability
follow-up:

```text
branch: frontier-rl-exper
commit: bd251b9dcd7e9e70cb22a292774e5592c59acd09
git status --short: empty
```

The authoritative benchmark and mini-CI bound this source state:

```text
branch: frontier-rl-exper
commit: bd251b9dcd7e9e70cb22a292774e5592c59acd09
dirty_worktree: true
source_worktree_digest: sha256:d3e5776338636c4e08ce4bff883b21d80a88aff45e374b3d09efd24e93206db2
```

The source digest remained unchanged at the benchmark's initial,
before-timing, and after-artifact snapshots; this protects run integrity but
does not make the revision clean. `git diff --check` passed after the
documentation update. No commit or push was performed by this task.
