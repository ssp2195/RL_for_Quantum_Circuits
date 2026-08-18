# Article V1 exact feature-evaluator contract

## Version separation

The mathematical feature schema and its evaluator implementation have independent versions:

```text
primary feature schema: article-v1-31d
target-free ablation: article-v1-no-target-28d
no-standardization ablation: article-v1-no-z-21d
reference evaluator: article-v1-reference-all-pairs-v1
production evaluator: article-v1-exact-incremental-v2
```

An evaluator version change does not authorize a feature equation change. Checkpoints, progress records, run manifests, raw records, and resume keys must serialize both versions and fail closed on a mismatch.

## Immutable scientific equation

For every open persistent record `v`, the ordered candidate vector remains:

```text
x(v) = [
  normalized T count,
  normalized two-qubit count,
  normalized gate count,
  normalized depth,
  normalized rotation count,
  normalized anticommuting-pair count,
  normalized mean Pauli weight,
  target process infidelity,
  exact frontier resource-dominance fraction,
  exact archive novelty,
]
```

For the complete current frontier, columns are sorted before the existing population mean and standard deviation reduction. With fixed `eta=1e-8` and remaining budget fraction `b_hat`, the full row is exactly:

```text
z(v)   = (x(v) - mean) / (std + eta)
phi(v) = [1, x(v), z(v), b_hat * x(v)]
```

The primary row therefore remains 31-dimensional `float64`. The pre-transition selected row is frozen for its SARSA update.

## Static and dynamic data

The following values are immutable while a record is open and are cached once:

```text
persistent record ID
complete canonical semantic key
complete resource tuple
first eight normalized candidate coordinates
target process infidelity reconstructed from the authoritative DAG
```

The dynamic coordinates are:

```text
resource-position r_P, because frontier membership changes
novelty, because generation_count[semantic_key] changes
```

The feature index is an accelerator only. `Frontier` and `ParetoArchive` remain authoritative for membership, acceptance, pruning, resource feasibility, terminal state, circuit semantics, and certification. A debug reconciliation must detect disagreement rather than letting the index alter the search.

## Exact resource-group identity

Let `rho(v)` be:

```text
(T count, two-qubit count, total gate count, depth on wire 0, ..., depth on wire n-1)
```

For each unique open resource tuple `r`, maintain:

```text
f(r) = number of active open records at r
D(r) = sum(f(s) for active s if s <= r componentwise)
```

Then every record at `r` has:

```text
r_P(v; F) = (D(r) - 1) / max(1, |F| - 1)
```

`D(r)` includes the candidate itself and all other records with the same tuple, so subtracting one is exactly the reference definition over `F \ {v}`. This is not a same-key Pareto rank and is not approximate.

For an insertion at tuple `a`:

1. Increment `D(r)` for every active group with `a <= r`.
2. If `a` is new, initialize its count from every active group `s <= a`, plus the new record itself.
3. Increment `f(a)`.

For a removal at `a`:

1. Decrement `D(r)` for every active group with `a <= r`.
2. Decrement `f(a)`.
3. Tombstone or remove the group only when its frequency becomes zero.

Removals and additions are applied in ascending persistent-record-ID order. Vectorized comparisons operate across unique resource groups, never a Python-level pair of loops over all open records. With `Delta` membership changes, `G` unique resource groups, and resource dimension `d_r`, synchronization costs approximately `O(Delta * G * d_r)` and decision gathering costs `O(F)`.

## Exact novelty deltas

For complete canonical key `k`:

```text
g(k) = number of resource-feasible generated records with key k
novelty(v) = 1 / sqrt(max(1, g(key(v))))
```

The root is counted once. Every resource-feasible generated record contributes even when rejected as a duplicate or dominated record. The index retains key-to-active-slot membership and applies generation-count deltas only to affected keys. A full key, not merely a hash, remains the identity.

## Compact batch and exact linear score

Partition linear weights as:

```text
theta = [theta_0, theta_x(10), theta_z(10), theta_b(10)]
```

For one decision state:

```text
w = theta_x + b_hat * theta_b + theta_z / (std + eta)
c = theta_0 - dot(theta_z, mean / (std + eta))
Q(v) = c + dot(x(v), w)
```

This is algebraically identical to `dot(theta, phi(v))`. The optimized evaluator may score the contiguous `F x 10` candidate matrix without constructing `F` separate 31-D arrays. It must still materialize the exact full row for the selected current and selected next SARSA records. Greedy ties resolve by lowest persistent record ID, and seeded epsilon exploration must consume random values in the existing order.

The no-target and no-standardization ablations apply the corresponding exact reduced block algebra; they do not fall back to approximate features.

## Deterministic standardization

The first optimized version retains the reference reduction order:

```python
sorted_matrix = np.sort(candidate_matrix, axis=0)
mean = np.mean(sorted_matrix, axis=0, dtype=np.float64)
std = np.std(sorted_matrix, axis=0, ddof=0, dtype=np.float64)
```

Incremental floating sums are outside this evaluator version because a different reduction order could alter near-ties. Snapshot hashing may be made cheaper because it is metadata, but candidate values and record ordering may not change.

## Benchmark integration API

`experiments.article_v1_feature_benchmark` intentionally has no CLI. The runner supplies an object implementing:

```python
measure_microbenchmark(frontier_size, *, include_reference)
measure_end_to_end(expansion_cap, *, include_reference)
```

and calls:

```python
benchmark_feature_evaluator(
    adapter,
    output_directory,
    correctness_gate=gate,
    implementation_checks={
        "no_python_nested_frontier_record_loop": True,
    },
    profile_writer=write_profiles,
    feasibility_criteria=criteria,
    pilot_relaunch_checks={
        "source_revision_committed_and_clean": True,
        "structured_progress_verified": True,
        "checkpoint_recovery_verified": True,
        "new_schema_mini_ci_passed_twice_byte_stable": True,
        "no_held_out_publication_test_outcomes_inspected": True,
    },
    benchmark_provenance={
        "code_version": commit,
        "source_worktree_digest": source_digest,
        "worktree_clean": True,
    },
)
```

The production integration uses `ArticleV1FeatureProvider.build_compact_batch(...)`, `CompactArticleDecisionBatch.scores(theta)`, `features_for_record(record_id)`, and provider `instrumentation()`. The oracle uses `ArticleV1ReferenceFeatureProvider.build_snapshot(...)`. `include_reference` is true only through the configured safe frontier size (256 by default) and for end-to-end trace caps 32 and 64.

The repository-backed factory is `create_repository_feature_benchmark_adapter(config="pilot", **kwargs)`. A thin CLI can call `run_repository_feature_benchmark(...)`; the latter constructs the adapter, runs the fixed axes, optionally writes profiles, and delegates artifact validation to the same generic harness. The fixed workload is the unique pilot `train/hard/3q` target `sha256:dfd960b7be1309661b720bb31eaf4fd97589b52fd3b11c7f25eb68dada3dafbf`, its transfer-corpus index is 5, and its effective policy/environment seed is `19 + 5 = 24`. The generator witness is stripped through `evaluation_target()` before the environment is constructed.

For microbenchmarks, one optimized rollout captures a shared representative frontier of at least 2,048 records; deterministic prefixes of that same frontier supply all requested sizes. The capture reached `F=2,053` after 139 expansions in the initial integration check. Static-cache construction and target-distance cache warmup occur outside the steady-state component timers. The reference batch and optimized synchronization, compact-batch, score, and selected-row calls then receive the same records, generation counts, completed-expansion value, and unchanged 8,192-step horizon.

For staged episodes, the environment config and budget-dependent feature coordinate retain the 8,192-step scientific horizon. A wrapper reports truncation to the trainer at the engineering cap only after the complete selected-record expansion has finished. Reference and optimized runs use fresh contexts/providers with identical seed, epsilon schedule, learning rate, reward, certifier tolerance, and exhaustive transition code. The adapter compares selected IDs plus rewards, final weights, terminal state, and non-timing search counters.

The adapter must time medians or another stated aggregation after adapter-owned warmup and return seconds for synchronization, compact-batch construction, vectorized scoring, and selected-row materialization separately. Memory is the evaluator/index structural memory, while process peak RSS is an additional optional field. Setup, target generation, and profile serialization must not be hidden inside a component timer.

The harness emits:

```text
baseline.json
microbenchmarks.csv
end_to_end_scaling.csv
profiles/
scaling_report.md
projected_pilot_cost.json
```

It refuses to start timing by default unless all required correctness checks pass. The artifact writer can record failed-gate diagnostics, but marks qualification false and cannot authorize a pilot relaunch. Even passing timing and explicit cost bounds produce only `insufficient performance evidence to relaunch` until source cleanliness, structured progress, checkpoint recovery, clean-schema mini-CI, and no-held-out-access checks are supplied and true.

## Required parity evidence

The correctness gate requires all of:

```text
snapshot equivalence
full-score equivalence
selected-record equivalence
seeded SARSA trace equivalence
final-weight equivalence
terminal-status equivalence
witness and independent-certification equivalence
```

Bounded cap-32 and cap-64 hard-target runs additionally require identical traces, weights, status, and deterministic counters. Timing fields are excluded from parity.

The fixed performance gates are at least 10× speedup near `F=1024`, no worse than 2.5× compact-batch-plus-score growth from 512 to 1,024, no production Python all-pairs record loop, and approximately linear index memory in `F+G`. A passing speed gate cannot override failed parity.

## Scientific boundary and limitations

The optimization does not cap, truncate, sample, or reorder the scientific frontier; omit generated gates; approximate `r_P`; make novelty stale; alter standardization; change SARSA; or bypass certification. Full-frontier exact ranking remains at least linear in `F`, and the underlying exact synthesis search remains exponential. Performance projection is an engineering log-space fit, not a confidence interval and not scheduler evidence.
