# Article V1 feature-scaling preflight

## Scope and evidence boundary

This document records both the clean engineering baseline used to optimize the
Article V1 frontier-feature evaluator and the subsequent authoritative local
engineering qualification. The measurements below are performance diagnostics.
They are not scheduler-comparison observations, are not scientific raw runs,
and must not be appended to `raw_runs.jsonl`.

The pilot was not relaunched during this preflight. No pilot checkpoint or pilot
scientific result existed at the starting revision. The post-optimization
measurements were made from an uncommitted, dirty worktree, so they are
engineering evidence only and cannot authorize a pilot relaunch.

## Starting repository state

```text
branch: frontier-rl
commit: f653193ec1fd15b17b948a476a0e89a343cbf062
worktree at preflight: clean
pilot process: not running
pilot checkpoint: none
pilot scientific raw results: none
certifier calibration: passed
clean mini-CI: passed twice with byte-stable resume
```

The recorded preflight checks were:

```text
python -m compileall -q .                         passed
python -m pytest -q tests/article_v1             240 passed
python -m pytest -q                              408 passed
git diff --check                                 passed
```

The corresponding reproducibility commands are:

```powershell
git status --short
git branch --show-current
git rev-parse HEAD
& '.\.venv\Scripts\python.exe' -m compileall -q .
& '.\.venv\Scripts\python.exe' -m pytest -q tests/article_v1
& '.\.venv\Scripts\python.exe' -m pytest -q
git diff --check
```

The post-optimization local validation completed successfully:

```text
python -m compileall -q .                         exit 0
python -m pytest -q tests/article_v1             340 passed in 160.65 s
python -m pytest -q                              509 passed in 188.84 s
```

These larger test results were obtained from the dirty qualification worktree,
not from a clean committed campaign revision.

## Runtime environment

The local preflight environment reported:

```text
Python: CPython 3.12.13
NumPy: 2.5.2
OS: Windows 11, build 26200, AMD64
CPU identifier: Intel64 Family 6 Model 142 Stepping 11, GenuineIntel
NumPy BLAS: scipy-openblas 0.3.34.0.0
OpenBLAS build: USE64BITINT, DYNAMIC_ARCH, NO_AFFINITY, SkylakeX, MAX_THREADS=24
NumPy SIMD baseline/found: X86_V2 / X86_V3
OMP_NUM_THREADS: unset during environment inspection
MKL_NUM_THREADS: unset during environment inspection
OPENBLAS_NUM_THREADS: unset during environment inspection
```

Performance qualification runs must set and record all three thread variables explicitly. The harness captures their actual values in `baseline.json`; it does not infer a thread count from the BLAS build. CPU and BLAS metadata identify the measurement environment only and do not confer scientific validity.

## Supplied baseline measurements

The fixed hard three-qubit training target showed:

| Expansion cap | Representative runtime | Reported range | Peak frontier | Feature-time share |
|---:|---:|---:|---:|---:|
| 32 | 8.589 s | 8.589–11.8 s | 543 | 84.7% |
| 64 | 61.837 s | 59.9–61.837 s | 1,039 | 94.5% |

Isolated reference feature batches showed:

| Frontier size | Feature-batch runtime |
|---:|---:|
| 534 | 0.790 s |
| 1,021 | 2.330 s |

Doubling the episode cap from 32 to 64 increased representative runtime by about 7.2 times while the frontier nearly doubled. This behavior is consistent with a frontier-wide quadratic feature batch accumulated over a frontier that grows with expansion count.

## Profiling record

The preserved baseline establishes the following component evidence:

```text
cap 32: frontier-wide feature evaluation = 84.7% of runtime
cap 64: frontier-wide feature evaluation = 94.5% of runtime
```

The raw baseline `cProfile` top-function table was not included in the supplied evidence, so this preflight does not invent function-level percentages. New runs must preserve `.prof` or text profile artifacts under:

```text
outputs/article_v1/<performance-run-id>/profiles/
```

At minimum, the integration adapter should separately profile semantic-key/resource extraction, target metric lookup, anticommutation counting, dominance maintenance, candidate gathering, standardization, compact scoring, selected-row materialization, and environment expansion. The generated scaling report inventories any files placed in `profiles/`.

## Qualification API and output layout

The `benchmark-features` runner command invokes the repository-backed
`run_repository_feature_benchmark` API. Before timing, it runs fixed pytest node
IDs through `run_focused_correctness_gate`, derives each semantic Boolean from
the resulting JUnit cases, and AST-inspects the two production dominance-update
methods through `inspect_production_dominance_update`. It retains pytest/JUnit
and source-check evidence under `profiles/` and aborts before constructing the
timing adapter if either preflight fails. The lower-level API still accepts data
objects, so direct callers must not hard-code their fields to true.

The high-level command accepts only a resolved configuration identical to the
checked-in canonical pilot digest/content, refuses a nonempty destination, and
takes fresh uncached source snapshots. Source and config are bound into the
JUnit/AST evidence and rechecked immediately before and after timing. A final
`benchmark_status.json` is written last with the artifact hashes/lengths and
separate `engineering_qualification_passed` and `pilot_relaunch_ready` fields.

The command surface is:

```powershell
python article_benchmark.py benchmark-features `
  --config configs/article_v1_pilot.json `
  --output-root outputs/article_v1 `
  --run-id article-v1-feature-index-v2
```

The default steady-state microbenchmark policy is 31 timed repetitions after
five untimed warmups at every frontier size. This makes the median-based
512-to-1,024 compact-score scaling gate robust to short scheduler, cache, and
clock noise; it replaces the earlier three-sample/one-warmup engineering
diagnostic, which was insufficiently stable for a launch gate. Command-line
overrides remain visible in the artifact metadata and must be justified during
review.

The command produced the passing engineering bundle
`outputs/article_v1/article-v1-feature-index-v2-local-20260818-a`. Its canonical
pilot configuration and source remained unchanged across the run, and the final
status records `engineering_qualification_passed: true`. The same status records
`pilot_relaunch_ready: false` because the bound source snapshot is dirty.

After real gates pass, the API writes a fresh run directory, for example:

```text
outputs/article_v1/article-v1-feature-index-v2/
├── baseline.json
├── microbenchmarks.csv
├── end_to_end_scaling.csv
├── profiles/
├── scaling_report.md
├── projected_pilot_cost.json
└── benchmark_status.json
```

The required isolated sizes are `32, 64, 128, 256, 512, 1024, 2048`. A
current-host reference is required through size 1,024 and skipped at 2,048; the
preserved size-1,021 historical baseline is context only and has explicitly
unknown environment metadata. The fixed hard target must be staged at caps
`32, 64, 128, 256, 512, 1024`. Caps 32 and 64 require exact reference trace
parity, at least 2x end-to-end current-host speedup, and explicit optimized and
reference feature-time shares.

## Post-optimization engineering results

The isolated current-host F=1,024 reference took 2.0747946 seconds, while the
complete optimized decision path took 0.0021255 seconds, a 976.144x speedup.
The compact-batch 512-to-1,024 ratio was 1.3042, below the 2.5x limit, and the
fitted index-memory exponent was 0.64119, below the 1.25 approximately-linear
limit.

| Hard-target cap | Reference runtime | Optimized runtime | Speedup | Reference feature share | Optimized feature share |
|---:|---:|---:|---:|---:|---:|
| 32 | 6.7472089 s | 1.4496152 s | 4.65448x | 0.86755 | 0.03397 |
| 64 | 44.6477131 s | 2.4842977 s | 17.97197x | 0.94525 | 0.04982 |

Reference/optimized trace, final-weight, terminal-status, and deterministic
counter parity passed at both reference caps. Optimized staged coverage also
completed at caps 128, 256, 512, and 1,024:

| Hard-target cap | Optimized runtime |
|---:|---:|
| 128 | 5.47783 s |
| 256 | 16.47306 s |
| 512 | 58.68196 s |
| 1,024 | 223.12976 s |

The cap-runtime model has exponent 1.4744 and projects one fixed hard-target
cap-8,192 episode at 3,515.33798 seconds with projected peak frontier
F=87,002.64. This is an engineering extrapolation fitted beyond the largest
measured cap, not a confidence interval, feasibility decision, or scientific
result.

All six required top-level artifacts and the profile/evidence files are complete
in the run directory: `baseline.json`, `microbenchmarks.csv`,
`end_to_end_scaling.csv`, `profiles/`, `scaling_report.md`, and
`projected_pilot_cost.json`; `benchmark_status.json` was written last with their
SHA-256 manifest.

A post-run artifact audit found two presentation/profiling defects in this
otherwise complete bundle: the Markdown report rendered unmeasured parity above
cap 64 as `fail`, and the file named `frontier-F1024-optimized` included the
current-host reference calculation. The worktree now represents absent parity
as not applicable and profiles only the optimized path under that filename.
Neither defect changes the CSV timings or qualification gates, but the final
bundle must be rerun after these corrections before it is used as the reviewed
profile artifact.

## Mini-CI resume evidence

The local run
`article-v1-feature-index-v2-mini-ci-local-20260818-a` passed twice. The first
invocation trained and appended nine records. The second did not retrain,
appended zero records, and skipped the nine existing records. The checkpoint
file remained byte-stable at
`8bfaa8a22ac4c54faccc85cb1fa2fa19aa6499d71a3de0bd64325579250abf49`,
and `raw_runs.jsonl` remained byte-stable at
`f8d49cb1d67e2b3c40f3a5ececaf861bddda755f83e03566c038af4a6346214d`.
The emitted progress event/status use v2 schemas, bind evaluator
`article-v1-exact-incremental-v2`, and contain positive measured timing.

This double pass was also produced from the uncommitted dirty worktree. It does
not satisfy the requirement for two clean committed mini-CI invocations.

No configured pilot may be relaunched merely because the optimized evaluator is faster. Correctness parity, scaling, trace parity, staged coverage, explicit feasibility bounds, progress, checkpoint recovery, a clean committed source revision, and clean-schema mini-CI must all pass first.

The engineering benchmark and dirty-worktree mini-CI evidence now pass. Pilot
relaunch readiness remains false: two clean committed mini-CI invocations and
the pilot itself are still pending. No publication or held-out scheduler result
is claimed here.
