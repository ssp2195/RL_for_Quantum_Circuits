# Article V1 feature-scaling preflight

## Scope and evidence boundary

This document records the engineering baseline used to optimize the Article V1 frontier-feature evaluator. The measurements below are performance diagnostics. They are not scheduler-comparison observations, are not scientific raw runs, and must not be appended to `raw_runs.jsonl`.

The pilot was not relaunched during this preflight. No pilot checkpoint or pilot scientific result existed at the starting revision. Existing clean mini-CI artifacts were left unchanged.

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

## Qualification commands and output layout

CLI wiring is intentionally deferred to the runner integration. Its command is expected to call `experiments.article_v1_feature_benchmark.benchmark_feature_evaluator` and write a fresh run directory, for example:

```text
outputs/article_v1/article-v1-feature-index-v2/
├── baseline.json
├── microbenchmarks.csv
├── end_to_end_scaling.csv
├── profiles/
├── scaling_report.md
└── projected_pilot_cost.json
```

The required isolated sizes are `32, 64, 128, 256, 512, 1024, 2048`. The reference provider is normally restricted to size 256 and below; the preserved size-1,021 baseline supplies the approximately-1,024 comparison when a current reference batch is intentionally skipped. The fixed hard target must be staged at caps `32, 64, 128, 256, 512, 1024`, with exact reference trace parity at caps 32 and 64.

No configured pilot may be relaunched merely because the optimized evaluator is faster. Correctness parity, scaling, trace parity, staged coverage, explicit feasibility bounds, progress, checkpoint recovery, a clean committed source revision, and clean-schema mini-CI must all pass first.
