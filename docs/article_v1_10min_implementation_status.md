# Article V1 ten-minute protocol implementation status

## Implemented

- V3 protocol configuration, inheritance, validation, digesting, and frozen
  publication guardrails; legacy V2 pilot/publication configurations are not
  edited.
- Exact process-CPU and wall timing, a completed-boundary operability watchdog,
  explicit incomplete `OPERABILITY_TIMEOUT`, and separate equal-CPU evidence.
- Exact archive/frontier optimizations: constant-time active counters,
  prepare-once canonical/resource insertion payloads, record-ID indexing, and a
  cached reviewed priority-then-record-ID selection view.
- Fixed-total seeded-round-robin easy/medium curriculum with no convergence
  stopping and no hard-target training.
- V5 transferable checkpoints plus internal journal replay. Mid-episode resume
  reproduces schedule, selection trace, policy updates, final weights, and final
  checkpoint digest. Incomplete checkpoints are recovery-only.
- Fixed-maximum-horizon anytime evaluation, one raw physical run per identity,
  derived threshold rows, validation-only protocol-sensitivity comparison, and
  separate equal-CPU evaluation.
- Validation-only hard-cap selection and training-interaction selection,
  no-test-access assertions, clean-worktree freeze enforcement, and campaign
  compute/cardinality planning.
- Strict campaign audit and `report-10min`, including all nine required SVG
  artifact names and family-separated aggregation.

## Verification completed

- Repository-wide suite: 564 passed.
- Final focused protocol/runner/reporting/instrumentation suite: 101 passed.
- Ten-minute-specific suite: 34 tests after the final recovery-only checkpoint
  guard (the focused final suite includes these tests).
- Mini-CI: 9/9 bounded physical runs completed; every semantic check passed.
- Exact optimized/reference hard-target trace parity: passed at caps 8, 16, 32,
  and 64.
- End-to-end engineering diagnostic versus the reviewed clean baseline:
  1.305x at cap 512 and 2.057x at cap 1024.

## Intentionally pending publication execution

The full hard-cap calibration has not been launched. Its configured matrix is
45 physical hard-target runs (three train/validation targets, five candidate
caps, and three schedulers), and it is expected to take multiple hours locally.
No candidate cap is selected or frozen from the two engineering measurements.

The publication freeze, five-seed training campaign, held-out pilot/publication
evaluation, secondary equal-CPU campaign, and final scientific report remain
blocked on reviewed/committed clean source plus completed validation-only
calibration and training-budget evidence. This is deliberate: held-out results
must not influence either selection.
