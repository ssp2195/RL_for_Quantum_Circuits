# Article V1 raw-metric finalization preflight

Recorded 2026-08-17 before publication qualification.

- Branch/base: `frontier-rl` at `3a0dcad167b326ee2ce1556a650eab0c6f8bb268`
- Worktree: dirty by design with the uncommitted finalization delta; campaign evidence is provisional until reviewed and committed
- Runtime: Python 3.12.13, NumPy 2.5.2, Windows 11 build 26200, Intel64 Family 6 Model 142 Stepping 11
- Focused/full suites: 137 / 305 passed
- Schemas: `projective-unitary-metrics-v2`, `phase-frobenius-raw-v2`, `article-v1-native-corpus-v2`, `article-v1-corpus-config-v2`, `article-v1-transferable-linear-checkpoint-v3`, `article-v1-publication-runner-v2`, `article-v1-raw-run-v2`, `article-v1-publication-report-v2`
- Frozen tolerances: `tau_cert=1e-6`; separately named `tau_identity=1e-7`
- Calibration: equivalent floor `2.5809568279517847e-8`; minimum adversarial non-equivalent discrepancy `0.00023385357996901645`
- Mini-CI: passed twice under `article-v1-raw-metric-v2-coverage-complete-mini-ci`; second run appended 0, skipped 9, and preserved raw-ledger/checkpoint bytes
- Plans: pilot 300 expected primary/OOD raw keys (1,212,928 mechanically enumerated worst-case expansions); publication 10,000 keys (791,895,000 mechanically enumerated worst-case expansions)

The historical `outputs/article_v1/final-mini-ci-v5/` artifact was not overwritten or reinterpreted.
