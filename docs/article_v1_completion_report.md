# Article V1 completion report

## Raw-metric V2 finalization inventory

| Path | Changed scientific behavior | Preserved behavior and coverage |
|---|---|---|
| `certification/unitary_phase_metrics.py`, `certification/article_v1.py` | Shared raw metric, no hidden rescaling, V2 certifier | global-phase quotient and fresh DAG replay; certifier tests/calibration |
| `rl/article_features.py`, `rl/target_context.py` | Shared `d_tar`, direct scheduler metric, and dense compatibility discrepancy | witness-keyed cache; direct delegation and cache-poisoning tests |
| `benchmarks/article_native_corpus.py`, `benchmarks/article_v1_calibration.py` | Shared identity decision, aliases, tolerance calibration | deterministic witness generation; corpus tests |
| `experiments/profiles.py`, `experiments/article_v1_runner.py`, `article_benchmark.py` | V2 provenance plus plan/calibration CLI | seven schedulers and fail-closed resume; runner tests/mini-CI |
| `reporting/article_v1.py` | V2 raw-ledger/report schemas and target-metric-bound resume keys | failure-preserving aggregation; reporting and old-schema rejection tests |
| `benchmarks/toffoli.py`, `ghz3_smoke.py`, `ghz3_rl.py`, `toffoli_certify.py` | Resource-baseline/lower-bound terminology with deprecated compatibility aliases | benchmark resource values and certification behavior |
| Article configs and focused tests | Frozen `tau_cert=1e-6`, V2 expectations | corpus/budgets/seeds unchanged |

Dirty-tree engineering qualification: focused 137 passed; full 305 passed; calibration passed; campaign planners passed; mini-CI passed twice with byte-stable resume. Pilot/publication evidence remains gated on a reviewed, committed clean version.

Status: **implementation and mini-CI qualification complete; pilot and
publication campaigns not run**

Date: 2026-08-15

The Article V1 implementation has passed its focused and full repository test
suites and a deterministic mini-CI workflow. This is an implementation
qualification, not a completed experiment. Neither the configured pilot nor
the full publication campaign has been executed, so the repository is not yet
fully publication-compliant and no comparative scientific result is claimed.

The external `Article_limited_scope.md` was treated as a reference and was not
edited. Operational completions and amendments to that reference are recorded
in `docs/article_v1_manuscript_patch.md`.

## A. Baseline and attribution

| Field | Recorded evidence |
|---|---|
| Initial branch | `frontier-rl` |
| Initial commit | `df2ad6a036538b701cc3eb70b80f906d99d34a9a` |
| Initial working tree | Dirty; pre-existing article/QFT alignment work was local and partly untracked |
| Initial compile check | Exit 0, with access warnings only for pre-existing `.stage2-*` / `.stage3-*` artifact directories |
| Initial full tests | 168 passed in 35.94s |
| Final focused Article V1 tests | 115 passed in 5.07s |
| Final full repository tests | 283 passed in 44.59s |

The baseline is recorded in `docs/article_v1_preflight.md`. In particular,
`algebra/pauli_rotation.py`, `algebra/tableau.py`,
`circuit/circuit_state.py`, and `circuit/dag.py` were already modified before
the Article V1 task began. Their dirty status must not be attributed to this
task. Article V1 did not require new DAG or Pauli/tableau semantics; the final
full-suite result qualifies the current combined working tree but does not
erase that provenance distinction.

## B. Task-scoped file inventory

The following is a grouped inventory, because several integration files were
already dirty at preflight and a flat `git diff` cannot reliably attribute
their complete contents to Article V1.

| Group | Paths | Article V1 purpose / attribution |
|---|---|---|
| Article V1 scientific modules | `certification/article_v1.py`; `rl/article_v1_reward.py`; `benchmarks/article_native_corpus.py`; `experiments/article_v1_runner.py`; `experiments/article_v1_ablations.py`; `reporting/article_v1.py` | New task-scoped certifier, reward, corpus, experiment, ablation, and reporting implementations |
| Profiles and features | `experiments/profiles.py`; `rl/article_features.py`; `rl/policy.py` | Versioned 31D Article V1 profile, common target context/features, frozen decision batches, and scope-bound transferable checkpoint V2; `rl/article_features.py` existed untracked at preflight and was extended rather than newly originated here |
| Frozen configs | `configs/article_v1_pilot.json`; `configs/article_v1_publication.json` | Pilot and publication corpus/experiment definitions; config presence is not evidence that either campaign ran |
| Shared integration surfaces | `article_benchmark.py`; `benchmarks/__init__.py`; `canonical/canonicalizer.py`; `config.py`; `env/rl_env.py`; `evaluate.py`; `pyproject.toml`; `reporting/__init__.py`; `search/archive.py`; `train.py` | CLI dispatch, exports, Article reward/metric integration, exact counters and total-evaluation timing, decision-state sampling, ablation switches, fail-closed identity/resume, and packaging; most were already modified or untracked at preflight, so only their Article V1 portions are task-scoped |
| Article V1 tests | `tests/article_v1/` | Focused unit, soundness, leakage, resume, reporting, scheduler, instrumentation, corpus, and end-to-end tests |
| Contracts and evidence | `README.md`; `docs/article_v1_*.md` | User workflow, preflight, contracts, protocol, alignment, manuscript amendment, and this completion report |
| Generated mini evidence | `outputs/article_v1/final-mini-ci-v5/` | Passing one-target, nine-scheduler-instance audited mini-CI artifact; not a pilot or publication result |

Other dirty paths listed by `git status` include pre-existing DAG/algebra,
QFT, GHZ/Toffoli, native-corpus, and earlier audit/test-artifact work. They are
outside this inventory unless explicitly named above. No attempt was made to
rewrite, discard, or claim ownership of those changes.

## C. Preserved scope and controlled ablations

The Article V1 primary profile remains the small, exact, ancilla-free,
all-to-all native grammar: H, S, SDG, T, TDG, and every directed CNOT. It does
not add a target-specific reachability oracle. Dense target information is
used for the declared metric, reward, ranking/features, and certification, not
for replaying a corpus generator witness.

Primary enhanced canonicalization remains enabled. The `raw_witness`
canonicalizer is an ablation-only control: it disables commuting reorder,
cross-position fusion, and Clifford-angle absorption, thereby preserving the
literal DAG word and preventing the ablation from falsely merging distinct
words. It is not the primary Article V1 canonicalizer and must not be used to
reinterpret the primary results.

Similarly, the Pareto-off ablation retains exact same-key/resource duplicate
suppression but disables dominance rejection and retirement. The primary
profile retains the full Pareto rule.

Learned policies use `article-v1-transferable-linear-checkpoint-v2`. Its digest
binds training seed, feature schema, standard/OOD family, corpus config,
training-scope mode, ordered training target IDs, learning rate, epsilon
schedule, training `beta`, certification tolerance, episodes per target,
optional expansion cap, budget policy, ordered effective per-target training
budgets, and weights. The evaluation scope additionally binds the allowed seed
set, held-out/permitted evaluation IDs, and an exact training-partition/protocol
match. Standard, length-OOD, and ablation campaign evaluations require their
complete declared training partition. Only mini-CI is explicitly marked
`explicit_partial_smoke`; its checkpoint cannot pass a complete campaign scope.

GHZ-3 remains a small state-preparation benchmark; Toffoli remains constrained
to the declared parity-network normal form; and QFT-3 remains a
reference/capability boundary rather than an Article V1 learned-synthesis
target. Article V1 does not broaden any of those claims.

## D. Equation and protocol disposition

| Contract | Implementation | Final disposition |
|---|---|---|
| Eq. 80 linear Q | `rl/policy.py`; `experiments/profiles.py` | Implemented and mini-CI qualified; campaign conclusions pending |
| Eqs. 81–92 features | `rl/article_features.py` | Operational \(r_P\), novelty, normalization, and \(\eta\) implemented under the documented plan completions |
| Eqs. 95–100 SARSA | `rl/policy.py`; `train.py`; `experiments/article_v1_runner.py` | Frozen-transition feature path and full protocol-/partition-/seed-bound checkpoint V2 qualified; multi-seed campaign unrun |
| Eqs. 101–103 objectives | `config.py`; `env/rl_env.py`; `reporting/article_v1.py` | Expansion-budget curves use only exactly executed caps; exact counters, paired target aggregation, and reporting implemented |
| Amended Eq. 104 | `rl/article_v1_reward.py` | Explicit operational amendment; not literal unamended-manuscript compliance |
| Eqs. 105–107 shaping | `rl/article_v1_reward.py` | Direct \(-\min d_{tar}\), terminal zero, no composite legacy terms |
| Eq. 129 certification | `certification/article_v1.py`; `experiments/article_v1_runner.py` | All seven schedulers use the same Article V1 certification engine |
| Eqs. 137–144 corpus | `benchmarks/article_native_corpus.py`; Article V1 configs | Pilot definition is exactly 5 easy/5 medium/5 hard primary targets plus four labelled OOD targets; OOD checkpoints train only on train cases of generator length at most four |
| Eqs. 145–154 experiment | runner, ablation, reporting, environment, evaluation, and policy modules | Seven schedulers, exclusive timing, total evaluation wall time, decision-state frontier metrics, immutable fail-closed resume, per-/between-learner reporting, portable paths, and six validation ablations implemented; pilot/publication execution pending |

Every evaluation scheduler uses `article_v1_expansion_potential`, the common
target metric, and configured `beta`. Scheduler ranking and feature timing are
exclusive, and dense target-metric time is excluded from feature time. Article
`frontier_sum` samples the root plus every nonterminal/nontruncated next
decision state; terminal/truncated result states are excluded. The separately
named legacy `frontier_mean` retains its historical post-expansion convention
and must not be mixed with the Article statistic.

The zero-weight Article control still computes the full 31D feature batches and
dense target coordinates; only its policy weights are zero. `wall_time_ns` is
the complete evaluation envelope, while `environment_step_time_ns` is the
separately named subset spent inside environment step bodies. Raw identity
includes both `training_seed` and `source_worktree_digest`, and report curves
omit a budget unless that exact external cap was executed. Per-learner output
and target-first between-learner statistics are separate tables. Artifact maps
under the repository use relative POSIX paths.

Resume is fail-closed for checkpoints as well as manifests and the raw ledger.
A compatible existing checkpoint is validated and loaded without retraining or
rewriting; missing, corrupt, stale, or conflicting checkpoint content aborts.

The detailed contract table is `docs/article_v1_alignment_matrix.md`.

## E. Final executable qualification

| Check | Command / scope | Result |
|---|---|---|
| Scoped compile | `python -m compileall -q` over the task-scoped Article V1 Python paths | Exit 0 |
| Focused Article V1 | `python -m pytest -q tests/article_v1` | 115 passed in 5.07s |
| Full repository suite | `python -m pytest -q` | 283 passed in 44.59s |
| Diff hygiene | `git diff --check` | Exit 0 |
| Deterministic mini benchmark | `python article_benchmark.py mini-ci --output-root outputs/article_v1 --run-id final-mini-ci-v5` | Passed; first execution appended 9 raw runs; all 11 semantic checks true; checkpoint training seed 19 and digest `sha256:f62a4f7595e284e70d52d9e56623972e0409cbe33e9f3f97f3a49feb246bc25e` |
| Resume check | Repeat the same mini-CI command | `checkpoint_trained_this_run: false`, `appended: 0`, `skipped: 9`, `completed: 9`; raw ledger and checkpoint bytes unchanged |
| Configured pilot | `python article_benchmark.py pilot --config configs/article_v1_pilot.json` | Not run |
| Full publication campaign | Publication config and documented runner workflow | Not run |

The resume byte-stability assertion covers the scientific raw ledger and
checkpoint file. Regenerated `report_metadata.json` timing may change, so this
report does not claim that every derived report artifact is byte-identical
across executions.

Resume is nevertheless fail-closed with respect to scientific identity:
existing run/environment/corpus/per-split manifests must exactly match the
current config, profile, code/worktree, and corpus contract. The raw key now
includes `training_seed` and `source_worktree_digest`; old pre-change ledgers or
pre-V2 checkpoints are intentionally incompatible and require a new run ID.
The existing checkpoint is contract-validated before reuse, then loaded without
training or rewriting.

## F. Final mini-CI artifacts

The evidence directory is `outputs/article_v1/final-mini-ci-v5/`. Its final
inventory is:

```text
outputs/article_v1/final-mini-ci-v5/
├── run_manifest.json
├── environment.json
├── mini_ci_summary.json
├── report_metadata.json
├── raw_runs.jsonl
├── per_target.csv
├── success_curves.csv
├── timing_breakdown.csv
├── completion_summary.md
├── corpus_manifest/
│   ├── manifest.json
│   ├── train.json
│   ├── validation.json
│   ├── test.json
│   └── ood_test.json
├── checkpoints/
│   └── seed-0.json
├── figures/
│   ├── conditional_expansions.svg
│   ├── success_curves.svg
│   └── wall_time.svg
└── tables/
    ├── learner_seed_results.csv
    ├── learner_seed_summary.csv
    ├── paired_differences.csv
    ├── paired_per_target.csv
    ├── summary.csv
    └── summary.md
```

`mini_ci_summary.json` records `passed: true`, checkpoint training seed 19, the
checkpoint digest shown above, and all eleven semantic checks as true:

- exact target set, raw-record count, and seven scheduler labels;
- exact Article schemas/reward/certifier parameters and SARSA checkpoint
  binding;
- deterministic scheduler seeds and seeded-random trajectories;
- persistent-frontier action semantics;
- independent FIFO certification of the known reachable target; and
- no generator/reference-witness fallback.

The first run appended exactly nine records. After the resume run the summary
records `checkpoint_trained_this_run: false`, 0 appended, 9 skipped, and 9
completed. The raw ledger SHA-256 remained
`332F18D9187BF4258C0AD4C1809C4C4F6FEEBB5EC5EF3A94188FBFCAF7607EB0`; the
checkpoint-file SHA-256 remained
`D364AA529D69D5CD06E080D9A7AA6BE8D25237FD85D5485E791FFD9A357761F9`.
The checkpoint's scientific digest is the distinct protocol/weight digest
shown above. `report_metadata.json` records nine raw inputs, 300 bootstrap
samples, and separates reporting time from scientific timing. The artifact map
uses repository-relative POSIX paths rather than checkout-specific absolute
Windows paths. `tables/learner_seed_results.csv` preserves the seed-19 learner
row; `tables/learner_seed_summary.csv` contains the separately named
between-learner aggregation surface (one learner in this mini only).

The mini checkpoint declares `explicit_partial_smoke`, one training target,
one episode per target, and an expansion cap/effective budget of 16. This is an
intentional smoke exception and must not be described as the complete standard
training partition required by the pilot/publication campaign.

This mini directory deliberately does not contain `validation_audit.json` or
ablation outputs: those are command-specific outputs of the pilot/validation
and `ablations` workflows, neither of which this one-target mini run purports
to replace. The implemented full run layout and commands remain documented in
`docs/article_v1_experiment_protocol.md`.

## G. Qualification boundary and remaining work

Completed:

- Article V1 feature, reward, certifier, corpus, scope-bound standard/OOD and
  ablation checkpoint V2, seven-scheduler, instrumentation, reporting, and
  ablation paths are implemented.
- Focused and full tests pass, scoped compilation succeeds, and diff hygiene
  is clean.
- The final mini-CI passes all eleven semantic checks, independently certifies
  the target under FIFO without reference-witness fallback, produces nine raw
  records and portable aggregate artifacts, and resumes without duplicating
  raw records, retraining its checkpoint, or changing raw/checkpoint bytes.
- Reporting is exact-executed-budget-only, preserves per-seed and
  between-learner results, and distinguishes total evaluation wall time from
  environment-step time. The zero-weight control executes the Article feature
  path, and run identity/resume now fail closed on seed and source-worktree
  provenance.
- Checkpoint identity/evaluation now fail closed on the complete training
  protocol and partition, including learning rate, epsilon schedule, allowed
  seed set, beta/tolerance, episodes, cap, and effective budgets. The mini's
  partial scope is explicit and cannot masquerade as a campaign checkpoint.

Not completed:

- The configured pilot has not been run.
- The full 75-held-out-target, five-training-seed, repeated-random publication
  campaign has not been run.
- Consequently there is no campaign-level success, budget-efficiency,
  scheduler-superiority, ablation, timing, or OOD result to report.
- The codebase is implementation-complete for the documented Article V1 plan,
  but it is **not fully publication-compliant** until the frozen pilot,
  validation audit/ablations, full campaign, artifact review, and corresponding
  manuscript tables/figures are completed.

Even after those runs, the supported domain remains small 2/3-qubit, exact,
ancilla-free, all-to-all Clifford+T search with dense verification. There is no
circuit-optimality, large-scale scalability, polynomial-time, unrestricted
Toffoli, learned QFT, or universal RL-superiority claim.

## H. Git state

The final qualification was performed on the dirty `frontier-rl` working tree
at baseline commit `df2ad6a036538b701cc3eb70b80f906d99d34a9a`. The dirty tree
contains both Article V1 work and explicitly pre-existing local work described
in `docs/article_v1_preflight.md`; status alone cannot attribute every hunk.
No production or test file was changed during this final documentation pass.
Changes remain local and uncommitted unless the user separately directs a
commit or push.

## Final scientific boundary

> The repository implements reinforcement-learning-guided ranking of
> persistent frontier records for small-scale, exact, ancilla-free, all-to-all
> Clifford+T synthesis. Every selected record is expanded exhaustively through
> the same deterministic native grammar; symbolic semantics, canonicalization,
> Pareto pruning, and final dense certification remain outside the learned
> policy. The Article V1 experiment asks whether linear SARSA allocates a
> finite expansion budget more effectively than fixed non-learning schedulers,
> especially a direct target-distance heuristic, when every other synthesizer
> component is held fixed.

This boundary does not authorize claims of general circuit generation,
large-scale synthesis, approximate QFT synthesis, universal RL superiority,
circuit optimality, unrestricted Toffoli discovery, or polynomial-time search.
