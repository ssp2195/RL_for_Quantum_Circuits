# Article V1 alignment matrix

This is an equation-to-repository implementation matrix, not a statement that
the external `Article_limited_scope.md` manuscript was edited. The article is
the scientific source; formulas and conventions identified as **plan
amendments** are operational completions introduced by
`codex_plan_article_v1_codebase_alignment.md`. A checked-in interface or config
is not evidence that the full publication experiment was run.

Status meanings:

- **Implemented and focused-tested**: the named Article V1 path and focused
  tests exist.
- **Partially implemented**: a relevant path exists, but one or more plan
  requirements or publication integrations remain open.
- **Campaign pending**: implementation and focused/mini qualification exist,
  but neither the configured pilot nor the full publication campaign has been
  executed and recorded.

| Article contract | Code location | Test location | Current status | Known qualification |
|---|---|---|---|---|
| Eq. 80: linear \(Q_\theta(s,v)=\theta^T\varphi(s,v)\) over frontier records | `rl/policy.py` (`LinearQPolicy`, `PolicyFeatureBatch`) and `experiments/profiles.py` (`ARTICLE_V1_PROFILE`) | `tests/article_v1/test_profiles_and_policy.py`; legacy SARSA regression in `tests/test_env_and_policy.py` | Implemented and focused-tested | Article V1 depends on an explicitly bound `article-v1-31d` provider. The older 37D provider remains a different schema and its checkpoints cannot be reinterpreted as 31D. |
| Eqs. 81–92: ten candidate coordinates, frontier standardization, remaining-budget interactions, and exact 31D feature map | `rl/article_features.py` (`ArticleV1FeatureProvider`, `ArticleTargetContext`); frozen coordinate contract in `docs/article_v1_feature_contract.md` | `tests/article_v1/test_features.py`; `tests/article_v1/test_target_metric.py` | Implemented and focused-tested | The executable formulas for \(r_{\mathrm P}\), \(\nu\), fixed normalizations, and \(\eta=10^{-8}\) are plan operational completions, not claimed verbatim manuscript text. Feature distance is process infidelity and is not the terminal predicate. |
| Eqs. 95–100: epsilon-greedy semi-gradient SARSA using the actual next behavior action | `rl/policy.py` (`select_from_batch`, `update_from_features`), `train.py`, `experiments/profiles.py`, and frozen evaluation in `experiments/article_v1_runner.py` | `tests/article_v1/test_profiles_and_policy.py`; `tests/article_v1/test_runner.py`; `tests/test_env_and_policy.py` | Implemented and focused/mini-tested | Frozen pre-transition features, actual behavior-action bootstrap, nonzero schema-validated checkpoints, epsilon-zero/fairness-zero evaluation, fresh environments, and immutable evaluation weights are enforced. Checkpoint V3 binds seed, schema/family/corpus, exact training IDs/scope, learning rate, epsilon schedule, beta/tolerance, episodes, cap/budget policy/effective budgets, and weights; an explicit scope binds the allowed seed set and rejects incomplete primary/OOD/ablation/foreign-corpus provenance. Compatible resume loads the byte-stable checkpoint without retraining. The five-independent-seed publication campaign has not been run. Expected SARSA/contextual bandit code remains supplementary. |
| Eqs. 101–103: finite expansion-budget objectives, first-hit success, and conditional expansions | `config.py`, `env/rl_env.py`, `evaluate.py`, and `reporting/article_v1.py` | `tests/test_article_expansion_cost.py`; `tests/article_v1/test_reporting.py`; `tests/article_v1/test_runner.py` | Implemented and focused/mini-tested | Expansion counting, budget-success curves \(S_\pi(B)\), paired conditional \(E_\pi(B)\), target-level aggregation, and failure-preserving raw records exist. A curve point is emitted only for an exactly executed external cap; a larger-cap trajectory never fabricates an unexecuted lower-budget point. The repository mini-CI artifact exercises the path; configured pilot/publication conclusions remain unmeasured. |
| **Amended** Eq. 104: expansion cost, certified-success correction, and early-failure padding | `rl/article_v1_reward.py` (`ArticleV1RewardModel`) and `env/rl_env.py` (`article_v1_expansion_potential`) | `tests/article_v1/test_reward.py` | Implemented and focused-tested | This is an explicit plan amendment to the manuscript equation. It yields \(B_{\mathrm{exp}}-T\) for success after \(T\) expansions and \(-B_{\mathrm{exp}}\) for every failure. It must not be described as literal compliance with unamended Eq. 104. |
| Eqs. 105–107: potential shaping \(\Psi(s)=-\min_{v\in\mathcal F(s)}d_{\mathrm{tar}}(v)\), terminal potential zero | `rl/article_v1_reward.py`; integration in `env/rl_env.py` | `tests/article_v1/test_reward.py` | Implemented and focused-tested | `gamma=1`, configurable `beta`, no clipping, visit/pruning/dead-end bonus, support metric, entanglement metric, or best-child bonus. Composite legacy shaping remains separately named. |
| Eq. 129: phase-quotiented trace/Frobenius discrepancy and independent DAG replay | `certification/unitary_phase_metrics.py`, `certification/article_v1.py`, `rl/article_features.py`, and binding in `experiments/article_v1_runner.py` | `tests/article_v1/test_certification.py`; `tests/article_v1/test_target_metric.py`; legacy certifier regressions in `tests/test_certification_simulator.py` | Implemented and focused/mini-tested | Acceptance uses the raw \(\Delta_\phi=\sqrt{1-|\mathrm{Tr}(V^\dagger U)|/d}\) without matrix normalization and calibrated `tau_cert=1e-6`; feature distance \(d_{\mathrm{tar}}=1-c_\phi^2\), direct scheduling, certification, and corpus identity share the V2 primitive. Final certification independently replays the DAG and ignores the feature cache. |
| Eqs. 137–144: deterministic native 2/3-qubit corpus, easy/medium/hard strata, identity-deduplicated train/validation/test/OOD splits | `benchmarks/article_native_corpus.py`, `configs/article_v1_pilot.json`, `configs/article_v1_publication.json`, and per-split export in `experiments/article_v1_runner.py` | `tests/article_v1/test_corpus.py`; `tests/article_v1/test_runner.py` | Implemented and focused/mini-tested | Generator witnesses prove reachability and remain audit metadata; `evaluation_target()` omits them. Witness length is not an optimal circuit length. The pilot primary corpus is exactly five easy, five medium, and five hard targets across train/validation/test, plus four separately labelled OOD targets. OOD SARSA uses separately trained checkpoints restricted to train witnesses of length at most four. The configured pilot/full campaigns remain unrun. |
| Eqs. 145–154: seven fixed schedulers, frozen evaluation, counters/timings, raw runs, curves, bootstrap, ablations, and experiment freeze | `experiments/article_v1_runner.py`, `experiments/article_v1_ablations.py`, `reporting/article_v1.py`, Article V1 configs, `evaluate.py`, `env/rl_env.py`, and `rl/policy.py`; root dispatch in `article_benchmark.py` | `tests/article_v1/test_runner.py`; `tests/article_v1/test_scheduler.py`; `tests/article_v1/test_reporting.py`; `tests/article_v1/test_instrumentation.py`; `tests/article_v1/test_ablations.py`; `tests/test_search_metrics.py` | Implemented and focused/mini-tested; campaigns pending | The seven schedulers share the amended Article V1 reward, `beta`, target metric, certifier, grammar, archive, and budgets. The zero-weight control executes the same 31D feature path. Ranking, structural feature, and dense target-metric timings are exclusive; total evaluation `wall_time_ns` includes reset/selection/steps/stopping and the narrower raw diagnostic is `environment_step_time_ns`. Article `frontier_sum` samples actual decision states only, while legacy `frontier_mean` retains its older post-expansion convention. Fail-closed immutable resume keys on training seed and source-worktree digest, preserves compatible checkpoint bytes, and requires a new run ID for old ledgers. Reporting emits exact executed budgets, portable artifact paths, per-learner and between-learner tables, per-split manifests, `report_metadata.json`, the six required validation ablations, and `ablations.csv`. `final-mini-ci-v5` passes all 11 semantic checks and resumes with 0 appended/9 skipped, unchanged raw and checkpoint files, and no retraining; the configured pilot and full publication campaign have not been run, so no comparative scientific result is claimed. |

## Preserved scientific boundaries

The primary Article V1 claim is restricted to small-scale, exact,
ancilla-free, all-to-all Clifford+T synthesis in which RL ranks persistent
frontier records and every selected record is expanded exhaustively through
the same deterministic native grammar. The learned policy does not control
native gate generation, symbolic updates, canonicalization, Pareto acceptance,
or dense certification.

GHZ-3 remains a small state-preparation smoke/case-study benchmark. The known
Toffoli witness is certification evidence, while learned Toffoli search remains
restricted to its declared seven-term parity-network normal form. QFT-3 is a
reference and capability-guard artifact; canonical QFT-3 is not reported as a
learned exact native synthesis result, and AQFT-3 fidelity is not exact
certification.

This matrix supports no claim of circuit optimality, unrestricted Toffoli
discovery, approximate QFT synthesis, large-scale scalability, polynomial-time
search, or universal RL superiority.
