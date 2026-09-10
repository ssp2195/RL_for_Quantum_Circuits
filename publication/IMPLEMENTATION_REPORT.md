# Publication-stage implementation and experimental report

## Executive result

A complete bounded phase-oracle synthesis study has been implemented and executed. It uses a 20-coordinate linear semi-gradient SARSA outer ranker and four disjoint 20-coordinate residual LinUCB continuation models. It retains the exact frontier, limits learned scoring to an indexed 32-candidate panel, preserves all feasible continuations through deterministic fairness, and exports independently checkable bounded exclusions.

**The central empirical hypothesis was not supported.** The trained hierarchy underperformed its identical untrained prior on the locked primary experiment and on a fresh confirmation set. Validation selected **zero learned residual**. The artifact does not relabel that fallback as successful learning, discard failed runs, let the auditor repair failed discovery, or claim a novel combination of RL and proof.

The positive results are correctness, bounded-scoring preservation, post-training circuit generation, strict incumbent improvements, independently verified bounded resource claims, a real clean-ancilla/T-depth/CNOT trade-off, and coherent use of a synthesized multi-marked Boolean-network phase oracle. These results do not establish unrestricted Clifford+T optimality or state-of-the-art synthesis.

## 1. Defensible central contribution and related work

The research question is narrowed to **bounded-scoring, proof-preserving linear scheduling for resource-constrained phase-oracle synthesis**, with explicit attribution of quantum correctness, resource proof, and the learned controller's effect.

An indexed candidate panel is not a beam: candidates outside the panel are retained and remain eligible through global fairness. Its scoring bound is O(K log K + K d_outer + b d_inner), excluding semantic transition, feature construction, archive and certification costs; heap updates still cost O(log F). The manuscript proves the relevant interface obligations instead of treating elementary induction/dominance as newly invented mathematics.

The primary-source comparison in `RELATED_WORK.md` covers phase-polynomial synthesis, T-par, Selinger's T-depth constructions, GraySynth, Reqomp, Pauli-network RL, AlphaTensor-Quantum and its reusability study, learned branch-and-bound, Vista, exact T-library synthesis, weighted model-counting synthesis, and lightweight learned beam search. AlphaTensor-Quantum already used Z3 to establish small tensor-domain T-count optima. Therefore “RL plus optimality checking” is explicitly rejected as a novelty claim. Current-source verification is dated September 9, 2026. Citation scope is stated, including where only an abstract was inspected.

The new phase domain fixes the polynomial and emits each prescribed phase block once. Its T-count is fixed; it optimizes CNOT count under native-depth, T-depth, native-gate and clean-workspace caps. It does not silently replace the existing evaluator optimizer or call these results unrestricted T-count optimization.

## 2. Training on relevant decisions

Each of five independent models completes 184 staged episodes: 64 outer, 96 inner with the outer frozen, and 24 outer adjustment with the inner frozen. The curriculum contains 26 specification-only problems, including non-Clifford phase placement, data parity changes, clean-workspace computation and restoration, and tight resource caps. All four continuation classes receive training updates in every final model. No target witness or auditor output is supplied to training or discovery.

The final rate is 0.01, selected on validation before the primary campaign. The original rate-grid and the earlier exploratory grid are both retained. The mean final-model CPU training cost was 7.868 seconds, excluding validation selection, ablation training and the later guard selection. These costs are separately recorded. Training coverage is demonstrated; global convergence of linear SARSA or this search-induced bandit is not claimed.

## 3. Attributable performance measurement

The original protocol and core source hashes were locked before held-out evaluation. Its digest is:

`1d89f27d811f779e7b79b1cb4f95155700ea2578b3095c04a1b2f4a88d35e115`

The primary experiment contains 25 logical targets, five training seeds, three timings and seven methods: **2,625 runs**. Every job shares 2,048-edge, 20,000-record, 0.25-second wall and CPU caps; these are cooperative limits, not an OS-enforced hard bound. Results retain all failures and actual observed times. Candidate look-ahead, scoring, transitions, certification and audit effort are recorded separately. Uniform-cost search is an actual goal-on-pop Dijkstra label-setting control, not merely a greedy cost ranker.

| Method | Successful / 375 | Mean normalized CNOT savings, all runs |
|---|---:|---:|
| Trained hierarchy | 244 | 20.51% |
| Identical untrained prior | 375 | 30.64% |
| Greedy | 341 | 25.30% |
| Outer-only | 261 | 21.92% |
| Inner-only | 336 | 24.05% |
| Uniform-cost | 15 | 0.00% |
| Audit-only | 15 | 0.00% |

The primary trained-minus-untrained savings difference is **−10.13 percentage points**, paired target-cluster bootstrap 95% interval **[−16.72, −4.11]**. Seed/timing rows are averaged within logical targets before resampling; they are not treated as independent targets. Other contrasts are descriptive, not multiplicity-adjusted significance tests.

The trained hierarchy has **121 runs with strict incumbent improvement**, but the untrained prior also has 103. Discovery improvements alone therefore do not identify a training benefit. Audits are excluded from discovery, and any auditor witness remains auditor-labelled.

Three additional constructive controls yield 225 runs. The adapted GraySynth control achieves 72/75 in-contract successes and 33.61% normalized savings with about 0.0041 seconds mean observed call time. It is stronger than the learned hierarchy here. The rank-partition reference produces correct circuits but exceeds the tight original CNOT caps; it is useful on separately widened rank-tight contracts, not falsely counted as successful on the primary contracts.

Separately retrained no-budget and no-workspace ablations and an identical-weight full-scoring control add 375 runs. The bounded panel has 2.11 percentage points higher normalized savings than full scoring in the matched secondary comparison, descriptive interval [0.60, 3.95]. The maximum scored panel is 32 rather than the observed 542. This is evidence about ranking overhead, not evidence that learning beats the untrained prior.

### Explicit post-primary amendment

The primary negative result was preserved. A separately dated shrinkage grid uses only the original validation set, with five seeds and two timings: 560 validation jobs. The selection rule requires at least the untrained success rate, then maximizes savings, breaking ties toward less learned correction. It selects **beta=0**, explicitly a nonlearned fallback.

Only after that selection is frozen are twelve new target orbits generated, excluding every original train/validation/test/regression orbit. They are crossed with 0, 1 and 2 ancillas, five seeds, three timings, and three methods: **1,620 new confirmation runs**. This also removes the original random test set's width/ancilla confounding for this confirmation comparison.

| Confirmation method | Successful / 540 | Mean savings |
|---|---:|---:|
| Raw trained hierarchy | 438 | 20.65% |
| Untrained prior | 539 | 27.06% |
| Validation-selected guard, beta=0 | 538 | 27.00% |

Raw trained-minus-prior difference is −6.41 percentage points, interval [−11.61, −1.62], clustered by twelve logical targets. At beta=0 the guarded and prior scores are identical; small deadline-dependent result differences do not constitute learned performance. No confirmation result reselects beta. The guard is not a theorem of distribution-free non-regression.

## 4. A genuine certified ancilla/resource trade-off

All five trained seeds synthesize the controlled-S calibration under each workspace budget:

| Available clean workspace | Used workspace | T count | T-depth | CNOTs | Native depth | Native gates |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 3 | 2 | 2 | 4 | 5 |
| 1 | 1 | 3 | 1 | 4 | 5 | 7 |
| 2 | 1 | 3 | 1 | 4 | 5 | 7 |

A clean ancilla saves one T-layer while costing two CNOTs and two native gates. Independent rank bounds prove the corresponding T-depth minima; checked closed covers exclude tighter CNOT caps. This is a known identity and a declared training calibration, not a novel quantum decomposition or held-out learning claim. Clean workspace is coherently restored. The minimum one clean ancilla is relative to the T-depth-one, fixed-polynomial domain.

## 5. Manuscript theorems and proof checks

`main.tex` gives full arguments for the promised-input semantic invariant; continuation-safe resource dominance including every wire-depth coordinate; admissible completion bounds; closed-cover exclusion soundness; bounded CNOT optimality and clean-width minimum; coverage despite arbitrary finite learned priorities; the indexed-panel scoring bound; and the terminal-potential shaping identity. Existing rank/matroid lower-bound ideas are attributed to prior work.

The auditor and checker use separately written scalar transitions. Exact basis/phase replay is distinguished from numerical native isometry/leakage checks. Python checker code and specified native gate identities remain trusted; this is not a proof-assistant development.

Post-campaign proofs close **11/25** primary CNOT gaps. The remaining **14/25** are explicitly upper bounds, not timeouts relabelled as infeasibility. Complete artifact verification covers **5,501 recorded outcomes**, **597 unique native witness/contract pairs**, **26 closed covers**, and **120 rank-bound records**. These are verification records, not 5,501 independent targets. Full records include primary, secondary, constructive, rank-tight, calibration, application, guard validation and confirmation jobs.

## 6. Application, tests and reproducibility

The BNN-style example is a fully specified three-input binary threshold network marking exactly the three weight-one inputs. The seed-zero trained hierarchy receives only its phase specification and generates a certified direct oracle:

- 0 auxiliary qubits; 7 T/TDG gates; 6 CNOTs; native depth 11; 16 native gates; T-depth 5.
- Actual native oracle gates are applied to a superposition and followed by diffusion. The marked probability is 0.8437499999999996, matching 27/32 within 4.45e−16, with zero simulated workspace leakage.
- A separately generated, coherently checked evaluator-wrapper reference needs two auxiliaries, 42 T gates, 42 CNOTs, depth 66 and 98 gates. This improvement is a direct-phase representation comparison, not an RL ablation. Diffusion costs are not included in oracle counts. No end-to-end BNN quantum advantage is claimed.

The complete local suite has **481 passing tests**. **320 tests (66.53%)** actually generate circuits after policy training: 300 target-orbit/seed combinations, five coherent BNN-oracle uses, and fifteen learned ancilla-trade-off generations followed by independent checks. All 135 earlier regression tests remain. Of the 346 new tests, 320 (92.49%) are post-training generation tests. No fabricated test duplication or removal of proof-safety regressions is used to obtain a majority.

All full raw records, unsuccessful runs, trained checkpoints, training trajectories, selection logs, environment metadata, certificates, source hashes, results tables and reproduction entry points are included. `outcome_capsule.json` additionally deduplicates repeated circuit/contract payloads while preserving every outcome and incumbent reference; omitted detailed profiling remains in the full raw files.

`REPRODUCE.md` explains fresh-training tests, complete campaign reproduction, archived proof replay, numerical version pins, Linux/WSL assumptions, cooperative deadlines, and lack of a hard byte-memory guarantee. Core engine files were not changed after the lock; report/harness additions are distinguished by their own files and timestamps. Earlier manuscripts and results are kept as historical provenance with their earlier claim correction documented.

## Publication interpretation

The completed artifact supports a transparent bounded synthesis-and-scheduling study, including a negative result about the trained controller under the measured budgets. It **does not support a paper claiming that the proposed RL hierarchy beats strong untrained/algebraic controls**. Related-work comparison and explicit scope make the claim defensible, but do not certify novelty, reviewer acceptance, or state-of-the-art performance. Further tuning against the already exposed test set would not repair this evidence; any new empirical claim would need new independently held-out data.

## Delivery and remote publishing

Base branch: `resource-ancilla-optimality-v1`, exact base `dfa7f9081a289a0b645eea70d8192c89cc17c1eb`.
Local delivery branch: `publication-linear-certified-qcs-v1`.

The GitHub actions exposed to this session are read-only, and direct Git network access fails DNS resolution. **No remote publication or remote CI success is claimed.** The committed Git bundle, reviewable patch, complete source/evidence archive and prepared CI workflow are provided. `PUSH_TO_GITHUB.md` gives the exact safe, non-force push. The original remote branches were not changed.
