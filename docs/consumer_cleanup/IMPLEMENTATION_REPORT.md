# Consumer-aware cleanup: proofs, implementation and measured results

## Main conclusion

The new default uses exact algebra to decide which temporary values are needed, and exact finite scheduling to reuse their qubits. It does not ask reinforcement learning to rediscover a known workspace minimum. All original native search and theorem-cleanup code remains unchanged; these are new optional entry points.

The finite comparison preserves full-bank and bank-free controls. Its selected circuits reduce T-count and workspace on all five existing development problems. The benefit comes from the constructions, not trained coefficients. Keeping fewer values can require more measurement rounds and worse physical latency.

Base commit: `84c55c94f4b4c33ca55cbdbb9c7738222d6b32b8`.

## The three requested changes

**A. Use the analytic workspace result first.** For the prescribed full bank, the checked smaller-side circuit attains rm+min(r,m); this closes that workspace coordinate without search. Failure of that particular schedule under another cap is not an infeasibility proof. Larger-helper layouts are retained for other resource trade-offs. Optional trained search is reserved for scheduling.

**B. Simplify the complete oracle before materialization.** Substitute the temporary definitions into the consumer and cancel identical Boolean terms. An exact binary matrix factorization replaces collections of products by products of parities. An additional deterministic quadratic elimination avoids computing every quadratic monomial separately. Neither elimination uses fitted weights or benchmark-selected pivots. All CNOTs for computing and undoing parities are emitted and charged.

**C. Reuse qubits at first/last use.** For fixed factors, a subset recurrence finds the exact minimum peak among non-recomputing, one-factor-at-a-time schedules. It has at most 2^d states for d factors; the default exact limit is d=16. Larger instances keep a labelled fixed-order upper bound. A separate recomputation construction is retained when it provides a different trade-off.

Proofs are in `PROOFS.md`: the full-bank bound-attainment proposition; the complete-oracle equivalence lemma; the separated-factor construction; its quadratic-compression corollary; the exact streaming recurrence; and clean block composition. The rank and schedule minima are restricted mathematical statements, not global T-count or global oracle-workspace optima.

## Quantum resources on the original five problems

Each comparison below is old frozen SARSA/LinUCB full-bank generation versus the new deterministic complete-oracle selection. The default lexicographic objective is T-count, auxiliary peak, modelled latency, native gates, then CNOTs. All nondominated candidates remain accessible for other objectives.

| Bank | T: old -> new | Auxiliaries: old -> new | CNOTs: old -> new | Native gates: old -> new |
|---|---:|---:|---:|---:|
| 3 x 1 | 16 -> 8 | 4 -> 2 | 19 -> 21 | 83 -> 51 |
| 4 x 2 | 40 -> 8 | 10 -> 1 | 47 -> 11 | 205 -> 45 |
| 5 x 2 | 48 -> 12 | 12 -> 2 | 60 -> 22 | 258 -> 68 |
| 6 x 3 | 84 -> 36 | 21 -> 4 | 108 -> 83 | 462 -> 219 |
| 8 x 3 | 108 -> 56 | 27 -> 5 | 138 -> 120 | 592 -> 324 |

T-count reductions are 50.00%, 80.00%, 75.00%, 57.14%, 48.15%.

These improvements do not contradict the old theorem. The new circuits do not materialize the complete bank, so its lower bound does not apply to them.

## Measurement and latency trade-offs

The clock model is inherited unchanged: one tick per native gate, five for Z measurement, and two for result availability, with explicit stage barriers. It is an all-to-all illustrative model, not a hardware experiment.

| Bank | Old native depth | New native depth | Old ticks | New ticks | New measurement rounds |
|---|---:|---:|---:|---:|---:|
| 3 x 1 | 57 | 44 | 71 | 58 | 2 |
| 4 x 2 | 79--84 | 45 | 93--98 | 59 | 2 |
| 5 x 2 | 93--103 | 58 | 107--117 | 79 | 3 |
| 6 x 3 | 117--143 | 164 | 131--157 | 213 | 7 |
| 8 x 3 | 145--179 | 192 | 159--193 | 248 | 8 |

The two largest T/space-first results have worse latency than the old construction. Selecting latency first retains the larger-workspace full-bank alternative; no componentwise superiority is claimed.

## Same-machine classical computation

All figures include construction, independent checking, and a persistent hybrid-block circuit witness. The menu times charge all considered constructions, not just the selected one. Serialization is subsequent. Target-independent primitive preparation and training are recorded separately. Method order was shuffled with a fixed seed.

| Method | Correct / runs | Median milliseconds |
|---|---:|---:|
| Previous SARSA/LinUCB | 75 / 75 | 19.033 |
| Same trained policies, proven helper side | 75 / 75 | 14.185 |
| Untrained, proven helper side | 15 / 15 | 11.377 |
| Direct bound-attaining full bank | 15 / 15 | 5.913 |
| New deterministic complete-oracle menu | 15 / 15 | 10.081 |
| Same menu plus learned schedule | 75 / 75 | 26.331 |
| Direct factor-and-stream construction | 15 / 15 | 1.273 |
| Direct factor construction, all features | 15 / 15 | 1.213 |
| Rank factors without quadratic compression | 15 / 15 | 1.371 |
| Old bank-free monomial control | 15 / 15 | 0.962 |
| Rank factors with feature recomputation | 15 / 15 | 1.194 |

The ratio of pooled medians, previous hierarchy / new deterministic menu, is 1.89. This is descriptive, not a paired population estimate. There are only five logical targets. Seeds and timing repetitions are not additional independent tasks.

Every trained variant completed both learning stages before circuit generation. However, every selected result from the new menu with an optional learned candidate is still algebraically constructed. Its additional policy calls do not improve T-count or workspace on these cases, and cost time. This study does not establish SARSA/LinUCB superiority.

## A genuine shared-lifetime schedule

A separate constructed example has four phase factors, sharing four quadratic temporary values in alternating groups. Keeping all values or using the canonical factor order needs five auxiliary qubits. The exact first/last-use order needs three, with T-count unchanged at 32. Recomputing each factor separately also uses three qubits, but needs 48 T gates. All complete circuits and branches are independently checked. This is a mechanism diagnostic, not a held-out performance claim.

## Correctness and testing

The primary study has 345 records; independent replay checks 67 distinct returned protocols and 2,425,280 input/outcome-generator pairs.

Each accepted protocol has the branch identity K_s J = 2^(-M/2) J O_p for every measurement transcript. The implementation checks exact Boolean functions, every byproduct coefficient, coherent input restoration, and all clean auxiliary outputs. The local gates are independently verified over cyclotomic integers. Small cases additionally replay every native Kraus branch as an exact matrix. The numerical counts in the study are from emitted instructions, not supplied theoretical estimates.

The source adds 198 tests, including 136 actual post-training circuit generation tests that exercise both frozen controllers before comparing with the new analytic constructions. These analytic winners are not mislabelled as RL discoveries. The 1,074 previous tests are preserved. The complete suite therefore has 1,272 tests; its execution result and timing are recorded separately in the validation logs.

Independent tests cover all 64 two-by-three binary matrices, all 1,024 four-variable quadratic polynomials without a constant term, exhaustive order comparisons on 24 five-factor examples, every outcome of small native circuits, shared-input block composition, corrupted receipts, measurement/cleanup ordering, and incompatible resource caps.

## Reproduction and limitations

Run `python -m hybrid_qcs.cleanup.consumer_runner --output-dir outputs/new-study` for a full fresh campaign, and add `--stage verify` for independent archived replay. Use `python -m hybrid_qcs.cleanup.consumer_diagnostics --output-dir outputs/new-study` for the shared-lifetime example.

The original five targets are reused development problems, purposefully amenable to the source theorem. No model or elimination pivot was tuned on their new comparison outcomes; the algorithms were specified from the proofs. This is not preregistered independent confirmation. The full-native search remains a separate engine with its original small-width dense-state limit. The new method is structured phase-oracle compilation, not arbitrary large-register noncommuting synthesis.

There is no comparison against executed AlphaTensor-Quantum, Exact T Library, or other optimized external compilers. There is no claim of a new temporary-AND identity, a new quadratic normal form, globally optimal T-count, or established publication priority. The gains show that the preceding architecture should sometimes be avoided, and that its proven local optimum should otherwise be used directly.

## Sources and proof dependencies

The supplied *H - Prove Cleanup Theorem(3).pdf*, pp. 24--25 and 59--62, provides the retained-parent rule, the restricted full-bank lower bound, and the counterexample to unrestricted oracle optimality. New derivations are stated and proved separately in `PROOFS.md`.

Craig Gidney, *Halving the cost of quantum addition*, Quantum 2, 74 (2018), DOI 10.22331/q-2018-06-18-74: existing four-T temporary AND and measured cleanup.

Meuli et al., *The Role of Multiplicative Complexity in Compiling Low T-count Oracle Circuits*, arXiv:1908.01609: existing Boolean factorization and low-T oracle compilation. These corollaries are not claimed as a new general Boolean-synthesis principle.

Paradis et al., *Reqomp: Space-Constrained Uncomputation*, Quantum 8, 1258 (2024), DOI 10.22331/q-2024-02-19-1258: relevant prior space/recomputation trade-offs. The schedule optimum here is explicitly limited to the fixed factors, atomic factor execution, and no feature recomputation.
