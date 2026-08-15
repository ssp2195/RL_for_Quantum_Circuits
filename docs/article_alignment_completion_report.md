# Article alignment and QFT-boundary completion report

Date: 2026-08-15
Branch: `frontier-rl`
Frozen pre-change commit: `df2ad6a036538b701cc3eb70b80f906d99d34a9a`

## Stage 0 — frozen baseline

- Python 3.12.13; NumPy 2.5.2; Gymnasium 1.3.0; pytest 9.1.1.
- Pre-change command: `python -m pytest -q`.
- Result: 118 passed in 43.83 seconds.
- Reproducibility metadata, exact budgets, seeds, target/policy digests, and
  existing GHZ/Toffoli artifact locations are in
  `baselines/article_alignment_baseline.json`.
- The Toffoli claim is limited to the fixed seven-term parity-network normal
  form. No benchmark substitutes a reference witness on failed search.

## Stage 1 — core invariants

- The archive requires a full immutable `semantic_key`; a hash-only provider
  raises `TypeError`, and forced diagnostic hash collisions cannot merge
  states.
- `CircuitDAG.add_gate` hard-fails. Public mutation is
  `CircuitState.apply_gate`; detached replay uses `CircuitDAG.from_gates`.
  `CircuitState.validate_consistency()` replays and checks semantics/resources.
- Duplicate, dominance, incomparability, reopening, frontier, archive, and
  Pareto-width events have one tested instrumentation layer shared by all
  schedulers.

## Stage 2 — canonicalization and pruning contract

- Exact Clifford-angle absorption is a fixed-point pass over the ordered
  signed Pauli-rotation word. It uses integer quarter/eighth turns and signed
  Clifford transport; anticommuting rotations are not silently reordered.
- On the diagnostic set `{T;T, S, TDG;TDG, SDG}`, absorption reduces four
  canonical keys to two. With absorption disabled, all four keys remain.
- Random short dense/symbolic properties and literal-phase regressions show no
  false merge.
- Pareto pruning uses semantic-interface equality plus one-sided monotone
  resource simulation. The theorem, assumptions, and normative manuscript
  amendment are in `docs/continuation_pruning_contract.md` and
  `docs/article_continuation_contract_amendment.md`.

## Stage 3 — article-compatible RL experiment

- `expansion_cost` implements Article equation (24): success returns
  `1-T_hit`; any failure returns `-B`, including early frontier exhaustion.
  It has no pruning, target-potential, visitation bonus, or reward clipping.
- The Equation (19) feature map is 37-dimensional:
  `[1, x_v, z_t(v), (b_t/B) x_v]`. The environment binds and advances the
  remaining expansion horizon. Frontier statistics are order invariant.
- Persistent `record_id` selection is authoritative; the fixed Gym action is
  a compatibility adapter only.
- FIFO, LIFO, uniform-cost, seeded random, target-potential, zero-policy,
  linear SARSA, Expected SARSA, and contextual-bandit implementations share
  deterministic expansion and dense certification.
- Canonicalization, duplicate-only versus Pareto reduction, Clifford
  absorption, target features, reward shaping, fairness, and visit-bonus
  ablations are executable. The last four train and freeze real scorers.

## Stage 4 — held-out native corpus

- Fixed split seeds: train 1729, validation 2753, test 3769.
- Training seed: 20260815. Validation seeds: 5 and 13. Untouched test
  evaluation seeds: 11, 23, and 37.
- Budget per bounded case: T <= 4, two-qubit <= 4, gates <= 4, depth <= 4;
  test horizon 64 expansions.
- Learning rates 0.001 and 0.0005 were compared only on validation. Both had
  validation success 1.0 and mean 1.667 successful expansions; stable declared
  order selected 0.001.
- Selected policy: linear SARSA, gamma 1, three episodes per training target,
  37 features, weight norm 0.0121274, digest
  `sha256:580e2c6fe6ae092d2322a151a5dd1f1304dc9a2c5f09c21f02c27e29cba9c54d`.

Held-out results (three targets times three seeds):

| Scheduler | Success | Mean expansions on successes | Std | Median |
|---|---:|---:|---:|---:|
| FIFO | 9/9 | 7.000 | 7.789 | 2.0 |
| LIFO | 3/9 | 1.000 | 0.000 | 1.0 |
| Uniform cost | 6/9 | 1.500 | 0.500 | 1.5 |
| Seeded random | 6/9 | 2.833 | 2.267 | 1.5 |
| Target potential | 9/9 | 1.667 | 0.471 | 2.0 |
| Zero-weight linear | 9/9 | 7.000 | 7.789 | 2.0 |
| Trained SARSA | 9/9 | 1.667 | 0.471 | 2.0 |

Every reported success was reconstructed from its returned search node and
independently dense-certified. Search consumed only each dense target, never
the retained replay witness. The corpus exposes no target-specific
reachability oracle. The analytical CCZ reference remains separately labelled
known-witness certification evidence.

## Stage 5/6 — QFT boundary and decision

- High-level, SDK-neutral `H`, `ControlledPhase(angle)`, and `SWAP` references
  are separate from the native action grammar.
- Forward/inverse analytical QFT matrices and swap/no-swap conventions pass.
- QFT-1 is `EXACT_NATIVE`; QFT-2 is `EXACT_DECOMPOSABLE`; canonical QFT-3 is
  `APPROXIMATION_REQUIRED`. Its guarded exact request contains no target.
- AQFT-3 omits only the pi/4 controlled phase. Process fidelity is
  0.8901650429449551 and maximum matrix error is 0.27059805007309856.
- Decision: retain the implemented high-level reference layer (Option A).
  Approximate Clifford+T synthesis (Option B) is the next research direction;
  ancilla-enabled exact synthesis (Option C) remains deferred.

## Validation and artifacts

Final commands:

```powershell
.\.venv\Scripts\python -m compileall -q article_benchmark.py algebra benchmarks canonical certification circuit env experiments reporting rl search qft_benchmark.py evaluate.py train.py
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python article_benchmark.py --artifacts-dir outputs\article-native-heldout
.\.venv\Scripts\python qft_benchmark.py --artifacts-dir outputs\qft3-reference
```

Final result: 168 tests passed in 41.90 seconds; compileall and
`git diff --check` passed. Generated reports are under
`outputs/article-native-heldout` and `outputs/qft3-reference`.

## Limitations and supported claim

- Dense target features/certification are deliberately small-register tools.
- The bounded native corpus has one- and two-gate targets; it establishes
  plumbing and held-out scheduling evidence, not large-circuit generalization
  or optimality.
- Literal-phase canonicalization is sound but intentionally incomplete.
- QFT diagrams are high-level references, not search witnesses; AQFT metrics
  are not exact certification.
- The repository carries the normative continuation-contract amendment; the
  user's external manuscript source was not overwritten by this code change.

The supported milestone is: the exact native engine has hardened archive,
witness, canonicalization, and Pareto invariants; its record-selection RL
baseline now matches Articles equations (19) and (24); its metrics, baselines,
splits, validation selection, and ablations are reproducible; it certifies
small held-out native targets without witness shortcuts; and it refuses to
misrepresent QFT-3 or AQFT-3 as exact native synthesis.
