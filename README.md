# RL for Quantum Circuits

This branch implements RL-guided **frontier-record selection** for small exact
Clifford+T synthesis. The policy chooses an open search record; the symbolic
engine always enumerates every legal one-gate continuation.

The exact symbolic invariant is:

```text
U = exp(i * phi * pi/8) · CliffordFrame · ordered PauliRotation word
```

`CircuitDAG` remains the authoritative witness. The dense NumPy verifier
checks a returned witness independently, up to global phase.

## Setup and verification

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e '.[dev]'
.\.venv\Scripts\python -m compileall -q .
.\.venv\Scripts\python -m pytest -q
```

The former `SanityTests.ipynb` predates the frontier/Pauli-rotation API and is
kept only as historical reference. Its executable regressions now live in
`tests/`.

## Small deterministic evaluation

```powershell
.\.venv\Scripts\python evaluate.py `
  --qubits 2 `
  --target H:0,T:1,CNOT:0-1 `
  --max-t 4 --max-depth 6 --max-gates 6 --max-steps 100
```

The simulator is deliberately small-instance only; its purpose is final
certification, not large-scale search pruning.

The article-aligned reward, record-action, metric, baseline, ablation, and
held-out evaluation definitions are fixed in
[`docs/article_experiment_contract.md`](docs/article_experiment_contract.md).
The stage-by-stage implementation evidence and final measurements are in
[`docs/article_alignment_completion_report.md`](docs/article_alignment_completion_report.md).
The one-sided continuation/resource-simulation theorem used by Pareto pruning
is documented separately in
[`docs/continuation_pruning_contract.md`](docs/continuation_pruning_contract.md).
Its normative replacement for the manuscript's earlier strict symmetric
wording is in
[`docs/article_continuation_contract_amendment.md`](docs/article_continuation_contract_amendment.md).

## GHZ-3 state-preparation smoke test

The deterministic GHZ-3 runner checks the native state-preparation witness
`H(0), CNOT(0,1), CNOT(0,2)` against the analytical
`(|000> + |111>)/sqrt(2)` state. It also asks the existing FIFO frontier
baseline to rediscover that witness under the tight three-gate resource budget.
This is a reproducible smoke test, not a trained-policy benchmark or a claim
of general state/unitary synthesis.

```powershell
.\.venv\Scripts\python ghz3_smoke.py --artifacts-dir outputs\ghz3-smoke
```

The selected directory receives JSON/CSV data, SVG probability and frontier
charts, a native-gate circuit diagram, and a Markdown summary. These files use
only the standard library and NumPy; no plotting or quantum SDK dependency is
added.

## Known exact Toffoli-3 certification

`toffoli_certify.py` is a deterministic reference-certification runner for the
fixed 15-gate native Clifford+T witness of `CCX(0,1 -> 2)`. The target is built
analytically from the LSB basis mapping (only indices `3` and `7` exchange),
while the candidate is reconstructed from the authoritative `CircuitDAG` and
checked by the independent dense simulator up to one global phase.

```powershell
.\.venv\Scripts\python toffoli_certify.py --artifacts-dir outputs\toffoli-known
```

The selected directory receives `summary.json`, `summary.md`, exhaustive
`truth_table.csv`, resource/DAG diagnostics, candidate and analytical target
matrix CSVs, and an SVG diagram generated from the actual candidate DAG. It
also records relative-phase, malformed-witness, and resource-budget negative
controls. This is a known-witness certification milestone only: it does not
train a policy, run frontier search, or claim synthesis discovery or optimality
proofs.

## Learned constrained Toffoli parity-network search

`toffoli_search.py` is the separate Stage 3 frontier-record benchmark. It
constructs the CCX target analytically, trains a policy only to rank existing
frontier records, and lets `ToffoliParityNetworkProblem` enumerate every legal
constrained continuation. It never substitutes a reference circuit: the
learned SVG is reconstructed only from a fresh learned `solution_node`.

Exact Toffoli synthesis within the fixed seven-term CCZ parity-network normal
form; not a proof of general unconstrained Clifford+T synthesis.

```powershell
.\.venv\Scripts\python toffoli_search.py `
  --artifacts-dir outputs\toffoli-search `
  --train --seed 23
```

The command returns zero only after the analytical phase/truth-table oracle,
FIFO/uniform/zero-policy baselines, seeded-random reproducibility, bounded
negative resource controls, a fresh learned witness, independent dense
validation, exact gate-resource profile, and the artifact contract all pass.
The output directory contains `summary.json`, `summary.md`, phase and resource
diagnostics, traces for every scheduler, policy weights, the learned truth
table, and generated `circuit.svg` / `frontier_size.svg`. On an unsuccessful
learned evaluation, `circuit.svg` explicitly states that no certified learned
witness was returned and shows no reference circuit.

## Linear-policy training plan

The linear SARSA policy schedules persistent frontier records; it never emits
gates.  The deterministic regression in `tests/test_linear_policy_training.py`
implements the first training stage:

1. Measure a frozen zero-weight policy on a tight two-qubit curriculum target.
2. Train the real `Trainer` for five seeded exploratory episodes.
3. Freeze exploration and require the learned scheduler to recover the same
   dense-certified witness in fewer expansions.
4. Verify that tied-priority records preserve the exact object chosen by the
   policy when the Gymnasium positional action is formed.

This verifies the SARSA scheduling loop, not trained GHZ-3 synthesis. The
target-free 16-feature schema intentionally remains available for this
baseline.

## Learned GHZ-3 frontier benchmark

`ghz3_rl.py` is a separate direct-training benchmark. It adds an opt-in
three-qubit dense target context, so the linear record ranker can distinguish
the target-labelled qubits and directed CNOT prefixes. Its 60-coordinate
schema includes the original 16 resource/context features plus process and
probe progress, labelled wire/gate data, directed CNOT slots, last-operation
data, target-relative frontier context, and a bias term.

```powershell
.\.venv\Scripts\python ghz3_rl.py --artifacts-dir outputs\ghz3-rl
```

The default direct protocol trains for 50 seeded four-selection episodes with
potential-shaped rewards, then evaluates a frozen policy with `epsilon=0` and
fairness disabled. The zero-weight and learned evaluations both receive a
32-selection cap; on the calibrated fixed seed, the zero policy certifies in
25 expansions and the learned policy certifies in 3. The returned witness is
reconstructed only from the search `solution_node` and independently checked
against the full target unitary and GHZ-positive state.

The artifact directory contains a JSON summary, training history and evaluation
trace CSV files, policy metadata/weights, SVG circuit and frontier charts, and
a Markdown handoff. No reference circuit is substituted if learned search
fails: the report is unsuccessful and the diagram explicitly says that no
certified witness was returned.

This is evidence for a target-aware linear ranking policy on this labelled,
small GHZ-3 search only. It is not evidence of generalization to arbitrary
circuits, larger registers, or state-preparation targets.

## Legacy held-out native Clifford+T target corpus

`benchmarks/native_corpus.py` generates two deterministic target suites from
the actual unrestricted native grammar (`H`, `S`, `SDG`, `T`, `TDG`, and every
directed `CNOT`) on one, two, and three qubits. The semantic/property suite uses
longer short witnesses; the bounded-synthesis suite uses one- or two-gate
witnesses so scheduler comparisons remain practical. Its multiqubit cases are
transparently conditioned to contain at least one directed CNOT, while gate
selection otherwise remains seeded and uniform over the native action list.

The fixed partition seeds are `1729` (train), `2753` (validation), and `3769`
(test). Partition identity is a global-phase-normalized dense-unitary digest,
not witness syntax, so semantically equal generator circuits cannot cross a
split. Each record retains its generator witness solely for replay and audit;
generic search receives `case.synthesis_target()` and has no target-specific
reachability oracle. The module also exposes a separately labelled analytical
CCZ target and the fixed native diagonal witness obtained from the established
`CCZ = H(2) CCX H(2)` relation. This reference witness is not generic-search
evidence.

Run this retained pre-Article-V1 train/test benchmark and its historical tiny
ablations with:

```powershell
.\.venv\Scripts\python article_benchmark.py `
  --artifacts-dir outputs\article-native-heldout
```

The default uses training seed `20260815` and evaluation seeds `11`, `23`, and
`37`. Learning rates `0.001` and `0.0005` are compared only on the validation
split before the untouched test split is opened. The runner writes the model
selection record, complete corpus replay manifest, TD/weight diagnostics,
individual and aggregate held-out scheduler results, trained-policy
ablations, and a Markdown summary. Search receives dense targets only; corpus
witnesses appear solely in the replay manifest.

## Article V1 publication workflow

The versioned Article V1 path is separate from the legacy command above.
`benchmarks/article_native_corpus.py` generates only two- and three-qubit
targets from the unchanged native grammar. The pilot primary corpus is exactly
five easy (witness length 2–3), five medium (4–5), and five hard (6–8) targets
across train/validation/test, plus four separately labelled length-OOD targets.
Generator witnesses are reachability/audit metadata only and are absent from
the evaluation surface.

The seven primary schedulers are FIFO, LIFO, uniform cost, seeded random,
zero-weight Article linear, direct Article target distance, and trained Article
SARSA. Every scheduler uses the same `ArticleTargetContext`, amended
`article_v1_expansion_potential` reward and configured `beta`, native grammar,
resource/expansion budgets, canonicalizer, Pareto archive, and independent
Article certifier. Frozen policies do not consume evaluation rewards. OOD SARSA
uses separate `ood-seed-*.json` checkpoints trained only on train targets whose
generator length is at most four.

Article timing separates record ranking, structural feature construction, and
dense target-metric evaluation. `wall_time_ns` is the complete evaluation
envelope, including reset, selection, features, environment steps, and stopping
logic; the narrower accumulated step-body timer is separately named
`environment_step_time_ns`. `frontier_sum` samples actual selection states: the
root and each nonterminal/nontruncated successor. The compatibility
`frontier_mean` intentionally retains the older root-plus-post-expansion
convention and is not the Article decision-state mean. The zero-weight linear
control still materializes the same Article 31D feature/target-metric pipeline;
only its coefficients are zero.

Learned checkpoints use the fail-closed
`article-v1-transferable-linear-checkpoint-v2` contract. The checkpoint digest
binds its feature schema, standard/OOD family, corpus config, training-scope
mode, ordered training target IDs, training `beta`, certification tolerance,
episodes per target, learning rate, epsilon schedule, training seed, optional
expansion cap, budget policy, effective per-target training budgets, and
weights. Evaluation additionally binds an explicit scope, its allowed training
seed set, and an exact training-partition match, so a foreign-corpus,
held-out-leaking, incomplete, OOD, or feature-ablation checkpoint cannot be
silently used as the primary learner. Standard, OOD, and ablation campaign
checkpoints use their complete declared train partition; mini-CI alone is
explicitly labelled `explicit_partial_smoke`. Raw-run identity includes both
the checkpoint training seed and source-worktree digest. Resume validates
immutable run/environment/corpus manifests and existing checkpoints before
reuse; a compatible checkpoint is loaded, not retrained or rewritten, while
stale/conflicting content fails closed. Pre-V2/pre-identity ledgers require a
new run ID.

The root CLI dispatches the Article V1 workflow:

```powershell
.\.venv\Scripts\python article_benchmark.py mini-ci `
  --output-root outputs\article_v1 --run-id mini-ci-v5-new
.\.venv\Scripts\python article_benchmark.py pilot `
  --config configs\article_v1_pilot.json `
  --output-root outputs\article_v1 --run-id pilot
.\.venv\Scripts\python article_benchmark.py generate-corpus `
  --config configs\article_v1_publication.json
.\.venv\Scripts\python article_benchmark.py train `
  --config configs\article_v1_publication.json
.\.venv\Scripts\python article_benchmark.py evaluate `
  --config configs\article_v1_publication.json
.\.venv\Scripts\python article_benchmark.py aggregate `
  --config configs\article_v1_publication.json
.\.venv\Scripts\python article_benchmark.py ablations `
  --config configs\article_v1_publication.json
```

The ablation command writes `ablations.csv`, `ablation_results.json`, and
variant checkpoints. Its Pareto-off arm retains exact duplicate suppression
but disables resource-dominance rejection/removal. Its `raw_witness`
canonicalization arm disables commuting reorder, cross-position fusion, and
Clifford-angle absorption while remaining sound because distinct DAG words are
never merged.

An Article V1 run directory includes run/environment/report metadata,
combined and per-split corpus manifests, standard and OOD checkpoints,
resumable `raw_runs.jsonl`, per-target/success/timing CSVs, paired tables, SVG
figures, per-learner `tables/learner_seed_results.csv`, between-learner
`tables/learner_seed_summary.csv`, and the completion summary. Success curves
contain only budgets actually executed for that target/method; reporting never
manufactures a lower-budget point from a larger-cap trajectory. Artifact-map
paths below the repository are serialized as portable repository-relative
POSIX paths.

The current `outputs/article_v1/final-mini-ci-v5` artifact passes all eleven
mini semantic checks. It contains nine raw scheduler records, uses a seed-19
scope-validated SARSA checkpoint explicitly marked as partial-smoke training,
independently certifies the reachable target under FIFO, and uses no
generator/reference-witness fallback. A same-ID resume appended zero records
and skipped all nine without changing `raw_runs.jsonl`; it also loaded the
existing checkpoint without training or rewriting it.
The configured pilot and full publication campaigns have not been run, so the
repository makes no scheduler-superiority, optimality, or scalability claim.

## QFT reference and exact-capability boundary

`benchmarks/qft.py` defines SDK-neutral reference-only `H`,
`ControlledPhase(angle_pi)`, and `SWAP` operations. Qubit 0 is the
least-significant basis bit, and the forward swapped convention is

```text
F[j,k] = exp(+2*pi*i*j*k/N) / sqrt(N).
```

Forward QFT without swaps has bit-reversed outputs; its inverse has
bit-reversed inputs. Both conventions have separate declared target matrices.
The operation-derived QFT-3 forward/inverse matrices are tested against the
independently constructed analytical 8x8 matrices and phase-sensitive basis
columns.

Before an exact target reaches native search,
`prepare_native_exact_search(...)` applies a machine-readable capability
guard. QFT-1 is `EXACT_NATIVE`, QFT-2 is `EXACT_DECOMPOSABLE`, and canonical
QFT-3 is `APPROXIMATION_REQUIRED` because the present no-ancilla model has no
registered exact lowering for its controlled phase below pi/2. Consequently a
QFT-3 request carries no exact `SynthesisTarget` and cannot be reported as a
false native-search success.

`aqft3_metrics(...)` is a distinct approximate benchmark that omits exactly
the pi/4 controlled phase and reports the complete operation/omission metadata,
process fidelity, maximum elementwise matrix error, selected state fidelities,
and swap/permutation convention. Approximate metrics are never accepted by the
exact certifier. These reference gates do not modify `GateType` or native
frontier expansion.

Generate the machine-readable report and high-level diagrams with:

```powershell
.\.venv\Scripts\python qft_benchmark.py `
  --artifacts-dir outputs\qft3-reference
```

The command exits successfully only when analytical forward/inverse checks,
swap and no-swap conventions, the exact native capability guard, the declared
AQFT omission/fidelity, and the unchanged native action grammar all pass. It
writes `summary.json`, `summary.md`, `exact_qft3.svg`, and `aqft3.svg`; both
diagrams explicitly identify themselves as high-level references rather than
native witnesses.
