# Article V1 experiment protocol

## Status and source boundary

This document is the preregistered repository protocol for Article V1. It
combines the scientific requirements of the external
`Article_limited_scope.md` with explicit operational completions from
`codex_plan_article_v1_codebase_alignment.md`. The latter include the exact
feature normalizations, executable \(r_{\mathrm P}\) and \(\nu\) formulas,
the amended Eq. 104 reward, metric-counter boundaries, schema names, and the
statistical interval convention. Those additions are not represented as
verbatim external-manuscript text. The external manuscript has not been
modified.

The checked-in `configs/article_v1_pilot.json` and
`configs/article_v1_publication.json` freeze intended settings.
`experiments/article_v1_runner.py`, `experiments/article_v1_ablations.py`, and
`reporting/article_v1.py` implement the Article V1 command, seven-scheduler,
ablation, raw-store, aggregation, and artifact surfaces; `article_benchmark.py`
dispatches recognized subcommands to that runner while retaining its older
no-subcommand workflow. Focused tests include an end-to-end, resumable mini-CI
matrix, and `outputs/article_v1/final-mini-ci-v5/mini_ci_summary.json` records
`passed: true` with all eleven semantic checks true. That smoke result is not a
pilot or publication result. The configured pilot and full publication
campaigns have not been run.

## Scientific question and fixed search system

The experiment asks whether a linear semi-gradient SARSA policy allocates a
finite record-expansion budget more effectively than fixed non-learning
schedulers, especially a direct target-distance heuristic, while every other
synthesizer component is fixed.

The experimental domain is small-scale, exact, ancilla-free, all-to-all
Clifford+T synthesis on two- and three-qubit targets. The native grammar is

```text
H(q), S(q), SDG(q), T(q), TDG(q) for every logical qubit q
CNOT(control, target) for every ordered pair control != target
```

There are no ancillas, connectivity/routing constraints, SWAP insertion,
arbitrary-angle gates, approximate synthesis, target-specific reachability
oracles, or learned gate choices. An action selects one persistent frontier
record. Expanding it deterministically considers every native gate, constructs
every resource-feasible one-gate child, updates the authoritative `CircuitDAG`
and symbolic state, canonicalizes, applies same-key Pareto logic, and performs
independent dense certification according to the fixed terminal policy. RL
controls only the next-record ranking.

All schedulers receive the same target and `ArticleTargetContext`, amended
`article_v1_expansion_potential` reward with the configured `beta`, native
grammar, resource budget, external expansion budget, canonicalizer, archive,
Article V1 certification engine/tolerance, stopping rule, and instrumentation
mode. A frozen policy does not consume evaluation rewards, but reward and
target-metric computation remain common environment behavior and common cache
accounting. Stable record ID is used only to break exact ranking ties and is
never a feature.

## Targets, splits, and witness prohibition

`benchmarks/article_native_corpus.py` deterministically generates reachable
targets from the native grammar. Primary targets contain exactly two or three
qubits, with strata

```text
easy:   generator witness length 2-3
medium: generator witness length 4-5
hard:   generator witness length 6-8 in the checked-in profiles
```

Generator length is reachability metadata, not an optimal circuit length.
The generator witness may be retained for manifest replay and audit, but it
must not be exposed through the evaluation target, features, scheduler,
legality, preferred path, stopping rule, terminal predicate, or failure
fallback. A failed search remains failed; no reference witness is substituted.

The identity-disjoint splits are `train`, `validation`, `test`, and
`ood_test`. Identity and the projective identity target are rejected using the
global-phase quotient discrepancy \(\Delta_\phi\) with the separately named
`tau_identity`. The length-OOD comparison uses `ood_test` targets of length
5–8 and separately trained `ood-seed-*.json` SARSA checkpoints whose training
targets are restricted to the `train` split with generator length at most 4.
The ordinary in-distribution checkpoints remain trained on the complete train
split; the two checkpoint families are never silently interchanged. Corpus
manifests serialize target ID, split, difficulty, qubit count, seeds, witness
audit metadata, length, target/dense digests, shape, resource and expansion
budgets, generation schema, identity tolerance, and
`target_specific_reachability_oracle=false`.

The pilot primary corpus is exactly five easy, five medium, and five hard
targets across `train` (six), `validation` (three), and `test` (six), plus four
separately labelled OOD targets, for 19 total manifest targets. The publication
config contains 75 train, 30 validation, 75 held-out test (25 per difficulty),
and 50 OOD targets. These are configured corpus counts, not claims that every
scheduler run has been executed.

## Article target metric, feature map, and tolerances

For \(d=2^n\), target \(U_\star\), and candidate \(U_v\) reconstructed from
the complete DAG witness,

\[
d_{\mathrm{tar}}(v)=1-
\frac{|\operatorname{Tr}(U_\star^\dagger U_v)|^2}{d^2}.
\]

The `complex128` computation is global-phase invariant, clips only harmless
roundoff into \([0,1]\), rejects nonfinite/dimension-mismatched inputs, and is
cached by target fingerprint plus complete DAG-witness identity, never record
ID. It is a ranking/reward metric, not terminal acceptance.

The exact ten ordered candidate coordinates are

```text
t_count
two_qubit_count
gate_count
depth
rotation_count
anticommuting_pair_count
mean_pauli_weight
target_process_infidelity
frontier_resource_dominance_fraction
archive_novelty
```

Set `B_T=max_t_count`, `B_g=max_gates`, `B_D=max_depth`, and
`B_2q=max_two_qubit_count` when present (otherwise `B_2q=B_g`). Resource
coordinates are divided by these fixed corresponding budgets using
`a / max(1, budget)`. Rotation count is divided by `max(1, B_T)`,
anticommuting pairs by `max(1, binomial(B_T, 2))`, mean Pauli weight by
`max(1, n)`, and target process infidelity is already normalized.

For an immutable frontier snapshot,

\[
r_{\mathrm P}(v;\mathcal F_t)=
\frac{|\{u\ne v:\boldsymbol\rho(u)\preceq\boldsymbol\rho(v)\}|}
{\max(1,|\mathcal F_t|-1)}.
\]

It compares resources across open records for ranking only; lower is better,
a singleton gets zero, and it does not change same-key pruning semantics.

Let \(g_t(k)\) count every resource-feasible generation of canonical key
\(k\), including accepted, exact-duplicate-rejected, and
dominance-rejected records; count the root once. Novelty is

\[
\nu_t(v)=1/\sqrt{\max(1,g_t(\kappa(v)))}.
\]

It is not a policy visit count or reward. Population frontier mean and standard
deviation use a deterministic reduction and fixed \(\eta=10^{-8}\):

\[
\mathbf z_t(v)=
(\mathbf x_v-\boldsymbol\mu_t)/(\boldsymbol\sigma_t+\eta\mathbf1).
\]

The singleton result is all-zero. After exactly \(t\) expansions,
\(\widehat b_t=(B_{\mathrm{exp}}-t)/B_{\mathrm{exp}}\), and the immutable
`float64` feature vector is

\[
\varphi(s_t,v)=[1,\mathbf x_v,\mathbf z_t(v),\widehat b_t\mathbf x_v]^T
\in\mathbb R^{31}.
\]

The selected pre-transition vector is frozen for the SARSA update. The full
ordering lives in `docs/article_v1_feature_contract.md`; the target-free 28D
and no-frontier-context 21D ablations have distinct schemas.

For final certification,

\[
c_\phi(U,V)=|\operatorname{Tr}(V^\dagger U)|/d,
\qquad
\Delta_\phi(U,V)=\sqrt{1-c_\phi(U,V)}.
\]

`certification/article_v1.py` accepts only when
\(\Delta_\phi\le\tau_{\mathrm{cert}}\) after fresh DAG replay. The checked-in
profiles set `tau_cert=1e-9` through `experiment.certification_tolerance` and
set corpus `tau_identity=1e-7`. These tolerances have separate scientific
roles and must not be treated as interchangeable. Also,
\(d_{\mathrm{tar}}=1-c_\phi^2\) is not \(\Delta_\phi\).

## Learning and reward

Article V1 uses linear semi-gradient SARSA:

\[
Q_\theta(s_t,v)=\theta^T\varphi(s_t,v),\qquad
\delta_t=R_{t+1}+Q_\theta(s_{t+1},v_{t+1})-Q_\theta(s_t,v_t),
\]

\[
\theta\leftarrow\theta+\alpha\delta_t\varphi(s_t,v_t),
\qquad \gamma=1.
\]

The bootstrap record is the actual next epsilon-greedy behavior action and is
reused on the next iteration. If an environment override changes the selected
record, learning uses the record actually expanded. Frozen evaluation uses
`epsilon=0`, fairness interval 0, immutable weights, and a fresh environment,
frontier, and target-metric cache. Missing or zero-weight checkpoints may not
be labelled `article_sarsa`.

Article learned policies serialize as
`article-v1-transferable-linear-checkpoint-v2`. The digest binds the feature
schema, checkpoint family, training-scope mode, corpus-config digest, ordered
training-target IDs, training `beta`, certification tolerance, episodes per
target, learning rate, epsilon schedule, training seed, optional expansion cap,
budget-policy name, ordered effective per-target expansion budgets, and
weights. Before evaluation, a separate fail-closed scope binds those fields,
the allowed training-seed set, and held-out/permitted evaluation IDs. The
checkpoint training IDs and effective budgets must exactly equal the scope—not
merely be subsets—and the checkpoint seed must belong to the declared seed set.

Primary standard evaluation requires the complete configured train partition.
Length-OOD requires the complete eligible train partition with generator length
at most four. Ablation evaluation binds the complete declared train partition,
the variant feature schema/`beta`, and validation-only held-out scope. Only the
mini-CI uses `explicit_partial_smoke`, with explicit training IDs and cap; that
checkpoint cannot pass a complete-training scope and is not publication learner
evidence. A structurally valid checkpoint from another scope is rejected. A V1
or otherwise pre-scope checkpoint is not upgraded by inference and must be
retrained/serialized as V2.

The explicit plan amendment to Eq. 104 is

\[
R_{t+1}^{(0)}=-1+B_{\mathrm{exp}}\mathbb1[\text{certified success}]
-(B_{\mathrm{exp}}-(t+1))\mathbb1[\text{terminal failure}].
\]

It gives base return \(B_{\mathrm{exp}}-T\) for success after \(T\)
expansions and \(-B_{\mathrm{exp}}\) for every failure, including early
frontier exhaustion. Shaping is

\[
R_{t+1}=R_{t+1}^{(0)}+\beta(\Psi(s_{t+1})-\Psi(s_t)),
\qquad \Psi(s)=-\min_{v\in\mathcal F(s)}d_{\mathrm{tar}}(v),
\]

with terminal potential zero. There is no clipping, trainer visit bonus,
pruning/dead-end bonus, second success bonus, best-generated-child term, or
support/entanglement term in this profile. `beta=0` is the no-shaping ablation.

## Seven primary schedulers

Run exactly these primary schedulers for each held-out target and each
preregistered expansion budget:

1. `fifo`: minimum persistent record ID.
2. `lifo`: maximum persistent record ID.
3. `uniform_cost`: lexicographically minimize
   `(t_count, two_qubit_count, gate_count, depth, record_id)`; this freezes the
   existing repository cost rule.
4. `seeded_random`: uniform record sampling with its serialized evaluation
   seed.
5. `zero_weight_linear`: the Article V1 31D linear policy with every weight
   exactly zero and stable-ID tie-breaking. It still materializes the same
   Article feature batches and dense target-metric coordinates as the trained
   linear scheduler, and records their counts/times; zero weight is a scoring
   control, not permission to bypass feature computation.
6. `article_target_distance`: minimize \(d_{\mathrm{tar}}\), breaking ties by
   lowest persistent record ID; it has no learned weights or witness access.
7. `article_sarsa`: a nonzero trained Article V1 checkpoint, evaluated frozen.

Expected SARSA, a contextual bandit, the legacy 37D policy, and the composite
`target_potential` scheduler are not primary Article V1 schedulers. They may be
reported only as clearly labelled supplementary methods. The exact seven-name
entry point and common-environment contract are covered by focused runner and
scheduler tests. The configured pilot and publication matrices remain unrun.

## Counter and frontier-sampling taxonomy

The event counters have nonoverlapping definitions:

- `num_expanded`: valid selected records actually expanded.
- `num_gate_attempts`: native gates considered before resource feasibility.
- `num_generated`: successfully constructed, resource-feasible children before
  canonical/Pareto rejection.
- `num_exact_duplicate_rejections`: same semantic key and exactly equal
  resource vector; equality is not also dominance rejection.
- `num_dominance_rejections`: a same-key active record strictly weakly
  dominates the new record, excluding equality.
- `num_dominance_replacements`: number of active records tombstoned because a
  new accepted record strictly dominates them.
- `num_pareto_incomparable_acceptances`: accepted at an already-seen key and
  incomparable with every currently active record there.
- `num_reopenings`: a newly accepted open record at a key where any historical
  record was previously expanded. It may coexist with replacement or
  incomparability and is not synonymous with either.

Article V1 `frontier_sum` is a decision-state statistic. Sample \(F_0\)
immediately after root insertion, then sample the resulting frontier after a
valid expansion only when that result is nonterminal and nontruncated and will
therefore be the next selection state. Terminal/truncated states are not
decision states and are excluded; invalid compatibility-layer Gym actions do
not expand a record and add no sample. If \(\mathcal D\) is the ordered set of
actual decision states,

\[
\texttt{frontier_sum}=\sum_{s_t\in\mathcal D}|\mathcal F(s_t)|,\quad
\texttt{frontier_observation_count}=|\mathcal D|,\quad
\texttt{frontier\_decision\_mean}=\texttt{frontier_sum}/
\texttt{frontier_observation_count}.
\]

For an episode whose root is not already terminal and that terminates on its
\(T\)-th valid expansion, there are \(T\) sampled decision states: the root
plus the \(T-1\) nonterminal successors. The compatibility metric
`frontier_mean` deliberately retains its
older convention: root plus a post-expansion sample after every valid
expansion, including the terminal/truncated result. Never substitute
`frontier_mean` for the Article decision-state mean or combine their
denominators.

Also retain `frontier_peak`, `archive_record_count` (historical admitted
records), `active_archive_peak`, and `maximum_pareto_antichain_width`. Minimum
memory reporting is `peak_frontier_records`, `peak_active_archive_records`,
and maximum antichain width; an optional process-RSS or `tracemalloc` measure
must serialize the measurement method.

## Timing taxonomy and invariance

Use `time.perf_counter_ns()`. `wall_time_ns` is the end-to-end evaluation
envelope from evaluation start through reset, frontier ranking, feature
construction, every environment step, stopping logic, and result extraction.
`runtime_seconds` is this same value in seconds. The narrower sum of executed
environment `step()` bodies is separately named `environment_step_time_ns`;
it must never be presented as total evaluation wall time. Wall time is reported
separately and is not the sum of phase counters. Phase counters are exclusive:

- `ranking_time_ns`: scheduler ordering, scoring comparisons, and tie-breaking,
  including externally timed fixed-scheduler selection but excluding feature
  construction and target-metric evaluation.
- `feature_time_ns`: structural candidate/batch construction and
  standardization, excluding separately timed dense target metrics.
- `target_metric_time_ns`: dense process-infidelity cache misses/evaluations.
- `symbolic_update_time_ns`: native legality/resource checks and symbolic/DAG
  child construction.
- `canonicalization_time_ns`: semantic-key/canonical-form computation only.
- `archive_time_ns`: archive comparison, insertion, rejection, and retirement
  after a key is available.
- `certification_time_ns`: independent DAG replay and final comparison.
- `reporting_time_ns`: serialization and report construction outside search
  wall.

Serialize `feature_evaluation_count`, `target_metric_evaluation_count`, cache
hits/misses and hit rate, and `certification_count`. Raw runs retain the
separately named environment-step diagnostic; the aggregate timing table
reports total wall, ranking, feature, target-metric, canonicalization, archive,
and certification time, plus feature/dense evaluations and peak
frontier/archive. Keep expansion-normalized and wall-clock-normalized
conclusions separate. Turning instrumentation off must preserve tie-breaking,
archive order, random-number consumption, features, success/failure, and
generated witnesses; only timing fields may differ.

## Seeds, raw records, and frozen evaluation

The publication config declares five independent SARSA training seeds
`[19, 23, 29, 31, 37]` and ten random scheduler seeds per target
`[3, 7, 11, 13, 17, 19, 23, 29, 31, 37]`. The pilot uses two and three,
respectively. Every seed must appear in raw output.

Use append-only JSONL with atomic writes and fail-closed resume. The unique run
key contains target identity, scheduler, resource and expansion budgets,
training seed, checkpoint digest, evaluation seed, feature/reward/certification
schemas and parameters, search-reduction settings, commit version, and the
source-worktree digest. Training seed remains an independent identity field
even when two learner seeds happen to produce identical weight digests.

On reuse of a nonempty run ID, the runner compares the existing immutable run,
environment, combined-corpus, and per-split manifests against the current
config/profile/code-worktree/corpus contract before loading the raw ledger.
Missing, corrupt, stale, or conflicting immutable content is rejected, not
overwritten. A partial final JSONL line is not a completed record and may be
repaired atomically; a conflicting completed key is an error. Old ledgers
created before checkpoint V2 or before the training-seed/source-worktree
identity fields are intentionally incompatible and require a new run ID.
Existing checkpoints are also fail-closed immutable artifacts: a compatible
resume validates and loads them without retraining or rewriting, while a
missing, corrupt, or contract-conflicting checkpoint aborts the resume.

## Metrics and statistics

For every scheduler and budget \(B\), report

\[
S_\pi(B)=\frac{1}{N_{\mathrm{test}}}
\sum_i\mathbb1[T_{\mathrm{hit}}^{(i,\pi)}\le B],
\]

with failures in the denominator, and

\[
E_\pi(B)=\mathbb E[T_{\mathrm{hit}}\mid T_{\mathrm{hit}}\le B]
\]

only beside its success curve. Never rank methods by successful-run expansions
while hiding failures.

Every reported budget point must have a raw trajectory actually executed with
that exact external expansion cap for that target, scheduler, seed, and
checkpoint. A trajectory run at a larger cap may not be retrospectively sliced
to synthesize an unexecuted smaller-budget point, even if its recorded hit time
would fall below that smaller value. Requested budgets absent from the raw
ledger are omitted rather than inferred.

The held-out target is the primary paired statistical unit. Use a deterministic
95% percentile bootstrap with 10,000 target resamples and the serialized
`statistics_seed`. Retain every target-by-random-seed record, average random
seeds within each target, and bootstrap targets rather than treating random
trajectories as independent targets. For SARSA, preserve and report every
checkpoint seed plus mean, median, standard deviation, and uncertainty across
targets and learner seeds; do not pool five checkpoints without documenting
the aggregation. `tables/learner_seed_results.csv` reports each frozen learner
seed/checkpoint separately across targets;
`tables/learner_seed_summary.csv` reports the preregistered target-first
aggregation and explicitly named between-learner statistics. Preserve paired
per-target differences. Confidence-interval overlap or non-overlap alone does
not establish superiority.

## Ablations

`experiments/article_v1_ablations.py` registers and executes the required
versioned comparisons on validation data only: remove the three
target-distance coordinates (28D); remove frontier \(\mathbf z\) context
(21D); set `beta=0`; use `article_target_distance` without learning; disable
Pareto dominance; and disable enhanced Pauli canonicalization. The two
expensive search-reduction toggles use an outcome-free, preregistered balanced
validation subset.

The Pareto-off arm remains duplicate-only: it suppresses an exact
same-key/same-resource record but performs no dominance rejection or
retirement. The canonicalization-off arm uses the sound, deliberately weak
`raw_witness` key, which disables commuting reorder, cross-position fusion,
and Clifford-angle absorption and therefore cannot falsely merge distinct DAG
words. The command writes `ablations.csv`, `ablation_results.json`, and variant
checkpoints. The 37D/composite, GHZ-specific, and Toffoli-parity modes are
registered as supplementary case studies and must never be relabelled
`article_v1`.

## Runner and artifacts

The root CLI dispatches these Article V1 commands:

```powershell
python article_benchmark.py pilot --config configs/article_v1_pilot.json
python article_benchmark.py generate-corpus --config configs/article_v1_publication.json
python article_benchmark.py train --config configs/article_v1_publication.json
python article_benchmark.py evaluate --config configs/article_v1_publication.json
python article_benchmark.py aggregate --config configs/article_v1_publication.json
python article_benchmark.py ablations --config configs/article_v1_publication.json
```

It also provides
`python article_benchmark.py mini-ci --output-root outputs/article_v1 --run-id mini-ci-v5-new`.
Use a fresh run ID when an older pre-V2 ledger is present. The implemented
output root is `outputs/article_v1/<run_id>/` and may contain:

```text
run_manifest.json
environment.json
validation_audit.json
report_metadata.json
mini_ci_summary.json                 # mini only
raw_runs.jsonl
per_target.csv
success_curves.csv
timing_breakdown.csv
ablations.csv
ablation_results.json
corpus_manifest/manifest.json
corpus_manifest/train.json
corpus_manifest/validation.json
corpus_manifest/test.json
corpus_manifest/ood_test.json
checkpoints/seed-*.json
checkpoints/ood-seed-*.json
checkpoints/ablations/
figures/*.svg
tables/learner_seed_results.csv
tables/learner_seed_summary.csv
tables/paired_differences.csv
tables/paired_per_target.csv
tables/summary.csv
tables/summary.md
completion_summary.md
```

Focused tests assert resumable seven-scheduler mini execution and that every
aggregate artifact is rebuilt from `raw_runs.jsonl`. Artifact-map paths below
the working tree use repository-relative POSIX spelling, and the recorded run
working directory is `.`; this avoids embedding one developer's absolute
Windows checkout in portable report metadata.

The final mini evidence is
`outputs/article_v1/final-mini-ci-v5/mini_ci_summary.json`. It records a passing
nine-record matrix, all eleven semantic checks true, no reference-witness
fallback, an independently certified FIFO success on the known reachable
target, and a V2 standard checkpoint trained with seed 19 under the explicit
partial-smoke scope (one training target, one episode, cap 16). A same-ID
resume appended zero, skipped nine, left `raw_runs.jsonl` unchanged, reported
`checkpoint_trained_this_run: false`, and left the checkpoint file unchanged.
Neither the configured pilot nor the full publication campaign has been run.
The full campaign must not be an ordinary blocking unit test.

## GHZ, Toffoli, and QFT boundaries

- GHZ-3 is a small state-preparation smoke/case-study benchmark. It is not
  evidence of unrestricted unitary synthesis, scaling, or optimality.
- The known Toffoli circuit is replay/certification evidence. Learned Toffoli
  search is limited to the explicitly declared seven-term CCZ parity-network
  normal form and is not unrestricted Toffoli discovery.
- QFT-3 artifacts are analytical/high-level references and capability guards.
  Canonical QFT-3 is `APPROXIMATION_REQUIRED` under the current native model;
  no RL-generated exact QFT-3 circuit is claimed. AQFT-3 fidelity is a separate
  approximation score, not exact synthesis or certification.

## Allowed claim and prohibited extrapolations

> The repository implements reinforcement-learning-guided ranking of
> persistent frontier records for small-scale, exact, ancilla-free, all-to-all
> Clifford+T synthesis. Every selected record is expanded exhaustively through
> the same deterministic native grammar; symbolic semantics, canonicalization,
> Pareto pruning, and final dense certification remain outside the learned
> policy. The article-v1 experiment asks whether linear SARSA allocates a
> finite expansion budget more effectively than fixed non-learning schedulers,
> especially a direct target-distance heuristic, when every other synthesizer
> component is held fixed.

Do not claim general quantum-circuit generation, large-scale or polynomial-time
synthesis, approximate QFT synthesis, universal RL superiority, circuit
optimality, unrestricted Toffoli discovery, or scalability beyond the measured
small-register dense-verification setting.
