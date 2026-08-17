# Proposed Article V1 manuscript patch

## Final raw-metric reconciliation

Equation (129) is implemented literally after finite/unitary validation: neither matrix is normalized or projected, and only `|Tr(V†U)|/d` is clipped. Deterministic calibration freezes `tau_cert=1e-6` (equivalent floor `2.5809568279517847e-8`); identity deduplication separately freezes `tau_identity=1e-7` while calling the same primitive. The amended reward remains normative: every unsuccessful episode has total base return `-B_exp`, including early frontier exhaustion.

This repository document records implementation-plan amendments and
operational completions. It is not part of the external manuscript, and the
external `Article_limited_scope.md` file has not been edited. In particular,
the replacement for Equation (104), the executable definitions of
\(r_{\mathrm P}\) and \(\nu\), the exact normalization constants, and the
bootstrap convention below must not be attributed verbatim to the external
article unless that manuscript is separately amended by its author.

## Replacement for Equation (104)

Replace the base transition reward after expansion \(t+1\) with

\[
\boxed{
R_{t+1}^{(0)}
=-1
+B_{\mathrm{exp}}\,\mathbb 1[\text{certified success}]
-\left(B_{\mathrm{exp}}-(t+1)\right)
\mathbb 1[\text{terminal failure}].
}
\]

Here, terminal failure means frontier exhaustion without certification or any
other unsuccessful terminal condition before the external expansion cap. On
an ordinary nonterminal transition both indicators are zero. At unsuccessful
budget exhaustion after exactly \(B_{\mathrm{exp}}\) expansions, the padding
term is zero.

The literal manuscript reward \(-1+B_{\mathrm{exp}}\mathbb 1[\text{success}]\)
would return \(-T\) when an unsuccessful frontier exhausts after
\(T<B_{\mathrm{exp}}\) expansions. It would therefore make early failure look
better than a failure that explored longer. The padding term removes that
incentive. Summing the amended base reward gives

\[
G_{\mathrm{success}}=B_{\mathrm{exp}}-T
\]

for certification after \(T\) expansions, and

\[
G_{\mathrm{failure}}=-B_{\mathrm{exp}}
\]

for every unsuccessful episode. Thus every success outranks every failure,
and earlier certification outranks later certification.

Potential shaping remains

\[
R_{t+1}=R_{t+1}^{(0)}+
\beta\left(\Psi(s_{t+1})-\Psi(s_t)\right),
\qquad
\Psi(s)=-\min_{v\in\mathcal F(s)}d_{\mathrm{tar}}(v),
\]

with \(\Psi(s_T)=0\) for every terminal state and \(\gamma=1\). Therefore the
shaping terms telescope to \(\beta(\Psi(s_T)-\Psi(s_0))\) for episodes sharing
the same root. The repository schema is
`article-v1-expansion-potential-amended`.

### Repository application

The Article V1 training and evaluation runner now requests
`article_v1_expansion_potential` for every scheduler, supplies the configured
`beta`, and shares one target-metric context within each fresh run. Frozen
schedulers do not learn from or consume evaluation rewards, but they execute
under the same amended reward computation and target-metric cache accounting.
Focused runner and reward tests cover this binding. This implementation status
does not imply that the external manuscript was edited or that the configured
pilot/publication campaigns were executed.

### Audited repository execution semantics

The repository's transferable learner schema is
`article-v1-transferable-linear-checkpoint-v2`. Its digest binds the feature
schema, standard/OOD family, corpus-config digest, training-scope mode, ordered
training-target IDs, training seed, learning rate, epsilon schedule, training
`beta`, certification tolerance, episodes per target, optional expansion cap,
budget policy, ordered effective per-target training budgets, and weights.
Evaluation also requires an explicit corpus-partition scope with an allowed
training-seed set and exact training-partition/protocol match, so a foreign,
incomplete, held-out-leaking, OOD-length, or feature-ablation checkpoint cannot
be silently substituted for the primary learner. Standard/OOD/ablation
campaign scopes require their complete declared training partition; only the
mini smoke explicitly permits a named partial-training scope. The zero-weight
linear control still computes the same Article 31D feature batches and
target-distance coordinates; only its coefficients are zero.

Raw-run identity includes both `training_seed` and
`source_worktree_digest`, in addition to checkpoint and experiment semantics.
Resume is fail-closed: immutable run, environment, corpus, and per-split
manifests must match before an existing ledger is reused. An existing
checkpoint must also validate exactly; a compatible resume loads it without
retraining or rewriting, while missing/corrupt/conflicting checkpoint content
aborts. Pre-V2 or pre-identity ledgers require a new run ID rather than implicit
migration.

Reported success-curve points are restricted to external expansion caps that
were actually executed for the corresponding target/method/seed/checkpoint.
The reporter never infers an unexecuted smaller budget from a larger-cap
trajectory. `wall_time_ns` denotes total evaluation wall time, including reset,
ranking/features, environment steps, stopping, and result extraction; the
narrower accumulated step-body time is separately named
`environment_step_time_ns`. Artifact maps for outputs under the working tree
use repository-relative POSIX paths.

## Operational feature definitions

For a candidate \(v\) in one immutable frontier snapshot \(\mathcal F_t\),
the plan completes the manuscript's Pareto-position feature as

\[
r_{\mathrm P}(v;\mathcal F_t)=
\frac{
\left|\{u\in\mathcal F_t\setminus\{v\}:\boldsymbol\rho(u)\preceq
\boldsymbol\rho(v)\}\right|
}{\max(1,|\mathcal F_t|-1)}.
\]

It is a cross-key ranking statistic only: lower is more resource-efficient, a
singleton receives zero, and it is never used to justify cross-key pruning.
Persistent record IDs do not enter the formula.

Let \(g_t(k)\) be the cumulative number of resource-feasible records generated
with canonical key \(k\), including accepted records, exact-duplicate
rejections, and dominance rejections; count the root once. The plan defines

\[
\nu_t(v)=\frac{1}{\sqrt{\max(1,g_t(\kappa(v)))}}.
\]

This is deterministic search-history state, not a policy-visit count, reward,
or target-specific oracle.

Let \(B_T=\texttt{max\_t\_count}\), \(B_g=\texttt{max\_gates}\),
\(B_D=\texttt{max\_depth}\), and
\(B_{2q}=\texttt{max\_two\_qubit\_count}\) when present, otherwise
\(B_{2q}=B_g\). For qubit count \(n\), rotation count \(m_R\),
anticommuting-pair count \(n_{\mathrm{ac}}\), and mean Pauli weight \(w_R\),
the exact normalizations are

\[
\widehat n_T=\frac{n_T}{\max(1,B_T)},\quad
\widehat n_{2q}=\frac{n_{2q}}{\max(1,B_{2q})},\quad
\widehat n_g=\frac{n_g}{\max(1,B_g)},\quad
\widehat D=\frac{D}{\max(1,B_D)},
\]

\[
\widehat m_R=\frac{m_R}{\max(1,B_T)},\quad
\widehat n_{\mathrm{ac}}=
\frac{n_{\mathrm{ac}}}{\max(1,\binom{B_T}{2})},\quad
\widehat w_R=\frac{w_R}{\max(1,n)},
\]

with \(w_R=0\) for an empty rotation word. The target coordinate is already
normalized:

\[
d_{\mathrm{tar}}(v)=1-
\frac{|\operatorname{Tr}(U_\star^\dagger U_v)|^2}{d^2},
\qquad d=2^n.
\]

Set \(\widehat r_{\mathrm P}=r_{\mathrm P}\) and
\(\widehat\nu=\nu\). Population frontier standardization uses

\[
\mathbf z_t(v)=\frac{\mathbf x_v-\boldsymbol\mu_t}
{\boldsymbol\sigma_t+\eta\mathbf 1},
\qquad \eta=10^{-8},
\]

not sample variance. A singleton frontier produces \(\mathbf z=0\). The final
ordered map is the 31-dimensional `float64` vector

\[
\boldsymbol\varphi(s_t,v)=
[1,\mathbf x_v,\mathbf z_t(v),\widehat b_t\mathbf x_v]^T,
\qquad
\widehat b_t=\frac{B_{\mathrm{exp}}-t}{B_{\mathrm{exp}}}.
\]

The exact coordinate names are frozen in
`docs/article_v1_feature_contract.md` and `rl/article_features.py`.

## Certification and identity tolerances

For target \(V\), candidate \(U\), and dimension \(d\), define

\[
c_\phi(U,V)=\frac{|\operatorname{Tr}(V^\dagger U)|}{d},\qquad
\Delta_\phi(U,V)=\sqrt{1-c_\phi(U,V)}.
\]

Final Article V1 acceptance is \(\Delta_\phi\le\tau_{\mathrm{cert}}\), after
reconstructing the candidate from its authoritative DAG. Corpus identity and
semantic duplicate rejection use the same discrepancy implementation with the
separately named \(\tau_{\mathrm{identity}}\). The tolerances have different
roles and are not interchangeable even if a future profile assigns the same
number.

The current checked-in pilot/publication settings are
`tau_cert = 1e-9` (stored as `experiment.certification_tolerance`) and
`tau_identity = 1e-7`. Feature distance remains
\(d_{\mathrm{tar}}=1-c_\phi^2\), not \(\Delta_\phi\).

## Statistical interval convention

The held-out target is the primary paired statistical unit. Use a deterministic
95% percentile bootstrap with 10,000 target-level resamples and a serialized
statistics seed. Retain every target-by-random-seed raw record, average random
scheduler seeds within target, and then resample targets. Retain every SARSA
checkpoint seed and report each seed plus mean, median, standard deviation,
and uncertainty across held-out targets and learner seeds without undocumented
pooling. The repository materializes those two levels separately as
`tables/learner_seed_results.csv` (one row per frozen learner seed/checkpoint
within an experiment stratum) and `tables/learner_seed_summary.csv`
(target-first and explicitly named between-learner statistics). Preserve paired
target-level differences; confidence-interval overlap or non-overlap alone is
not a superiority test.
