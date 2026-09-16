# Theorem-certified cleanup backend

## Contract and status

This optional backend implements the **deterministic endpoint of Theorem 2,
“Lifecycle-accounted materialization optimum,” pages 59–60 of the supplied
`H - Prove Cleanup Theorem(1).pdf`**. It is based on the native branch at
`6c2f15ffc2e0ce524c6492f40f234a2862dba51a`. The original native search, its
benchmark registry, its exact algebra, and its historical evidence are unchanged.

The logical inputs are `(a,b[0:r],x[0:m])`, independently ranging over the full
Boolean cube. The complete oracle has phase

`p(a,b,x) = q(a,b,x,F(a,b,x))`, where `F[i,j] = a*b[i]*x[j]`.

The supplied consumer `q` is a quadratic Boolean sign phase on the displayed
input and product wires. It is implemented by native decompositions of Z and CZ
(and an explicit scalar -I word for a constant phase). No output flag is omitted
from a cost comparison: the target is explicitly a phase oracle from the start.
The implemented consumer language does not include arbitrary non-diagonal gates.

For a full-bank architecture, inactive-clean helpers of the class
`H=a(Pb+Qx+c)`, one main product-bank instrument and Clifford-feedback stage,
and one final direct-X/helper-feedback stage, the source theorem gives the
auxiliary peak lower bound `rm+min(r,m)`. This excludes original logical inputs.
The implemented attaining witnesses use coordinate helpers `a*x[j]` or `a*b[i]`;
the software does not enumerate every mixed encoding or arbitrary instrument.
The supplied mathematical converse is a **trusted theorem**, not a theorem proved
inside a proof assistant by this software. Its applicability and an attaining
physical protocol are checked explicitly.

## Integration architecture

`hybrid_qcs.cleanup` is a separate, explicitly typed **structured dynamic-protocol
backend**. It does not add a fictitious measurement to `HybridState.apply`, replace
`NativeSearch`, or relabel the former fixed-phase-polynomial experiments.

Each persistent protocol event stores an actual local coherent `HybridState`,
its physical-wire embedding, quantum predecessors, classical predecessors and a
native word. The local state retains the original persistent DAG, Clifford
tableau, ordered Pauli rotations, scalar phase and resource coordinates. All
coherent primitives occupy at most three active wires.

A complete instrument needs more information than a unitary tuple. Its enclosing
record additionally stores the helper layout, certified live-value roles,
completed preparation/consumer obligations, cleanup stage, classical outcomes,
and the full native resource/dependency boundary. A measurement is represented
as a Kraus operation with an explicit reset, not as an invertible unitary.

There is **no dense global HybridState/isometry** for a 39-wire example. Large
banks use exact structured-interface composition of the bounded coherent blocks.
This is not unrestricted large-register gate-by-gate synthesis. The original
native problem's eight-physical-qubit dense-cache guard is deliberately retained.
The restricted Boolean identities used by the cleanup verifier are local promise
certificates, not a replacement algebraic representation of arbitrary circuits.

## Implemented functions

- `CleanupProblem`, `Consumer`, `Limits`, `Hardware`: explicit target, theorem,
  native-resource and timing contracts, with versioned serialization.
- `CleanupSearch`: persistent frontier, indexed scoring panel, global fairness,
  exact guarded transitions, continuation-safe cost dominance and cancellation.
- `CleanupHierarchy`: 20-feature linear semi-gradient SARSA and six disjoint
  24-feature linear ridge/LinUCB continuation models. Old native/phase checkpoints
  are rejected rather than implicitly reinterpreted.
- `optimize_cleanup`: incumbent-based workspace tightening plus the theorem's
  lower bound, returning scoped optimum, upper bound, unknown, or scoped
  infeasibility. It does not prove T/CNOT/depth optimality.
- `verify_protocol`, `verify_workspace_certificate`: exact independent branch
  and scope checking. Neither calls the search queue or trained policy.
- `verify_small_matrix`: all-outcome native/Kraus cyclotomic replay for explicitly
  small registers. `verify_truth_table_generators` supplies an additional,
  independently implemented packed-truth-table differential check.
- `compile_deterministic`: matched row-only or smaller-side constructive control.
- `compile_bank_free`: direct ANF-monomial control that does not materialize the
  bank and is **outside** the workspace theorem. It is not an optimized Boolean
  factorization package and is not presented as state of the art.
- `python -m hybrid_qcs.cleanup.runner`: train, generate five candidates, compare,
  verify and archive the complete study. Installed alias: `hybrid-qcs-cleanup`.

## Native construction and exact phases

For column helpers, preparation writes `H[j]=a*x[j]` and then
`F[i,j]=b[i]*H[j]`. Row helpers use `H[i]=a*b[i]` and `F[i,j]=H[i]*x[j]`.

The default primitive is an exact **clean-target AND isometry** with four T-type
gates. Its native word is

`H(t), T(t), CX(b,t), Tdg(t), CX(a,t), T(t), CX(b,t), Tdg(t), H(t), Sdg(t)`.

The last S-dagger removes the relative phase on the occupied target. All four
clean-input columns and the inverse isometry are independently checked over
`Z[exp(i*pi/4),1/2]`. This is **not** a four-T Toffoli on arbitrary target inputs.
The established four-T temporary-AND mechanism must be credited to prior work,
not claimed as this implementation's invention. The source's conservative exact
seven-T Toffoli primitive is also supported and tested.

After the complete consumer, each product is measured in X. Outcome `s[i,j]`
introduces phase `(-1)^(s[i,j]*F[i,j])`. Conditional
`CZ(b[i],H[j])` (or the symmetric row form) cancels it. The helpers are then
measured and corrected by conditional `CZ(a,x[j])` or `CZ(a,b[i])`.

All products are measured before main feedback; helpers are preserved until that
feedback is complete. Stage barriers prevent the exporter from silently merging
the two feedback stages or reusing a helper prematurely.

For N=`rm+k` measurements, every corrected branch is exactly
`K_s J_in = 2^(-N/2) J_out O_p`. The probability is input-independent `2^-N`, all
outcomes are retained, and summing over outcomes gives one. Thus the channel is
correct on arbitrary logical superpositions entangled with a reference. There
is no retry, postselection, assumed harmless failure or measurement choice by RL.

## Verification and scaling

The exact verifier replays live Boolean values and the constant/outcome-dependent
phase coefficients. It checks the actual supplied consumer, clean-target
preconditions and resets, and requires every final outcome coefficient to vanish.
Because the constructed phase is linear in transcript bits, this verifies all
`2^N` outcomes without enumerating them. It also checks the actual native
isometries for the coherent primitives. This is compositional exact evidence,
not numeric sampling or a dense simulation disguised as a symbolic certificate.

For this coordinate-helper construction, live wire functions are single
monomials and the consumer is an explicit list of quadratic terms. Verification
scales with the emitted program/consumer size and polynomial bit-mask arithmetic,
not with `2^(physical width)` or `2^N`. General inference of such interfaces from
arbitrary quantum circuits is not implemented. The optional truth-table and
dense-matrix diagnostics have explicit size guards.

The independent verifier checks the precise protocol class and rejects unknown
operations, incorrect guards, hidden input promises, changed phase conventions,
missing branches/corrections, bad lifetimes and reused outcome bits. It never
accepts a native closed-cover certificate as proof over dynamic protocols.

## Resource accounting

Every returned count is reconstructed from the emitted instructions. Z is
lowered to S,S; CZ to H,CX,H. X measurement/reset is explicitly H, Z measurement,
and outcome-conditioned X lowered as H,S,S,H. Thus there is no free reset hidden
in the reported native-gate or latency cost. A backend with a native X/reset can
use a different cost model, but must not mix its counts with this study.

The report includes auxiliary peak, physical width, T and CNOT counts, worst-case
native gates, expected native gates, per-wire native/T-depth, measurements,
measurement rounds, retained classical bits and worst-case schedule ticks.
Expected gate counts rely on the independently verified uniform transcript.
The all-ones outcome is possible and activates every conditional correction, so
worst-case count accounting is attainable for this construction. Schedule values
are conservative values for the emitted stage-barrier program, not optimized
hardware execution times. Classical outcomes are retained, not freely erased.

Default illustrative latencies are one tick per native gate, five per Z
measurement and two for outcome availability. Routing is not modeled; all-to-all
connectivity is declared. Measurement duration and feed-forward are included in
`worst_case_ticks`, not confused with native quantum-gate depth. No hardware
noise or physical fault-tolerance claim is made.

With the four-T primitive and a fixed identical consumer, replacing row helpers
by column helpers when `r>m` saves exactly `r-m` peak auxiliaries, `4(r-m)` T gates,
`4(r-m)` native CNOTs and `18(r-m)` worst-case native gates in this implementation.
Those equalities are also checked on generated circuits; they are not substituted
for generation during evaluation. The remaining resource coordinates can trade
off, and these formulas do not establish global resource optima.

## Learning semantics and training

SARSA still chooses a persistent frontier record, not a physical measurement
outcome. LinUCB ranks eligible macro continuations: layout, helper preparation,
product preparation, consumer term, product-bank cleanup, and helper cleanup.
The deterministic legality predicates decide applicability. A macro is one
classical search allocation; its entire emitted native work and verification
cost are accounted separately. The formulation uses gamma=1 and does not equate
physical gate duration with the number of search-MDP steps.

The outer update is normalized, clipped semi-gradient SARSA using the **actually
selected next frontier context**. During inner fitting the outer weights are
fixed; the inner response is the one-step frontier TD residual, including the
same terminal-safe potential shaping. There is no parent/child substitution.
The two blocks are trained in stages. Standard stationary-bandit regret and
linear-SARSA convergence are not asserted for this environment.

The reward combines certified success/resource quality, a charge per classical
allocation and a progress potential whose terminal value is zero. Hard resource
bounds and certificates remain deterministic. The theorem does not imply that
learning is useful: choosing the smaller side alone is analytically solved.
The trained policy additionally chooses operation schedules, but that only
becomes a research contribution if a matched comparison supports it.

## Five-problem experiment

The fixed five shapes are `(3,1),(4,2),(5,2),(6,3),(8,3)`. They represent synthetic
common-enable pairwise-feature phase oracles; they are not five end-to-end BNN
verification applications. All `r>m` was an explicit selection criterion because
the source theorem predicts a helper advantage over the row-only construction.
The supplied q expressions and their simplified logical ANFs are archived.
Development included correctness smoke tests; this is not a preregistered
population-level superiority trial.

Training uses logical widths 3,4,6; testing uses 5,7,8,10,12. Hence train and test
operators cannot coincide under a wire permutation, complement or scalar phase
at equal width. Five models use seeds 11,19,23,31,47 and 64+96+24 staged episodes.
Regression fixtures complete shorter 8+12+4 schedules; completion does not mean
convergence. Missing or incompatible checkpoints fail closed. Evaluation never
updates fitted coefficients.

The 285-job experiment comprises five problems, three timing repetitions,
three trained variants on each of five seeds, and four deterministic controls.
The trained variants are theorem-enabled measured search, row-helper measured
search and smaller-side coherent inverse search. Controls are a feature-matched
untrained theorem-enabled search, direct smaller-side compilation, direct
row-side compilation, and bank-free ANF compilation. Auditors never repair a
failed discovery and deterministic outputs are never labelled trained outputs.

The full cost of generation and its certificate is timed. Subsequent artifact
serialization is outside that endpoint. A shared verified primitive cache is
prepared separately and disclosed; it contains no target-specific solution.
Timing repetitions and training seeds are not independent logical problems.
No population superiority claim or confidence interval is inferred from five
purposefully selected specifications.

## Comparison boundary

The attached theorem itself gives bank-free counterexamples. Here the bank-free
control is actually executed and independently verified. It can use fewer
qubits and, on some candidates, fewer T gates. This outcome must remain in the
report. A materialization optimum cannot be used to rule out a bank-free circuit.

The direct smaller-side control gets exactly the same construction as learning.
It can reproduce the resource savings without any training and may have lower
classical runtime and depth. Therefore a gain over row helpers does **not**
establish SARSA–LinUCB superiority, novelty of the smaller-side factorization,
or an improvement over every prior compiler. AlphaTensor, Exact T Library,
Reqomp and other external optimized solvers were not executed in this campaign.

The existing general phase-oracle/QFT/Toffoli/SWAP benchmarks and 848 regression
tests are preserved. The cleanup theorem does not imply improvements on them.

## Reproduction

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m hybrid_qcs.cleanup.runner --output-dir outputs/cleanup-fresh
python -m hybrid_qcs.cleanup.runner --stage verify --output-dir outputs/cleanup-fresh
```

A completed study includes the locked specifications, source hashes, environment,
frozen checkpoints, full compressed training trajectories, all successes/failures,
individual JSON protocols and fully lowered OpenQASM 3 exports, workspace
certificates, per-run allocations/costs and independent verification receipts.
The producer refuses to overwrite a previous campaign. Verification checks
source/evidence hashes, completeness of the job grid, fitting update counts,
frozen checkpoint provenance and every distinct returned protocol.

## Source attribution

Theorem source: `H - Prove Cleanup Theorem(1).pdf`, Theorem 2 and its construction,
pp. 59–60; scope and bank-free counterexample, pp. 61–62. The source is a supplied
research note, not assumed to be a published or peer-reviewed theorem.

Craig Gidney, *Halving the cost of quantum addition*, Quantum 2, 74 (2018),
DOI 10.22331/q-2018-06-18-74, is the resource-primitive reference for four-T
clean-target AND computation and Clifford-only measured termination.

Richard S. Sutton, Doina Precup, Satinder Singh, *Between MDPs and semi-MDPs:
A framework for temporal abstraction in reinforcement learning*, Artificial
Intelligence 112 (1999), DOI 10.1016/S0004-3702(99)00052-1, provides the established
framework when a future planner models variable-duration classical options.
This version retains one macro per classical decision and charges physical
resources separately.
