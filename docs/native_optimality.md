# Proof-carrying optimization of the restored hybrid frontier

## Representation and objective

Every discovery and audit record retains the authoritative native state
`v = (G, Theta, R, phi, rho)`: a persistent dependency DAG, signed Clifford
tableau, ordered signed Pauli rotations, exact scalar coordinate, and consumed
resources. This change does not introduce a phase-obligation representation,
change the gate grammar, or inject reference circuits into discovery. The
historical 25 phase-oracle and 18 named target contracts are unchanged.

For a fixed logical unitary and clean-workspace embedding J_a, acceptance remains
`U_C J_a = J_a U_target` to the declared isometry tolerance (or the explicitly
selected projective phase convention). Arbitrary feasible native H, S, SDG, T,
TDG and CNOT continuations are available. T-count is an optimization variable,
not the fixed number of odd coefficients in a phase-polynomial specification.

The default lexicographic objective is

    (T count, CNOT count, native depth, native gate count).

`t_depth` can be inserted anywhere in the ordered objective list; a single
objective is also supported. These are minima under ALL the remaining finite
resource caps, the fixed register/ancilla contract and the stated numerical
acceptance predicate, not unrestricted exact-synthesis optima. This is not a
weighted-sum objective or an enumeration of the entire resource Pareto frontier.

## What was missing before this change

The restored branch already had `optimize_native`, which repeatedly tightened
one resource cap. It only returned certified upper bounds, except for the
trivial zero-cost nonnegativity bound. `audit_native` and `verify_native_cover`
existed separately but were not connected to that optimization loop. The active
runner performed first-feasible discovery rather than resource optimization.

## Integrated procedure

`optimize_native_resources` now performs the following procedure for each
objective, advancing to the next only after the current minimum is proved.

1. Run frozen-policy native discovery within the current resource bounds.
2. After a certified incumbent with cost k, search with that objective at k-1.
3. If discovery does not resolve the tighter subproblem, run deterministic
   native auditing. Re-check every exclusion with `verify_native_cover` before
   accepting it. An unverified or interrupted cover cannot prove anything.
4. A feasible incumbent at k and a verified exclusion at k-1 establish the
   bounded optimum k. At k=0, nonnegative resource increments suffice instead.
5. Fix the established coordinate at its optimum and proceed lexicographically.

The justification is elementary: the witness gives the upper bound, the closed
cover excludes the strictly smaller integer costs, and subsequent objectives
retain all previously proved upper caps. A circuit violating an earlier minimum
would contradict its checked exclusion. Keeping the same operator semantics and
componentwise per-wire resource boundaries preserves the existing continuation
simulation argument. Learned values are never treated as lower bounds.

The result includes ordered stages, exact subproblem digests, exclusion
certificates, verification decisions, all discovery/audit attempts, and an
incumbent trace. `verify_native_optimization(problem, result)` independently
replays the final native word, recomputes its resources, derives each k-1
subproblem from the caller's original problem, and re-checks the covers. It does
not call either the learned policy or the auditor. Serialized JSON round trips
are supported. The checker shares the native symbolic algebra and retains the
existing fail-closed near-threshold numerical predicate; this is not a
proof-assistant or exact algebraic certification of arbitrary floating targets.

## Provenance and limits

Discovery, deterministic auditing and cover verification have separate work and
time fields. They share one cumulative ledger of edge operations, admitted
records, wall time and CPU time across ALL bound-tightening rounds and objectives.
Discovery/audit quotas are per round; verification replay and all closure checks
are charged to the same total. These are cooperative limits, not OS-enforced CPU
or physical-memory guarantees. Record accounting is cumulative, not peak bytes.

If a deterministic audit finds a circuit, it is explicitly attributed to
`deterministic_native_audit`; `audit_witness_used` records that fact. The separate
`discovery_incumbent` never includes an audit-produced word. Nevertheless, an
adopted audit witness can change later resource caps, so subsequent discovery is
not an independent discovery-only learning experiment. Set
`allow_audit_witness=False` to prohibit adoption; set `audit=False` for a pure
upper-bound discovery run. Neither setting disguises unresolved gaps as optima.
Constructive benchmark references are not used by this optimizer.

Result states are `optimal_under_numerical_contract`,
`optimal_nonnegative_resource_bound`, `infeasible_under_numerical_contract`,
`upper_bound`, or `unknown`. An exhausted discovery frontier is not a proof.
The final witness's own correctness certificate continues to say that correctness
alone does not establish optimality; the surrounding optimization certificate
supplies the independently checked lower bound.

## Clean-workspace budgets

`optimize_native_ancillas` solves each requested clean-width contract separately,
with the same logical unitary and other caps. Each width has its own work ledger
and archive. Limits are explicitly PER WIDTH. A minimum a is reported only when
a feasible witness exists at a and every smaller width 0,...,a-1 has a checked
native exclusion. Omitting a smaller width or timing out there leaves the
minimum unproved. Width zero requires no lower-width exclusion. The first
implementation does not mix borrowed and clean ancilla budgets or merge
intermediate states across physical register dimensions.

## Usage

The existing single-resource API now invokes the integrated optimizer:

```python
from hybrid_qcs.native_search import optimize_native
from hybrid_qcs.native_optimize import (
    optimize_native_resources, verify_native_optimization, optimize_native_ancillas,
)
from hybrid_qcs.resource_search import WorkLimits

# problem: NativeProblem; model: completed, frozen native checkpoint
result = optimize_native_resources(
    problem, model,
    objectives=('t_count', 'cnot', 'depth', 'gates'),
    limits=WorkLimits(50000, 100000, 60., 60.),
    discovery_edges=512, audit_edges=10000, verification_edges=20000,
)
checked = verify_native_optimization(problem, result)
# Inspect checked['optimality_verified'], not merely witness['success'].
```

Train and optimize the unchanged 43-target registry:

```bash
python -m hybrid_qcs.native_optimality_runner --output-dir outputs/native-optimality
```

Run small post-training optimization/certificate calibrations:

```bash
python -m hybrid_qcs.native_optimality_runner --targets calibration \
  --seeds 0 1 2 --seconds 15 --output-dir outputs/native-optimality-calibration
```

The default `hybrid-qcs`, `hybrid-qcs-native`, `hybrid-qcs-publication` and
`python -m hybrid_qcs.native_runner` now select optimization. The dedicated
optimality runner exposes the per-phase quotas and a larger default total-edge
budget. The old first-feasible qualification remains available explicitly:

```bash
python -m hybrid_qcs.native_runner --mode discover --output-dir outputs/native-discovery
```

All new artifacts are written outside `experiments/publication_v1`; no historical
phase-study numbers, checkpoints, proofs or manuscript claims are overwritten.

## Validation boundary

The new tests emphasize real post-training circuit optimization (63 generated
cases), including positive T-count minima, non-diagonal HTH, SWAP CNOT minima,
lexicographic objectives, coherent clean-wire return, and clean-width sweeps.
Other tests reject forged resource counts, changed target or tolerance domains,
phase-polynomial covers, malformed stages, and false nonnegativity proofs.
Audit-assisted witnesses are tested separately from learned discovery.

This implementation completes the optimizer/checker integration. It does not
assert that QFT-2, QFT-3, CCX, C^3X or every retained oracle can be synthesized or
proved optimal within a short qualification budget, nor establish an advantage
of SARSA/LinUCB over a deterministic scheduler. Hard-target unresolved outcomes
must remain visible as unknown or upper bounds.
