# Resource-conditioned search and clean-ancilla optimality

## Scope and implementation map

This extension is based on `oracle-optimality-certification-v1` at
`a41cd7fc99f98982fd22a618a47e55815d7baad9`. It preserves the old native
DAG/tableau/signed-Pauli-rotation machinery and the legacy search entry points.
The new finite optimizer is intentionally explicit about its two domains:

| Module | Responsibility |
| --- | --- |
| `resource_domain.py` | Exact promised-input mappings/phases, fixed native lowerings, resource bounds, native/DAG witness replay |
| `resource_policy.py` | 34-coordinate linear SARSA and 22-coordinate disjoint LinUCB contexts; versioned checkpoints |
| `resource_search.py` | Persistent-record deferred expansion, full pending coverage, resource antichains, fairness, staged training |
| `resource_audit.py` | Separate deterministic exploration and a closed-cover certificate checker |
| `resource_optimize.py` | Bound tightening, lexicographic stages, separate clean-workspace sweeps and status accounting |
| `resource_runner.py` | Qualification, custom truth-table optimization and standalone certificate verification |

The evaluator domain uses the existing role-aware X/CNOT/Toffoli grammar with
fixed native Clifford+T lowerings. Data X, flag X and data/workspace-controlled
CNOT/Toffoli into flag/workspace are allowed. The flag is never a control;
workspace X and CNOT into data are not in this declared grammar. These
restrictions matter to any minimum-ancilla claim.

The direct-phase domain uses affine X/CNOT transformations and T, T-dagger,
S, S-dagger phases, with exact exponents modulo eight. It is H-free at the
operation level; the fixed native lowering of X uses HSSH. It is not an
unrestricted native Clifford+T search with arbitrary H insertions.

Targets have one to three Boolean inputs. Each contract allows zero to two
clean workspace qubits. Borrowed ancillas, measurement/reset, arbitrary
connectivity, dynamic allocation and a full graph neural policy are not added.
QFT-3 remains the earlier certified upper-bound benchmark, not a newly proved
unrestricted optimum. Existing Toffoli/CCZ normal-form and four-wire CNOT-only
audits remain available with their original domain qualifications.

## Contract and resource accounting

For an evaluator the logical interface is `(x,y)` and correctness requires
`|x,y,0^a> -> |x,y xor f(x),0^a>` for both flag values. Search can propagate
only `y=0` because its grammar never controls on y; every operation commutes
with flipping the flag, so agreement for y=0 implies the specified y=1 action.
The independent native certifier nevertheless checks both values explicitly.

For a direct phase oracle the interface is `x` and the required map is
`|x,0^a> -> (-1)^f(x)|x,0^a>`. There is no hidden output flag. Exact relative
and global phases are retained. No floating-point fingerprint authorizes a
merge or an infeasibility proof.

The native witness is replayed as a dense clean-input isometry and through the
existing persistent DAG/tableau/Pauli representation. The report distinguishes
exact discrete semantics, numerical isometry error/tolerance, and workspace
leakage. Native replay is an independent numerical cross-check of the exact
macro semantics, not a claim of exact symbolic arithmetic for floating-point
matrix entries.

T count, CNOT count, gate count and per-wire depths count the emitted native
witness. Resource counts do not shrink after an inverse cancellation or a
symbolic reduction. The wire-slot cap is fixed per run, and every fixed native
macro has zero extra decomposition scratch. A workspace wire is available only
when it is zero on every promised input. Uncomputation permits reuse of that
same wire; it does not count as a second qubit allocation.

Reports separate available workspace from touched workspace. The evaluator's
required output bit is logical for that contract, but it counts as one extra
auxiliary qubit when an evaluator is used in a compute-phase-uncompute wrapper.
The reported wrapper count is accounting only; the new direct-phase optimizer
does not mistake evaluator optimization for direct-phase optimality.

## Learned scheduling, not learned optimality

The outer action remains a persistent frontier-record ID. The inner action
chooses one still-pending operation on that record. Every resource-feasible
operation is retained until attempted or covered by sound dominance. Every
32nd allocation uses deterministic oldest-record/lowest-token fairness.

The policy uses cached target discrepancy, certified clean-workspace count,
wire-depth summaries, operand overlap, local inverse opportunities, projected
resource usage and progress/slack interactions. No full circuit DAG is
reconstructed for policy scoring. Full witness/DAG replay happens at terminal
certification. Richer native semantic objects remain authoritative in native
replay; they are not falsely described as raw policy inputs in the mapping
search.

Normalized usage and candidate-dependent slack interactions ensure budget
changes can alter relative linear scores. A global cap alone would cancel
between candidates. Incumbent, pending-count and remaining-work features are
refreshed separately from static record features and never enter semantic keys.

Training uses fixed-count stages: outer SARSA with a deterministic projected-
distance inner baseline, then LinUCB with frozen outer values, then a short
outer adjustment with frozen LinUCB. The inner training response uses a frozen
outer-value difference, not an admissible lower bound. Evaluation freezes both
policies. Saved models include ordered feature schemas, RNG state, covariance
inverses and training update counts; incompatible checkpoints are rejected.

The undiscounted bounded-episode reward is success credit minus classical edge
work plus a potential difference. Terminal potential is zero for success,
exhaustion, cancellation and resource/work truncation. Attempted T extensions
are not summed as though the search trajectory were one quantum circuit.

## Proof rule

Each record carries exact semantics z and a resource vector
`r=(T,CNOT,G,operation_count,wire_depth[0],...,wire_depth[w-1])`.
Within a fixed contract, `(z,r1)` covers `(z,r2)` only when `r1 <= r2`
componentwise. All wire depths are needed: maximum depth alone does not
preserve the resource effect of a common suffix. Historical gate patterns are
not used to mask continuations in this subsystem.

A closed-cover certificate contains a complete domain manifest and a finite
set H of such labels. The independent checker verifies:

1. H covers the zero-cost root.
2. H contains no target semantics.
3. For every label in H and every operation, every resource-feasible successor
   is covered by another label in H with the same exact semantics and no
   greater resources.

These conditions are an inductive over-approximation proof. By induction on
circuit length, every feasible reachable label is covered by H. Equality of
semantics and monotonicity of each native resource transition preserve this
coverage under a common suffix. A feasible target would therefore require a
target label in H, contradicting condition 2. Reachability of every individual
cover label is not required; an over-approximation is sufficient for exclusion.

The checker implements a scalar reference transition and native resource
replay separately from the learned engine. It does not trust queue exhaustion,
a stored success flag, a learned score or a certificate checksum. SHA-256 binds
payload integrity, while actual closure checks establish the proof. The trust
base includes the declared native lowerings and the verifier implementation;
this is not an externally formalized proof-assistant development.

## Deterministic optimization

The default priority is `(T,CNOT,depth,native gate count)`, not macro count.
Operation count remains a declared finite search bound. After a certified
incumbent with primary cost k, the controller searches the same contract with
cap k-1. It continues until it finds a better witness, verifies a closed-cover
proof of absence, or reaches a work limit. If the best cost is zero, the
nonnegative-integer resource bound supplies its matching lower bound.

Only a completed primary proof permits fixing that cap and proceeding to the
next lexicographic objective. Independent audit-discovered witnesses are
identified separately from linear-policy-discovered witnesses. Certificates
bind the target, native grammar, phase convention, width and every resource
cap. A proof for one width or bound is never silently reused for another.

Statuses are `optimal`, `infeasible`, `upper_bound` and `unknown`. Learned
frontier exhaustion alone is `unknown` until audited. Interruption during an
audit or proof verification cannot be promoted to infeasibility. A certified
incumbent survives as an upper bound when proof is incomplete.

Each clean-workspace budget has a separate archive. A minimum workspace budget
is proved only when a witness exists at that budget and all smaller budgets
have verified infeasibility. Missing or timed-out smaller budgets do not
support a minimum. Any such minimum is relative to this grammar and the other
finite bounds. It is not an assertion that arbitrary quantum circuits for the
same target require that many ancillas.

## Reproducible commands

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m hybrid_qcs.resource_runner qualify --output-dir outputs/resource-optimality

# Three-input conjunction evaluator; q0 is the least-significant input bit.
python -m hybrid_qcs.resource_runner optimize \
  --truth-table 00000001 --mode evaluator --ancillas 0,1,2 \
  --t-cap 21 --cnot-cap 18 --gate-cap 45 --depth-cap 45 --operation-cap 3 \
  --policy outputs/resource-optimality/policy.json \
  --total-edges 100000 --discovery-edges 512 --audit-edges 40000 \
  --max-records 15000 --wall-seconds 12 --cpu-seconds 12 \
  --output-dir outputs/conjunction-optimality

python -m hybrid_qcs.resource_runner verify outputs/conjunction-optimality/certificates/*.json
python -m hybrid_qcs.optimality_runner --output-dir outputs/legacy-optimality
python -m hybrid_qcs.oracle_runner --output-dir outputs/original-oracle --timing-repeats 1
```

Without `--policy`, custom optimization uses an explicitly untrained linear
warm start. `--scheduler distance` and `--scheduler cost` are deterministic
baselines. The cost scheduler is not Dijkstra with a first-hit optimality
claim; it uses the same independent bound-tightening proof protocol.

## Qualification and remaining limitations

Qualification trains on one-input predicates and freezes evaluation on
held-out two-/three-input functions, including multi-marked predicates and a
cubic conjunction. It compares distance, cost, outer-only and hierarchical
schedulers with identical per-width work caps. Outputs include policy and
curriculum files, complete manifests, JSON/CSV results, incumbent provenance,
separate discovery/audit/verification times, and standalone certificates.
Exported certificates receive an additional delivery-integrity recheck outside
the optimization timings; those recheck times are recorded in each file entry.

There is no promised learned speedup. Small policy fitting cost does not imply
small search cost. Record limits bound stored records, not exact bytes; an
increased workspace budget can still substantially expand search and proof
work. CPU/wall deadlines are cooperative at operation boundaries, including
after scoring. They are not hard OS-level interrupts, and a native replay or
one scoring operation can overrun the deadline. No worker, training job or
pilot is scheduled to run after the current command returns.

No dynamic reversible-pebbling scheduler, automatic full-graph encoder,
borrowed-workspace optimization, arbitrary native-H search, large-register
symbolic Boolean compiler, or statistical superiority claim is included.

## Correction to inherited optimality results

The source branch failed five baseline tests when reproduced. Two independent
issues were found:

- Minimal-T phase-polynomial coefficients can be 3 or 5 modulo eight, not only
  1 or 7. Lowering now handles every residue, with exactly one T/TDG for odd
  coefficients. T-optimal coefficient ties are broken by fixed native phase
  cost and coefficient order for reproducibility.
- The old evaluator auditor allowed flag controls absent from the learned
  grammar. Its grammar is now independently constructed and regression-tested
  against the search grammar. Even in the aligned grammar, the original
  learned evaluator is not the CNOT-before-gate-count optimum.

The exact audited witness is:

```text
X(1), TOFFOLI(0,1,4), CNOT(4,3), TOFFOLI(2,4,3), TOFFOLI(0,1,4), X(1)
```

Its `(macros,T,CNOT,native gates)` cost is `(6,21,19,54)`. Independent native
clean-input replay validates both logical flag values. The earlier learned
witness remains correct at `(6,21,21,48)` and is a different resource trade-off.
Tests, current reports and CI expectations now state this distinction. Earlier
experiment/manuscript artifacts are retained as historical provenance and must
not be used to repeat the superseded lexicographic-optimum claim.
