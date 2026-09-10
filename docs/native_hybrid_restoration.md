# Restored native hybrid frontier

Base: `publication-linear-certified-qcs-v1` at
`4e5d1a429f9ec6514a0afbae7a29929ffe4845ef`.
Restoration branch: `hybrid-frontier-restored-v1`.

## Authoritative semantics

Every `NativeRecord.state` is a `HybridState`

    v = (G, Theta, R, phi, rho),
    U(v) = exp(i*pi*phi/8) C(Theta) prod_j exp(-i*pi*k_j*P_j/8).

G is the persistent dependency DAG; Theta contains the complete signed forward
and inverse Clifford generator images; R is an ORDERED signed-Pauli-rotation
word. Noncommuting factors are not sorted past one another. The phase coordinate
is an integer modulo 16. Resources include native T/T-dagger count, CNOT count,
native gate count, individual wire depths and (in the surrounding search
record) individual T-depth boundaries. The DAG shares predecessor events; it is
not reconstructed during policy scoring.

The old tableau represented a Clifford only projectively. The restored core
fixes its scalar representative by making the first nonzero amplitude of C|0>
positive real. `CliffordColumn` is a DERIVED small-register cache of this column,
using Gaussian integers and a common sqrt(2) denominator. Clifford updates
extract their phase by exact integer arithmetic and add it to phi. It is not a
phase polynomial. Its cache has 2^w entries and is explicitly not a scalable
large-register stabilizer-state implementation. The current workload already
uses dense isometry features and is limited to eight physical qubits.

This convention makes (Theta,R,phi) an exact operator representation. Tests
compare its independently rendered matrix with native DAG replay, including
scalar-distinct Clifford cycles. `exact_key` uses raw ordered factors and the
canonical frame scalar; the projective canonicalizer's Clifford-extraction
key must not be combined with an uncorrected scalar. Exact equality is
sufficient for merging, not a claimed complete Clifford+T normal form.

## Search and learning

`NativeProblem` accepts a logical unitary, resource bounds and an explicit
`AncillaContract`. Clean-workspace correctness is U_C J = J U_target; a borrowed
wire, when supplied through the API, remains a free input with identity action
in the contract. Exact and projective global-phase modes are distinct.
The benchmark registry uses exact phase unless explicitly labelled otherwise.

Every frontier record enumerates native H,S,SDG,T,TDG,CNOT continuations. No
phase-obligation mask, prescribed parity-emission order, hidden reference path,
macro witness, target-specific Clifford scaffold, or PhaseState is used. A phase
polynomial, when present in an archived specification, is used ONLY to construct
the diagonal logical target matrix. It neither fixes the realized rotation word
nor restricts native continuations. Its old T count becomes an upper bound,
not a requirement that each phase coefficient be emitted once.

The outer policy is a 20-coordinate linear semi-gradient SARSA action-value
approximation over frontier record IDs. The inner policy has six disjoint
24-coordinate LinUCB models for native gate families. Its action chooses the
next still-pending continuation; it cannot discard another continuation.
The full frontier is retained, with a maximum scoring panel of 32 records and
a globally fair oldest-record allocation every 32 attempted edges. No learned
value or floating-point isometry hash authorizes pruning. Pruning compares
exact symbolic equality and componentwise consumed resources; it is conservative
on clean-input subspaces because it compares full physical operators.

The dense isometry cache supplies target discrepancy and projected gate
features. This is a numerical scoring cache, not a substitute for HybridState.
Every attempted child performs the real tableau, ordered-rotation, scalar and
DAG updates. Feature-lookahead work, symbolic transitions and certification are
reported separately. No full-DAG neural policy is introduced.

The episodic base reward is certified success credit minus 0.002 per attempted
native edge. Frontier-potential differences supply shaping, with zero potential
at terminal outcomes. Staged training freezes the outer values while fitting
the inner response, then briefly adjusts the outer policy. Ranking and SARSA use the same decision-time feature vector, including the
current pending-action fraction. Evaluation freezes
all model arrays and checks checkpoint digests. Old phase-study checkpoints are
rejected by the native schema; the models must be retrained.

## Experiments retained

The 25 archived test phase specifications, logical unitaries, ancilla budgets,
CNOT/native-depth/native-gate/T-depth bounds are read from the frozen protocol,
not regenerated. The BNN exactly-one predicate remains among these targets.
The additional named registry contains QFT-2 and QFT-3 under 0/1/2 clean-qubit
budgets, CCX and four-logical-qubit C^3X under 0/1/2 clean-qubit budgets, SWAP on
2/3/4-qubit registers, and mixed-axis parity targets under 0/1/2 clean budgets.
There are 43 total restoration-qualification targets.

All QFT targets use the conventional forward Fourier matrix including output
bit reversal, with q0 the least-significant bit. QFT-3 is not replaced with an
approximate or truncated Fourier transform. Four-qubit Toffoli means three
controls and one logical target; an extra clean workspace wire makes FIVE
physical qubits. The independent reference set contains a one-clean-ancilla
QFT-3 construction and a one-clean-ancilla C^3X construction. There is no
ancilla-free exact reference supplied for those two targets. Failure to find a
circuit in the allowed work budget is UNKNOWN, not an infeasibility theorem.

References are certified AFTER discovery in separate result records with
`source=constructive_reference_not_RL`. They never seed the frontier, train a
policy, repair a failed learned run or count as learned circuit generation.

The default qualification trains five seeds for 64+96+24 episodes, matching the
previous staged episode counts, with short native-edge training budgets. The
curriculum adds genuine mixed-basis and noncommuting native tasks to the original
phase specifications. It then compares frozen hierarchy and identical untrained
initialization under 512-edge/3-second/20,000-record discovery bounds. This is a
restoration qualification, NOT a replacement publication-scale performance
claim. Wider native branching makes the old phase-network runtime results
inapplicable. The CLI exposes stage, seed and work budgets for larger campaigns.

The API also provides `optimize_native` for deterministic T-count/CNOT/depth/
T-depth/gate-count tightening. Each incumbent must be found by native discovery.
It now connects tighter-bound trials to native auditing and independent cover
checking, including nonzero optima and ordered lexicographic objectives.
Unfinished proofs still return upper bounds. See `native_optimality.md`. Ancilla budgets remain separate problems/archives.

## Independent verification and proof boundary

`certify_native` reconstructs the lossless DAG, replays HybridState including
its exact scalar lift, independently applies dense native gate matrices to every
promised input column, checks the selected phase convention and clean return,
and reconstructs all native resources. The numerical comparison tolerance is
explicit. Numerical checks are not labelled exact algebraic equality proofs
for arbitrary floating-point target matrices.

`audit_native` and `verify_native_cover` provide an optional independent-cover
verification path for tiny bounded native domains. The checker does not invoke
policy selection, discovery caches or the auditor's queue. It validates the
native schema, digest, root coverage, terminal exclusion and EVERY feasible
native continuation, using native symbolic equality and resource dominance.
The exact HybridState algebra remains shared trusted code. Terminal predicates
are numerical isometry comparisons with a fail-closed near-threshold check;
these are labelled tolerance-domain certificates, not proof-assistant or
unrestricted exact-synthesis proofs. Phase-study certificates are rejected.
Cancellation, exhausted work/record budgets and verification interruption are
never reported as infeasibility or optimality. Limits are cooperative, not a
hard operating-system CPU or byte-memory kill.

## Tests and historical evidence

The new suite has 181 cases. 126 actually synthesize circuits using completed,
frozen native training runs: 96 distinct target/seed combinations, 18 clean/
entangling cases, nine SWAP cases and three exact -I cases. A further 18 tests
exercise named hard-target frontiers without pretending that a truncated run is
a successful synthesis. Sixteen known reference circuits and two deliberately
missing references are tested separately. Essential exact-phase, noncommutation,
coherent ancilla-return, schema, coverage and interruption tests are retained.
The pre-existing 481 tests are not deleted, and their old phase-network results
are not relabelled native results. Unit-regression success budgets are deliberately
more generous than performance qualification budgets.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m hybrid_qcs.native_runner --output-dir outputs/native-hybrid-restoration
# Public API: NativeProblem, NativeHierarchy, NativeSearch, optimize_native
```

No negative or positive scientific finding from the archived phase study is
silently transferred to this restored environment. Read the new qualification
outcomes separately and distinguish certified native discovery from constructive
reference replay and from a closed optimality gap.
