# Consumer-aware cleanup: constructive results and exact scheduling

Proof-first specification, 28 September 2026.
Base: theorem-cleanup-sarsa-linucb-v1, commit 84c55c94f4b4c33ca55cbdbb9c7738222d6b32b8.

## 0. Scope and existing results

The supplied *H - Prove Cleanup Theorem(3).pdf*, Theorem 2, pp. 59--60,
proves auxiliary peak rm+min(r,m) only when all products F_ij=a b_i x_j
must coexist, the stated helper class is used, and the specified two-stage
cleanup is required. Its pp. 61--62 explicitly show that a complete phase
oracle can avoid that bank. The batch criterion on pp. 24--25 requires every
measured value to be quadratic in values that survive the batch.

This note derives consequences and constructive extensions. It does not claim
new temporary-AND gates, literature priority, global Clifford+T optimality,
or a theorem guaranteeing that a learned policy improves performance.
All logical inputs are arbitrary, possibly entangled with a reference.
Measurements are internal; no subsequent external consumer reads their bits.
Clifford+T operations are exact. Connectivity and timing are explicitly specified.

The native coherent representation remains (G,Theta,R,phi,rho). The enclosing
protocol additionally contains measurements and their classical corrections.
Boolean polynomials here certify this structured interface; they do not replace
the general native noncommuting-Pauli circuit representation.

## 1. Analytically closed workspace problems

**Proposition 1 (bound attainment).** In the original full-bank architecture,
let L=rm+min(r,m). If a complete, independently checked construction obeys all
requested resource caps and uses L auxiliary qubits, its workspace optimum is
closed. No second discovery search is required to establish that coordinate.

**Proof.** The supplied theorem gives A>=L for every admitted implementation.
The checked construction gives A<=L. Thus A*=L. Other costs have not been
minimized by this argument. If the construction violates a depth or other cap,
its infeasibility does not imply that every L-workspace schedule is infeasible.
A larger-helper construction is not excluded from a different Pareto trade-off.
QED.

Implementation consequence: compile and verify the known smaller-side witness
first. Search is optional for unmet constraints or improved depth, rather than
mandatory for rediscovering the already determined helper count. Keep the old
search unchanged for historical reproduction. For a complete-oracle contract,
the number L is not a valid lower bound.

## 2. Consumer substitution and a separated-factor construction

Use square-free Boolean polynomials: addition is XOR; multiplication satisfies
z_i^2=z_i. Substituting F_ij=a b_i x_j into the given consumer q gives a unique
algebraic normal form p. Cancelling equal monomials is exact on the full Boolean
cube. Consequently the complete oracle is O_p|z>=(-1)^p(z)|z>.

**Proposition 2 (exact consumer replacement).** A complete measured construction
whose every corrected branch is an input-independent scalar times O_p may
replace the full compute--consumer--cleanup block, whether or not it ever
materializes F. It may not replace that block if external consumers inspect
F, helpers, or internal measurement outcomes.

**Proof.** Equality of Boolean phases on every basis input gives equality by
linearity on arbitrary superpositions, including a reference. Branch scalars
c_s independent of the input and sum_s |c_s|^2=1 give the same unitary channel.
QED.

For the source consumer (degree at most two in a,b,x,F), write

    p(a,b,x)=q0(a,b,x) XOR a B(b)^T M X(x),

where q0 contains exactly the monomials of degree at most two. B and X list the
nonempty, square-free monomials of the respective operand block that occur in
the remaining terms. Each has degree one or two. No monomial mixes b and x
inside a feature. Every high-degree monomial produced by the source consumer
has a factor a and contains a nonempty b part and x part, so this representation
exists. The code must reject unsupported polynomials rather than pretend they
belong to this class.

Take an exact binary rank factorization

    M = XOR_(ell=1..d) u_ell v_ell^T,    d=rank_F2(M).

Define L_ell=u_ell^T B and R_ell=v_ell^T X. Let s be the number of distinct
quadratic features used by these factors (linear features are original inputs).

**Theorem 3 (feature-factor phase construction).** There is an exact complete
oracle for p using d+s clean-target AND computations, zero additional T gates
in cleanup, and at most s+1 auxiliary qubits. With the verified four-T
clean-target primitive its T-count is 4(d+s). If all features are linear,
T=4d and one reused helper suffices (zero when d=0).

**Proof.** Compute each required quadratic feature once into a clean qubit with
an exact clean-target AND. Form L_ell by CNOTs into one of its already available
feature wires; the pivot choice is fixed, not a heuristic. Independently form
R_ell in its disjoint block. Compute t=a L_ell into a clean helper; apply CZ(t,R_ell).
Measure t in X, with outcome mu, reset it, and apply CZ(a,L_ell)^mu before undoing
the parity transformations. Measurement contributes (-1)^(mu a L_ell), exactly
cancelled by that correction. The remaining intentional phase is
(-1)^(a L_ell R_ell). Both parity transformations restore all original feature
values. Repeat, reusing t. After a quadratic feature's final use, measure it and
correct with CZ on its original raw operands, which are restored and retained.
Apply the quadratic phase q0 with Z/CZ (and a scalar minus identity if needed).
Every one of the d+s measurements contributes 1/sqrt(2), independent of logical
input. All corrected branches are 2^(-(d+s)/2) O_p, and all auxiliaries return zero.
Counts and workspace follow. Extra CNOTs, measurements, resets and feed-forward
latency are charged; this is not a free-parity or zero-latency statement. QED.

**Restricted minimality.** d is the minimum number of separated terms L(b)R(x)
when L and R are linear combinations of the fixed B and X features. Each term
has matrix rank at most one; rank subadditivity gives at least d, and the
factorization attains d. This is NOT a global T lower bound: another nonlinear
factorization, mixed b/x factors, catalysts or native realization can do better.
The added cost s is not claimed minimal over all feature choices or rank bases.

### 2.1 Exact quadratic compression before storing features

Theorem 3 can be improved without adding a heuristic. A quadratic Boolean
function Q(z) has an exact form

    Q(z) = ell(z) XOR XOR_(j=1..k) U_j(z) V_j(z),

where ell is affine and U_j,V_j are linear. For zero constant input, ell is
linear. A constructive elimination selects any remaining term z_i z_j, defines
U=z_i XOR sum_(h!=i,j) Q_jh z_h and
V=z_j XOR sum_(h!=i,j) Q_ih z_h, and replaces Q by Q XOR U V.
No quadratic term involving i or j remains. Any linear terms created by
z_h^2=z_h remain in the affine remainder. Induction finishes in at most
floor(n/2) steps. This is established quadratic-form elimination, not a new
normal form or a claimed optimal Clifford+T synthesis.

**Corollary 3a (quadratic factors need not store each monomial).** Apply this
elimination separately to every L_ell and R_ell of Theorem 3. Share identical
products U V between factors. If s' distinct products are used, the same
construction has T=4(d+s'), with peak at most s'+1. Theorem 4 applies to their
actual first and last uses, replacing monomial-feature labels by product labels.

**Proof.** For each eliminated product, U has a pivot i absent from V and V has
a pivot j absent from U. CNOTs from the other raw variables into i and j place
U,V on those wires without an additional parity ancilla. Compute their AND into
a clean feature qubit and undo the CNOTs. The feature now stores U V while the
raw input register is restored. A factor is the XOR of these features and its
linear remainder and can be formed in-place as before. On feature erasure,
recreate U,V on the raw pivot wires, apply the outcome-conditioned CZ, then
undo their parity networks. This cancels exactly the phase from measuring U V.
All parity costs are included. Feature sharing and lifetime proofs require only
that each feature is quadratic in restored raw inputs, so the earlier argument
applies unchanged. QED.

The elimination is deterministic (first remaining pair), not selected using
benchmark performance. The catalogue retains the original monomial compiler
as a control, since a smaller number of ANDs can still give more Clifford gates.
No claim is made that this quadratic basis minimizes all total resource costs.

## 3. Exact first-use/last-use streaming

Fix the factors and primitive construction in Theorem 3. Let S_ell be the set
of quadratic features needed by factor ell; all its parity transforms are
undone before moving to another factor. Each feature is computed at most once;
there is no encoded storage, no regeneration, and no partial processing of one
factor interleaved with another. The one factor helper is separate and reused.

For a set A of already completed factors, define

    L(A) = (union_(ell in A) S_ell) intersect
           (union_(ell not in A) S_ell).

These are precisely the features whose first use occurred but whose last use
has not occurred. The exact peak required for next factor j is

    w(A,j) = 1 + | L(A) union S_j |.

Let V(All)=0 and

    V(A) = min_(j not in A) max( w(A,j), V(A union {j}) ).

**Theorem 4 (no-recomputation schedule optimum).** V(empty) is the minimum
auxiliary peak among the above schedules for the fixed factorization.
An order obtained from the recurrence attains it. Every feature is computed
once; T=4(d+s) is unchanged. The recurrence visits 2^d subsets with at most d
successors each, using polynomial-size bit-set operations.

**Proof.** In any such schedule, a feature used both before and after the cut A
must remain stored: deletion would require forbidden recomputation. All S_j must
be available during j, and its helper must also exist, giving the lower bound
w(A,j). It is attainable by delaying each feature's preparation to its first
use and deleting it immediately after its last use, correcting on its unchanged
raw parents. Thus every order has peak max_j w(A_j,j). Conditioning on the first
remaining factor yields exactly the minimax recurrence; induction on remaining
factors proves both directions. QED.

The limit d<=D is explicit in the implementation. Beyond it, the exact optimizer
reports that its domain is too large; a fixed-order construction can still be
returned as an upper bound, never labelled an optimum. A cancellation or
incomplete table is likewise not a proof.

Resource trade-offs: recomputing features independently for each factor uses
T=4(d+sum_ell |S_ell|), peak at most 1+max_ell |S_ell|, and possibly more
measurement stages. A catalogue should retain both this construction and the
no-recomputation optimum when neither dominates. Minimum workspace alone does
not guarantee minimum CNOTs, T-depth, or physical latency.

## 4. Complete block composition

**Corollary 5 (clean workspace reuse).** Suppose O_p is a product of phase
oracles O_(p_j), each implemented with every corrected branch c_(j,s) O_(p_j),
clean workspace at its boundary, and internal outcomes. The blocks may share
original input qubits; they may not share live unclean auxiliaries at boundaries.
Sequential composition has peak at most max_j A_j, counts equal to the sums,
and a valid complete coherent channel. For independently materialized rectangles,
A_j=r_j m_j+min(r_j,m_j) is one available construction, not a global lower bound.

**Proof.** Compose branch maps on the same original inputs. Their product is the
product of the scalar amplitudes times O_(XOR_j p_j). Clean workspace can be
reallocated, and scalar norm sums multiply to one. This proof also allows a
reference entangled across all blocks. QED.

For overlapping factor consumers within a block, Theorem 4, rather than the
independent-rectangle formula, accounts for shared feature lifetimes. Never
measure a value while a required correction or future consumer depends on it.

## 5. Explicit selection and optional learning

**Proposition 6 (finite-menu non-regression).** Verify each completed candidate
against the same logical phase and resource caps. Preserve the nondominated
resource vectors. Selecting the minimum by a declared lexicographic objective
is no worse in that objective than every included feasible baseline.

**Proof.** It is minimization over a finite set containing those baselines.
It does not imply componentwise dominance of the selected point, global circuit
optimality, or a faster compilation time. Interrupted candidate generation must
be disclosed. QED.

The implemented menu can include the old smaller-side full bank, its row-side
alternative, bank-free monomials, rank factors with simultaneous features,
rank factors with exact streaming, and recomputed factor features. Independent
verification decides validity. SARSA/LinUCB can supply an additional fully
costed candidate or search for a full-bank schedule under tighter depth caps;
they do not decide identities, rank, legality, or proof validity. An analytically
constructed winner is not counted as a learned discovery merely because a
checkpoint was loaded. No policy convergence or improvement theorem is asserted.

## 6. Attribution and testing boundary

Gidney, Halving the cost of quantum addition, Quantum 2, 74 (2018),
https://doi.org/10.22331/q-2018-06-18-74: established temporary-AND cost mechanism.
Meuli et al., The Role of Multiplicative Complexity in Compiling Low T-count
Oracle Circuits, arXiv:1908.01609: existing Boolean factoring and low-T compilation.
Paradis et al., Reqomp, Quantum 8, 1258 (2024),
https://doi.org/10.22331/q-2024-02-19-1258: existing space-constrained uncomputation.
The supplied cleanup note is the source of Proposition 1's restricted converse.
The other results are proof-first derivations here, built from these established
principles, not certified novelty claims. Finite tests support implementation
correctness but do not substitute for the arguments above.
