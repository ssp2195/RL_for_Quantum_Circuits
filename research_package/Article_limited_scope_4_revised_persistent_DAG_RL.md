# Reinforcement-Learning-Guided Frontier Ranking for Certified Small-Scale Clifford+\(T\) Circuit Synthesis

## Abstract

Exact synthesis over discrete quantum gate libraries is a combinatorial search problem in which the number of reachable circuit prefixes grows rapidly with the available gate count. This work develops a deliberately restricted synthesis architecture for **small-scale, exact, ancilla-free Clifford+\(T\) circuit synthesis with all-to-all CNOT connectivity**, in which reinforcement learning controls the **order of symbolic search rather than the generation of gates**. The learned action at decision time \(t\) is the selection of one currently open frontier record,

\[
a_t=v_t,\qquad v_t\in\mathcal F_t,
\]

after which the deterministic synthesis engine exhaustively generates every resource-feasible one-gate continuation of \(v_t\). Gate legality, circuit semantics, canonicalisation, resource dominance, and final acceptance are therefore outside the learned policy.

Each search record retains a lossless persistent dependency-DAG witness together with a complete forward/inverse Clifford tableau, an ordered word of signed Pauli rotations, explicit global phase in eighth-turns, and a multidimensional monotone resource vector. The DAG is represented by immutable path-shared witness steps and is materialized only when reconstruction or independent certification is required. For a circuit prefix \(v\), the algebraic invariant is written

\[
\operatorname{Sem}(v)
=
e^{i\phi_v}
C(\Theta_v)
\prod_{j=1}^{m_v}
R_{P_{v,j}}(\theta_{v,j}),
\qquad
R_P(\theta)=e^{-i\theta P/2},
\]

where \(\Theta_v\) represents the Clifford frame and the ordered rotation word represents the non-Clifford component. In the Clifford+\(T\) library, \(T\) and \(T^\dagger\) become \(\pm\pi/4\) rotations about Pauli axes transported exactly through the evolving Clifford frame. Canonicalisation uses only sound algebraic identities: signed-axis normalisation, same-axis fusion, cancellation, commuting reordering, and exact absorption of emergent Clifford rotations.

A central contribution of the restricted formulation is a **one-sided resource-simulation theorem** for Pareto pruning. Equality of implemented transformations does not require two records to possess identical feasible continuation sets. Instead, if \(u\) and \(v\) implement the same transformation and the resource state of \(u\) componentwise dominates that of \(v\), then monotonicity of the resource transition implies

\[
\mathcal L_{\mathbf B}(v)\subseteq\mathcal L_{\mathbf B}(u),
\]

where \(\mathcal L_{\mathbf B}(v)\) denotes the suffixes feasible from \(v\) under the fixed resource budget \(\mathbf B\). Every feasible solution continuation of the dominated record is therefore reproducible from the dominating record with no worse resources. This is sufficient for sound Pareto pruning and does not require symmetric continuation-language equivalence.

The variable-cardinality frontier is ranked by a shared linear action-value function trained using online semi-gradient SARSA. The implemented 24-dimensional feature vector combines normalized resources, per-wire depth, Pauli-rotation structure, exact symbolic target mismatch, last-gate indicators, target structure, and register size. Features are cached per persistent search record and all frontier scores are evaluated by one vectorized matrix-vector product. The learned policy can alter which record is investigated first but cannot alter the symbolic search space or certify a circuit. Final acceptance is performed independently from the authoritative DAG witness by dense unitary reconstruction and equality testing up to global phase.

The study is intentionally confined to the working exact-synthesis baseline rather than extending prematurely to ancillas, physical connectivity, routing, approximate synthesis, arbitrary rotations, scalable verification, neural ranking architectures, or DAG-derived policy embeddings. A publication-scale evaluation protocol is specified around unrestricted native Clifford+\(T\) targets on primarily two and three qubits, multi-seed training, strong deterministic scheduler baselines, budget-success curves, compute-normalised measurements, and controlled ablations. This separation produces a compact experimental object in which the scientific question is precise: **does learned frontier ranking allocate finite exact-search effort more effectively than non-learned scheduling when every other component of the synthesiser is held fixed?**

The qualified implementation was exercised with five independent training seeds (11, 19, 23, 31, and 47), 160 online-SARSA episodes per seed, and a hard 1,800-s campaign limit. The clean-source qualification completed all 800 training episodes with independent certification, all 35 unrestricted frozen-policy evaluations, five held-out exact QFT-2 evaluations, and five structured Toffoli parity-network stress tests. Fifteen regression tests passed; the recorded process-CPU time for that qualification was 22.703 s. Runtime is reported only as environment-specific engineering evidence and is not interpreted as an asymptotic performance claim.

**Keywords:** quantum circuit synthesis; Clifford+\(T\); reinforcement learning; SARSA; symbolic search; frontier ranking; Pauli rotations; Pareto pruning; exact synthesis; circuit equivalence.

## Introduction

Quantum circuit synthesis seeks a circuit over an admissible gate alphabet whose induced transformation equals a prescribed target transformation, normally up to physically irrelevant global phase. For an \(n\)-qubit circuit

\[
C=(g_1,g_2,\ldots,g_L),
\]

with gate matrices \(U_{g_i}\), we use the chronological convention

\[
U(C)
=
U_{g_L}U_{g_{L-1}}\cdots U_{g_1}.
\tag{1}
\]

Given \(U_\star\in U(2^n)\), exact synthesis requires

\[
U(C)\sim U_\star,
\tag{2}
\]

where

\[
U\sim V
\iff
\exists\phi\in\mathbb R:
U=e^{i\phi}V.
\tag{3}
\]

The discrete nature of exact synthesis makes the problem fundamentally combinatorial. Even highly structured subclasses demonstrate a substantial gap between compact algebraic representations and brute-force search. SAT-based optimal Clifford synthesis has been reported for instances substantially larger than those tractable by earlier exhaustive methods, while recent work on CNOT-optimal Clifford synthesis explicitly notes the severe scaling limit of exhaustive search over its relevant normal forms. At the same time, exact two-qubit Clifford+\(T\) synthesis remains an active specialised research problem, reinforcing the importance of matching claims to a sharply bounded circuit class rather than treating “quantum synthesis” as a single homogeneous scaling regime. [1]–[3] citeturn14academia1turn13academia0turn14academia3

The present work therefore fixes a deliberately narrow exact-synthesis problem. The logical system contains primarily

\[
n\in\{2,3\}
\]

qubits; no ancillas are available; every ordered pair of distinct logical qubits admits a CNOT; all targets are required to be exactly representable within the frozen native grammar and resource envelope; and the terminal circuit is independently certified through dense unitary reconstruction. The gate alphabet is

\[
\boxed{
\mathcal G_n
=
\left\{
H_i,S_i,S_i^\dagger,T_i,T_i^\dagger
\right\}_{i=0}^{n-1}
\cup
\left\{
\operatorname{CNOT}_{i\rightarrow j}:i\neq j
\right\}.
}
\tag{4}
\]

Consequently,

\[
|\mathcal G_n|
=
5n+n(n-1).
\tag{5}
\]

Exact Clifford+\(T\) representability is itself a nontrivial algebraic restriction; the literature characterises the multiqubit exact Clifford+\(T\) domain both with and without local ancillas. The present study avoids conflating exact and approximate synthesis by generating the main target corpus from the same native grammar and by admitting no synthesis ancillas. [4] citeturn14academia2

The second restriction concerns the role assigned to learning. Reinforcement learning has already been investigated at several points in the quantum compilation and synthesis pipeline. Kölle *et al.* formulate directed Clifford+\(T\) circuit construction as an RL environment; Kremer *et al.* learn synthesis and routing strategies for structured circuit classes; Riu *et al.* choose ZX-calculus transformations through reinforcement learning; Dubal *et al.* use learned decision making for Pauli-network synthesis; and recent equivariant work learns elementary reductions of Clifford symplectic representations. Other current research combines learned residual-cost models with explicit beam search rather than model-free gate-by-gate RL. [5]–[10] citeturn12academia2turn16academia12turn16academia13turn12academia0turn14academia0turn16academia14

The action studied here is different. Let

\[
\mathcal F_t
\]

be the set of discovered, accepted, but not yet expanded search records at decision step \(t\). The reinforcement-learning action is

\[
\boxed{
a_t=v_t,\qquad v_t\in\mathcal F_t.
}
\tag{6}
\]

Once \(v_t\) has been selected, the learned component has no further role in that expansion. The environment constructs

\[
\boxed{
\operatorname{Expand}_{\mathbf B}(v_t)
=
\left\{
\delta(v_t,g):
g\in\mathcal G_n,\;
\boldsymbol\rho\!\left(\delta(v_t,g)\right)
\preceq
\mathbf B
\right\},
}
\tag{7}
\]

where \(\boldsymbol\rho\) is the resource record and \(\mathbf B\) is the fixed resource budget. **Every** one-gate child satisfying the deterministic resource constraints is generated. Hence the RL action set and the symbolic branching set are different objects:

\[
|\mathcal A(s_t)|=|\mathcal F_t|,
\qquad
b(v_t)
=
\left|
\operatorname{Expand}_{\mathbf B}(v_t)
\right|.
\tag{8}
\]

The agent chooses which existing record receives the next unit of search effort; it does not choose which quantum gate is allowed to exist.

This decomposition has a useful precedent outside quantum synthesis. Learned open-node scheduling has been formulated as an RL problem for branch-and-bound, where a policy decides which currently available search-tree node should be processed next while the underlying optimisation machinery retains responsibility for generating and evaluating descendants. [11] citeturn12academia1 The analogous decomposition in the present setting permits the quantum-symbolic machinery to remain exact while making search priority learnable.

Based on the primary quantum-synthesis literature surveyed through 15 August 2026, recent learned methods predominantly make gate-level synthesis decisions, graph-rewrite decisions, Pauli-network synthesis decisions, Clifford-reduction decisions, or learned heuristic estimates inside another search mechanism. We have not identified a published quantum-synthesis system with precisely the combination adopted here: a learned action over a **persistent global frontier**, exhaustive deterministic generation of **all** native one-gate children of the selected record, a semantic canonical/Pareto archive, and an independent terminal certifier. This is a literature-based positioning rather than a claim that unpublished or unindexed implementations cannot share some or all of these elements. citeturn12academia0turn12academia2turn14academia0turn16academia13turn16academia14

The central research question is therefore not whether reinforcement learning can invent quantum circuits in the unrestricted sense. It is

\[
\boxed{
\begin{minipage}{0.88\linewidth}
Can a learned ranking policy improve finite-budget allocation of exact symbolic search effort by choosing which already-discovered Clifford+\(T\) frontier record is expanded next, when gate generation, semantics, resource pruning, and certification are held deterministic and identical across schedulers?
\end{minipage}
}
\tag{9}
\]

The restricted architecture is scientifically useful precisely because it isolates this question. FIFO, LIFO, uniform-cost selection, random scheduling, a deterministic target-potential heuristic, a zero-weight linear policy, and a trained SARSA policy can all act on the same frontier generated by the same search engine. Consequently, scheduler comparisons do not also change the native gate set, semantic representation, canonicalisation rules, Pareto test, or terminal verifier.

The symbolic core follows established representations. Stabiliser and Clifford evolution admit efficient tableau or binary symplectic descriptions; Clifford operators possess strong canonical structure; and Clifford-conjugated \(T\) gates can be treated exactly as \(\pi/4\) rotations about Pauli operators. [12]–[15] citeturn13academia1turn13academia2turn12academia3 A circuit DAG is retained as the authoritative reconstructable witness. DAG circuit representations are also standard in current compiler infrastructure; for example, Qiskit explicitly uses `DAGCircuit` objects in its transpilation stages because operation dependencies and information flow are represented directly. [16] citeturn15search0

The paper makes four tightly coupled methodological contributions. First, it specifies an exact Clifford-frame/ordered-Pauli-rotation state representation whose algebraic summary is maintained inductively alongside an authoritative DAG. Second, it formulates search control as variable-cardinality frontier ranking with linear semi-gradient SARSA rather than gate generation. Third, it states a resource-dominance theorem appropriate to the actual bounded search: the required relation is **one-sided continuation simulation**, not equality of feasible continuation languages. Fourth, it separates learned search efficiency from semantic correctness through independent dense certification and an evaluation design in which all schedulers operate on one frozen deterministic synthesis core.

## Exact-Synthesis Baseline and Symbolic Representation

The synthesis instance is denoted

\[
\xi
=
\left(
n,
U_\star,
\mathcal G_n,
\mathbf B
\right),
\tag{10}
\]

where \(n\in\{2,3\}\) in the primary regime, \(U_\star\) is the exact target, \(\mathcal G_n\) is given by (4), and \(\mathbf B\) contains the resource limits imposed on each search record. No placement variable, hardware coupling graph, ancilla state, approximate error tolerance, or arbitrary-angle gate parameter is part of the Version-1 synthesis state.

Define

\[
\mathcal G_n^\ast
=
\bigcup_{\ell=0}^{\infty}\mathcal G_n^\ell.
\]

The bounded exact-synthesis problem is

\[
\boxed{
\text{find }C\in\mathcal G_n^\ast
\text{ such that }
U(C)\sim U_\star
\text{ and }
\boldsymbol\rho(C)\preceq\mathbf B.
}
\tag{11}
\]

At least one component of \(\mathbf B\) bounds total gate count. Therefore the raw search space used in a single synthesis instance is finite.

A search record is represented as

\[
\boxed{
v=
\left(
C_v,
D_v,
\Theta_v,
\Phi_v,
\boldsymbol\rho_v,
\mathbf x_v,
\zeta_v
\right).
}
\tag{12}
\]

Here \(C_v\) is a reconstructable gate witness; \(D_v\) is its circuit DAG; \(\Theta_v\) is the Clifford-frame representation; \(\Phi_v\) is the non-Clifford Pauli-rotation representation together with tracked phase; \(\boldsymbol\rho_v\) contains synthesis resources; \(\mathbf x_v\) contains policy features; and \(\zeta_v\) contains search metadata such as the parent record, generating gate, persistent identity, insertion order, and expanded status. The policy features and metadata are not part of the circuit semantics.

The DAG is authoritative. In particular, the final certifier does not accept a candidate because its learned features, canonical key, or cached symbolic distance indicate success. It reconstructs the candidate from the witness/DAG and independently evaluates the corresponding unitary. This distinction makes the algebraic representation an exact search accelerator rather than an unchecked source of terminal truth.

For the Clifford component, write an \(n\)-qubit Pauli as

\[
P(\mathbf x,\mathbf z)
=
i^{\mathbf x^{\mathsf T}\mathbf z}
X^{\mathbf x}Z^{\mathbf z},
\tag{13}
\]

with

\[
\mathbf x,\mathbf z\in\mathbb F_2^n,
\qquad
\mathbf p
=
\begin{bmatrix}
\mathbf x\\
\mathbf z
\end{bmatrix}
\in\mathbb F_2^{2n}.
\tag{14}
\]

The binary symplectic form is

\[
J
=
\begin{bmatrix}
0&I_n\\
I_n&0
\end{bmatrix}.
\tag{15}
\]

A Clifford transformation acts linearly on binary Pauli labels through

\[
M\in\mathbb F_2^{2n\times2n},
\qquad
M^{\mathsf T}JM=J,
\tag{16}
\]

supplemented by phase/sign information \(\mathbf r\). We therefore write

\[
\Theta=(M,\mathbf r)
\tag{17}
\]

for the Clifford-frame state. Binary tableau and symplectic descriptions are standard exact tools for stabiliser/Clifford computation, and canonical structural descriptions of Clifford operators can be obtained in polynomial time. [12], [13] citeturn13academia1turn13academia2

A \(T\) gate is not Clifford and cannot be absorbed into \(\Theta\). The non-Clifford component is instead represented by Pauli rotations. For a signed Hermitian Pauli

\[
P\in\mathcal P_n^\pm
=
\pm\{I,X,Y,Z\}^{\otimes n},
\]

define

\[
\boxed{
R_P(\theta)
=
\exp\!\left(-\frac{i\theta}{2}P\right)
=
\cos\!\left(\frac{\theta}{2}\right)I
-
i\sin\!\left(\frac{\theta}{2}\right)P.
}
\tag{18}
\]

For every Clifford \(C\),

\[
\boxed{
C R_P(\theta)C^\dagger
=
R_{CPC^\dagger}(\theta),
}
\tag{19}
\]

because Clifford conjugation maps Pauli operators to signed Pauli operators. Equivalently,

\[
C R_P(\theta)
=
R_{CPC^\dagger}(\theta)C.
\tag{20}
\]

This closure is the essential reason that Clifford+\(T\) circuits admit a compact Clifford-frame/Pauli-rotation decomposition. Zhang and Chen exploit the same representation to optimise \(T\) gates as \(\pi/4\) rotations around Paulis. [14] citeturn12academia3

Under the convention (18),

\[
R_Z(\pi/4)
=
\begin{bmatrix}
e^{-i\pi/8}&0\\
0&e^{i\pi/8}
\end{bmatrix},
\]

and therefore

\[
\boxed{
T
=
e^{i\pi/8}R_Z(\pi/4),
\qquad
T^\dagger
=
e^{-i\pi/8}R_Z(-\pi/4).
}
\tag{21}
\]

The Clifford phase gate similarly obeys

\[
S=e^{i\pi/4}R_Z(\pi/2).
\tag{22}
\]

Thus explicit \(\pm\pi/4\) rotations identify the non-Clifford contributions, whereas rotations by integer multiples of \(\pi/2\) are Clifford.

The rotation axes need not remain single-qubit. For example,

\[
HXH=Z,
\qquad
HZH=X,
\qquad
HYH=-Y,
\tag{23}
\]

hence

\[
\boxed{
HTH
=
e^{i\pi/8}R_X(\pi/4).
}
\tag{24}
\]

Likewise, for \(C=\operatorname{CNOT}_{c\rightarrow t}\),

\[
C Z_c C^\dagger=Z_c,
\qquad
C Z_t C^\dagger=Z_cZ_t,
\tag{25}
\]

and therefore

\[
\boxed{
\operatorname{CNOT}_{c\rightarrow t}
\,T_t\,
\operatorname{CNOT}_{c\rightarrow t}
=
e^{i\pi/8}
R_{Z_cZ_t}(\pi/4).
}
\tag{26}
\]

Multi-qubit Pauli axes are thus not an auxiliary generalisation introduced for future hardware synthesis; they arise intrinsically from exact Clifford conjugation of the native \(T\) gate. Pauli-network synthesis research likewise treats such multi-qubit Pauli rotations as explicit synthesis objects. [10], [14] citeturn12academia0turn12academia3

The order of rotations matters. For Hermitian Paulis \(P,Q\),

\[
[P,Q]=0
\Longrightarrow
R_P(\alpha)R_Q(\beta)
=
R_Q(\beta)R_P(\alpha),
\tag{27}
\]

whereas anticommuting axes cannot generally be exchanged. In binary symplectic form,

\[
\boxed{
\mathbf p_i^{\mathsf T}J\mathbf p_j
=
\begin{cases}
0,&[P_i,P_j]=0,\\
1,&\{P_i,P_j\}=0
\end{cases}
\pmod 2.
}
\tag{28}
\]

Accordingly, the non-Clifford part is stored as an **ordered** word

\[
\mathcal R_v
=
\bigl(
(P_{v,1},\theta_{v,1}),
\ldots,
(P_{v,m_v},\theta_{v,m_v})
\bigr).
\tag{29}
\]

Its anticommutation dependency graph is

\[
G_R(v)
=
(V_R,E_R),
\tag{30}
\]

with

\[
V_R=\{1,\ldots,m_v\}
\]

and

\[
E_R
=
\left\{
(i,j):
i<j,\;
\mathbf p_i^{\mathsf T}J\mathbf p_j=1
\right\}.
\tag{31}
\]

For the \(\pi/4\) Pauli-rotation setting generated by Clifford+\(T\) circuits, commuting exchanges can be represented through the corresponding partial-order structure; anticommuting dependencies must remain ordered. [14] citeturn12academia3

Several elementary identities are exact:

\[
\boxed{
R_{-P}(\theta)=R_P(-\theta),
}
\tag{32}
\]

\[
\boxed{
R_P(\alpha)R_P(\beta)
=
R_P(\alpha+\beta),
}
\tag{33}
\]

and hence

\[
R_P(\theta)R_P(-\theta)=I.
\tag{34}
\]

For the non-Clifford angles relevant here,

\[
R_P(\pi/4)^2
=
R_P(\pi/2)
\in\mathcal C_n.
\tag{35}
\]

These equalities permit exact local reductions of the rotation word. Established Clifford+\(T\) optimisation procedures use precisely such Pauli-rotation algebra, and the known pair-scanning approach of Zhang and Chen has worst-case complexity \(O(nk^2)\) for \(k\) initial \(T\)-type rotations. [14] citeturn12academia3

The phase convention is retained explicitly in the semantic record. Since

\[
R_P(\theta+2\pi)
=
-R_P(\theta),
\tag{36}
\]

angle reduction by \(2\pi\) requires

\[
\theta\leftarrow\theta-2\pi,
\qquad
\phi\leftarrow\phi+\pi
\pmod{2\pi}.
\tag{37}
\]

Because the terminal synthesis relation (3) quotients global phase, \(\phi_v\) need not distinguish canonical state identities, but it remains part of the exact semantic bookkeeping and is not discarded from the authoritative witness.

The algebraic invariant of a record is

\[
\boxed{
\operatorname{Sem}(v)
=
e^{i\phi_v}
C(\Theta_v)
\prod_{j=1}^{m_v}
R_{P_{v,j}}(\theta_{v,j}),
}
\tag{38}
\]

where the product is ordered from left to right as displayed, and

\[
\Phi_v=(\phi_v,\mathcal R_v).
\tag{39}
\]

Appending a gate \(g\) to the chronological circuit sequence means

\[
U(C_v\oplus g)=U_gU(C_v).
\tag{40}
\]

If \(G\) is Clifford,

\[
G\operatorname{Sem}(v)
=
e^{i\phi_v}
\bigl(GC(\Theta_v)\bigr)
\prod_jR_{P_{v,j}}(\theta_{v,j}),
\]

so the update is simply

\[
\boxed{
C(\Theta_v)
\leftarrow
G C(\Theta_v),
\qquad
\mathcal R_v\leftarrow\mathcal R_v.
}
\tag{41}
\]

If the appended operation is \(R_Q(\vartheta)\),

\[
\begin{aligned}
R_Q(\vartheta)\operatorname{Sem}(v)
&=
e^{i\phi_v}
R_Q(\vartheta)C(\Theta_v)
\prod_jR_{P_{v,j}}(\theta_{v,j})\\
&=
e^{i\phi_v}
C(\Theta_v)
R_{C(\Theta_v)^\dagger Q C(\Theta_v)}(\vartheta)
\prod_jR_{P_{v,j}}(\theta_{v,j}),
\end{aligned}
\tag{42}
\]

and therefore

\[
\boxed{
P_{\mathrm{new}}
=
C(\Theta_v)^\dagger Q C(\Theta_v).
}
\tag{43}
\]

For \(T_i\) and \(T_i^\dagger\),

\[
P_{\mathrm{new}}
=
C(\Theta_v)^\dagger Z_i C(\Theta_v),
\qquad
\theta_{\mathrm{new}}
=
\pm\frac{\pi}{4},
\tag{44}
\]

with the corresponding phase increment \(\pm\pi/8\).

At the root,

\[
C(\Theta_\varnothing)=I,
\qquad
\phi_\varnothing=0,
\qquad
\mathcal R_\varnothing=().
\tag{45}
\]

Equations (41)–(44) then establish (38) inductively for every native gate sequence. This hybrid representation is closely related to established Clifford+\(T\) Pauli-rotation formulations and to work on synthesising Pauli-rotation sequences with controlled Hadamard count. [14], [15] citeturn12academia3


### Persistent dependency-DAG implementation

The implementation stores the circuit witness persistently rather than copying a complete gate list or adjacency structure into every frontier state. Let a witness step be

\[
w_k=(w_{k-1},g_k,\operatorname{par}_k,\ell_k,k),
\tag{39a}
\]

where \(w_{k-1}\) is the previous chronological step, \(g_k\) is the appended gate, \(\operatorname{par}_k\) is the tuple of latest predecessor steps on the wires touched by \(g_k\), \(\ell_k\) is its DAG level, and \(k\) is the persistent gate index. The hybrid state stores only the tail pointer

\[
\tau_v=w_{|C_v|-1}
\tag{39b}
\]

and one latest-step pointer per wire,

\[
\boldsymbol\tau_v=(\tau_{v,0},\ldots,\tau_{v,n-1}).
\tag{39c}
\]

For a gate on wire set \(S(g)\), its parents are

\[
\operatorname{par}(g)=\operatorname{sort}
\left(\{\tau_{v,q}:q\in S(g),\;\tau_{v,q}\neq\varnothing\}\right),
\tag{39d}
\]

and its level is

\[
\ell(g)=1+\max_{q\in S(g)}d_q(v).
\tag{39e}
\]

Appending a child allocates one immutable witness step and reuses the entire prefix through structural sharing. If \(M\) accepted frontier/search records have mean gate count \(\bar L\), materializing a full length-\(L\) gate list per state would require witness storage proportional to \(\Theta(M\bar L)\). With path sharing, the incremental witness allocation is one node per accepted transition, while every state stores only \(O(n)\) wire-tail references in addition to its symbolic algebraic data. This optimization changes representation cost but not search semantics.

The full DAG is reconstructed only on demand. Traversing the chronological predecessor chain yields the gate sequence, after which parent indices are recovered from the stored predecessor identities. The materialized DAG is validated against three invariants: contiguous node indices, exact latest-wire dependency parents, and the recurrence (39e). The replay validator then reconstructs the symbolic state from the materialized DAG with partial-order pruning disabled and checks both the canonical key and the resource vector. Thus the DAG remains the lossless source of reconstructable circuit truth even though most search decisions never materialize it.

### Deterministic transition-level reductions and caches

Two target-independent partial-order reductions are applied before symbolic child construction. First, if all touched wires share the same latest parent and the candidate gate is the exact inverse of that parent on the same wires, the child is rejected. Second, if the candidate gate is disjoint from the chronologically last gate and has a smaller deterministic gate sort key, it is rejected. The second rule chooses one representative ordering for commuting disjoint gates and removes redundant interleavings without introducing target information. Both rules can be disabled for replay validation.

The implementation also keeps the following incremental caches inside each immutable hybrid state: \(T\)-count, CNOT count, total gate count, per-wire depth, maximum depth, canonical key, persistent wire tails, ordered Pauli rotations, and the complete Clifford tableau. Consequently, a child transition updates only the quantities affected by the appended gate; no complete DAG traversal is required in the inner expansion loop.

Canonicalisation is deliberately **sound but not required to be complete**. Let

\[
\kappa:V\rightarrow\mathcal K
\tag{46}
\]

be the canonical-key function. The necessary correctness condition is

\[
\boxed{
\kappa(u)=\kappa(v)
\Longrightarrow
\operatorname{Sem}(u)\sim\operatorname{Sem}(v).
}
\tag{47}
\]

The converse is unnecessary:

\[
\operatorname{Sem}(u)\sim\operatorname{Sem}(v)
\nRightarrow
\kappa(u)=\kappa(v).
\tag{48}
\]

Failure to recognise some equivalence increases search cost but does not remove a valid branch. By contrast, merging semantically different records would be unsound.

For the frozen baseline, define

\[
\kappa(v)
=
\operatorname{Serialize}
\left[
\mathcal N_{\mathrm{PR}}
\left(
\Theta_v,
\Phi_v
\right)
\right],
\tag{49}
\]

where \(\mathcal N_{\mathrm{PR}}\) is restricted to transformations whose correctness follows from exact Pauli-rotation identities. It first chooses a deterministic signed-Pauli convention using (32). It reduces angles with the phase correction (37), fuses adjacent equal axes using (33), removes zero rotations, and exchanges adjacent rotations only where (27) proves that the exchange is valid. It never globally sorts anticommuting rotations.

When reduction produces a Clifford-angle factor

\[
K=R_P(k\pi/2),
\qquad
k\in\mathbb Z,
\tag{50}
\]

that factor may be transported exactly into the Clifford frame. If \(K\) lies to the right of \(R_Q(\theta)\), then

\[
R_Q(\theta)K
=
K R_{K^\dagger QK}(\theta),
\tag{51}
\]

and \(K^\dagger QK\) remains a signed Pauli. Repeated application of (51) moves \(K\) leftward until it can be absorbed into \(C(\Theta_v)\). Every crossed Pauli axis is conjugated explicitly. Thus two identical \(\pi/4\) rotations may collapse into a Clifford operation without losing the correct action on later noncommuting rotations.

The identity relation is equality of the canonical payload, not equality of a finite-width hash. An implementation may use

\[
h(\kappa(v))
\tag{52}
\]

to locate an archive bucket, but two records are merged only after verifying

\[
\kappa(u)=\kappa(v).
\tag{53}
\]

The resource state is flattened into a vector containing additive gate resources and per-wire scheduling state:

\[
\boxed{
\boldsymbol\rho(v)
=
\left(
n_T(v),
n_{2q}(v),
n_g(v),
d_0(v),\ldots,d_{n-1}(v)
\right).
}
\tag{54}
\]

Here \(n_T\) counts \(T/T^\dagger\) gates, \(n_{2q}\) counts CNOTs, \(n_g\) counts all native gates, and

\[
\mathbf d(v)
=
(d_0(v),\ldots,d_{n-1}(v))
\tag{55}
\]

is the per-wire depth state. The ordinary reported depth is

\[
D(v)=\max_q d_q(v).
\tag{56}
\]

For a native gate \(g\) acting on the wire set \(S(g)\), let

\[
\tau_g(\mathbf d)
=
1+\max_{q\in S(g)}d_q.
\tag{57}
\]

The depth transition is

\[
[T_g(\mathbf d)]_q
=
\begin{cases}
\tau_g(\mathbf d),&q\in S(g),\\
d_q,&q\notin S(g).
\end{cases}
\tag{58}
\]

This vector form is important because scalar total depth alone does not, in general, reveal how much scheduling slack remains on each qubit.

Resource feasibility is defined componentwise by

\[
\boldsymbol\rho(v)\preceq\mathbf B.
\tag{59}
\]

For two records,

\[
\boldsymbol\rho(u)\preceq\boldsymbol\rho(v)
\iff
\rho_i(u)\le \rho_i(v)
\quad
\forall i,
\tag{60}
\]

and

\[
\boldsymbol\rho(u)\prec\boldsymbol\rho(v)
\]

means (60) holds and is strict in at least one coordinate.

The use of the per-wire vector makes depth monotonicity explicit. If

\[
\mathbf d^{(u)}\le\mathbf d^{(v)}
\tag{61}
\]

componentwise, then

\[
\max_{q\in S(g)}d_q^{(u)}
\le
\max_{q\in S(g)}d_q^{(v)},
\]

hence

\[
T_g(\mathbf d^{(u)})
\le
T_g(\mathbf d^{(v)}).
\tag{62}
\]

Induction over a common suffix \(K\) yields

\[
\boxed{
\mathbf d^{(u)}\le\mathbf d^{(v)}
\Longrightarrow
\mathbf d(u\oplus K)
\le
\mathbf d(v\oplus K).
}
\tag{63}
\]

The additive coordinates \(n_T,n_{2q},n_g\) satisfy the same monotonicity immediately. This property is the foundation of the one-sided pruning theorem developed below.

## Frontier-Ranking Reinforcement Learning

The implemented learning problem is online ranking of persistent frontier records. The policy never chooses a quantum gate. For target \(\xi\), let \(\mathcal F_t\) be the active non-dominated frontier at decision step \(t\). The action is

\[
a_t=v_t,\qquad v_t\in\mathcal F_t.
\tag{80}
\]

After \(v_t\) is selected, the deterministic environment attempts every native gate in the frozen grammar, applies the target-independent legality and resource tests, updates the persistent DAG and symbolic state, and submits surviving children to the canonical/Pareto archive. Thus scheduling is learned, whereas branching is exhaustive.

### Persistent frontier and Pareto archive

The implementation stores search records in dictionaries keyed by persistent integer record identifiers. A record is

\[
r=(i,s,d_{\mathrm{sym}},e,\mathbf x),
\tag{81}
\]

where \(i\) is a monotonically increasing identifier, \(s\) is the immutable hybrid state, \(d_{\mathrm{sym}}\) is the cached symbolic target distance, \(e\) is the expansion flag, and \(\mathbf x\) is an optional cached policy feature vector. The Pareto archive maps each semantic canonical key \(\kappa(s)\) to a list of pairs

\[
(\boldsymbol\rho(s),i).
\tag{82}
\]

A generated state is rejected when an archived resource vector weakly dominates it. If the new resource vector strictly dominates archived representatives of the same semantic key, those representatives are removed from the active frontier before the new record is inserted.

A practical optimization is that the active frontier is not re-sorted at every decision. Record identifiers are inserted monotonically; deletion of expanded or dominated records preserves the relative insertion order of survivors. The frontier snapshot is therefore the tuple of dictionary values, reducing the snapshot operation from an unnecessary \(O(F_t\log F_t)\) sort to \(O(F_t)\) materialization, where \(F_t=|\mathcal F_t|\). Persistent record identifiers also provide deterministic tie breaking.

### Exact symbolic target distance

The current ranker uses a symbolic mismatch rather than a dense unitary feature in the inner loop. Let \(\Theta_v^{\mathrm{can}}\) and \(\Theta_\star^{\mathrm{can}}\) denote the canonical tableau payloads. Define

\[
d_\Theta(v)=\sum_j\mathbf 1\!\left[\Theta_{v,j}^{\mathrm{can}}\neq\Theta_{\star,j}^{\mathrm{can}}\right].
\tag{83}
\]

Let \(R_v=(r_1,\ldots,r_m)\) and \(R_\star=(r_1^\star,\ldots,r_{m_\star}^\star)\) be canonical rotation payload sequences. The sequence mismatch is

\[
d_{\mathrm{seq}}(v)=|m-m_\star|+\sum_{j=1}^{\min(m,m_\star)}\mathbf 1[r_j\neq r_j^\star].
\tag{84}
\]

If \(N_v(a)\) and \(N_\star(a)\) count occurrences of canonical rotation payload \(a\), the multiset mismatch is

\[
d_{\mathrm{multi}}(v)=\sum_a |N_v(a)-N_\star(a)|.
\tag{85}
\]

The cached total symbolic distance is

\[
\boxed{d_{\mathrm{sym}}(v)=2d_\Theta(v)+d_{\mathrm{seq}}(v)+d_{\mathrm{multi}}(v).}
\tag{86}
\]

The frontier potential used for reward shaping is

\[
\Psi(s_t)=
\begin{cases}
0,&\text{if a symbolic solution record has been found},\\
-\min_{v\in\mathcal F_t} d_{\mathrm{sym}}(v),&\mathcal F_t\neq\varnothing,\\
-(4B_g+1),&\mathcal F_t=\varnothing.
\end{cases}
\tag{87}
\]

### Implemented 24-dimensional feature map

For each persistent record the current implementation computes exactly one 24-dimensional vector and caches it on the record. Let the gate/resource budgets be \(B_T,B_C,B_g,B_D\), let \(m(v)\) be rotation-word length, \(A(v)\) the number of anticommuting rotation pairs, and \(\bar w_P(v)\) the mean Pauli weight. With absent wires padded by zero to three positions, the feature vector is

\[
\mathbf x(v)=\bigl[1,
\frac{n_T}{B_T},\frac{n_C}{B_C},\frac{n_g}{B_g},\frac{D}{B_D},
\frac{d_0}{B_D},\frac{d_1}{B_D},\frac{d_2}{B_D},
\frac{m}{B_T},\frac{A}{\max(1,{B_T\choose2})},\frac{\bar w_P}{n},
\widehat d_\Theta,\widehat d_{\mathrm{seq}},\widehat d_{\mathrm{multi}},\widehat d_{\mathrm{sym}},
\frac{m_\star}{B_T},\widehat c_\Theta^\star,
\mathbf 1[g_{\mathrm{last}}=H],\mathbf 1[g_{\mathrm{last}}=S],\mathbf 1[g_{\mathrm{last}}=S^\dagger],
\mathbf 1[g_{\mathrm{last}}=T],\mathbf 1[g_{\mathrm{last}}=T^\dagger],\mathbf 1[g_{\mathrm{last}}=\mathrm{CNOT}],
\frac{n}{3}\bigr]^\mathsf T.
\tag{88}
\]

The normalized mismatch terms use fixed denominators determined by qubit count, rotation budget, and target rotation length. \(\widehat c_\Theta^\star\) is the normalized number of non-identity target-tableau generator payloads. The feature map contains no raw DAG adjacency encoding; DAG structure influences resources and reconstruction but is not yet supplied directly to the policy. This separation is deliberate and motivates a later DAG-feature extension.

### Linear scoring, vectorized frontier evaluation, and caching

The action-value approximation is

\[
\boxed{Q_{\boldsymbol\theta}(v)=\boldsymbol\theta^\mathsf T\mathbf x(v).}
\tag{89}
\]

If the frontier matrix is

\[
X_t=\begin{bmatrix}\mathbf x(v_1)^\mathsf T\\\vdots\\\mathbf x(v_{F_t})^\mathsf T\end{bmatrix},
\tag{90}
\]

all candidate scores are evaluated together as

\[
\mathbf q_t=X_t\boldsymbol\theta.
\tag{91}
\]

Feature construction is skipped for any record whose cached vector is already present. The implementation separately profiles feature-construction time and matrix-vector scoring time. This is important because persistent frontier ranking may involve many repeated observations of the same surviving records.

With \(d_x=24\), uncached evaluation costs \(O(F_t d_x)\) feature assembly plus \(O(F_t d_x)\) scoring, whereas repeated observations of an unchanged frontier subset avoid recomputing record-local features. The dominant variable cost becomes one dense matrix-vector product and tuple construction.

### Epsilon-greedy action selection and semi-gradient SARSA

With exploration probability \(\varepsilon_t\), the policy chooses a uniformly random frontier index with probability \(\varepsilon_t\); otherwise it chooses a score maximizer. Numerical score ties within absolute tolerance \(10^{-12}\) are resolved by the smallest persistent record identifier. The training schedule linearly anneals \(\varepsilon\) from 0.35 to 0.03 by default.

For selected feature vector \(\mathbf x_t\), current score \(q_t\), reward \(r_{t+1}\), and next on-policy score \(q_{t+1}\), the update is

\[
\delta_t=r_{t+1}+\gamma q_{t+1}-q_t,
\tag{92}
\]

\[
\boxed{\boldsymbol\theta\leftarrow\boldsymbol\theta+\alpha\delta_t\mathbf x_t.}
\tag{93}
\]

The terminal bootstrap term is omitted. The implementation uses \(\gamma=1\) and clips each learned coefficient to the interval \([-50,50]\) after every update as a numerical safeguard.

The environment reward is

\[
r_{t+1}=-1+\beta[\Psi(s_{t+1})-\Psi(s_t)]
+R_{\mathrm{succ}}\mathbf 1[\mathrm{success}]
-R_{\mathrm{fail}}\mathbf 1[\mathrm{exhausted\ or\ truncated}],
\tag{94}
\]

with default \(\beta=0.5\) and success/failure magnitudes 20 in the compact implementation. Because the policy only selects frontier records, this reward can change search order but cannot alter child legality or certify an incorrect circuit.

### Training deadline and instrumentation

Online training checks both process CPU time and wall time against a hard deadline and stops when either reaches the configured limit. Per-episode logs include target, success, independent certification status, expansions, total reward, mean absolute TD error, epsilon, CPU time, and wall time. Aggregate instrumentation records transition time in gate validation, persistent DAG append, tableau update, inverse Pauli transport, rotation insertion, canonical-key construction, archive maintenance, frontier snapshot construction, and independent certification. Search metrics additionally record generated, accepted, and rejected children, peak frontier size, total records, maximum rotation-word length, and mean generated-state rotation length.

The deterministic search loop is therefore:

```text
construct identity HybridState and persistent root record
insert root into frontier and Pareto archive
while frontier nonempty and expansion/deadline budget remains:
    snapshot active records without re-sorting
    obtain cached/new 24D features for each record
    score frontier by one matrix-vector product
    select persistent record ID by scheduler or epsilon-greedy SARSA
    remove selected record from frontier and mark expanded
    for every native gate in deterministic grammar order:
        reject invalid/resource-infeasible/partial-order-redundant child
        update Clifford tableau or transported Pauli rotation
        allocate one persistent DAG witness step
        update incremental resources and canonical key
        apply semantic-key Pareto dominance
        insert surviving child as a new persistent record
        mark symbolic target-key matches as terminal candidates
    update SARSA during training
reconstruct and independently certify every returned witness from the DAG
```

## Soundness, Resource Simulation, and Search Guarantees

The first property is preservation of the symbolic semantic invariant.

**Theorem — exact preservation of the Clifford-frame/rotation representation.**  
Let \(v\) be any resource-feasible search record obtained from the root by native gates from \(\mathcal G_n\). If the root satisfies (38) and each Clifford and \(T/T^\dagger\) append operation is updated according to (41)–(44), then

\[
\operatorname{Sem}(v)\sim U(C_v).
\tag{108}
\]

**Proof.** The root is immediate from (45). Assume the claim holds for \(v\). If the next gate \(G\) is Clifford, left multiplication gives

\[
U(C_v\oplus G)
=
G\,U(C_v)
\sim
G\,\operatorname{Sem}(v),
\]

and (41) represents this expression exactly. If the next gate is \(T_i\) or \(T_i^\dagger\), use (21) and

\[
R_Q(\vartheta)C
=
C R_{C^\dagger Q C}(\vartheta)
\tag{109}
\]

to obtain (42)–(44). Since \(C^\dagger Q C\) is a signed Pauli, the updated record remains in the same representation class. Induction on circuit length completes the proof. \(\square\)

The theorem relies on standard Clifford conjugation and Pauli-rotation structure rather than a learned approximation. [12], [14] citeturn13academia1turn12academia3

**Proposition — sound canonical merging.**  
If every rewrite used by \(\mathcal N_{\mathrm{PR}}\) is an exact identity under \(\sim\), and the serialisation is deterministic, then

\[
\kappa(u)=\kappa(v)
\Longrightarrow
U(C_u)\sim U(C_v).
\tag{110}
\]

**Proof.** By construction, equal canonical payloads are outputs of sequences of semantics-preserving transformations from the exact semantic summaries of \(u\) and \(v\). Both therefore represent the same canonical transformation up to global phase. The semantic-invariant theorem transfers this equality to the authoritative witnesses. \(\square\)

The statement is intentionally one-directional. Completeness of \(\mathcal N_{\mathrm{PR}}\) as a decision procedure for equality of arbitrary ordered Pauli-rotation words is neither assumed nor required. This conservatism is important because anticommuting Pauli rotations cannot generally be reordered, even though commuting rotations can. [14] citeturn12academia3

The central pruning argument is also one-directional. Define the set of suffixes feasible from \(v\) under resource budget \(\mathbf B\) by

\[
\boxed{
\mathcal L_{\mathbf B}(v)
=
\left\{
K\in\mathcal G_n^\ast:
\boldsymbol\rho(v\oplus K)\preceq\mathbf B
\right\}.
}
\tag{111}
\]

There is no requirement that two semantically identical records have equal sets \(\mathcal L_{\mathbf B}\). Resource consumption can make one set a strict subset of another.

**Lemma — monotonicity under a common continuation.**  
Let \(u,v\) satisfy

\[
\boldsymbol\rho(u)\preceq\boldsymbol\rho(v).
\tag{112}
\]

For any gate \(g\) that is syntactically native to the all-to-all baseline,

\[
\boldsymbol\rho(u\oplus g)
\preceq
\boldsymbol\rho(v\oplus g).
\tag{113}
\]

Consequently, for every finite common suffix \(K\),

\[
\boxed{
\boldsymbol\rho(u\oplus K)
\preceq
\boldsymbol\rho(v\oplus K).
}
\tag{114}
\]

**Proof.** The \(T\)-count, CNOT-count, and total-gate-count coordinates receive identical non-negative increments when the same gate is appended. The per-wire depth transition is monotone by (62). The one-gate implication therefore holds in every resource coordinate. Induction over the gates of \(K\) proves (114). \(\square\)

**Theorem — one-sided resource simulation and sound Pareto pruning.**  
Suppose

\[
\kappa(u)=\kappa(v)
\tag{115}
\]

and

\[
\boldsymbol\rho(u)\preceq\boldsymbol\rho(v).
\tag{116}
\]

Then

\[
\boxed{
\mathcal L_{\mathbf B}(v)
\subseteq
\mathcal L_{\mathbf B}(u).
}
\tag{117}
\]

Moreover, for every suffix

\[
K\in\mathcal L_{\mathbf B}(v),
\]

if

\[
U(C_v\oplus K)\sim U_\star,
\tag{118}
\]

then

\[
U(C_u\oplus K)\sim U_\star
\tag{119}
\]

and

\[
\boldsymbol\rho(u\oplus K)
\preceq
\boldsymbol\rho(v\oplus K).
\tag{120}
\]

Therefore \(v\) can be discarded without removing a resource-superior solution.

**Proof.** Let \(K\in\mathcal L_{\mathbf B}(v)\). By definition,

\[
\boldsymbol\rho(v\oplus K)\preceq\mathbf B.
\tag{121}
\]

From (116) and Lemma (114),

\[
\boldsymbol\rho(u\oplus K)
\preceq
\boldsymbol\rho(v\oplus K)
\preceq
\mathbf B.
\tag{122}
\]

Thus \(K\in\mathcal L_{\mathbf B}(u)\), establishing (117).

By canonical soundness,

\[
U(C_u)\sim U(C_v).
\tag{123}
\]

Hence there exists \(\phi\) with

\[
U(C_u)=e^{i\phi}U(C_v).
\]

For any common suffix \(K=(h_1,\ldots,h_\ell)\), let

\[
U(K)=U_{h_\ell}\cdots U_{h_1}.
\]

Then

\[
U(C_u\oplus K)
=
U(K)U(C_u)
=
e^{i\phi}U(K)U(C_v)
=
e^{i\phi}U(C_v\oplus K).
\tag{124}
\]

Therefore (118) implies (119), while (120) follows from (114). \(\square\)

The important logical relation is consequently

\[
\boxed{
\mathcal L_{\mathbf B}(v)
\subseteq
\mathcal L_{\mathbf B}(u),
}
\]

not

\[
\mathcal L_{\mathbf B}(v)
=
\mathcal L_{\mathbf B}(u).
\tag{125}
\]

A lower-resource representative can have **more** feasible continuations than the record it replaces. Requiring symmetric equality would be unnecessarily strong for the bounded, ancilla-free, all-to-all problem and would fail to capture the actual reason that dominance pruning is safe.

This one-sided simulation rule is especially clean because the frozen baseline contains no state-dependent connectivity or ancilla availability. Every syntactically native gate is available from every semantic state; only resource feasibility can remove a continuation. Extensions in which future legality depends on placement, liveness, hardware state, or classical control would require those variables to re-enter the formal state and would need a correspondingly stronger simulation relation. They are deliberately excluded from this paper.

The learned component is also separated from circuit correctness.

**Theorem — policy-independent terminal correctness.**  
Let

\[
\operatorname{Cert}(D_v,U_\star)\in\{0,1\}
\]

be a sound independent certification procedure satisfying

\[
\operatorname{Cert}(D_v,U_\star)=1
\Longrightarrow
U(D_v)\sim U_\star.
\tag{126}
\]

If the synthesiser returns a circuit only when \(\operatorname{Cert}=1\), then every returned circuit is correct independently of

\[
\boldsymbol\theta,
\quad
\varepsilon,
\quad
\alpha,
\quad
R,
\quad
\boldsymbol\varphi,
\]

or the scheduler used to select frontier records.

**Proof.** The scheduler can only select an element of \(\mathcal F_t\). It cannot alter the DAG associated with an already generated child, the exhaustive gate generator, or the terminal predicate. A returned witness therefore satisfies (126) regardless of the search ordering that caused it to be discovered. \(\square\)

For the small-\(n\) baseline, certification is deliberately dense. Let

\[
U=U(D_v),\qquad V=U_\star,
\qquad d=2^n.
\]

A phase-invariant Frobenius discrepancy is

\[
\Delta_\phi(U,V)
=
\min_{\phi}
\frac{
\|U-e^{i\phi}V\|_F
}{
\sqrt{2d}
}.
\tag{127}
\]

Since

\[
\|U-e^{i\phi}V\|_F^2
=
2d
-
2\operatorname{Re}
\left(
e^{-i\phi}
\operatorname{Tr}(V^\dagger U)
\right),
\tag{128}
\]

the optimal phase gives

\[
\boxed{
\Delta_\phi(U,V)
=
\sqrt{
1-
\frac{
|\operatorname{Tr}(V^\dagger U)|
}{d}
}.
}
\tag{129}
\]

In exact arithmetic,

\[
\Delta_\phi(U,V)=0
\iff
U\sim V.
\tag{130}
\]

The implementation uses a fixed numerical tolerance only to accommodate floating-point evaluation. Crucially, the unitary supplied to this test is reconstructed independently from the DAG witness rather than accepted from a canonical key or policy feature cache. Dense verification scales exponentially in \(n\), which is precisely why it is appropriate here only after the first paper has explicitly frozen the problem at two and three qubits. More scalable equivalence-checking methods exist and remain natural later extensions. [18] citeturn15academia12

The finite resource bound also simplifies the global search guarantee. Let \(B_g\) be the gate-count coordinate of \(\mathbf B\). The number of raw syntactic prefixes is at most

\[
N_{\mathrm{raw}}
\le
\sum_{\ell=0}^{B_g}
|\mathcal G_n|^\ell
=
\begin{cases}
B_g+1,&|\mathcal G_n|=1,\\[2mm]
\dfrac{
|\mathcal G_n|^{B_g+1}-1
}{
|\mathcal G_n|-1
},
&|\mathcal G_n|>1.
\end{cases}
\tag{131}
\]

The bound is exponential in \(B_g\), not a claim of polynomial synthesis complexity.

**Theorem — scheduler-independent exhaustion completeness for the frozen bounded domain.**  
Assume that every selected record is removed from the unexpanded frontier after expansion; every resource-feasible native child is generated; canonical merging satisfies (47); Pareto pruning satisfies the one-sided simulation theorem; and dense goal certification is complete within its numerical exactness convention. If the search is allowed to continue until the finite frontier is exhausted, then any solution satisfying the fixed resource bound is eventually discovered, irrespective of how the scheduler orders the remaining frontier records.

**Proof sketch.** The raw bounded search contains only finitely many prefixes by (131). Canonicalisation and Pareto pruning can only decrease the number retained, and reopening can occur only for records derived from this finite raw set. Because every selected record is permanently marked expanded and every iteration selects one unexpanded accepted record, a scheduler cannot indefinitely postpone one record while expanding infinitely many others: there are only finitely many accepted records. Consider a resource-feasible raw solution path. If none of its prefixes is pruned, exhaustive expansion eventually reaches its terminal record. If a prefix is removed through Pareto dominance, the one-sided simulation theorem supplies a semantically identical no-worse representative from which the remaining suffix remains feasible. Repeating this replacement argument preserves a feasible solution path in the retained archive. Frontier exhaustion must therefore generate a certifiable representative. \(\square\)

This theorem is intentionally different from a claim about experiments performed under a finite external expansion cap. When

\[
B_{\mathrm{exp}}
<
N_{\mathrm{accepted}},
\]

a scheduler can fail simply because the budget expires before it reaches a useful record. The correct empirical quantity is therefore the success probability as a function of \(B_{\mathrm{exp}}\), not an unconditional completeness claim.

The cost of an RL decision also matters. Let

\[
F_t=|\mathcal F_t|,
\qquad
d_\varphi=\dim\boldsymbol\varphi.
\]

A linear ranking pass costs

\[
O(F_t d_\varphi).
\tag{132}
\]

If

\[
C_{\mathrm{upd}},
\quad
C_{\mathrm{can}},
\quad
C_{\mathrm{arch}},
\quad
C_{\mathrm{cert}}
\]

denote one-child symbolic update, canonicalisation, archive, and certification costs, an expansion has approximate cost

\[
\boxed{
O\!\left(
F_t d_\varphi
+
|\mathcal G_n|
\left(
C_{\mathrm{upd}}
+
C_{\mathrm{can}}
+
C_{\mathrm{arch}}
\right)
+
N_{\mathrm{cert},t}C_{\mathrm{cert}}
\right).
}
\tag{133}
\]

This expression already shows why expansion count and wall-clock time must both be measured: ranking a large frontier can cost more than the expansion saved by a better policy.

If the ordered rotation word has

\[
m_R=|\mathcal R_v|
\]

entries, its packed binary Pauli axes require \(O(nm_R)\) bit storage up to fixed metadata. A fully materialised dependency graph has at most

\[
\binom{m_R}{2}
\]

edges and therefore \(O(m_R^2)\) worst-case edge storage. A direct symplectic commutation test costs \(O(n)\) bit operations, so naïvely constructing every pairwise dependency costs

\[
O(nm_R^2).
\tag{134}
\]

Incremental maintenance avoids rebuilding this graph after every append.

Dense unitary storage requires

\[
O(d^2)=O(4^n)
\tag{135}
\]

complex numbers, while conventional dense matrix products can require

\[
O(d^3)=O(8^n)
\tag{136}
\]

arithmetic operations. These costs are unacceptable as a general scalable verification strategy but modest for the deliberately frozen \(n=2,3\) experimental domain. The restricted baseline therefore does not conceal this exponential component behind a claim of future scalability.

## Experimental Results and Reproducibility

The implementation branch used for the reported qualification is `hybrid-frontier-online-rl-30min` of `ssp2195/RL_for_Quantum_Circuits`, package version 0.3.0. The scientific scope recorded by the source manifest is the persistent dependency DAG, complete forward/inverse Clifford tableau, ordered signed Pauli rotations, explicit global phase and monotone resources, persistent frontier-record selection as the RL action, the unrestricted native grammar \(`H,S,SDG,T,TDG,CNOT`\), and independent dense certification by replaying the DAG up to global phase.

### Qualification protocol

The bounded campaign uses five independent learner seeds,

\[
\mathcal S=\{11,19,23,31,47\},
\tag{137}
\]

with 160 online-SARSA episodes per seed, giving

\[
N_{\mathrm{train}}=5\times160=800
\tag{138}
\]

training episodes. The command-line qualification uses a hard deadline of 1,800 s, a training expansion cap of 2,048, an unrestricted evaluation expansion cap of 8,192, and the same 8,192 cap for the structured Toffoli stress test. The runner returns success only when all declared training/evaluation jobs complete inside the hard deadline and every returned circuit passes independent dense certification.

### Unrestricted held-out targets and exact QFT-2

The unrestricted evaluation uses the same native gate grammar used by search; search-facing targets do not expose their generator witnesses. In addition to the ordinary held-out target set, the branch includes a conventional forward two-qubit QFT target

\[
(F_4)_{jk}=\frac12 e^{2\pi i jk/4},\qquad j,k\in\{0,1,2,3\},
\tag{139}
\]

including the final SWAP represented through its exact three-CNOT decomposition. Because this particular two-qubit target is exactly representable in the frozen grammar, it can be used as an unrestricted exact-synthesis benchmark rather than as an approximate-QFT task.

Across the clean-source five-seed qualification, all

\[
35/35
\tag{140}
\]

unrestricted frozen-policy evaluations returned independently certified circuits, and all

\[
5/5
\tag{141}
\]

held-out QFT-2 evaluations were certified. These counts establish correctness and completion for the bounded qualification; they do not by themselves establish superiority over every deterministic scheduler.

### Structured Toffoli stress test

The Toffoli benchmark is intentionally separated from unrestricted native discovery. The implementation ranks complete hybrid frontier records inside an exact seven-term CCZ parity-network normal form and surrounds the CCZ core by deterministic Hadamards on the target qubit. Every accepted result is reconstructed from the persistent DAG and compared with the analytical CCX unitary. The branch reports

\[
5/5
\tag{142}
\]

certified structured-Toffoli evaluations, each returning a 15-gate witness. The scientifically correct interpretation is therefore that frontier ranking successfully schedules search inside the certified parity-network formulation; it is not evidence of unrestricted native Toffoli discovery.

A compact circuit-level representation of this benchmark can be reproduced in editable LaTeX with Quantikz as follows:

```latex
\begin{quantikz}
\lstick{$q_0$} & \ctrl{2} & \qw \\
\lstick{$q_1$} & \ctrl{1} & \qw \\
\lstick{$q_2$} & \gate{H} & \targ{} & \gate{H} & \qw
\end{quantikz}
```

The actual decomposition evaluated by the code is the exact Clifford+\(T\) parity-network witness generated by the structured benchmark rather than the three-symbol logical diagram above.

### Aggregate qualification outcome

The branch-level qualification record is:

| Quantity | Measured outcome |
|---|---:|
| Independent training seeds | 5 |
| Episodes per seed | 160 |
| Certified online-SARSA training episodes | 800/800 |
| Certified unrestricted frozen-policy evaluations | 35/35 |
| Certified exact QFT-2 evaluations | 5/5 |
| Certified structured-Toffoli evaluations | 5/5 |
| Regression tests | 15 passed |
| Hard campaign deadline | 1,800 s |
| Recorded process-CPU time | 22.703 s |

The process-CPU value is machine- and environment-dependent engineering evidence. It is useful for demonstrating that the compact persistent-state implementation is operationally lightweight in the qualified environment, but it must not be extrapolated as a hardware-independent runtime guarantee or scaling law.

### Research-quality plots and reproducible figure generation

The implementation records per-episode expansions, rewards, TD errors, success/certification flags, and CPU/wall times, as well as search-level frontier and transition profiling. Publication plots should be generated from the runner's emitted records rather than from manually transcribed values. A vector-friendly PGFPlots template for a multi-seed learning curve is:

```latex
\begin{tikzpicture}
\begin{axis}[
  width=0.92\linewidth,
  xlabel={Training episode},
  ylabel={Expansions to termination},
  grid=major,
  legend pos=north east]
\addplot table [x=episode,y=expansions,col sep=comma] {seed11.csv};
\addlegendentry{seed 11}
% Repeat for seeds 19, 23, 31, and 47.
\end{axis}
\end{tikzpicture}
```

For raster export from Matplotlib, the reproducibility script should save the same plotted data as PDF/SVG and, when a raster copy is required by a venue, use `savefig(..., dpi=1200, bbox_inches="tight")`. The editable vector source should be retained with the paper artifact.

### What the qualification establishes

The completed qualification supports four concrete claims. First, the persistent DAG, symbolic semantics, Pareto archive, online SARSA ranker, and dense certifier operate together without loss of witness reconstructability. Second, the target-independent expansion engine successfully handles unrestricted native held-out targets and the exact QFT-2 instance under the stated finite budgets. Third, the structured Toffoli normal form is certified end-to-end. Fourth, the optimized implementation is sufficiently lightweight to complete the declared five-seed campaign far inside its hard deadline in the recorded environment.

The qualification does not establish asymptotic quantum-synthesis scalability, global circuit optimality, or universal superiority of learned scheduling over every deterministic heuristic. Those questions require larger controlled comparative datasets and are separate from correctness qualification.

## Conclusion

This work formulates reinforcement learning as a search-scheduling mechanism for exact small-scale Clifford+\(T\) synthesis rather than as a gate generator. The search state combines a persistent lossless dependency-DAG witness, a complete Clifford tableau, an ordered signed-Pauli-rotation word, explicit phase bookkeeping, and monotone resources. The RL action selects one active frontier record; deterministic code then attempts every target-independent native gate, applies resource and partial-order reductions, updates exact symbolic semantics, performs canonical/Pareto pruning, and independently certifies any returned witness by replaying the DAG.

The implementation choices are important to the practicality of this separation. Structural sharing allocates only one new witness step per accepted transition instead of copying complete circuits, per-record features are computed once and cached, the active frontier is snapshotted without repeated sorting, and all linear action values are evaluated by a single matrix-vector multiplication. These optimizations reduce engineering overhead without changing the mathematical search space defined by the deterministic engine.

The clean-source qualification completed 800/800 certified training episodes over five independent seeds, 35/35 certified unrestricted frozen-policy evaluations, 5/5 exact QFT-2 evaluations, and 5/5 structured-Toffoli stress tests, with all 15 regression tests passing. The recorded 22.703 process-CPU seconds demonstrate that the current small-scale implementation can execute the complete bounded protocol efficiently in its qualification environment; the value is not presented as an architecture-independent performance guarantee.

Two extensions are especially important. The first is ancilla-assisted synthesis. Once clean, dirty, borrowed, or measurement-conditioned ancillas are admitted, the state must encode ancilla liveness, initialization assumptions, restoration obligations, and resource ownership. A dominance relation based only on logical-unitary equality and gate/depth resources is then insufficient; safe pruning must also preserve the ancilla interface of all feasible continuations.

The second extension is to expose richer circuit-DAG information directly to the ranking policy. The present policy deliberately does not consume adjacency or dependency-graph embeddings. Future work can derive low-cost DAG features such as critical-path slack, per-wire dependency depth, recent interaction structure, gate-type histograms by layer, frontier width, or compact message-passing embeddings. Such features should be added only when their improvement in search allocation justifies their scoring cost. The persistent DAG already provides the exact structural substrate needed for this extension without requiring a duplicated graph per frontier record.

## Limitations and Future Scope

The first limitation is intrinsic to exact search. Learning changes the **order** in which a finite search space is explored; it does not remove the exponential worst-case growth of that space. The bound in (131) remains exponential in the allowed gate count. Exact-synthesis literature repeatedly relies on specialised algebraic, SAT, meet-in-the-middle, or normal-form structure to push tractable problem sizes beyond direct enumeration. [1]–[3] citeturn14academia1turn13academia0turn14academia3 A reduction in expansions on the present target distribution must therefore not be interpreted as a new polynomial-time exact-synthesis algorithm.

The second limitation is the deliberately small system size. Dense target features and dense independent certification exploit

\[
d=2^n
\]

matrices and therefore become exponentially expensive. Their use is defensible because the baseline is explicitly frozen at primarily two and three qubits; it is not a scalable verification architecture. Quantum-circuit equivalence checking based on reversibility and more specialised symbolic methods provides a natural later replacement once verification rather than frontier ranking becomes the limiting factor. [18] citeturn15academia12

The third limitation is canonicalisation completeness. The canonicaliser is designed to satisfy the sound implication

\[
\kappa(u)=\kappa(v)
\Longrightarrow
U(C_u)\sim U(C_v),
\]

but it does not claim

\[
U(C_u)\sim U(C_v)
\Longrightarrow
\kappa(u)=\kappa(v).
\]

In particular, commuting reordering, equal-axis fusion, inverse cancellation, and Clifford absorption do not form a complete equivalence decision procedure for arbitrary noncommuting Pauli-rotation words. The consequence is additional search work, not incorrect pruning. Pauli-rotation optimisation literature supports these algebraic transformations while also making clear that commutation structure constrains admissible reorderings. [14] citeturn12academia3

The fourth limitation is Pareto-archive growth. Even with only gate counts and a per-wire depth vector, a canonical identity can admit several mutually non-dominating resource states. Thus

\[
|\mathcal P_t(k)|
\]

need not be one and may increase with search depth. Retaining these records is mathematically preferable to an unsound scalar collapse, but it can increase both memory consumption and ranking cost.

The fifth limitation concerns the linear policy. The state observation compresses the complete archive into candidate features and symmetric frontier statistics; distinct archives can therefore induce similar observations. A linear \(Q_{\boldsymbol\theta}\) may also fail to represent important nonlinear relationships between target discrepancy, resource state, and rotation dependencies. This limitation is intentional in the first publication. Recent work demonstrates much higher-capacity graph and equivariant models for quantum synthesis and optimisation, including graph-based ZX policies and size-agnostic Clifford policies. [7], [9] citeturn16academia13turn14academia0 Introducing a neural ranker before establishing whether frontier ranking itself provides value would confound architectural capacity with the central scheduling hypothesis.

The sixth limitation is that the dense target-relative feature gives the learner information that itself costs computation. An expansion reduction is therefore not sufficient evidence of a practical speed-up. The explicit measurement of \(T_{\mathrm{ranking}}\), dense target evaluations, cache behaviour, and total wall-clock time is necessary to establish whether improved ordering outweighs its computational overhead.

The seventh limitation concerns optimisation claims. The method is an exact **correctness** synthesiser inside a finite resource domain, but a first solution returned by an arbitrary frontier scheduler is not automatically globally \(T\)-optimal, CNOT-optimal, gate-optimal, or depth-optimal. Exact optimisation methods in the Clifford literature establish optimality under much more specialised search or SAT constructions. [1], [2] citeturn14academia1turn13academia0 The present Pareto archive prevents locally dominated semantic duplicates from consuming search effort; it does not by itself turn early stopping into a proof of global circuit optimality.

The eighth limitation is that the structured Toffoli experiment and unrestricted native-target experiment answer different questions. The former measures learned ordering within a parity-network representation whose search structure already encodes strong knowledge of Toffoli. The latter tests the generic native Clifford+\(T\) frontier. Their measurements should therefore remain separate rather than being pooled as though they sampled one homogeneous synthesis distribution.

The broader Pauli-rotation research programme remains the ultimate direction, but it is intentionally deferred until the exact baseline has been empirically characterised. The first extension is state-dependent hardware connectivity. A future record could augment (12) with a placement variable

\[
\pi_v:
\mathcal Q_L\rightarrow\mathcal Q_P
\]

and define native CNOT legality through a coupling graph. Contemporary RL synthesis and Pauli-network methods already demonstrate the relevance of topology-aware learned compilation, but such an extension changes the continuation theorem because future gate legality would then depend on state as well as resource budget. [6], [10] citeturn16academia12turn12academia0

Ancilla-assisted synthesis is a separate extension for the same reason. Once an ancilla can be allocated, borrowed, dirtied, restored, or released, a record must contain a liveness/restoration interface such as

\[
\boldsymbol\ell_v\in\{0,1\}^{m}
\]

and semantic equality of logical unitaries alone is not enough to justify pruning. Exact Clifford+\(T\) theory shows that ancillas can change representability itself, so incorporating them is not a cosmetic implementation feature. [4] citeturn14academia2 The present no-ancilla restriction avoids introducing this separate correctness problem into the first experimental paper.

Approximate synthesis and arbitrary-angle targets form another distinct programme. Their terminal relation would replace (2) with a tolerance criterion such as

\[
d_{\mathrm{tar}}(C)
\le\epsilon,
\tag{155}
\]

and resource-efficiency claims would have to be reported jointly with approximation error. Quantum Fourier transform targets containing non-native controlled rotations naturally belong to this extension rather than to the exact finite-grammar baseline.

Scalable equivalence checking can subsequently replace dense certification while preserving the architectural rule

\[
\boxed{
\text{the policy proposes search priority; an independent verifier decides semantic acceptance}.
}
\tag{156}
\]

Existing circuit-equivalence research provides several routes for this transition. [18] citeturn15academia12 The crucial requirement is that the verifier remain outside the learned ranking model.

A further extension concerns direct DAG-derived policy features. In the current implementation, the persistent dependency DAG is authoritative for reconstruction and certification but the linear ranker receives only resource, symbolic-mismatch, rotation-structure, last-gate, target, and register-size features. Future policies can add compact dependency summaries or graph encodings without changing the action definition. Candidate additions include normalized critical-path length, wire-specific slack, interaction recency, layerwise gate histograms, local ancestor counts, and eventually permutation-equivariant message-passing embeddings. The research criterion should remain compute-normalized: a richer DAG encoder is useful only if the reduction in expansions exceeds its feature-extraction and scoring overhead.

Finally, once the controlled linear-SARSA baseline has established the empirical value—or lack of value—of learned frontier ordering, the shared scorer in (80) can be replaced by a permutation-equivariant set model or graph model without altering the action definition

\[
a_t\in\mathcal F_t.
\tag{157}
\]

Recent equivariant Clifford synthesis demonstrates that learned policies can generalise across qubit relabellings and unseen sizes when their architectures respect algebraic symmetries, while current learned-search work also shows the viability of combining a learned cost estimate with an explicit search algorithm. [9], [10] citeturn14academia0turn16academia14 These directions are natural successors to the present baseline, not prerequisites for validating it.

The ultimate programme is therefore deliberately staged:

\[
\boxed{
\begin{aligned}
\text{present work:}\quad&
\text{exact small-scale frontier ranking}
\\
&\text{with deterministic symbolic semantics and dense certification},
\\[1mm]
\text{later work:}\quad&
\text{ancillas, topology, scalable verification,}
\\
&\text{approximate targets, larger corpora, and richer learned rankers}.
\end{aligned}
}
\tag{158}
\]

Freezing the first line is methodologically important. It makes it possible to determine whether reinforcement learning contributes evidence of improved **search scheduling** before the research programme is enlarged into a system in which changes in representation, legality, verification, hardware constraints, and policy capacity can no longer be experimentally disentangled.

### Implementation provenance

The implementation-specific statements and qualification counts in this version are grounded in the repository branch `hybrid-frontier-online-rl-30min`, particularly `hybrid_qcs/model.py`, `hybrid_qcs/search.py`, `hybrid_qcs/rl.py`, `README.md`, and `SOURCE_MANIFEST.json` at the qualified branch head.

## References

[1] S. Schneider, L. Burgholzer, and R. Wille, “A SAT Encoding for Optimal Clifford Circuit Synthesis,” *arXiv:2208.11713*, 2022. The work presents SAT-based optimal Clifford synthesis and reports synthesis for instances up to 26 qubits. citeturn14academia1

[2] I. Shaik and J. van de Pol, “CNOT-Optimal Clifford Synthesis as SAT,” *arXiv:2504.00634*, 2025. The work develops SAT encodings for CNOT-count and CNOT-depth optimisation and discusses the scaling limit of exhaustive Clifford normal-form search. citeturn13academia0

[3] A. N. Glaudell, M. Jarret, S. Klein, S. S. Mendelson, T. C. Mooney, M. Tian, *et al.*, “High-Performance Exact Synthesis of Two-Qubit Quantum Circuits,” *arXiv:2601.19166*, 2026. citeturn14academia3

[4] B. Giles and P. Selinger, “Exact Synthesis of Multiqubit Clifford+\(T\) Circuits,” *arXiv:1212.0506*, 2012. The paper characterises exact Clifford+\(T\) representability with local ancillas and also treats the no-ancilla case. citeturn14academia2

[5] M. Kölle, T. Schubert, P. Altmann, M. Zorn, J. Stein, and C. Linnhoff-Popien, “A Reinforcement Learning Environment for Directed Quantum Circuit Synthesis,” *arXiv:2401.07054*, 2024. citeturn12academia2

[6] D. Kremer, V. Villar, H. Paik, I. Duran, I. Faro, and J. Cruz-Benito, “Practical and Efficient Quantum Circuit Synthesis and Transpiling with Reinforcement Learning,” *arXiv:2405.13196*, 2024. citeturn16academia12

[7] J. Riu, J. Nogué, G. Vilaplana, A. Garcia-Saez, and M. P. Estarellas, “Reinforcement Learning Based Quantum Circuit Optimization via ZX-Calculus,” *arXiv:2312.11597*, 2023. citeturn16academia13

[8] A. Dubal, D. Kremer, S. Martiel, V. Villar, D. Wang, and J. Cruz-Benito, “Pauli Network Circuit Synthesis with Reinforcement Learning,” *arXiv:2503.14448*, 2025. citeturn12academia0

[9] R. Yeung, A. Kissinger, and R. Cornish, “Equivariant Reinforcement Learning for Clifford Quantum Circuit Synthesis,” *arXiv:2605.10910*, 2026. citeturn14academia0

[10] L. Theissinger, T. Gerlach, D. Berghaus, and C. Bauckhage, “Beyond Reinforcement Learning: Fast and Scalable Quantum Circuit Synthesis,” *arXiv:2602.15146*, 2026. The method combines a learned residual-description estimate with stochastic beam search. citeturn16academia14

[11] A. Mattick and C. Mutschler, “Reinforcement Learning for Node Selection in Branch-and-Bound,” *arXiv:2310.00112*, 2023. citeturn12academia1

[12] S. Aaronson and D. Gottesman, “Improved Simulation of Stabilizer Circuits,” *arXiv:quant-ph/0406196*, 2004. citeturn13academia1

[13] S. Bravyi and D. Maslov, “Hadamard-Free Circuits Expose the Structure of the Clifford Group,” *arXiv:2003.09412*, 2020. citeturn13academia2

[14] F. Zhang and J. Chen, “Optimizing \(T\) Gates in Clifford+\(T\) Circuit as \(\pi/4\) Rotations around Paulis,” *arXiv:1903.12456*, 2019. citeturn12academia3

[15] N. Vandaele, S. Perdrix, C. Vuillot, and collaborators, “Optimal Hadamard Gate Count for Clifford+\(T\) Synthesis of Pauli Rotations Sequences,” *arXiv:2302.07040*, 2023.

[16] IBM Quantum, “Work with DAGs in Transpiler Passes,” Qiskit documentation, current documentation consulted August 2026. Qiskit represents circuits as `DAGCircuit` objects during transpilation. citeturn15search0

[17] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018. citeturn16search0

[18] L. Burgholzer and R. Wille, “Advanced Equivalence Checking for Quantum Circuits,” *arXiv:2004.08420*, 2020. citeturn15academia12

[19] Z. T. Wang, Q. Chen, Y. Du, Z. H. Yang, X. Cai, K. Huang, *et al.*, “Quantum Compiling with Reinforcement Learning on a Superconducting Processor,” *arXiv:2406.12195*, 2024. citeturn16academia15