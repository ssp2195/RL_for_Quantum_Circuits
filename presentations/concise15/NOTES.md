# Quantum phase oracles with fewer clean ancillas: notes for the 15-slide talk

## Structure

There are exactly 15 slides, with no cover-only slide, overlays, hidden frames, or appendix slides.

1--7: the phase-oracle problem; amplitude amplification and BNN verification as motivation; a coherent frontier record; Clifford tableau, Pauli rotations and the dependency graph; reinforcement learning; SARSA; LinUCB.

8--11: measured cleanup; the restricted full-bank workspace theorem; direct consumer factorization; exact fixed-feature lifetime ordering.

12--15: the benchmark contract; all six resource coordinates on the five development functions; the 36 additional-function comparisons; conclusion and the present role of learning.

## Source boundary

This is a new presentation of existing mathematics, code, and archived results. It does not claim a new synthesis experiment, a new trained policy, a new proof of publication priority, or a new global optimum. The source-code snapshot inspected is commit `2671f6984b87f1d89613ba81d8b0e99798cd3f58` on `consumer-aware-cleanup-20260928`.

The primary numerical source is the supplied `Consumer_Aware_Cleanup_Slides.pdf`, pages 31--44, which derives its tables from the archived 41-target comparison. The five development vectors are on page 34. The 36-target counts are on pages 39--44. These data are copied into `data/development.csv` and `data/additional_counts.csv` in the delivered project, with explicit current/control labels. The build checks their row totals. External optimizers were not rerun for this presentation.

### Short source keys used in the slides

- **[NC]** Michael A. Nielsen and Isaac L. Chuang, *Quantum Computation and Quantum Information*, Cambridge University Press, 10th Anniversary Edition, 2010; Chapter 6 for amplitude amplification, and the standard Pauli/Clifford and measurement formalism elsewhere in the book. The prose is newly written and is not a quotation from the textbook.
- **[SB]** Richard S. Sutton and Andrew G. Barto, *Reinforcement Learning: An Introduction*, second edition, MIT Press, 2018; SARSA and semi-gradient function approximation. The normalization and clipping in slide 6 are specific to the repository, not claimed as a textbook convergence result.
- **[G]** Craig Gidney, *Halving the cost of quantum addition*, Quantum 2, 74 (2018), DOI 10.22331/q-2018-06-18-74; arXiv:1709.06648. The four-T clean-target AND and measured termination are credited prior ingredients.
- **[AR]** Matthew Amy and Neil J. Ross, *The phase/state duality in reversible circuit design*, arXiv:2105.13410, especially measurement-assisted termination.
- **[MOS]** Maria Garcia de la Banda and Peter J. Stuckey, *Dynamic Programming to Minimize the Maximum Number of Open Stacks*, INFORMS Journal on Computing 19(4), 607--617 (2007), DOI 10.1287/ijoc.1060.0205. The fixed-feature ordering is an application of this established classical problem.
- **[H]** Supplied research note, `H - Prove Cleanup Theorem`, final common-enable outer-product theorem and lifecycle-accounted materialization theorem, PDF pages 53--60. This is the user's research theorem, not an assertion of external peer review or proof-assistant verification.
- **[N]** Repository `hybrid_qcs/model.py`, `rotation.py`, `tableau.py`, `clifford_lift.py`, `native_search.py`, and `native_policy.py` at the scientific snapshot above.
- **[L]** Repository `hybrid_qcs/cleanup/search.py`, `policy.py`, and `ir.py`. These give the protocol frontier, the linear policies, and the local coherent witnesses.
- **[C]** Repository `hybrid_qcs/cleanup/consumer.py`, `consumer_compile.py`, and `consumer_search.py`. The last file uses `include_learned=False` by default; the optional trained candidate searches full-bank schedules, not Boolean factorization or the exact subset recurrence.
- **[D]** Supplied 69-page presentation and its archived proof/evidence project, with primary experimental statements on pages 31--44 and qualifications on pages 46--50. The externally optimized control uses the authors' mockturtle algorithms followed by the project's explicitly described measured-oracle conversion. It is not an execution of every workspace-optimal external compilation algorithm.

## 1. Phase oracle, amplitude amplification, and verification

Let p(z) be one exactly specified Boolean predicate on a finite input domain. A phase oracle applies (-1)^p(z) to each basis state while leaving z unchanged. Its coherent implementation must work on arbitrary input amplitudes, not only uniformly distributed or classical inputs.

For a BNN application, p(z)=1 is the event that a particular output property is violated. The present benchmarks are synthetic structured predicates; they do not demonstrate an end-to-end verifier for a complete trained network.

For uniform initial state |psi> and M violating inputs among N=2^n inputs, write |psi>=cos(theta)|safe>+sin(theta)|violating>, with sin^2(theta)=M/N. In this two-dimensional basis, O_p=diag(1,-1). Multiplying the reflection 2|psi><psi|-I by O_p gives the real rotation matrix [[cos(2theta),-sin(2theta)],[sin(2theta),cos(2theta)]]. Its jth power therefore produces violating probability sin^2((2j+1)theta). The decomposition into two normalized component states assumes 0<M<N; the endpoint probabilities M=0 and M=N are handled directly. An unsuccessful finite run is not an exact proof of no violating input. Oracle resource counts omit state preparation and the other reflection.

## 2. What is represented and what is learned

A coherent circuit prefix is stored as (G,Theta,R,phi,rho). Its unitary has a Clifford frame on the left and a right-acting ordered product of signed Pauli rotations, with a separate exact scalar phase. A tableau records the signed images of the Pauli generators X_j and Z_j. The circuit dependency graph preserves the actual gate witness and per-wire ordering. It is not fed as a full graph into a graph neural network; small derived features are used.

The outer RL action is a frontier record. The full classical search situation S contains more information than one prefix: pending alternatives, recorded representatives, resource limits and remaining search budget. The inner choice is an eligible native gate in native search, or a certified protocol operation in cleanup search. Neither learner chooses measurement outcomes.

For the cleanup policy, outer coefficients are updated using the actual next selected frontier context, discount gamma=1, normalized semi-gradient steps, and componentwise clipping to [-20,20]. The slide gives this implemented update. All parameters are held fixed in testing. Training completion is not convergence.

The inner disjoint linear models fit corrections to a public prior. For a family f, A_f=I+sum zz^T, b_f=sum z(y-z^T beta_0), and fitted coefficients are beta_0+A_f^{-1}b_f. The square-root term is used for exploration during training, not to certify a completion or prune a mathematically feasible circuit. The feedback depends on the evolving search; no stationary-bandit regret guarantee is asserted.

In the current consumer-aware method, candidate full-bank, bank-free, and rank-factor constructions are generated analytically and compared by exact resource vectors. A learned full-bank schedule is optional. The selected deterministic circuits in the external comparison are not relabelled as RL discoveries.

## 3. Measurement and exact branch correctness

For any Boolean f, <s_X|f(z)>=2^{-1/2}(-1)^{s f(z)}. Thus measuring a temporary bit produces an input-dependent sign even though the outcome is uniform. A second application of that sign cancels it. The measured target is reset, and all input parities used in the correction must be restored in the correct order.

When f is quadratic in retained values, its sign is a product of Z and CZ phases. For f=uv, the correction is CZ(u,v)^s. This does not justify Clifford-only correction for arbitrary cubic functions of the remaining data. Every operand needed for correction must survive the cleanup stage.

For a protocol with N such measurements, the verified corrected branch identity is K_s J_A=2^{-N/2}J_A O_p. It proves coherent action, clean ancillas, and input-independent transcript probabilities. The complete process is an instrument, not a unitary; local coherent segments use the original hybrid representation, while the surrounding protocol records outcomes and conditional gates.

The primitive AND_0 is a clean-input isometry: |u,v,0> -> |u,v,uv>. It is not a four-T Toffoli on arbitrary target input. The repository's phase-correct local word, in chronological order on controls u,v and target t, is H(t), T(t), CNOT(v,t), T-dagger(t), CNOT(u,t), T(t), CNOT(v,t), T-dagger(t), H(t), S-dagger(t). Reset and CZ corrections are charged separately.

## 4. Scope and brief proof of the full-bank theorem

Raw inputs a,b,x range independently over their entire Boolean cubes. The product bank is F_ij=a b_i x_j and all rm product qubits must coexist. The k helper bits are computational-basis values H=a(Pb+Qx+c). They vanish when a=0 and can be erased at the end by direct X measurement followed by Clifford feedback on the raw inputs. In the main stage, garbage processing does not have arbitrary live-register access. Afterward a unitary Clifford correction preserves the promised live values. General garbage-only instruments are allowed; an arbitrary fixed diagonal live--garbage prefix is also covered by the supplied theorem. The shorter slide states the simpler no-extra-interaction case.

The upper bound stores H_j=a x_j when m<=r, or the symmetric row helpers otherwise. Products are bilinear in surviving inputs and helpers; measured cleanup of products and then helpers uses CZ feedback. All auxiliary qubits are clean at completion. Exactly rm+min(r,m) auxiliary qubits coexist at the peak.

For the converse, first consider G=(h,F), where h=ab is an extra proof register. Remove the affine helper constant with CNOTs. If k<min(r,m), choose b_0!=0 in ker P. A Clifford fixing the promised computational-basis values has affine basis permutation and a quadratic mod-four phase. Constancy on the a=0 fibre and on the zero-garbage configurations (a,b,x,H)=(1,0,x,0), x in ker Q, forces every eligible correction phase to be constant on B={(b_0,b_0 tensor x):x in ker Q}. The map x -> b_0 tensor x is injective, so |B|=2^(m-rank Q)>=2^(m-k)>1.

An exact successful refined branch is a garbage bra <v_alpha| followed by a clean reset and a Clifford on the live registers. On B its amplitudes must be proportional to the same phase profile. Testing measurement completeness on the normalized coherent sum over B gives |B| sum_alpha |gamma_alpha|^2<=1. Deterministic success is impossible.

If an F-only instrument succeeded with fewer helpers, add h only for the proof, run that instrument, and measure h in X. Its extra sign a(s dot b) is corrected by CZs on original inputs, giving the forbidden G cleanup. Thus the F-only helper threshold also follows. A fixed diagonal joint prefix cancels on dividing two successful-branch equations, leaving the same indistinguishability argument.

This is not a global lower bound for the final oracle. For example p=a(XOR_i b_i)(XOR_j x_j) can be implemented by parities and a CCZ/temporary-AND construction without materializing the bank. The lower bound also does not cover arbitrary quantum helper states, interleaved arbitrary live--garbage operations, or extra promises on raw inputs.

## 5. Consumer factorization and lifetime ordering

After substituting F, every remaining high-degree term in this supported consumer family can be written with one a factor and one nonempty degree-at-most-two feature from each of b and x. Thus p=q_0+a B^T M X. Binary rank factorization M=sum u_l v_l^T produces d=rank(M) separated factors. This d is minimal only among products of linear combinations of the fixed B and X lists.

Quadratic-form elimination writes each factor as an affine remainder plus products of independent parities. If s_f distinct nonlinear products are shared, computing each once costs 4 s_f T gates. Applying the d phase factors costs 4d more T gates. One temporary phase helper is reused, giving N_T=4(d+s_f), A_peak<=s_f+1. All parity operations and their inverses are actually included. Factor and feature measurements have corrected branch signs independent of the logical input.

For fixed feature sets S_j, a completed-factor set A forces L(A)=(union past S_i) intersect (union future S_i) to remain live when recomputation is forbidden. A step j requires all of L(A) union S_j and one phase helper. A schedule that prepares at first use and cleans at last use attains this lower bound for every fixed order. The subset recurrence minimizes the maximum over orders. This is precisely minimum-open-stacks ordering plus one helper, not a new classical dynamic program. Empty factor sequences need zero factor helpers and are handled separately. The exact implementation limit is 16 factors. Larger fixed-order results are upper bounds, not proved minima.

## 6. Experimental reading rules

The development table lists current/control vectors. The control is measured compilation of externally optimized XOR--AND networks; it uses three source forms and permits serial or parallel reverse cleanup. It is not the best possible recomputation schedule under every workspace cap.

Additional-target better/equal/worse counts are: clean ancillas 30/6/0; T-count 5/13/18; CNOT 22/5/9; native gates 14/2/20; latency 2/2/32; measurement stages 0/4/32. Each row totals 36. There are 41 distinct labelled target functions in total and 246 repeated records. Synthetic functions and repeated timings must not be called independent application problems or RL generalization evidence.

Both methods' reported selections prioritize T-count, then peak ancillas, then latency, native gates and CNOTs. A workspace-oriented interpretation must acknowledge that these are not solutions of a common workspace-first or fixed-cap optimization problem.

No claims of global ancilla/T optimality, general BNN verification speedup, hardware performance, external state-of-the-art dominance, or SARSA--LinUCB superiority follow. The useful observed result is a clean-workspace trade-off for this specified family.
