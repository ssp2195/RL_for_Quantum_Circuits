# Related-work audit and claim boundaries

Review date: 2026-09-09. Primary papers, author repositories and publisher records were used. This is a targeted technical comparison, not a claim of an exhaustive systematic review. Publication priority and acceptance remain matters for expert review. No absence of a capability is inferred merely from an abstract.

## Proposed central contribution

**A bounded-scoring, proof-preserving interface between linear search priorities and exact clean-workspace phase synthesis, evaluated with source-separated discovery and independently checkable resource exclusions.**

The core construction retains the full frontier but scores an indexed panel and globally services pending actions fairly. The paper proves semantic preservation, continuation-safe dominance, admissible completion cuts, finite-domain coverage, independent closed-cover soundness, bounded optimum certification, and a scoring-overhead bound. These are explicit specialized interface guarantees, not claims to invent induction, branch-and-bound, fairness, matroid partitioning, or RL.

## Closest comparisons

| Primary source | What the source establishes | Consequence for the claim |
|---|---|---|
| Amy, Maslov, Mosca (2014), arXiv:1303.2042; full domain/rank discussion inspected | Phase-based T-depth optimization, matroid partitioning, extra-ancilla trade-offs | Phase polynomials and rank/T-depth/ancilla identities are prior work. The calibration is not a new gate construction. |
| Amy, Azimzadeh, Mosca (2018/2019), arXiv:1712.01859; paper and Qiskit implementation inspected | CNOT-efficient parity-network synthesis, GraySynth | Must compare against an algebraic constructive method, not only BFS. An explicitly modified GraySynth is executed and labelled. |
| Selinger (2013), arXiv:1210.0974 | T-depth-one constructions with workspace | Ancilla-assisted depth reduction itself is not novel. |
| Paradis, Bichsel, Vechev (2024), arXiv:2212.10395 | Space-constrained uncomputation | Do not claim the first gate/space trade-off optimizer. Dynamic reversible pebbling is outside this implementation. |
| Dubal et al. (2025), arXiv:2503.14448; full HTML inspected | Learned Pauli-network resynthesis with a substantially broader Clifford/Pauli domain and a transpiler evaluation | Do not claim first RL Pauli synthesis, larger scale, or better performance without reproduction. |
| Ruiz et al. (2025), Nature Machine Intelligence, doi:10.1038/s42256-025-01001-1; full article inspected | Deep RL for T-count/tensor decomposition; Z3 verification of small T-count optima in its tensor domain | **The generic combination of learned discovery and independent optimality checking is already present. It is not the central novelty claim.** |
| Zen, Nägele, Marquardt (online 2025/issue 2026), doi:10.1038/s42256-025-01166-9, arXiv:2511.09951 | Reuse/generalization of AlphaTensor-Quantum agents | Frozen reusable agents are not novel by themselves. |
| Mattick and Mutschler (2023/2024), arXiv:2310.00112 | RL node selection in branch-and-bound, trained policy applied across problems | Learned ordering of exact search is not new. The retained-frontier/scoring interface needs its own explicit scope and cost argument. |
| Yu et al. (2026), Vista, doi:10.1145/3786335.3813148; publisher and author-institution abstract/metadata inspected | Quantum-program generation with staged verified rewards and budget-aware evaluator allocation | Verification-aware/cost-aware RL already exists. Our resource-coverage certificates differ from the described staged verifier feedback, but no uninspected claim about Vista's entire proof support is made. |
| Zak et al. (2025), doi:10.4230/LIPIcs.CP.2025.38 | Exact/approximate depth-optimal synthesis reduced to weighted #SAT | Independent bounded optimization is established; a hand-written enumerator is not presumed state of the art. |
| Wang et al. (2026), arXiv:2605.15476; full HTML inspected | Exact T library and Clifford-equivalence mapping, addressing shortcomings of fixed AND cost | The new fixed-polynomial experiments do not establish unrestricted T-count optimality. |
| Theissinger et al. (2026), arXiv:2602.15146; full HTML inspected | Lightweight supervised remaining-cost prediction with stochastic beam search | Low training overhead and zero-shot deployment are not exclusive to RL. Our panel retains alternatives instead of beam-pruning them. |

## Executed versus literature-only comparisons

Executed: identical untrained linear prior; greedy; component ablations; retrained feature ablations; actual Dijkstra uniform-cost; independently implemented audit-only; parity-star; small rank-partition reference; adapted GraySynth.

Not reproduced: modern Qiskit PMH GraySynth restoration, full T-par/Reqomp, neural Pauli synthesis, AlphaTensor, Vista, exact-T libraries, SAT/#SAT solvers, supervised beam-search systems. No numerical superiority over these systems is asserted. The GraySynth attribution and Apache license are retained separately.

## Findings and prohibited claims

The original locked campaign has negative transfer relative to the untrained identical prior. Preserve it. The amendment shrinks learned residuals using validation only, then uses entirely new logical target orbits for confirmation. Shrinkage is not labelled a new RL algorithm or a safe-policy-improvement theorem. If beta=0 is selected, deployment is explicitly nonlearned.

Do not claim first learning-plus-proof synthesis, universal learned superiority, new controlled-S physics, full Clifford+T optimality, arbitrary Boolean-oracle coverage, a neural-DAG policy, end-to-end BNN quantum advantage, or publication acceptance.
