# Source provenance

The article was developed from the two user-supplied benchmark projects `Hierarchical_QCS_Article_LaTeX_Project(1).zip` and `Research_project_latex_hierarchical_linucb_canonicalizer(1).zip`. The benchmark archives are not duplicated inside this repository directory.

Their organization, notation, exact symbolic representation, hierarchical outer-SARSA/inner-LinUCB framing, canonicalization pipeline, plots, and visual conventions were used as the baseline. The current project revises the article so that fixed-pool clean and borrowed ancillas are part of the method rather than only future work.

Implementation evidence is derived from:

- repository: `ssp2195/RL_for_Quantum_Circuits`
- base branch: `hierarchical-linucb-canonicalizer-v1`
- development branch: `ancilla-isometry-hierarchical-v1`

The ancilla runner records the deterministic contract tests, low-cost training protocol, held-out synthesis runs, QFT-3 witness certification, and bounded unrestricted QFT-3 search probe. Successful witnesses are reconstructed from the persistent DAG and independently certified on the permitted input subspace.
