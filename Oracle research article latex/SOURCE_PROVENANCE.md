# Source provenance

The article was developed from the user-supplied
`Hierarchical_QCS_Article_LaTeX_Project(2).zip`.  Its modular LaTeX organization,
notation, persistent-DAG/Clifford-tableau/Pauli-rotation representation,
hierarchical outer-SARSA/inner-LinUCB formulation, canonicalization pipeline,
and visual conventions were retained as the base.

The application context is the user-supplied paper
`QuantumAE_for_BNN_modified (1)(1).pdf`, specifically its Boolean robustness
predicate, clean-ancilla compute-phase-uncompute construction, and three-input
example in which `100` is the unique violating input.  Copies of the source ZIP
and PDF are deliberately not redistributed in this project.

Implementation evidence is derived from:

- repository: `ssp2195/RL_for_Quantum_Circuits`
- implementation base: `qft3-guided-ancilla-v1`
- development/publication branch: `bnn-oracle-hierarchical-v1`
- experiment directory: `experiments/bnn_oracle_20260905/`

The Boolean truth table is supplied to the synthesizer, but no evaluator gate
sequence is supplied.  Outer SARSA and inner LinUCB schedule exact reversible
mapping transitions.  The discovered evaluator is then lowered to native
Clifford+T and independently certified after the universal
`U_g^dagger Z_f U_g` phase wrapper is applied.
