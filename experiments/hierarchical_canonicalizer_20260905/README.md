# Hierarchical policy and canonicalizer validation

This directory records the validation evidence used by the corresponding
standalone LaTeX research presentation.

## Canonicalization audit

`canonicalization_audit/` contains bounded differential enumeration for one and
two qubits. Dense projective fingerprints are used only as a validation oracle;
they are not production archive keys.

## Mixed Clifford+T hierarchy

`mixed_crossover/` contains frozen-policy runs for the full native grammar
`H, S, SDG, T, TDG, CNOT` at four to six qubits. The outer controller is linear
SARSA and the inner controller is a disjoint linear LinUCB model. The reported
six-qubit target is out of distribution in register width relative to policy
training.

## Claim boundary

The recorded experiments establish a controlled continuation-cost crossover and
a reduction in unresolved symbolic duplicate excess. They do not establish a
complete Clifford+T normal form, arbitrary-unitary scalability, or unrestricted
ancilla-assisted synthesis.
