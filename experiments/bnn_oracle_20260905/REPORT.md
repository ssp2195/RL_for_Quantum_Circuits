# Hierarchical BNN verification-oracle synthesis

The experiment synthesizes the reversible predicate evaluator with outer linear SARSA and inner disjoint LinUCB over a generic NOT/CNOT/Toffoli macro grammar. The final phase oracle is formed through the universal compute-phase-uncompute identity and lowered to the unchanged native Clifford+T library.

## Target

- Inputs: `x1 x2 x3` on data qubits `q0 q1 q2`.
- Predicate: `g(x)=1` only for `x=100`.
- Flag: `q3`, initialized and restored to `|0>`.
- Clean workspace: `q4`, initialized and restored to `|0>`.
- Phase action: `|x> -> (-1)^g(x)|x>`.

## Certified result

- Success: **True**
- Outer allocations: **94**
- Attempted macro continuations: **352**
- Peak frontier: **236**
- Evaluator macro witness: `TOFFOLI(0,1,4); CNOT(0,4); CNOT(4,3); TOFFOLI(2,4,3); CNOT(0,4); TOFFOLI(0,1,4)`
- Evaluator native gates: **48**
- Complete phase-oracle native gates: **98**
- Phase-oracle T/TDG count: **42**
- Phase-oracle CNOT count: **42**
- Exact isometry error: **1.224e-15**
- Clean-workspace leakage: **3.265e-32**
- Policy-training time: **0.010947 s**

## Interpretation

The learned search does not receive a target-specific circuit. It sees the truth table and a target-independent reversible macro grammar. Exact domain mappings provide the oracle-layer canonical key; every returned macro witness is lowered through the strengthened Clifford-tableau/Pauli-rotation canonicalizer and independently certified under the clean-ancilla isometry contract.

The macro grammar is a restricted first implementation for small Boolean predicates. It is not a complete compiler for arbitrary large BNNs, and the results do not establish superiority over an exact target-potential scheduler.

## Frozen comparison rows

| Method | Success | Median wall time (s) | Macro edges | Allocations | Peak frontier |
|---|---:|---:|---:|---:|---:|
| oracle_deferred_fixed_order | True | 0.095339 | 371.0 | 100.0 | 252.0 |
| oracle_exact_target_potential | True | 0.038467 | 180.0 | 48.0 | 139.0 |
| oracle_hierarchical_sarsa_linucb | True | 0.103288 | 352.0 | 94.0 | 236.0 |
