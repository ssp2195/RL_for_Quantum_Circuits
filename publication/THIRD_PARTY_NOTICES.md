# Third-party source attribution

`hybrid_qcs/phase_graysynth.py` adapts the cofactor-recursion implementation in
Qiskit 0.46.3, `qiskit/synthesis/linear_phase/cnot_phase_synth.py`, blob
`f1b49cc8d8df1ffba2d25830deb2875106f5c3f5`, copyright IBM 2017, 2019,
licensed under Apache-2.0 (`LICENSE-QISKIT.txt`). Source inspected 2026-09-09.

Modifications: return exact continuation tokens instead of a QuantumCircuit;
replace floating-point phase angles with the declared native phase blocks;
use deterministic integer index order; update aliased pending subsets once;
replace PMH restoration with the shorter of Gaussian elimination and inversion
of the synthesized CNOT prefix. This is identified in the results as the
GraySynth/Gaussian reference variant, not the current Qiskit implementation or
an exact reproduction of its performance. Correctness is independently replayed
for every generated circuit. No Qiskit runtime dependency is added.

Algorithm: M. Amy, P. Azimzadeh, M. Mosca, On the controlled-NOT complexity of
controlled-NOT-phase circuits, Quantum Science and Technology 4, 015002 (2019),
arXiv:1712.01859. The other rank/layer reference baseline follows established
rank and matroid-partition ideas from Amy, Maslov, Mosca (arXiv:1303.2042); it is
not a reproduction of the complete T-par optimizer.
