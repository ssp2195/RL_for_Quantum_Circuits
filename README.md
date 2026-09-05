# Hierarchical Learning-Guided Exact Clifford+T Synthesis

This branch implements an exact, ancilla-free Clifford+T synthesis framework in
which learning allocates search effort but never determines circuit semantics or
correctness.

Each accepted circuit prefix retains

\[
(D,\Theta,\mathcal R,\phi,\rho),
\]

where `D` is a losslessly shared dependency DAG, `Theta` is a complete
forward/inverse Clifford tableau, `R` is an ordered word of signed Pauli
rotations, `phi` records global phase in eighth-turn units, and `rho` stores
monotone resources. The exact execution invariant is

\[
U=e^{i\phi\pi/8}C(\Theta)
R_{P_1}(\theta_1)\cdots R_{P_m}(\theta_m).
\]

The deterministic engine owns gate legality, tableau and rotation updates,
resource checks, canonicalization, duplicate handling, Pareto dominance, and
independent dense certification.

## Hierarchical search control

The outer controller is a shared linear semi-gradient SARSA ranker over active
frontier records. In deferred-search experiments, every record additionally
retains a bit mask of native continuations that have not yet been processed.
A disjoint linear LinUCB contextual bandit ranks those pending continuations,
with one small linear model for each gate family
`H`, `S`, `SDG`, `T`, `TDG`, and `CNOT`.

Only a fixed batch of exact continuations is attempted before control returns to
the outer ranker. Unchosen continuations remain pending; a deterministic
fairness mechanism prevents learned scheduling from becoming learned pruning.
LinUCB exploration is used only during training. Frozen evaluation ranks by the
posterior mean.

## Strengthened deterministic canonicalization

The execution representation remains incremental, but duplicate and
resource-Pareto pruning now use a stronger projective canonical view:

1. normalize signs, angles, same-axis fusions, and commuting order;
2. identify every even-quarter-turn Pauli rotation, which is Clifford-valued;
3. move an embedded Clifford rotation to the Clifford boundary by exactly
   conjugating the preceding Pauli axes;
4. absorb that rotation into a copied Clifford tableau;
5. repeat until the residual rotation word contains only odd quarter turns.

Thus identities such as `T;T` and `S`, or `TDG;TDG` and `SDG`, receive the same
projective semantic key. The persistent witness and execution factorization are
unchanged.

The key remains intentionally conservative for general noncommuting odd
Clifford+T rotations. Equal keys are safe evidence of projective equivalence,
but the converse is not claimed.

## Validation

Install and run the complete regression suite:

```bash
python -m pip install -e '.[dev]'
python -m compileall -q hybrid_qcs tests
python -m pytest -q
```

Run the bounded canonicalization audit:

```bash
hybrid-qcs-canonicalization-audit \
  --output-dir outputs/canonicalization-audit
```

The recorded audit enumerates 3,906 one-qubit prefixes through depth five and
1,885 two-qubit prefixes through depth three. Against an independent dense
projective fingerprint used only as a differential-test oracle, the stronger
key reduced unresolved duplicate excess by approximately 51% and 50%,
respectively, with no observed false merge in those bounded samples.

## Mixed Clifford+T continuation-cost experiment

The principal hierarchical experiment uses the complete native grammar
`H, S, SDG, T, TDG, CNOT` on four to six qubits. Outer and inner policies are
trained on four- and five-qubit targets and evaluated on held-out four-, five-,
and six-qubit Clifford-frame phase motifs. Hidden witnesses construct and
validate targets but are not exposed to either policy.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  hybrid-qcs-mixed-crossover \
  --output-dir outputs/mixed-crossover
```

With the strengthened canonicalizer and frozen policies, the recorded medians
for deferred SARSA plus LinUCB were:

| Target | Certified wall time | Exact edge attempts | Speedup over eager SARSA |
|---|---:|---:|---:|
| 4-qubit mixed frame-phase | 0.01035 s | 52 | 1.13x |
| 5-qubit mixed basis echo | 0.03372 s | 162 | 1.36x |
| 6-qubit mixed frame-phase, OOD width | 0.01588 s | 50 | 1.33x |

The hierarchy also outperformed deferred native continuation order in the three
recorded cases. These are controlled crossover examples, not a claim of
universal superiority on arbitrary unitaries or of statistically established
OOD generalization.

Machine-readable data, frozen policies, reports, and figures are stored under
`experiments/hierarchical_canonicalizer_20260905/`.

## Additional benchmarks

The repository retains:

- exact unrestricted native QFT-2 evaluation;
- a certified structured seven-term CCZ parity-network Toffoli stress test;
- a directed-CNOT continuation-cost crossover on four to six qubits;
- an unrestricted native Toffoli feasibility probe that did not certify within
  its bounded cap.

The structured Toffoli experiment must not be interpreted as unrestricted
native Toffoli discovery.

## Ancilla boundary

Ancilla-assisted synthesis is not implemented in this branch. A clean-ancilla
extension requires a semantic object based on the restricted isometry

\[
V_C=U_C\bigl(I\otimes |0\rangle^{\otimes a}\bigr),
\]

rather than equality of full unitaries on arbitrary ancilla inputs. Ancilla
liveness, restoration, and resource coordinates must be specified before they
can participate in sound dominance pruning.
