# Hierarchical Learning-Guided Exact Clifford+T Synthesis

This branch implements exact unitary Clifford+T synthesis with hierarchical
outer-SARSA frontier allocation, role-aware inner LinUCB continuation
allocation, strengthened projective canonicalization, and fixed-pool clean or
borrowed ancilla contracts certified through logical isometries. Learning
allocates search effort but never determines gate legality, circuit semantics,
pruning validity, or correctness.

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

## Ancilla contract boundary

The branch now supports fixed, preallocated clean ancillas and optional borrowed
ancillas under unitary Clifford+T evolution. Measurement, reset, classical
feed-forward, discarded garbage channels, and learned ancilla release are still
outside the declared model.

## Fixed-pool clean and borrowed ancillas

The `bnn-oracle-hierarchical-v1` extension retains the explicit
`AncillaContract`. Physical wires are partitioned into logical data, clean
workspace initialized in `|0>`, and optional borrowed workspace whose arbitrary
input state must be returned unchanged. For a clean-input embedding `J`, a
candidate full-register unitary `U_C` is accepted when

\[
U_C J = e^{i\phi}J(U_\star\otimes I_{\mathrm{borrowed}})
\]

in projective mode, or without the scalar phase in exact mode. Independent
certification reports both phase-aligned isometry error and leakage outside the
clean-output subspace. Full symbolic-key equality is not required for terminal
acceptance.

Archive pruning deliberately remains conservative. Projective contracts use
the strengthened full-register Clifford/Pauli key, which is a sufficient but
not necessary test for clean-ancilla equivalence. Exact-phase contracts disable
semantic merging through a witness key because the current Clifford tableau is
projective. This sacrifices pruning efficiency rather than risking an unsound
phase merge.

The generic ancilla search retains outer linear SARSA and a role-aware disjoint
linear LinUCB policy. Gate contexts distinguish logical, clean-workspace, and
borrowed-workspace operands. Existing mixed-gate policies provide a warm start;
the recorded ancilla fine-tuning adds only three outer episodes and four inner
episodes.

Run the bounded qualification:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  hybrid-qcs-ancilla \
  --output-dir outputs/ancilla-qualification
```

The recorded qualification under
`experiments/ancilla_isometry_20260905/` reports certified
held-out clean-ancilla contracts on one to three logical qubits, and an 8.2%
training-time increase relative to the matched pre-ancilla hierarchy. It also
constructs and independently certifies an exact 47-gate QFT-3 witness using one
clean ancilla to implement the controlled-`T` phase through
compute-phase-uncompute. The unrestricted QFT-3 search probe did not certify
within its 1.5-second bound and is reported as a negative bounded result.

## Decomposition-guided QFT-3 mitigation

The negative unrestricted probe is a search-depth result, not a failure of the
ancilla semantics or certifier. Four physical qubits expose 32 native one-gate
continuations, whereas the original exact witness has 47 gates and contains a
long compute--phase--uncompute detour. Hundreds of attempted native edges are
therefore not a realistic unrestricted-discovery budget.

The branch adds a separate, explicit structured generator for analytically
recognized QFT targets. It derives the standard Hadamard, controlled-phase,
and final bit-reversal blocks from the logical-wire order, lowers each block to
the unchanged native grammar, and independently certifies the final DAG. It
does not inspect the hidden benchmark witness.

For controlled-`T`, the generator uses a nine-gate relative-phase AND compute
circuit `R`, applies `T` to the clean ancilla, and executes the exact inverse
`R^dagger`. The input-dependent phases of `R` cancel against `R^dagger`, so the
logical controlled phase is exact on the declared clean-input subspace. This
reduces the complete QFT-3 realization to:

| Quantity | Original certified witness | Guided generator |
|---|---:|---:|
| Native gates | 47 | 35 |
| `T`/`TDG` gates | 21 | 15 |
| CNOT gates | 19 | 13 |
| Depth | 34 | 26 |

Run the structured generator directly:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  hybrid-qcs-qft3 \
  --output outputs/qft3-guided/result.json
```

The current regression suite contains 59 passing tests. The guided path adds
no training episodes, so it does not increase policy-training time. The
unrestricted native-search result remains reported separately and must not be
relabelled as successful unrestricted discovery.

## Hierarchical RL synthesis of a BNN verification phase oracle

The branch adds a specification-level Boolean-oracle synthesizer.  The input is
an exact Boolean truth table rather than a target-specific gate sequence.  A
generic reversible macro grammar contains NOT, CNOT, and Toffoli operations;
outer linear SARSA selects a persistent mapping-frontier record and disjoint
linear LinUCB ranks the record's still-pending macros.  Exact mapping equality,
continuation masks, resource-Pareto pruning, deterministic fairness, native
Clifford+T lowering, and clean-ancilla certification remain outside learning.

After a reversible evaluator `U_g` is independently certified, the phase oracle
is assembled through the universal identity

\[
O_g = U_g^\dagger Z_f U_g.
\]

For the three-input BNN robustness example with unique violating input `100`,
the frozen hierarchy discovers a six-macro evaluator without receiving a
hidden evaluator circuit.  Native lowering produces a 48-gate evaluator and a
98-gate exact phase oracle with 42 `T`/`TDG` gates and 42 CNOTs.  The exact
clean-input isometry error is approximately `1.22e-15`, and clean-workspace
leakage is approximately `3.26e-32`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  hybrid-qcs-oracle \
  --output-dir outputs/bnn-oracle
```

The exact mapping target-potential control remains faster on this deliberately
small three-bit instance.  The result therefore establishes generation and
certification feasibility, not universal learned-policy superiority.  The
truth-table implementation is currently restricted to at most three inputs;
larger BNN and arithmetic predicates require a symbolic Boolean IR and
compositional reversible compiler rather than an exponentially listed table.

Evidence is stored under `experiments/bnn_oracle_20260905/`.

## Research articles

The complete standalone oracle-synthesis article is stored in
`Oracle research article latex/`.  It integrates the hierarchical exact search,
strengthened canonicalization, fixed clean/borrowed ancilla contracts, the
RL-generated BNN verification phase oracle, and the restricted QFT-3 stress
test.  The earlier ancilla-contract article remains under
`Ancilla research article latex/` for provenance.
