# Hybrid Clifford+T Frontier Ranking

This branch contains the compact hybrid implementation of online-RL frontier
ranking for exact, ancilla-free Clifford+T circuit synthesis. The learned action
selects one open frontier record; it never selects a gate. The deterministic
engine owns gate legality, semantic updates, canonical/Pareto pruning, and
independent final certification.

Each frontier record retains

\[
(G,\Theta,\mathcal R,\phi,\rho),
\]

where `G` is a lossless persistent dependency DAG, `Theta` is a complete
forward/inverse Clifford tableau, `R` is an ordered word of signed Pauli
rotations, `phi` tracks global phase in eighth-turns, and `rho` stores monotone
resources. The exact symbolic invariant is

\[
U=e^{i\phi\pi/8}C(\Theta)R_{P_1}(\theta_1)\cdots R_{P_m}(\theta_m).
\]

## Added benchmarks

### Exact QFT-2

`heldout-qft2-exact` is a conventional forward two-qubit QFT target,

\[
(F_4)_{jk}=\frac{1}{2}e^{2\pi i jk/4},
\]

including the final SWAP through its exact three-CNOT native decomposition.
It is evaluated through the unrestricted native gate grammar
`H, S, SDG, T, TDG, CNOT`; the search-facing target contains no generator gate
sequence.

### Structured Toffoli

`stress-toffoli3-structured-parity-network` is a separately reported certified
stress test. RL ranks full hybrid frontier records inside the exact seven-term
CCZ parity-network normal form, with deterministic outer `H(2)` gates. Every
returned 15-gate witness is reconstructed from its persistent DAG and compared
independently with the analytical CCX matrix. This is deliberately not claimed
as unrestricted native Toffoli discovery.

## Local qualification

The qualified five-seed campaign used:

- 800/800 certified online SARSA training episodes;
- 35/35 certified unrestricted frozen-policy evaluations;
- 5/5 exact QFT-2 evaluations;
- 5/5 structured-Toffoli stress tests;
- approximately 22.7 process-CPU seconds under a hard 1,800-second limit;
- 15 passing regression tests.

Runtime values are environment-specific engineering evidence rather than a
universal performance guarantee.

## Install and test

```bash
python -m pip install -e '.[dev]'
python -m compileall -q hybrid_qcs tests
python -m pytest -q
```

## Reproduce the bounded campaign

```bash
python -m hybrid_qcs.runner \
  --deadline-seconds 1800 \
  --episodes 160 \
  --seeds 11,19,23,31,47 \
  --train-expansion-cap 2048 \
  --eval-expansion-cap 8192 \
  --toffoli-expansion-cap 8192 \
  --output-dir outputs/qft2-toffoli-30min
```

The command returns success only when all declared training episodes,
unrestricted evaluations, and structured Toffoli runs complete within the hard
deadline and all returned witnesses pass independent dense certification up to
global phase.
