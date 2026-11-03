# Measured M1/M2 integration results (2026-11-03)

Reproduce with `python -m hybrid_qcs.cleanup.theorem_guided_runner --output results.json`.
All reported circuits were independently verified with `verify_phase` over
the full Boolean input domain and all measurement outcomes. Source branch:
`theorem-guided-causal-frontier-20261103`.

## BNN amplitude-amplification phase oracles (B1–B3)

The comparison is measured-AND cleanup before vs after M1-generated feedback.
These three are explicitly specified small BNNs from
`bnn_amplitude_study.py`. These counts exclude Hamming-ball preparation,
amplitude-amplification reflections, and Grover iterations.

| Phase oracle | Peak ancilla qubits before / after | T before / after | CNOT before / after |
| --- | ---: | ---: | ---: |
| B1: 3-input threshold BNN | 1 / 1 | 4 / 4 | 7 / 7 |
| B2: 4-input pattern detector | 2 / 2 | 16 / 16 | 20 / 20 |
| B3: 5-input, 3-hidden-neuron BNN | 3 / 3 | 36 / 36 | 42 / 42 |

M1 is used to **construct** the corrections and independently check them,
not to claim an ancilla saving absent from the data. The frozen outer SARSA /
inner LinUCB test on a held-out 4-qubit ANF phase finds the same best
1-ancilla / 15-T circuit in 6 expansions with and without M1.

## Complete staged correction circuits (M3 cubic-garbage family)

Unlike the theorem's narrower correction-only subproblem, these complete
identity-phase instruments include the physical AND preparation, X-measurement
and cleanup of the original cubic garbage. They are **not BNN oracles**.

| Family | Correction stages | Peak correction ancillas | Peak total ancillas | T | CNOT | Native depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| q=3 | 1 | 3 | 3 | 60 | 60 | 228 |
| q=3 | 3 | 1 | 2 | 60 | 60 | 264 |
| q=4 | 1 | 4 | 4 | 96 | 96 | 361 |
| q=4 | 2 | 2 | 2 | 96 | 96 | 382 |
| q=4 | 4 | 1 | 2 | 96 | 96 | 424 |

Ancilla qubits can be reused between correction stages. Peak **total**
ancilla occupancy has a floor of two here because the original garbage
preparation uses two temporary qubits. All stage partitions are exhaustively
enumerated within the declared family, with the same deterministic objective.

## M2-pruned frontier ablation

| Infeasible constraints | Expanded edges without M2 | With M2 | Pruned records |
| --- | ---: | ---: | ---: |
| q=4, one stage, one correction ancilla | 4 | 0 | 1 |
| q=6, two stages, one correction ancilla | 36 | 6 | 6 |

The q=6 example is nontrivial: its initial M2 capacity bound permits one
helper over two stages, but **after any first-stage helper choice** the
remaining cubic byproduct rank exceeds the last stage's capacity. In feasible
test cases M2 preserved the best independently verified circuit but did not
reduce expanded edges. M2 is a **necessary** bound, not a sufficient test.

**Boundaries:** (1) The correction stage permits one simultaneous helper
preparation/use/measurement cycle per feature; no repeated helper lifecycle
inside one declared stage. (2) M2 prunes the staged correction frontier,
not the original immediately-corrected ANF frontier. (3) No result claims a
global oracle resource optimum, a complete BNN synthesis speedup, or a
learned-policy advantage. (4) Measurement latency is a declared abstract
resource model, not measured quantum hardware performance.
