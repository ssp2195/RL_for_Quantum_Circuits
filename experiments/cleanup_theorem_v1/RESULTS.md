# Five-candidate cleanup study

measured resources of five selected full-bank oracle problems; not unrestricted optimality or learner superiority.

| Candidate | r x m | Aux: theorem / control | T: theorem / control | CNOT: theorem / control | Native gates: theorem / control |
|---|---:|---:|---:|---:|---:|
| enabled-three-feature-coupling | 3 x 1 | 4 / 6 | 16 / 24 | 19 / 27 | 83 / 119 |
| two-channel-feature-ring | 4 x 2 | 10 / 12 | 40 / 48 | 47 / 55 | 205 / 241 |
| mixed-five-row-predicate | 5 x 2 | 12 / 15 | 48 / 60 | 60 / 72 | 258 / 312 |
| three-channel-cross-feature-bank | 6 x 3 | 21 / 24 | 84 / 96 | 108 / 120 | 462 / 516 |
| wide-shared-mask-oracle | 8 x 3 | 27 / 32 | 108 / 128 | 138 / 158 | 592 / 682 |

| Method | Correct / runs | Timely | Median generation + verification time |
|---|---:|---:|---:|
| bank_free_anf | 15 / 15 | 15 | 0.001077 s |
| coherent_inverse_trained | 75 / 75 | 75 | 0.014569 s |
| control_helpers_trained | 75 / 75 | 75 | 0.017384 s |
| deterministic_control_side | 15 / 15 | 15 | 0.004764 s |
| deterministic_smaller_side | 15 / 15 | 15 | 0.004111 s |
| theorem_trained | 75 / 75 | 75 | 0.021600 s |
| theorem_untrained | 15 / 15 | 15 | 0.019235 s |

The deterministic smaller-side control receives the SAME construction and may be faster than learned search. Any common quantum-resource improvement is attributable to the helper/cleanup construction, not training.

All costs include the consumer and the native H/measurement/conditional-X reset implementation. T=4 per exact clean-target AND; this is an established resource primitive, not a new four-T full Toffoli.

Measurement has an explicit latency; two measurement layers do not mean constant total runtime. CNOT and gate counts are worst-case execution counts. The archived expected gate counts require the certified uniform transcript.

The workspace theorem is trusted mathematics from the attached PDF. Its hypotheses and the witness are machine-checked; this is not proof-assistant validation of the converse, and it does not prove global oracle optimality.
