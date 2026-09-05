# Mixed Clifford+T continuation-cost crossover

## Shortlisted problem

The benchmark synthesizes short Clifford-frame signed phase-pair motifs over the complete native grammar `H, S, SDG, T, TDG, CNOT`. Each held-out target contains a nontrivial Clifford scaffold and both positive and negative non-Clifford phase injections. The hidden native witness is used only to construct and validate the target; it is not exposed to either scheduler.

The old and new methods use the same frozen linear outer SARSA policy. The old method exhaustively attempts every native continuation after selecting a record. The new method retains pending exact continuations and uses a frozen disjoint linear LinUCB policy to rank a batch. A deferred native-order control isolates the benefit of lazy expansion from the benefit of learned continuation ordering.

## Aggregate results

| Target | Qubits | Actions/node | Method | Success consistency | Median observed wall (s) | Median certified wall (s) | Median attempted edges | Outer decisions | Peak frontier | Stop reason |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| heldout-mixed-frame-phase-4q | 4 | 32 | Deferred SARSA + LinUCB | 1.000 | 0.010347 | 0.010347 | 52 | 15 | 48 | certified |
| heldout-mixed-frame-phase-4q | 4 | 32 | Deferred SARSA + native order | 1.000 | 0.012000 | 0.012000 | 87 | 24 | 70 | certified |
| heldout-mixed-frame-phase-4q | 4 | 32 | Eager outer SARSA | 1.000 | 0.011671 | 0.011671 | 192 | 6 | 105 | certified |
| heldout-mixed-frame-phase-4q | 4 | 32 | Eager target potential | 1.000 | 0.151232 | 0.151232 | 5440 | 170 | 833 | certified |
| heldout-mixed-basis-echo-5q | 5 | 45 | Deferred SARSA + LinUCB | 1.000 | 0.033724 | 0.033724 | 162 | 44 | 154 | certified |
| heldout-mixed-basis-echo-5q | 5 | 45 | Deferred SARSA + native order | 1.000 | 0.041622 | 0.041622 | 245 | 66 | 229 | certified |
| heldout-mixed-basis-echo-5q | 5 | 45 | Eager outer SARSA | 1.000 | 0.045832 | 0.045832 | 585 | 13 | 281 | certified |
| heldout-mixed-basis-echo-5q | 5 | 45 | Eager target potential | 0.000 | 2.002262 | — | 29475 | 655 | 4775 | wall_limit |
| heldout-mixed-frame-phase-6q-ood | 6 | 60 | Deferred SARSA + LinUCB | 1.000 | 0.015884 | 0.015884 | 50 | 14 | 32 | certified |
| heldout-mixed-frame-phase-6q-ood | 6 | 60 | Deferred SARSA + native order | 1.000 | 0.020256 | 0.020256 | 102 | 27 | 84 | certified |
| heldout-mixed-frame-phase-6q-ood | 6 | 60 | Eager outer SARSA | 1.000 | 0.021108 | 0.021108 | 360 | 6 | 146 | certified |
| heldout-mixed-frame-phase-6q-ood | 6 | 60 | Eager target potential | 0.000 | 1.292049 | — | 30720 | 512 | 1376 | expansion_cap |

The repeated runs use the same frozen policies and targets; the success column therefore measures execution consistency, not independent-policy generalization uncertainty. Timings are medians, and the raw CSV retains every repetition.

## Crossover interpretation

- **heldout-mixed-frame-phase-4q:** 1.13x versus eager outer SARSA; 1.16x versus deferred native order; 14.62x versus eager target potential; 72.9% fewer exact edge attempts than eager SARSA.
- **heldout-mixed-basis-echo-5q:** 1.36x versus eager outer SARSA; 1.23x versus deferred native order; 72.3% fewer exact edge attempts than eager SARSA.
- **heldout-mixed-frame-phase-6q-ood:** 1.33x versus eager outer SARSA; 1.28x versus deferred native order; 86.1% fewer exact edge attempts than eager SARSA.

In this recorded frozen-policy run, LinUCB used fewer exact continuations than deferred native ordering on 3/3 targets and was faster in wall time on 3/3 certified comparisons. These selected cases demonstrate a continuation-cost crossover; they are not a universal or statistical claim over arbitrary target distributions.

## Scope and claim boundary

This is unrestricted search within the declared finite Clifford+T grammar and resource envelope; it is not a prescribed normal form. The target family is deliberately selected so that continuation cost grows with register width while exact solutions remain short enough to certify. It demonstrates a continuation-cost crossover, not universal superiority on arbitrary unitaries.

The four-qubit case is a lower-width control. The five- and six-qubit cases test whether the saved symbolic transitions, canonical keys, archive operations, and frontier growth amortize the extra LinUCB context cost. The six-qubit target is OOD in width because outer and inner training use only four- and five-qubit targets.

## Reproduction configuration

```json
{
  "batch_size": 4,
  "benchmark": "mixed-clifford-t-continuation-crossover-v1",
  "environment": {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "machine": "x86_64"
  },
  "evaluation_cap": 2048,
  "heldout_target_count": 3,
  "heldout_widths": [
    4,
    5,
    6
  ],
  "inner_episodes": 96,
  "inner_policy": "frozen disjoint linear LinUCB posterior mean",
  "linucb_alpha_during_evaluation": 0.0,
  "linucb_alpha_during_training": 0.5,
  "native_gate_set": [
    "H",
    "S",
    "SDG",
    "T",
    "TDG",
    "CNOT"
  ],
  "numpy": "2.3.5",
  "outer_episodes": 64,
  "outer_policy": "shared frozen linear semi-gradient SARSA",
  "outer_seed": 11,
  "platform": "Linux-6.18.35-x86_64-with-glibc2.41",
  "python": "3.13.5 (main, Jul 15 2026, 20:25:40) [GCC 14.2.0]",
  "repetitions": 3,
  "stress_cap": 512,
  "stress_repetitions": 2,
  "stress_wall_limit_seconds": 1.5,
  "target_potential_repetitions": 1,
  "train_expansion_cap": 128,
  "training_target_count": 16,
  "training_widths": [
    4,
    5
  ],
  "wall_limit_seconds": 2.0
}
```
