# Compact Online-RL Simulator for Frontier-Record Ranking

## Scope

This path is an exact, deliberately restricted training simulator for signed
unit-coefficient phase-polynomial cores. The learned action is still the
selection of one persistent frontier record. Legal child generation remains
deterministic and exhaustive within the declared grammar.

The searched core uses only `CNOT`, `T`, and `TDG`. A Toffoli experiment places
a fixed `H(2)` before and after the searched CCZ core. Consequently this path
does **not** claim arbitrary Clifford+T synthesis with interleaved Hadamards.

## Compact state

For a fixed target with signed parity terms, a training state is

```text
(basis_rows, emitted_mask, cnot_count)
```

- `basis_rows` is the current invertible GF(2) linear transformation carried by
  the physical wires.
- `emitted_mask` identifies required signed parity terms already emitted.
- `cnot_count` is the sole independent monotone resource counter. Phase count
  is `popcount(emitted_mask)` and core gate count is phase count plus CNOT count.

This removes per-child CircuitDAG copying, Clifford tableau/frame updates,
Pauli-axis transport and rotation normalization, general canonicalization, and
dense target evaluation.

## Exact deterministic transition

When a frontier record is expanded, the simulator generates:

1. one `T` or `TDG` child for every currently exposed, not-yet-emitted required
   parity; and
2. one child for every directed CNOT pair.

A memoized exact reachability predicate rejects only prefixes with no terminal
suffix inside the CNOT bound. On three qubits, the linear basis graph has 168
vertices. Exact duplicate identity is the tuple `(basis_rows, emitted_mask)`;
for that identity, the smaller CNOT count dominates.

## Online learner

A shared linear score ranks all active frontier records. The policy is trained
online by epsilon-greedy semi-gradient SARSA. It never observes or predicts a
next gate. The reward uses emitted target-term progress, a per-expansion cost,
and a terminal bonus; it does not consume a demonstration circuit or offline
dataset.

Training periodically freezes exploration and evaluates the current policy.
The best observed frozen checkpoint is retained, preventing a late unstable
update from replacing a better scheduler. A process-CPU guard defaults to 1800
seconds.

## Certification boundary

Only the final compact parent chain is converted into the repository's
`CircuitState`. The fixed shell is added, the full authoritative DAG/tableau/
Pauli-rotation summaries are reconstructed once, and the existing independent
dense simulator certifies the result against the analytical Toffoli unitary.

## Command

```bash
python -m experiments.compact_online_rl \
  --episodes 64 \
  --training-max-expansions 256 \
  --evaluation-max-expansions 3000 \
  --cpu-seconds 1800 \
  --seed 23 \
  --output outputs/compact-online-rl/summary.json
```

The report distinguishes compact-search success from independent full-pipeline
certification and records the exact claim boundary.
