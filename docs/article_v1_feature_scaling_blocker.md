# Article V1 frontier-feature scaling blocker

## Finding

The observed nontermination was not an infinite loop and was not a SARSA convergence failure. Article V1 training has a fixed two episodes per target and each episode has explicit terminal conditions: independent certification, frontier exhaustion, or the external expansion cap. It does not wait for a learned convergence criterion.

The blocker was the cost of constructing the exact policy features before each expansion.

## How the search is designed

At decision time the policy chooses one persistent record from the complete open frontier. It does not choose a gate. The selected record is then expanded exhaustively with every resource-feasible native one-gate continuation:

```text
single-qubit H, S, S†, T, T† on each wire
directed CNOT on every ordered pair of distinct wires
```

This yields 12 attempted native actions for two qubits and 21 for three qubits before resource rejection. Each feasible child is symbolically updated, canonicalized, passed through the Pareto archive, and independently certified when terminal. The real frontier is unbounded; `max_frontier=64` sizes only a compatibility mask.

The RL component is linear semi-gradient SARSA(0). It ranks frontier records with the same frozen 31-D feature equation and uses the actual next epsilon-greedy behavior action for the bootstrap. A hard episode may legally consume all 8,192 expansions without certifying a solution. That outcome is truncation under a finite budget, not lack of algorithmic termination.

## Unintended complexity

For an open record `v`, the resource-position coordinate is:

```text
r_P(v; F) = |{u in F \ {v}: rho(u) <= rho(v) componentwise}|
            / max(1, |F| - 1)
```

The reference evaluator rebuilt every record's intrinsic features and evaluated this coordinate with a nested comparison over all open records on every decision. A single batch therefore required quadratic frontier work:

```text
O(F² * d_r)
```

where `d_r` is only five for two qubits or six for three qubits, but `F` can exceed one thousand after 64 expansions. When frontier size grows approximately linearly with the number of expansions, repeatedly paying `O(F²)` produces an approximately cubic cumulative path.

The baseline confirms this diagnosis:

| Cap | Runtime | Peak F | Feature share |
|---:|---:|---:|---:|
| 32 | 8.589 s representative | 543 | 84.7% |
| 64 | 61.837 s representative | 1,039 | 94.5% |

An isolated all-pairs batch took 0.790 seconds at `F=534` and 2.330 seconds at `F=1,021`. The lack of output compounded the operational problem: episode output was suppressed and a transferable checkpoint was written only after all targets for a learner seed completed.

## Exact permitted correction

The mathematical feature does not require an all-pairs implementation. Grouping open records by immutable resource tuple gives:

```text
f(r) = number of open records with resource tuple r
D(r) = sum of f(s) over every s <= r
r_P(v) = (D(rho(v)) - 1) / max(1, F - 1)
```

Insertion and removal update exact group counts with vectorized comparisons over unique resource groups. A snapshot gathers each record's count by group in linear frontier time. Static record coordinates, target distance, canonical key, and resources are cached while the record remains open; novelty is updated only for semantic keys whose generation counts changed.

This changes the evaluator, not the scientific object:

```text
feature schema: article-v1-31d                     unchanged
reference evaluator: article-v1-reference-all-pairs-v1
optimized evaluator: article-v1-exact-incremental-v2
frontier/action/gate/certification semantics:      unchanged
```

## Qualification requirement

Optimization is acceptable only if the reference and optimized paths produce the same candidate coordinates, feature rows, greedy record, seeded SARSA trace, final weights, terminal status, witness, and deterministic counters on bounded runs. Performance does not count when that correctness gate is absent or false.

On one recorded environment, the optimized path must additionally demonstrate:

- at least 10× feature-decision speedup near frontier size 1,024;
- no more than 2.5× compact-batch-plus-score growth from 512 to 1,024;
- no Python nested loop over every pair of open records in production dominance maintenance;
- approximately linear feature-index memory in open records plus unique groups;
- exact trace parity at caps 32 and 64;
- staged hard-target measurements through at least cap 512 before feasibility review.

Exact full-frontier ranking remains at least linear in `F`, and exact circuit synthesis remains exponential in its gate-count budget. Removing this unintended cubic implementation path does not make exact synthesis polynomial. If the optimized 8,192 cap remains infeasible, a lower equal cap may be adopted only through a dated preregistration amendment made without inspecting held-out scheduler outcomes.
