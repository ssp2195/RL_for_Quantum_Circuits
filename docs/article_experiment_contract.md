# Article-aligned experiment contract

This document fixes the experimental meaning of the generic native
Clifford+T benchmark. Correctness continues to come only from dense
certification of the concrete `CircuitDAG`; learning changes search order but
never gate generation, legality, pruning semantics, or acceptance.

## Action and transition

The authoritative action is a persistent frontier-record identity:

```python
nodes = env.current_nodes()
selected = policy.select_node(nodes)
env.select_record(selected.record_id)
```

Selecting one record deterministically enumerates every legal one-gate child.
The Gymnasium `Discrete(max_frontier)` action is a compatibility view over the
current container only. Its indices are not persistent actions and records
beyond the mask remain selectable by `record_id`.

## Reward modes

`expansion_cost` implements Article equation (24) with `discount = 1`:

- every expanded record contributes `-1`;
- successful terminal generation contributes a `+1` correction;
- a success at expansion `T_hit` therefore returns `1 - T_hit`;
- exhausting a `B`-expansion episode returns `-B`;
- if the frontier empties after `k < B` expansions, the terminal transition
  charges the unused `B-k` horizon, so failure still returns exactly `-B`.

It contains no archive-pruning, target-potential, or visitation bonus.
`expansion_cost_plus_visit_bonus`, `legacy_archive_shaping`, and
`target_progress_shaping` are separately named ablation/task-shaping modes.
Reports and training histories record the normalized mode and policy digest.

The article feature provider implements equation (19) as a shared 37-value
linear map: bias, 12 candidate features, 12 frontier-normalized features, and
12 interactions between the candidate vector and the scalar remaining
record-expansion budget fraction `b_t / B`. The environment updates that
fraction after every expansion. Frontier mean and variance are symmetric
reductions, so container permutation cannot give an index semantic meaning.

## Search metrics

All schedulers consume the same environment counters. The initial frontier
size is sampled after root insertion; another sample is taken after every
valid record expansion. Invalid positional actions are not expansion samples.

- `generated`: legal children returned by deterministic expansion.
- `certification_nonmatch`: generated children not independently certified as
  successful. Structurally incomplete constrained prefixes count here without
  invoking a final certifier.
- `duplicate_rejected`: a child rejected because an active record at the same
  semantic key weakly dominates it.
- `dominated_retired`: active records tombstoned because the new record
  strictly dominates them.
- `pareto_incomparable_accepted`: an additional incomparable resource record
  accepted at an already present semantic key.
- `reopened`: an accepted selectable record at a key for which a concrete
  record was previously expanded. A mere dominating replacement is not by
  itself a reopening.
- `expanded`: frontier records actually selected and expanded.
- `frontier_peak` / `frontier_mean`: maximum and arithmetic mean of the
  specified frontier samples.
- `archive_size`: number of distinct semantic keys ever admitted.
- `pareto_width_peak`: largest number of active resource records at one key.

Reports additionally contain runtime, the terminal solution resource vector,
and, for learning, TD-error summaries and weight norms. Wall-clock time is not
placed inside deterministic episode histories.

## Shared baselines and ablations

`experiments.article_benchmark` evaluates FIFO, LIFO, uniform-cost, seeded
random, target-potential, zero-weight linear, and trained linear SARSA with one
grammar, archive, budget, stopping rule, and certifier. Expected SARSA and a
one-step contextual bandit use the same feature and record-selection APIs.
Aggregates contain each seed plus success rate and the mean, standard
deviation, and median successful expansion count.

The tiny-instance ablation entry point toggles canonicalization, Pareto
dominance, Clifford-angle absorption, target-aware features, reward shaping,
fairness interleaving, and visitation bonus. Production defaults keep
canonicalization, dominance, and Clifford absorption enabled. An ablation key
schema is never mixed with a production key schema. The Pareto-off arm still
rejects exact same-key/same-resource duplicates, but retains different
comparable resource profiles. Feature, reward, fairness, and visit-bonus arms
each train and freeze a real linear scorer; they are not FIFO/zero-weight
configuration smoke tests.

## Native held-out corpus

`benchmarks.native_corpus` fixes train/validation/test split seeds 1729, 2753,
and 3769. Split identity is the phase-normalized dense target digest rather
than generator syntax. Generic search receives only the dense target matrix;
the generator witness and seed are retained in metadata solely for replay and
audit. No target-specific reachability rule is available to the generic
problem. The constrained Toffoli parity-network runner remains a separately
labelled, target-specific normal-form experiment.

The default runner evaluates declared learning-rate candidates only on the
validation targets and seeds, selects lexicographically by validation success
rate and mean successful expansions, and opens the test split only after that
choice. The model-selection artifact explicitly records that no test target
was observed.

## Exact QFT boundary

The QFT operation model is reference-only and cannot enter native expansion.
Canonical QFT-3 is classified `APPROXIMATION_REQUIRED`; its exact search
request contains no target. AQFT-3 is a separate fidelity-scored result, never
an exact certification. The current milestone therefore supports a correct
QFT reference/capability layer, not learned QFT-3 synthesis.
