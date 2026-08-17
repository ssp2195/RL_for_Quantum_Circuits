# Article V1 feature contract

## V2 target metric binding

Process infidelity and direct target-distance scheduling are bound to `projective-unitary-metrics-v2`, which validates and compares the original matrices without normalization. Feature caching remains keyed by target fingerprint plus complete ordered DAG witness; terminal certification reconstructs a fresh DAG unitary and does not consume that cache.

`article-v1-31d` is the publication feature schema. It is distinct from the
pre-existing extended 37-coordinate and composite target-aware schemas.

## Candidate vector

The candidate vector has exactly ten ordered coordinates:

```text
t_count
two_qubit_count
gate_count
depth
rotation_count
anticommuting_pair_count
mean_pauli_weight
target_process_infidelity
frontier_resource_dominance_fraction
archive_novelty
```

Resources are divided by `max(1, B_T)`, `max(1, B_2q)`, `max(1, B_g)`, and
`max(1, B_D)`. Rotation count is divided by `max(1, B_T)`. Anticommuting pairs
are divided by `max(1, binomial(B_T, 2))`. Mean Pauli weight is divided by
`max(1, n)`. Target process infidelity is already in `[0, 1]`.

For one immutable frontier snapshot, the resource-dominance coordinate is the
fraction of other open records whose complete resource vector is no greater
componentwise. This is a ranking statistic only and is never used for
cross-semantic pruning.

The novelty coordinate is

```text
1 / sqrt(max(1, generated_count[semantic_key]))
```

where the deterministic generation count includes the root once and every
resource-feasible generated record, whether accepted, duplicate-rejected, or
dominance-rejected. It is not a policy visitation count.

## Target metric

For dimension `d`, target `V`, and a candidate reconstructed from its complete
DAG witness `U`, the feature metric is

```text
process_infidelity = 1 - abs(trace(V.conj().T @ U))**2 / d**2
```

Only floating-point overshoot is clipped into `[0, 1]`. The cache identity is
the target fingerprint plus the complete DAG gate witness, never record ID.

## Complete vector

Using population frontier mean and standard deviation with fixed stabilizer
`eta = 1e-8`, the complete map is

```text
[bias, x(10), z(10), (remaining_expansions / expansion_budget) * x(10)]
```

and therefore has exactly 31 `float64` coordinates. All rows are computed from
one immutable frontier/archive snapshot. The selected pre-transition vector is
frozen and supplied directly to the SARSA update.

The target-free and no-frontier-context ablations have distinct schema names
and dimensions; weights may never be resized or reinterpreted across schemas.
