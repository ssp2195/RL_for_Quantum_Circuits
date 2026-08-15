# Continuation and Pareto-Pruning Contract

This implementation deliberately separates circuit meaning from the cost of
the witness that reached it. This is the contract used by the canonicalizer,
resource records, and Pareto archive.

## Semantic equivalence

Two records are semantically equivalent when their canonical keys are equal.
The key contains:

- the exact projective Clifford-frame/irreducible-Pauli-rotation normal form;
- the register width; and
- the static continuation interface, including the gate grammar, topology,
  ancilla model, and the common resource-limit instance.

Consumed resources are not part of this key. In the current all-to-all,
no-ancilla implementation, equal keys therefore mean equal represented
unitaries and the same static rules for extending them. They do **not** imply
identical remaining-budget suffix languages.

## One-sided resource simulation

Let `u` and `v` have the same semantic key and let `rho(u) <= rho(v)` hold
componentwise for T count, two-qubit count, total gate count, and every wire's
depth. For every suffix `K` that is feasible from `v`, the implementation must
satisfy:

```text
K feasible from v  =>  K feasible from u
rho(u + K) <= rho(v + K)
semantic_key(u + K) == semantic_key(v + K)
```

The implication is intentionally one-sided. A cheaper record may admit a
suffix that has become infeasible from a costlier equivalent record.

The present budgets satisfy the assumptions because gate, T, and two-qubit
counts are additive and bounded above. Per-wire depth uses the monotone update

```text
layer = 1 + max(depth[q] for q touched by the gate)
depth[q] = layer for every touched q
```

which preserves componentwise order under a common suffix. Any future
resource or legality constraint may participate in Pareto pruning only after
the same extension-monotonicity property is specified and tested. State such
as placement, ancilla liveness, or classical-control status that changes the
static legal operations belongs in the continuation interface instead.

## Safe-pruning consequence

If `rho(u) <= rho(v)`, every certified solution reachable from `v` by a legal
suffix is also reachable from `u` by that suffix with no worse resources.
Discarding `v` therefore cannot remove a unique feasible or Pareto-superior
solution. Resource-incomparable records at one semantic key must both remain
in the archive antichain.

This is a resource-simulation theorem, not strict symmetric continuation
equivalence. The normative article replacement text is recorded in
`docs/article_continuation_contract_amendment.md`, and implementation claims
and tests use those terms consistently. The exhaustive small-suffix regressions in
`tests/test_continuation_resource_simulation.py` directly check the implication
for gate, T, two-qubit, total-gate, and per-wire-depth limits.

## Canonicalization scope

The canonicalizer may fuse same-axis rotations, cancel inverses, reorder only
commuting rotations, and absorb exact Clifford-angle factors into the residual
frame. Equality of canonical payloads is a sound sufficient condition for
semantic equivalence; the normal form is not claimed to decide all possible
Pauli-rotation word identities. Final target correctness remains the job of
the independent certifier operating on the authoritative circuit witness.

For a controlled tiny-instance ablation,
`Canonicalizer(absorb_clifford_angles=False)` keeps emergent Clifford factors
in the normalized rotation word. The flag is encoded in the key schema so
payloads from different experimental modes are never mixed accidentally.
