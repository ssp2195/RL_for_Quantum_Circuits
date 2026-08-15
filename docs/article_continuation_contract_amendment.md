# Normative amendment to the article's continuation contract

This is the implementation-aligned replacement for the strict symmetric
continuation-equivalence wording in `Article_Pauli_rotation.md`. It preserves
the article's safety result while distinguishing semantic identity from the
one-sided resource relation that actually justifies Pareto pruning.

## Replacement definitions

Let `lambda(v)` contain every **static** fact capable of changing available
gate types or transition semantics: grammar, topology, register width,
placement, ancilla liveness/restoration obligations, and relevant classical
or hardware state. Consumed monotone resources are stored in `rho(v)`, not in
the canonical semantic key.

Define semantic-interface equivalence by

```text
u ==sem v
    iff Sem(u) ~ Sem(v)
    and lambda(u) = lambda(v)
    and every common suffix has equivalent semantics.
```

The canonicalization obligation is the one-way implication

```text
kappa(u) = kappa(v)  =>  u ==sem v.
```

It does not assert equality of the budget-constrained suffix languages of two
witnesses that have consumed different resources.

For records with the same canonical key, define one-sided resource simulation

```text
u <=sim v
    iff rho(u) <= rho(v)
    and, for every suffix K feasible from v,
        K is feasible from u,
        Sem(u + K) ~ Sem(v + K), and
        rho(u + K) <= rho(v + K).
```

## Amended Pareto proposition

If `kappa(u) = kappa(v)`, `rho(u) <= rho(v)`, and every resource coordinate is
extension-monotone under a common legal suffix, then `u <=sim v`. Therefore
every certified solution continuation from `v` is also feasible from `u` and
has no worse resources. The dominated record `v` can be discarded without
losing a feasible or Pareto-superior solution. Resource-incomparable records
must remain in the archive antichain.

For the current implementation, gate count, T count, and two-qubit count are
additive. Per-wire depths use the same monotone layer update on both records,
so componentwise depth order is preserved. The exhaustive tests in
`tests/test_continuation_resource_simulation.py` exercise these assumptions.

Any future constraint that violates this implication must either enter the
static continuation interface or be excluded from dominance pruning until a
new simulation theorem and tests are supplied.
