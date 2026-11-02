# Generic Boolean phase synthesis (opt-in)

This branch extends the consumer-aware cleanup project without changing the
historical native unitary grammar or the default consumer-aware candidate menu.

## Exact mathematical components

- `contextual.py`: M1 causal Z/CZ correction membership and an explicit
  decomposition witness; M2 necessary bound on *additional correction ancilla
  qubits*. Neither proves globally minimal oracle ancilla qubits.
- `m3_corrections.py`: M3's declared multistage correction-subsystem family.
  Original garbage creation/measurement is **excluded** from these counts.
- `general_boolean.py`: arbitrary Boolean predicates represented by their
  exact algebraic normal form (ANF). Each nonlinear monomial has two legal
  options: temporary four-T ANDs + CZ + measured Clifford cleanup, or one fewer
  temporary AND followed by an exact seven-T CCZ phase. The latter can reduce
  peak ancilla qubits at increased T-count.
- `generic_search.py`: persistent frontier of exact macro continuations;
  coherent operations retain native DAG/tableau/Pauli-rotation witnesses.
  Outer SARSA selects a frontier record; inner disjoint LinUCB ranks eligible
  implementations. All feasible continuations remain available. Certification,
  resource caps, ANF identities and pruning are deterministic.

The general candidate menu is exhaustive only for the admitted monomial macro
choices and within an explicit nonlinear-term cap; it is **not** globally
optimal over all Clifford+T and adaptive circuits. Rank-factor compression in
the existing consumer backend can substantially outperform per-monomial ANF
construction in T-count and CNOTs.

## Commands

```bash
python -m pytest -q tests/test_generic_phase_cleanup.py
python -m pytest -q -m 'not generated' tests/test_consumer_cleanup.py tests/test_qft2_toffoli.py tests/test_qft3_guided.py tests/test_oracle_synthesis.py
python -m hybrid_qcs.cleanup.generic_runner --output outputs/generic_comparison.json
```

For structured consumer problems, call
`synthesize_oracle(problem, include_generic=True, objective=('peak_aux','t_count','cnot'))`
to retain all previous candidate constructions alongside both general ANF
endpoints. The original behavior is unchanged when `include_generic=False`.

For arbitrary functions, use `BooleanPhaseProblem.from_truth_table`,
`compile_boolean_phase`, `optimize_boolean_phase` or `GenericFrontierSearch`.
The independent `verify_phase` enforces the complete phase on all logical
inputs and all measurement outcomes, including reset and coherent ancilla
cleanup.

The 3-qubit Toffoli, QFT-3 and 4-wire SWAP tests are **non-diagonal unitary
controls**, not generic Boolean sign oracles; they use the project's existing
exact native witnesses. No performance improvement for those controls is
claimed from the generic phase construction.

## Limits

ANF may have exponentially many terms. Causal rank computations are exact but
can become expensive at large Boolean widths. The finite-menu optimizer has an
explicit maximum number of nonlinear terms; exceeding it raises, rather than
silently returning an incomplete optimum. Resource timings are modelled
all-to-all gate ticks, not physical device latency.

Existing tests and historical study code are preserved. See the GitHub Actions
workflow `generic-phase-ci.yml` for the exact reproduction commands and
downloadable JSON comparison artifact.
