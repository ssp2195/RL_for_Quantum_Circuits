# Theorem-guided causal cleanup search

This opt-in implementation extends the stable general Boolean phase-oracle
branch and preserves its independent exact branch verifier.

## Where theorems enter the search

* **M1 / generic ANF:** `causal_protocol.causalize_immediate_cleanup`
  tracks live Boolean ANF values through AND preparation and X measurements.
  For every measured temporary it invokes `contextual.causal_corrections`
  and emits actual measurement-conditioned Z/CZ gates. The
  `GenericFrontierSearch` uses this path by default
  (`m1_guided=True`); `m1_guided=False` is the exact historical control.
  Its outer policy remains SARSA and inner policy remains LinUCB.
* **M2 / staged cleanup:** `staged_frontier.CausalCorrectionFrontier`
  holds a persistent schedule prefix and unresolved measured cubic garbage.
  Before expanding a record it invokes `necessary_peak_ancillas` on the
  residual byproduct span and available future stages. If the bound exceeds
  the correction ancilla cap, the record is discarded **without expansion**.
  Exact GF(2) rank results are cached across equivalent suffixes.
* **M1 / staged cleanup:** For each complete schedule the solver constructs
  all guarded correction gates from stage live contexts via M1, constructs
  the complete source-garbage generation and cleanup circuit, and passes
  the resulting `PhaseProtocol` to the independent `verify_phase`.
  No correctness decision is delegated to a learned model.

The staged frontier is a separate deterministic subsearch from the general
ANF macro frontier, because the existing ANF grammar immediately corrects each
garbage measurement. The M2 residual-rank bound is **not** applied to that
immediate-correction grammar (its outstanding correction rank is zero).

## Explicit stage model

For the M3 diagnostic family, g[i,j] = x[i] y[2j] y[2j+1].
One correction stage permits one simultaneous preparation/use/X-measurement
cycle of each quadratic feature helper, with no intra-stage reuse. Separate
stages may reuse the same wires. The original q² cubic garbage measurements
are physically constructed (not declared free), including their temporary
workspace and 4T AND costs. Peak **correction helper occupancy** and total
physical peak auxiliary qubits are reported separately.

M2 is necessary, not sufficient, and the search makes no claim of a
globally minimum-cost oracle outside this precisely stated library/stage
grammar. For finite declared grammars, every possible nonempty subset of
uncorrected features fitting the stage cap is enumerated. There is no
learned or heuristically omitted action.

## Reproduction

```bash
python -m pytest -q tests/test_theorem_guided_frontier.py
python -m pytest -q
python -m hybrid_qcs.cleanup.theorem_guided_runner --output outputs/theorem_guided_results.json
```

The runner compares M1-off vs M1-on immediate cleanup on BNN robustness
oracles B1–B3; M2-off vs M2-on on finite staged correction instances;
and a held-out frozen SARSA/LinUCB evaluation. No end-to-end amplitude
amplification runtime or preparation/reflection cost is claimed.

The branch workflow archives `theorem_guided_results.json` as an artifact.
