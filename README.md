# Native hybrid quantum-circuit synthesis

The active native frontier is

`v = (persistent circuit DAG, Clifford tableau, ordered signed Pauli rotations, global phase, consumed resources)`.

It permits resource-feasible H, S, S-dagger, T, T-dagger, and CNOT extensions.
Phase-polynomial inputs in archived benchmarks only define logical target
matrices; they do not constrain the native rotation word or phase-emission order.

## Current publication study

The current native manuscript is [`publication_native/main.tex`](publication_native/main.tex).
Its compiled PDF and numerical tables are generated from one locked campaign.
The previous [`publication/`](publication/) manuscript and
[`experiments/publication_v1/`](experiments/publication_v1/) describe the HISTORICAL
fixed-phase-polynomial study and are not evidence for the native controller.
The preceding root README is preserved at
[`docs/HISTORICAL_ROOT_README_9f811ca.md`](docs/HISTORICAL_ROOT_README_9f811ca.md).

The native study adds an exact cyclotomic verifier, a disjoint intermediate-
difficulty unitary corpus, validation-only policy selection, equal-edge and
separate equal-wall experiments, untrained/one-level/deterministic controls,
independently retrained ablations, bounded-continuation diagnostics, clean-width
studies, and provenance-separated exact resource proofs. It does not replace the
hybrid state, introduce a neural network, or inject constructive reference paths
into discovery. Numerical results and limitations belong to the generated
[`experiments/native_publication_v1/RESULTS.md`](experiments/native_publication_v1/RESULTS.md),
not stale numbers copied from an earlier campaign.

```bash
python -m pip install -r publication_native/requirements-reproduction.txt
python -m pip install -e '.[dev]'
python -m pytest -q
# Fresh native training, all campaigns, independent replay and analysis:
bash scripts/reproduce_native_publication.sh
# Independent replay of an existing archive without retraining:
python -m hybrid_qcs.native_study_runner --stage verify --output-dir experiments/native_publication_v1
```

Use a new output directory for a new source version. Resume refuses a changed
source or target protocol. Fitting (including ablations and selection) is capped
at 1,800 aggregate CPU seconds. This is a training cap, not a promised total
campaign duration. All search time/record limits are cooperative, and late
witnesses are not counted as timely successes.

## Synthesis and resource optimization API

```python
from hybrid_qcs.native_search import NativeSearch
from hybrid_qcs.native_optimize import optimize_native_resources, verify_native_optimization
# NativeProblem supplies a logical unitary and an AncillaContract.
# A completed frozen native checkpoint is required for learned discovery.
result = optimize_native_resources(problem, model,
    objectives=('t_count', 'cnot', 'depth', 'gates'))
checked = verify_native_optimization(problem, result)
```

A correct circuit is an upper bound. Optimality additionally requires checked
native exclusions (or the zero-cost nonnegativity bound). Numerical-domain
certificates are not automatically exact algebraic proofs. For eligible exact
target descriptors, `native_exact.verify_exact_optimization` actually replays
and rechecks the receipt over the cyclotomic ring; its trusted symbolic algebra
is identified explicitly.

See [`docs/native_hybrid_restoration.md`](docs/native_hybrid_restoration.md),
[`docs/native_optimality.md`](docs/native_optimality.md), and
[`docs/native_publication_protocol.md`](docs/native_publication_protocol.md).

## Scientific boundaries

Full exact-phase QFT-3 and four-logical-qubit Toffoli have native determinant
obstructions without ancillas. `native_exact.determinant_obstruction` and its
checker report those known exact-domain facts separately. Checked one-clean-wire
references establish minimum clean width for those ideal targets; they are not
RL discoveries, gate-optimality proofs, or new circuit identities.

The full frontier remains available outside the scored panel. Dense scoring
isometries and a derived Clifford-column scalar cache still limit the code to
small registers. A successful regression suite does not establish learned
superiority, broad scalability, or publication acceptance. All unfavorable,
unresolved and audit-assisted outcomes remain in the evidence.
