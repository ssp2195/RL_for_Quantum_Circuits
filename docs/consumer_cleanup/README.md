# Use the theorem where it helps; avoid materializing values the oracle does not need

This extension is based on `theorem-cleanup-sarsa-linucb-v1` at
`84c55c94f4b4c33ca55cbdbb9c7738222d6b32b8`. All its existing code and evidence
are preserved. The new functions are explicit alternatives, not a silent change
to historical experiments.

## Complete oracle synthesis

```python
from hybrid_qcs.cleanup.benchmarks import candidate_problems
from hybrid_qcs.cleanup.consumer_search import synthesize_oracle

problem = candidate_problems()[3]
result = synthesize_oracle(problem)
print(result['selected']['verification']['resources'])
print(result['pareto_methods'])
```

The source problem supplies the logical phase and resource limits. This function
explicitly permits a different complete implementation: **it does not require the
original full product bank**. Its result is an upper bound, not a global optimum.
The default order minimizes T-count first, then auxiliary peak, modelled latency,
native gates, and CNOTs. Change `objective` to change that declared order:

```python
fast = synthesize_oracle(problem, objective=('worst_case_ticks', 't_count', 'peak_aux'))
```

No weighted heuristic combines these resources. Every complete construction is
verified. Nondominated alternatives and unsuccessful attempts are retained.
If a finite candidate search is interrupted, `menu_complete` is false; absence
of a circuit is never reinterpreted as unrestricted infeasibility.

## A full bank that is genuinely required

```python
from hybrid_qcs.cleanup.consumer_search import optimize_materialization

result = optimize_materialization(problem)
```

This directly constructs the known bound-attaining helper layout and verifies
it. The scoped workspace proof is closed without invoking a learner. A failure
under another cap is unknown, not a proof of infeasibility. With a frozen model,
`search_on_failure=True` searches for a different schedule, preserving larger
helper layouts where allowed. The old `optimize_cleanup` remains available for
unchanged historical reproduction.

## Keep SARSA and LinUCB for the still-undecided schedules

```python
from hybrid_qcs.cleanup.training import train_cleanup

policy, training = train_cleanup(11, stages=(64, 96, 24))
result = synthesize_oracle(problem, policy, include_learned=True)
print(result['selected_is_learned'])
```

Both stages finish before this call. SARSA still selects frontier records;
LinUCB still ranks eligible continuations. The model is frozen during generation.
The analytic helper choice, Boolean identities and exact schedule proof are NOT
learned decisions. The optional learned candidate is fully costed and retains
its own provenance. A deterministic winner is not counted as an RL discovery.

## Direct mathematical constructions

`compile_factors(problem, storage='all'|'stream'|'recompute')` returns a complete
protocol and a proof receipt. `compress_quadratics=False` keeps the uncompressed
monomial-feature control. `minimum_live_order` uses the proved subset recurrence
up to 16 factors. Larger problems keep a canonical-order upper bound, clearly
labelled; raising a limit does not make exponential search polynomial.

`compose_blocks` reuses clean workspace between verified blocks. Original inputs
may overlap between blocks, but unclean auxiliaries and internal transcripts may
not cross a block boundary.

`verify_phase` checks exact all-input/all-outcome semantics without invoking any
compiler, scheduler or policy. `verify_factor_receipt` checks factorization and
count claims. `verify_live_certificate` checks every edge of the subset recurrence.
Coherent instructions continue to have local persistent-DAG/Clifford-tableau/
ordered-Pauli-rotation/global-phase witnesses. The enclosing protocol represents
the measurements, reset and classical control explicitly.

## Reproduce

```bash
python -m pip install -r publication_native/requirements-reproduction.txt
python -m pip install -e '.[dev]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0 \
  python -m hybrid_qcs.cleanup.consumer_runner --output-dir outputs/new-study
python -m hybrid_qcs.cleanup.consumer_diagnostics --output-dir outputs/new-study
python -m hybrid_qcs.cleanup.consumer_report --output-dir outputs/new-study \
  --report-dir outputs/new-report
python -m hybrid_qcs.cleanup.consumer_runner --stage verify --output-dir outputs/new-study
```

Use `--smoke` for one seed, 24 training episodes and one repetition. A smoke run
is not the full five-seed campaign. The committed primary campaign uses five
184-episode training runs, three repetitions, five original target problems,
and 11 comparison methods: 345 records. All failures would be preserved.

The test suite preserves the 1,074 original cases and adds 198 cases. Of the new
cases, 136 perform actual frozen-SARSA/LinUCB generation and compare it with
independent analytic circuit constructions. Training schedules used in regression
fixtures are shorter than the full performance study and do not prove convergence.

The proofs do not establish global circuit optimality, learning superiority,
state-of-the-art compiler superiority, hardware advantage or publication novelty.
The report states which physical resources improve and which latency costs worsen.
