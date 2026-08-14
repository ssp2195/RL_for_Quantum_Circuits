# RL for Quantum Circuits

This branch implements RL-guided **frontier-record selection** for small exact
Clifford+T synthesis. The policy chooses an open search record; the symbolic
engine always enumerates every legal one-gate continuation.

The exact symbolic invariant is:

```text
U = exp(i * phi * pi/8) · CliffordFrame · ordered PauliRotation word
```

`CircuitDAG` remains the authoritative witness. The dense NumPy verifier
checks a returned witness independently, up to global phase.

## Setup and verification

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e '.[dev]'
.\.venv\Scripts\python -m compileall -q .
.\.venv\Scripts\python -m pytest -q
```

The former `SanityTests.ipynb` predates the frontier/Pauli-rotation API and is
kept only as historical reference. Its executable regressions now live in
`tests/`.

## Small deterministic evaluation

```powershell
.\.venv\Scripts\python evaluate.py `
  --qubits 2 `
  --target H:0,T:1,CNOT:0-1 `
  --max-t 4 --max-depth 6 --max-gates 6 --max-steps 100
```

The simulator is deliberately small-instance only; its purpose is final
certification, not large-scale search pruning.

## GHZ-3 state-preparation smoke test

The deterministic GHZ-3 runner checks the native state-preparation witness
`H(0), CNOT(0,1), CNOT(0,2)` against the analytical
`(|000> + |111>)/sqrt(2)` state. It also asks the existing FIFO frontier
baseline to rediscover that witness under the tight three-gate resource budget.
This is a reproducible smoke test, not a trained-policy benchmark or a claim
of general state/unitary synthesis.

```powershell
.\.venv\Scripts\python ghz3_smoke.py --artifacts-dir outputs\ghz3-smoke
```

The selected directory receives JSON/CSV data, SVG probability and frontier
charts, a native-gate circuit diagram, and a Markdown summary. These files use
only the standard library and NumPy; no plotting or quantum SDK dependency is
added.

## Linear-policy training plan

The linear SARSA policy schedules persistent frontier records; it never emits
gates.  The deterministic regression in `tests/test_linear_policy_training.py`
implements the first training stage:

1. Measure a frozen zero-weight policy on a tight two-qubit curriculum target.
2. Train the real `Trainer` for five seeded exploratory episodes.
3. Freeze exploration and require the learned scheduler to recover the same
   dense-certified witness in fewer expansions.
4. Verify that tied-priority records preserve the exact object chosen by the
   policy when the Gymnasium positional action is formed.

This verifies the SARSA scheduling loop, not trained GHZ-3 synthesis. The
target-free 16-feature schema intentionally remains available for this
baseline.

## Learned GHZ-3 frontier benchmark

`ghz3_rl.py` is a separate direct-training benchmark. It adds an opt-in
three-qubit dense target context, so the linear record ranker can distinguish
the target-labelled qubits and directed CNOT prefixes. Its 60-coordinate
schema includes the original 16 resource/context features plus process and
probe progress, labelled wire/gate data, directed CNOT slots, last-operation
data, target-relative frontier context, and a bias term.

```powershell
.\.venv\Scripts\python ghz3_rl.py --artifacts-dir outputs\ghz3-rl
```

The default direct protocol trains for 50 seeded four-selection episodes with
potential-shaped rewards, then evaluates a frozen policy with `epsilon=0` and
fairness disabled. The zero-weight and learned evaluations both receive a
32-selection cap; on the calibrated fixed seed, the zero policy certifies in
25 expansions and the learned policy certifies in 3. The returned witness is
reconstructed only from the search `solution_node` and independently checked
against the full target unitary and GHZ-positive state.

The artifact directory contains a JSON summary, training history and evaluation
trace CSV files, policy metadata/weights, SVG circuit and frontier charts, and
a Markdown handoff. No reference circuit is substituted if learned search
fails: the report is unsuccessful and the diagram explicitly says that no
certified witness was returned.

This is evidence for a target-aware linear ranking policy on this labelled,
small GHZ-3 search only. It is not evidence of generalization to arbitrary
circuits, larger registers, or state-preparation targets.
