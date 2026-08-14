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
