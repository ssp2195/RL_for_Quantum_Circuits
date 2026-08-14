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
