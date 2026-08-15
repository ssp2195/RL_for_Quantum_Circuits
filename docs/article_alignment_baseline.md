# Article-alignment baseline

This document freezes the pre-alignment state identified by
`baselines/article_alignment_baseline.json`.  It is a reproducibility record,
not an empirical comparison against the aligned implementation.

## Implemented semantic invariant

For the authoritative concrete witness `dag`, `CircuitState` stores an exact
symbolic summary of the form

```text
U(dag) = exp(i * phase_eighths * pi/8)
         * CliffordFrame
         * ordered PauliRotation word.
```

Appending a Clifford left-multiplies the frame.  Appending `T` or `TDG`
inserts the exactly transported signed Pauli axis and updates the integer
eighth-turn phase.  The DAG—not the symbolic summary, canonical key, reward,
or policy—is the concrete witness used by the independent dense certifier.

## Declared search model

- Logical, labelled qubits with all-to-all connectivity.
- No ancillas.
- Native grammar: `H`, `S`, `SDG`, `T`, `TDG`, and every directed `CNOT`.
- The RL action selects one persistent frontier record; deterministic expansion
  enumerates every legal one-gate continuation.
- Global phase is quotiented unless the configured certifier requires literal
  phase equality.

The Stage-3 Toffoli result is exact synthesis **inside the fixed seven-term CCZ
parity-network normal form**.  It is not unrestricted Clifford+T synthesis.
GHZ and Toffoli correctness are independent-certifier results; scheduler
expansion counts measure search efficiency only.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe ghz3_smoke.py --artifacts-dir outputs\ghz3-smoke-final
.\.venv\Scripts\python.exe ghz3_rl.py --artifacts-dir outputs\ghz3-rl-cli-check
.\.venv\Scripts\python.exe toffoli_certify.py --artifacts-dir outputs\toffoli-known
.\.venv\Scripts\python.exe toffoli_search.py --artifacts-dir outputs\toffoli-search --train --seed 23
```

Every successful runner renders its circuit from the returned certified search
witness.  Failure artifacts explicitly state that no certified witness was
returned; a reference circuit is never substituted into a failed search result.
