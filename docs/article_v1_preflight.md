# Article V1 preflight

Recorded: 2026-08-15 (Asia/Calcutta)

## Repository state

- Branch: `frontier-rl`
- Commit: `df2ad6a036538b701cc3eb70b80f906d99d34a9a`
- Working tree: dirty; the pre-existing article/QFT alignment implementation is local and partly untracked.
- Remote relation observed before this task: local `frontier-rl` was one commit ahead of `origin/frontier-rl`.

Tracked files already modified before Article V1 work:

```text
README.md
algebra/pauli_rotation.py
algebra/tableau.py
benchmarks/__init__.py
canonical/canonicalizer.py
circuit/circuit_state.py
circuit/dag.py
config.py
env/rl_env.py
evaluate.py
pyproject.toml
reporting/__init__.py
search/archive.py
search/frontier.py
search/problems/native.py
tests/test_certification_simulator.py
tests/test_env_and_policy.py
tests/test_search_archive.py
tests/test_symbolic_state.py
train.py
```

Untracked implementation paths already present before Article V1 work:

```text
.stage3-audit-baseline/
article_benchmark.py
baselines/
benchmarks/native_corpus.py
benchmarks/qft.py
docs/
experiments/
qft_benchmark.py
reporting/qft.py
rl/article_features.py
rl/baselines.py
tests/test_article_expansion_cost.py
tests/test_article_features.py
tests/test_article_native_benchmark.py
tests/test_article_rl_baselines.py
tests/test_clifford_angle_absorption.py
tests/test_continuation_resource_simulation.py
tests/test_native_target_corpus.py
tests/test_qft_benchmark_cli.py
tests/test_qft_reference.py
tests/test_search_metrics.py
```

The `.stage2-*` and `.stage3-*` test-artifact directories reported access warnings during recursive discovery. They were not modified or removed.

## Baseline runtime

- Python: `3.12.13`
- NumPy: `2.5.2`
- Gymnasium: `1.3.0`
- pytest: `9.1.1`
- pytest-cov: `7.1.0`

Commands used:

```powershell
.\.venv\Scripts\python.exe -m compileall -q .
.\.venv\Scripts\python.exe -m pytest -q
```

Results:

- `compileall`: exit 0. It emitted only access warnings for pre-existing `.stage2-*` and `.stage3-*` test-artifact directories.
- Full tests: `168 passed in 35.94s`.

This file records the baseline only. It does not imply that the pre-existing working tree already satisfied the Article V1 profile.
