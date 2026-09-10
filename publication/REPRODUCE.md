# Reproduction

## Environment and installation

The measured environment was Python 3.13.5, NumPy 2.3.5, pytest 9.0.2, Linux x86-64, with one OpenBLAS/OMP thread. Detailed metadata is in `experiments/publication_v1/campaign_environment.json`. Linux is the validated runtime; on Windows use WSL for these commands. The Git repository itself can be cloned on Windows. We do not claim native Windows execution has been validated: `resource` accounting is Unix-specific.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r publication/requirements-reproduction.txt
python -m pip install -e '.[dev]'
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
```

## Reproduce post-training circuit generation

To exercise the five archived frozen models:

```bash
python -m pytest -q
```

To retrain all five policies before generation:

```bash
python -m hybrid_qcs.publication_runner lock --output-dir outputs/fresh-tests
python -m hybrid_qcs.publication_runner train --output-dir outputs/fresh-tests
export QCS_PUBLICATION_MODELS=outputs/fresh-tests/checkpoints
python -m pytest -q
```

Training must finish before these tests start. Missing checkpoints fail rather than silently falling back to an untrained policy. The full suite has 481 cases; 320 are marked `generated`. These test correctness under generous generation caps, not success under the much tighter performance-campaign cap. The two purposes must not be conflated.

## Check the complete archived evidence

```bash
python -m hybrid_qcs.publication_evidence --capsule-only
python -m hybrid_qcs.publication_evidence --output-dir experiments/publication_v1
python -m hybrid_qcs.publication_analysis --output-dir experiments/publication_v1
python -m hybrid_qcs.publication_guard analyze --output-dir experiments/publication_v1
```

The first command replays 597 unique circuit/contract pairs from the deduplicated index. The second also verifies all campaign membership, discovery provenance, calibration/application witnesses, 26 closed covers and 120 rank-bound records. It writes fresh verification reports; copy the directory first to preserve original verification timing fields exactly. Original circuit and profiler records are not modified by verification.

## Re-run every stage

Use a NEW directory. No existing evidence is overwritten unless `--resume` is explicit.

```bash
./scripts/reproduce_publication.sh --output-dir outputs/new-publication-study
```

This completes staged training for five seeds; 2,625 primary jobs; the separately declared validation safeguard; separately trained feature ablations and 375 secondary jobs; constructive, rank-tight, ancilla and application experiments; independent post-audits; 1,620 fresh-orbit confirmation jobs; analysis; and independent evidence verification. The archived model/selection parameters are the declared study, not hyperparameters chosen again on test outcomes. Timings and deadline-limited incumbent success may vary with hardware.

The original validation-only learning-rate selection can be re-executed separately:

```bash
python scripts/reproduce_rate_selection.py --output outputs/new-rate-selection.json
```

It preserves the original grid and validation caps, but cannot change an existing locked campaign. A hardware-sensitive different winner belongs to a new experiment, not a retroactive replacement.

## Build the article

A standard LaTeX distribution with `pdflatex`, Latin Modern, AMS packages, `microtype`, `tabularx`, and `hyperref` is required.

```bash
python -m hybrid_qcs.publication_report
make -C publication
```

The result is `publication/main.pdf`; result tables are generated from the stored analysis, not edited by hand. The author field is deliberately anonymous pending the author's own metadata. `RELATED_WORK.md` records primary-source boundaries. `THIRD_PARTY_NOTICES.md` and `LICENSE-QISKIT.txt` accompany the adapted GraySynth implementation.

## Scientific and resource boundaries

The new phase domain emits each fixed-polynomial phase obligation once. Its T-count is fixed. It minimizes CNOT count under explicit clean-width, native-depth, native-gate and T-depth caps. A learned score or timeout is never a lower-bound proof. Bounded optimality is not unrestricted Clifford+T optimality.

The true frontier is not a beam. The 32-node panel limits scoring, not discovered-state retention. The 20,000-record performance cap is an interruption condition, not a hard 100 MB memory guarantee. Deadlines are cooperative per-operation checks, and actual measured time includes certification and overrun. The benchmark reports process high-water RSS, not a fictitious per-run allocation counter.

The trained hierarchy loses the primary and fresh-confirmation comparison to the untrained prior. The validation-selected residual coefficient is zero. CI must check correctness and provenance, not enforce a fictitious positive performance assertion. CI intentionally does not rerun the whole timing study on every push.
