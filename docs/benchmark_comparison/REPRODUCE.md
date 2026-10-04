# Reproduce the external comparison

The proposed synthesizer is unchanged. Run from the repository root with Python
3.13, the pinned reproduction requirements, and one numerical thread. Historical
experiments are read-only. Use a NEW output directory for a fresh run.

## Independent archived replay (no external compiler required)

```bash
python -m pip install -r publication_native/requirements-reproduction.txt
python -m pip install -e '.[dev]'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0 python -m pytest -q
python -m hybrid_qcs.cleanup.benchmark_comparison --verify \
  --output-dir experiments/benchmark_comparison_v1
python -m hybrid_qcs.cleanup.exact_t_comparison --verify \
  --output-dir experiments/exact_t_comparison_v1
```

The first command collects 1,341 tests, including the 1,272 preserved tests.
The added tests are comparison/circuit-generation tests, not newly trained RL
experiments. The stored local and remote logs establish actual execution.

## Build exact upstream algorithms

```bash
git clone --recursive https://github.com/lsils/mockturtle.git /tmp/mockturtle
git -C /tmp/mockturtle checkout --detach 47d1e70fdf775e1a295016c3c17a1ad206db24c0
git -C /tmp/mockturtle submodule update --init --recursive
bash tools/benchmark_baselines/build_mockturtle.sh /tmp/mockturtle /tmp/mockturtle-driver

git clone https://github.com/Nozidoali/exact-t-map.git /tmp/exact-t-map
git -C /tmp/exact-t-map checkout --detach bffe54c38b6bfd689a04e0d7d5afdbd949bdac3a
cmake -S /tmp/exact-t-map -B /tmp/exact-t-map/build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/exact-t-map/build -j2
```

Only the calling adapter is ours. The authors' library/mapper sources are not
modified or copied into this repository. Builds and external tool initialization
are reported separately from per-target runtime. The GitHub preparation workflow
provides a pinned artifact containing these sources for this recorded study.

## Fresh experiments

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0
python -m hybrid_qcs.cleanup.benchmark_comparison \
  --mockturtle /tmp/mockturtle-driver --repeats 3 \
  --output-dir outputs/fresh/benchmark_comparison_v1
python -m hybrid_qcs.cleanup.exact_t_comparison \
  --binary /tmp/exact-t-map/build/exact-t-synth \
  --output-dir outputs/fresh/exact_t_comparison_v1
python tools/benchmark_baselines/report.py --experiments outputs/fresh \
  --output outputs/fresh/report
```

The measured comparison has 246 jobs over 41 labelled target functions; the
Exact-T coherent-wrapper diagnostic has 15 jobs on the five original functions.
Every final oracle, optimized classical network, mapper QASM, and failure record
is saved. There is no external-auditor repair of failed synthesis. A missing
output is not an infeasibility proof. Exact replay has a declared sparse-column
support limit and fails instead of silently switching to sampled verification.

The benchmark runner refuses overwrites. `--resume` is allowed only with the
identical manifest and driver binary, and executes only missing job keys.
Resource comparisons are target-paired. Deterministic resource outcomes are
checked across repetitions; differing outcomes require an explicit distribution
analysis instead of selecting a favorable repetition.

## Control development and interpretation

The proposed method and target definitions remained fixed while the external
control was strengthened from four-input cuts to five-input cuts, then supplied
with a factored Davio source and a direct final-phase conversion. Initial pilot
sources and results are retained in the downloadable audit package. Pilot and
final rows are not pooled. This is a developmental benchmark audit, not an
independent preregistered RL confirmation experiment.

Read REPORT.md before quoting the Exact-T comparison: it includes a full coherent
inverse to convert the emitted evaluator to our clean phase-oracle contract.
The resulting counts must not be advertised as the best achievable counts of
measurement-enabled Exact-T Library compilation. Likewise the XAG control is
not the complete caterpillar or a globally optimal space-aware schedule.
