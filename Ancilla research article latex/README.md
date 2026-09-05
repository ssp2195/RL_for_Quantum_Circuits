# Ancilla-aware hierarchical Clifford+T synthesis article

This directory is a complete LaTeX project for the article

> **Hierarchical Learning-Guided Exact Clifford+T Circuit Synthesis under Clean-Ancilla Isometry Contracts**

The article is based on the supplied hierarchical article and Beamer projects, but presents the ancilla-aware framework as a standalone research contribution. It includes the exact physical-register representation, clean and borrowed workspace contracts, contract-relative isometry certification, strengthened full-unitary canonicalization, resource-Pareto pruning, outer linear SARSA, role-aware inner LinUCB, low-cost staged training, regression results, and the one-clean-ancilla QFT-3 qualification.

## Build

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Clean generated files with:

```bash
latexmk -C
```

## Project structure

```text
main.tex                 article entry point
preamble.tex             notation, packages, and TikZ styles
references.bib           bibliography
sections/                modular manuscript sections
data/                    frozen numerical results and configuration
figures/                 article figures and experiment plots
scripts/                 figure-regeneration support for legacy plots
source/                  supplied manuscript source used for traceability
validation/              build, test, rendering, and preflight records
```

## Reproduce code evidence

From the repository root on branch `ancilla-isometry-hierarchical-v1`:

```bash
python -m pip install -e '.[dev]'
python -m compileall -q hybrid_qcs tests
python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m hybrid_qcs.ancilla_runner \
  --output-dir experiments/ancilla_isometry_20260905
```

## Claim boundary

The implemented ancilla model covers fixed, preallocated clean ancillas and optional borrowed ancillas under unitary Clifford+T evolution. It does not include measurement, reset, classical feed-forward, dynamic allocation gates, or discarded garbage channels. The QFT-3 witness is independently certified; the bounded unrestricted search probe is reported as unsuccessful.
