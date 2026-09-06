# Hierarchical BNN-oracle synthesis article

This directory contains the editable standalone LaTeX project for the
clean-ancilla BNN verification-oracle experiment.

All architecture and circuit diagrams are written directly in TikZ. The
quantitative plots are written in PGFPlots and read the packaged CSV data in
`data/`; no externally generated image is required to compile the article.

Build from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The code, frozen experimental records, evaluator witness, native
Clifford+T lowering, and independent isometry certificate are stored in the
repository beside this project. The phase oracle follows the reversible
compute-phase-uncompute construction for the BNN verification predicate.
