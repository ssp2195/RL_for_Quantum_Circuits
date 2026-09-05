# Validation record

## Code evidence

The implementation used by this article is on branch `ancilla-isometry-hierarchical-v1`. The final local validation completed with:

```text
47 tests passed
python -m compileall -q hybrid_qcs tests: passed
git diff --check: passed
```

The 47 tests include all 37 pre-existing regression tests and ten new tests for clean-subspace equivalence, clean-workspace leakage, borrowed-ancilla restoration, projective/exact phase modes, compute-phase-uncompute, QFT-3 witness certification, deferred role-aware features, staged training, and wider symbolic registers.

## Numerical evidence

The packaged `data/ancilla_summary.json` and `data/ancilla_evaluations.csv` are copied from the frozen ancilla qualification run. The recorded training ratio is 1.0816485, corresponding to approximately 8.2% incremental training time. The QFT-3 witness is independently certified; the unrestricted 1.5-second QFT-3 search probe is recorded as unsuccessful.

## LaTeX and PDF validation

The project was rebuilt from source with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Final checks:

- PDF compilation succeeded;
- 15 pages on US Letter paper;
- all bibliography entries and cross-references resolved;
- no overfull boxes in the final log;
- the PDF is unencrypted and contains selectable text;
- all 15 pages were rendered at 160 dpi and visually inspected;
- no clipped text, overlapping objects, broken glyphs, or black rendering boxes were observed.

Build, font, metadata, and preflight outputs are retained under `validation/`.
