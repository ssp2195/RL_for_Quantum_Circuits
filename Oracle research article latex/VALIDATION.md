# Validation record

The complete source revision was validated before publication:

- `python -m compileall -q hybrid_qcs tests`: passed;
- `python -m pytest -q`: 59 tests passed;
- `git diff --check`: passed;
- bounded hierarchical oracle generation: certified;
- exact logical-isometry error: approximately `1.22e-15`;
- clean-workspace leakage: approximately `3.26e-32`;
- clean LaTeX rebuild: passed;
- article length in the validated local build: 17 pages;
- undefined citations/references: none;
- overfull boxes: none;
- rendered-page visual inspection: passed.

The repository tracks the editable article source, TikZ/PGFPlots definitions,
and frozen CSV/JSON data. The compiled PDF is distributed in the external
release bundle rather than duplicated in Git.
