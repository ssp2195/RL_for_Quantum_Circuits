#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
python -m hybrid_qcs.publication_reproduce "$@"
