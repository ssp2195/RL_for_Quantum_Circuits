#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONHASHSEED=0
output="${1:-experiments/native_publication_v1}"
python -m hybrid_qcs.native_study_runner --output-dir "$output"
test -f "$output/ALL_CAMPAIGNS_COMPLETE"
python -m hybrid_qcs.native_study_report --output-dir "$output" --publication-dir publication_native
make -C publication_native
python -m hybrid_qcs.native_study_runner --stage verify --output-dir "$output"
