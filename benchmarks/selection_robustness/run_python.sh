#!/usr/bin/env bash
set -euo pipefail
# The study/ directory is deployed inside a self-contained Sesame run root.
STUDY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$STUDY_ROOT/tools/gamma_smc_cu/python"
export LD_LIBRARY_PATH="$STUDY_ROOT/tools/gamma_smc_cu/python/gamma_smc_cu:/usr/local/cuda-12.1/lib64"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "$STUDY_ROOT/env/bin/python" "$@"
