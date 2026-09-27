#!/bin/bash
#SBATCH --job-name=tmrca_suite
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=stdpopsim_%A_%a.log
# From the immutable software checkout root, set host-appropriate resource
# overrides and a NEW output root, then submit:
# sbatch --array=0-8,10-14 this-script --results-dir /path/to/new/run
# Use BENCH_PYTHON/BENCH_ENV_PREFIX/GAMMA_SMC_BIN to select installed binaries.
set -euo pipefail
BENCH_REPO_ROOT=${BENCH_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}
BENCH_ENV_PREFIX=${BENCH_ENV_PREFIX:-${BENCH_REPO_ROOT}/.pixi/envs/default}
BENCH_PYTHON=${BENCH_PYTHON:-${BENCH_ENV_PREFIX}/bin/python}
export PATH="${BENCH_ENV_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${BENCH_ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export MPLBACKEND=Agg
cd "${BENCH_REPO_ROOT}"
# Preserve scheduler-assigned CUDA_VISIBLE_DEVICES; do not change device ownership.
"${BENCH_PYTHON}" benchmarks/test_suite_stdpopsim/run_one.py \
  --config-idx "${SLURM_ARRAY_TASK_ID:?submit as an array}" "$@"
