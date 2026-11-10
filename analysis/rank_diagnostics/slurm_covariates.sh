#!/usr/bin/env bash
#SBATCH --job-name=rank-covariates
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5632M
#SBATCH --time=01:00:00
set -euo pipefail
ROOT=/vast/projects/smathi/cohort/kkor/tmrca.cu/gamma_smc_cu
cd "$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="$ROOT/.pixi/envs/default/bin/python"
"$PY" analysis/rank_diagnostics/test_common.py
"$PY" analysis/rank_diagnostics/covariates.py \
  --input analysis/rank_diagnostics/inputs --output analysis/rank_diagnostics/outputs/summary
