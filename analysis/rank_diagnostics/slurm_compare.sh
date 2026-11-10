#!/usr/bin/env bash
#SBATCH --job-name=rank-compare
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5632M
#SBATCH --time=01:00:00
set -euo pipefail
ROOT=/vast/projects/smathi/cohort/kkor/tmrca.cu/gamma_smc_cu
cd "$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$ROOT/.pixi/envs/default/bin/python" analysis/rank_diagnostics/compare.py \
  --summary analysis/rank_diagnostics/outputs/summary --normalized analysis/rank_diagnostics/outputs/normalized
