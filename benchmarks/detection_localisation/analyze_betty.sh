#!/bin/bash
#SBATCH --job-name=detection-analysis
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5632M
#SBATCH --time=00:30:00
set -euo pipefail
ROOT=/vast/projects/smathi/cohort/kkor/tmrca.cu
RUN="$ROOT/reruns/detection-localisation-20260927"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="$ROOT/gamma_smc_cu/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
"$ROOT/gamma_smc_cu/.pixi/envs/default/bin/python" "$RUN/production/source/analyze.py" --root "$RUN/production" --out "$RUN/production/summary_v1"
