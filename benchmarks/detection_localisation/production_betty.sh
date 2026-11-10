#!/bin/bash
#SBATCH --job-name=detection-sim
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5632M
#SBATCH --time=00:30:00
set -euo pipefail
ROOT=/vast/projects/smathi/cohort/kkor/tmrca.cu
RUN="$ROOT/reruns/detection-localisation-20260927"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="$ROOT/gamma_smc_cu/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
export GAMMA_SMC_BIN="$ROOT/reruns/review-20260925/inputs/tools/gamma_smc/bin/gamma_smc"
export GAMMA_FLOW="$ROOT/reruns/review-20260925/source-runtime-v3/python/gamma_smc_cu/default_flow_field.txt"
"$ROOT/gamma_smc_cu/.pixi/envs/default/bin/python" "$RUN/production/source/task.py" --root "$RUN/production" --index "$SLURM_ARRAY_TASK_ID" --mode both --slim "$RUN/tools/slim43/slim"
