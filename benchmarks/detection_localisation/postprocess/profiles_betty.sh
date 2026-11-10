#!/bin/bash
#SBATCH --job-name=sweep-profiles
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5632M
#SBATCH --time=00:20:00
set -euo pipefail
ROOT=/vast/projects/smathi/cohort/kkor/tmrca.cu
RUN="$ROOT/reruns/detection-localisation-20260927"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="$ROOT/gamma_smc_cu/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
ARGS=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then ARGS=(--index "$SLURM_ARRAY_TASK_ID"); fi
"$ROOT/gamma_smc_cu/.pixi/envs/default/bin/python" "$RUN/postprocess/sweep_profiles.py" --root "$RUN/production" --out "$RUN/production/profiles_v1" "${ARGS[@]}"
