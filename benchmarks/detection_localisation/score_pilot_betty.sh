#!/bin/bash
#SBATCH --job-name=detection-score
#SBATCH --partition=b200-mig45
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00
set -euo pipefail
ROOT=/vast/projects/smathi/cohort/kkor/tmrca.cu
RUN="$ROOT/reruns/detection-localisation-20260927"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="$ROOT/gamma_smc_cu/.pixi/envs/default/lib:$ROOT/reruns/review-20260925/source-runtime-v3/python/gamma_smc_cu:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOT/reruns/review-20260925/source-runtime-v3/python"
for REGIME in neutral incomplete near_fixed completed older_completed; do
 "$ROOT/gamma_smc_cu/.pixi/envs/default/bin/python" "$RUN/source/score.py" --input "$RUN/pilot/$REGIME" --out "$RUN/pilot_scores/$REGIME" --audit
done
