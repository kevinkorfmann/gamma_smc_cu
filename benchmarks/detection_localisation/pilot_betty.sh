#!/bin/bash
#SBATCH --job-name=detection-pilot
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5632M
#SBATCH --time=00:30:00
set -euo pipefail
ROOT=/vast/projects/smathi/cohort/kkor/tmrca.cu
RUN="$ROOT/reruns/detection-localisation-20260927"
export PYTHONNOUSERSITE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export LD_LIBRARY_PATH="$ROOT/gamma_smc_cu/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
REGIMES=(neutral incomplete near_fixed completed older_completed)
REGIME="${REGIMES[$SLURM_ARRAY_TASK_ID]}"
"$ROOT/gamma_smc_cu/.pixi/envs/default/bin/python" "$RUN/source/simulate.py" --regime "$REGIME" --split pilot --replicate 0 --out "$RUN/pilot/$REGIME" --slim "$RUN/tools/slim43/slim"
