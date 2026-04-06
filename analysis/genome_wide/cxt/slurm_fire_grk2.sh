#!/bin/bash
#SBATCH --job-name=fire_grk2
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/fire_grk2_%A_%a.log
#SBATCH --array=0-1

set -euo pipefail

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12

mkdir -p ${BASE}/logs ${BASE}/results/fire

POPS=(SAS YRI)
POP=${POPS[$SLURM_ARRAY_TASK_ID]}

echo "=== Fire plot inference: ${POP} ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

${PYTHON} ${BASE}/run_fire_grk2.py \
    --pop ${POP} \
    --n-pairs 0 \
    --outdir ${BASE}/results/fire

echo "=== Done: ${POP} ==="
