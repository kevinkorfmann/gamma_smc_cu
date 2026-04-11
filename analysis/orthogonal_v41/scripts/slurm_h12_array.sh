#!/bin/bash
#SBATCH --job-name=v41-h12
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=analysis/orthogonal_v41/logs/h12_%A_%a.log

set -uo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/${PIXI_ENV}"
export PYTHONPATH="$(pwd)/python:$(pwd)"

TASK_LIST=analysis/orthogonal_v41/scripts/selscan_tasks.txt
TASK=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_LIST")
CHR=$(echo "$TASK" | awk '{print $1}')
POP=$(echo "$TASK" | awk '{print $2}')

echo "=== H12 task ${SLURM_ARRAY_TASK_ID}: chr${CHR} ${POP} ==="
echo "Node: $(hostname); Start: $(date)"

python analysis/orthogonal_v41/scripts/run_h12_task.py --chr "$CHR" --pop "$POP"

echo "=== Done at $(date) ==="
