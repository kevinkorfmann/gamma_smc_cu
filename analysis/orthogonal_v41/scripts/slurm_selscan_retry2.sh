#!/bin/bash
#SBATCH --job-name=v41-selscan-retry
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=analysis/orthogonal_v41/logs/selscan_retry_%A_%a.log

set -uo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/${PIXI_ENV}"
export PYTHONPATH="$(pwd)/python:$(pwd)"

TASK_LIST=analysis/orthogonal_v41/scripts/selscan_retry2_tasks.txt
TASK=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_LIST")
if [ -z "$TASK" ]; then
    echo "No task at index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

CHR=$(echo "$TASK" | awk '{print $1}')
POP=$(echo "$TASK" | awk '{print $2}')

echo "=== RETRY job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}: chr${CHR} ${POP} ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

rm -rf "analysis/orthogonal_v41/selscan/chr${CHR}_${POP}"
python analysis/orthogonal_v41/scripts/run_selscan_task.py \
    --chr "$CHR" --pop "$POP" --threads 4

echo "=== Done at $(date) ==="
