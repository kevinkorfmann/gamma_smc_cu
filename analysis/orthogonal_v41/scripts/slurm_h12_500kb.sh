#!/bin/bash
#SBATCH --job-name=h12-500kb
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=analysis/orthogonal_v41/logs/h12_500kb_%A_%a.log

set -uo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
export PATH="$(pwd)/.pixi/envs/default/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/.pixi/envs/default"
export PYTHONPATH="$(pwd)/python:$(pwd)"

# 66 tasks = 22 chromosomes × 3 populations (CDX, CHS, GIH)
TASK_ID=${SLURM_ARRAY_TASK_ID}
TASKS=()
for POP in CDX CHS GIH; do
  for CHR in $(seq 1 22); do
    TASKS+=("$CHR $POP")
  done
done
TASK=${TASKS[$((TASK_ID - 1))]}
CHR=$(echo "$TASK" | awk '{print $1}')
POP=$(echo "$TASK" | awk '{print $2}')

echo "=== H12 ±500kb task $TASK_ID: chr${CHR} ${POP} ==="
echo "Node: $(hostname); Start: $(date)"

python analysis/orthogonal_v41/scripts/run_h12_500kb_task.py --chr "$CHR" --pop "$POP"

echo "=== Done at $(date) ==="
