#!/bin/bash
#SBATCH --job-name=v41-variant
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/orthogonal_v41/logs/variant_%A_%a.log

set -uo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/${PIXI_ENV}"
export PYTHONPATH="$(pwd)/python:$(pwd)"

# 5 novel + 5 positive controls = 10 tasks (skip neutrals — they have no
# expected variant evidence). Each line: gene chr pop super
TASK_LIST=analysis/orthogonal_v41/scripts/variant_tasks.txt
TASK=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_LIST")
GENE=$(echo "$TASK"  | awk '{print $1}')
CHR=$(echo  "$TASK"  | awk '{print $2}')
POP=$(echo  "$TASK"  | awk '{print $3}')
SUPER=$(echo "$TASK" | awk '{print $4}')

echo "=== variant evidence ${GENE} chr${CHR} ${POP}/${SUPER} ==="

python analysis/orthogonal_v41/scripts/run_variant_evidence.py \
    --gene "$GENE" --chr "$CHR" --pop "$POP" --super "$SUPER"

echo "=== Done at $(date) ==="
