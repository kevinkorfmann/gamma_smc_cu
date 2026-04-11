#!/bin/bash
#SBATCH --job-name=v41-asmc
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --array=1-30
#SBATCH --output=analysis/orthogonal_v41/logs/asmc_%A_%a.log

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export PYTHONPATH="$(pwd)/python:$(pwd)"

NVCUDA_LIB="$(pwd)/${PIXI_ENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
export LD_LIBRARY_PATH="${NVCUDA_LIB}:${LD_LIBRARY_PATH:-}"

TASK_LIST=analysis/orthogonal_v41/scripts/tasks.txt
TASK=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_LIST")
GENE=$(echo "$TASK" | awk '{print $1}')
CHR=$(echo "$TASK" | awk '{print $2}')
POP=$(echo "$TASK" | awk '{print $3}')
GROUP=$(echo "$TASK" | awk '{print $4}')

echo "=== Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}: ASMC ${GENE} chr${CHR} ${POP} (${GROUP}) ==="
echo "Node: $(hostname)"
echo "Start: $(date)"

python analysis/orthogonal_v41/scripts/run_asmc_region.py \
    --gene "$GENE" --chr "$CHR" --pop "$POP" --group "$GROUP"

echo "=== Done at $(date) ==="
