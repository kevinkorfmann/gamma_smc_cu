#!/bin/bash
#SBATCH --job-name=v41-3method
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:B200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --array=1-30
#SBATCH --output=analysis/orthogonal_v41/logs/3method_%A_%a.log

set -uo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/${PIXI_ENV}"
export PYTHONPATH="$(pwd)/python:$(pwd)"
# Prepend the newer libcudart shipped by nvidia-cuda-runtime-cu12 12.8 so torch
# (built against CUDA 12.8) can find cudaGetDriverEntryPointByVersion. Without
# this, the bundled pixi libcudart 12.2 wins and torch fails to load.
NVCUDA_LIB="$(pwd)/${PIXI_ENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
export LD_LIBRARY_PATH="${NVCUDA_LIB}:$(pwd)/python/tmrca_cu:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

# Build the (gene, chr, pop, group) task list at runtime so it picks up
# the dynamic neutral controls. tasks.txt is one line per task:
#   gene  chr  pop  group
TASK_LIST=analysis/orthogonal_v41/scripts/tasks.txt

# Lines: focal_pop entries (15 genes), then YRI control entries (15 genes)
TASK=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$TASK_LIST")
if [ -z "$TASK" ]; then
    echo "No task at index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

GENE=$(echo "$TASK" | awk '{print $1}')
CHR=$(echo "$TASK" | awk '{print $2}')
POP=$(echo "$TASK" | awk '{print $3}')
GROUP=$(echo "$TASK" | awk '{print $4}')

echo "=== Job ${SLURM_JOB_ID} task ${SLURM_ARRAY_TASK_ID}: ${GENE} chr${CHR} ${POP} (${GROUP}) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

python analysis/orthogonal_v41/scripts/run_three_method.py \
    --gene "$GENE" --chr "$CHR" --pop "$POP" --group "$GROUP"

echo "=== Done at $(date) ==="
