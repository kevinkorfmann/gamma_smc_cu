#!/bin/bash
#SBATCH --job-name=tmrca-repro-dgx
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=1:30:00
#SBATCH --array=1-2
#SBATCH --output=analysis/genome_wide/reproducibility/logs/dgx_%a_%j.log

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/${PIXI_ENV}"
export PYTHONPATH="$(pwd)/python:$(pwd)"
export LD_LIBRARY_PATH="$(pwd)/python/tmrca_cu:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

SCRIPT=analysis/genome_wide/reproducibility/infer_chromosome_repro.py

echo "=== task ${SLURM_ARRAY_TASK_ID} on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="

case "${SLURM_ARRAY_TASK_ID}" in
  1)
    python $SCRIPT --chr 15 --populations YRI CEU CHB GIH PEL LWK \
        --pair-chunk 1000 --subdir default
    ;;
  2)
    python $SCRIPT --chr 7 --populations YRI CEU CHB GIH PEL \
        --pair-chunk 1000 --subdir default
    ;;
esac
echo "=== task ${SLURM_ARRAY_TASK_ID} finished at $(date) ==="
