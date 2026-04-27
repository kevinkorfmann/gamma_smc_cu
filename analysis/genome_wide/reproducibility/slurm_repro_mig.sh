#!/bin/bash
#SBATCH --job-name=tmrca-repro-mig
#SBATCH --partition=b200-mig90
#SBATCH --qos=mig
#SBATCH --gres=gpu:90gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=0:45:00
#SBATCH --array=3-4
#SBATCH --output=analysis/genome_wide/reproducibility/logs/mig_%a_%j.log

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
  3)
    python $SCRIPT --chr 22 --populations YRI CEU CHB \
        --pair-chunk 500 --subdir chunk500
    ;;
  4)
    python $SCRIPT --chr 22 --populations YRI CEU CHB \
        --pair-chunk 2000 --subdir chunk2000
    ;;
esac
echo "=== task ${SLURM_ARRAY_TASK_ID} finished at $(date) ==="
