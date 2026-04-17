#!/bin/bash
#SBATCH --job-name=akbari-tmrca
#SBATCH --partition=b200-mig45
#SBATCH --gres=gpu:45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=analysis/akbari_479_tmrca/logs/chr%a_%j.log

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/${PIXI_ENV}"
export PYTHONPATH="$(pwd)/python:$(pwd)"
export LD_LIBRARY_PATH="$(pwd)/python/tmrca_cu:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

CHR=${SLURM_ARRAY_TASK_ID}

echo "=== Job ${SLURM_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}: chr${CHR} ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

python analysis/akbari_479_tmrca/infer_akbari_windows.py --chr ${CHR}

echo "End: $(date)"
