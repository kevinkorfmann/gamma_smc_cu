#!/bin/bash
#SBATCH --job-name=tmrca-repro
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:90gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --array=21-22
#SBATCH --output=analysis/genome_wide/reproducibility/logs/chr%a_%j.log

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/${PIXI_ENV}"
export PYTHONPATH="$(pwd)/python:$(pwd)"
export LD_LIBRARY_PATH="$(pwd)/python/tmrca_cu:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

CHR=${SLURM_ARRAY_TASK_ID}

echo "=== Repro job ${SLURM_JOB_ID}, chr${CHR} ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

python analysis/genome_wide/reproducibility/infer_chromosome_repro.py \
    --chr ${CHR} --populations YRI CEU CHB

echo "=== chr${CHR} repro finished at $(date) ==="
