#!/bin/bash
#SBATCH --job-name=tmrca-repro-ext
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --array=1-4
#SBATCH --output=analysis/genome_wide/reproducibility/logs/ext_%a_%j.log

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/${PIXI_ENV}"
export PYTHONPATH="$(pwd)/python:$(pwd)"
export LD_LIBRARY_PATH="$(pwd)/python/tmrca_cu:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

TASK=${SLURM_ARRAY_TASK_ID}

# Configurations:
#   1: chr15  pops=YRI CEU CHB GIH PEL LWK  pair_chunk=1000  subdir=default/chr15
#   2: chr7   pops=YRI CEU CHB GIH PEL      pair_chunk=1000  subdir=default/chr7
#   3: chr22  pops=YRI CEU CHB              pair_chunk=500   subdir=chunk500
#   4: chr22  pops=YRI CEU CHB              pair_chunk=2000  subdir=chunk2000

SCRIPT=analysis/genome_wide/reproducibility/infer_chromosome_repro.py

case "$TASK" in
  1)
    python $SCRIPT --chr 15 --populations YRI CEU CHB GIH PEL LWK \
        --pair-chunk 1000 --subdir default
    ;;
  2)
    python $SCRIPT --chr 7 --populations YRI CEU CHB GIH PEL \
        --pair-chunk 1000 --subdir default
    ;;
  3)
    python $SCRIPT --chr 22 --populations YRI CEU CHB \
        --pair-chunk 500 --subdir chunk500
    ;;
  4)
    python $SCRIPT --chr 22 --populations YRI CEU CHB \
        --pair-chunk 2000 --subdir chunk2000
    ;;
esac

echo "=== task ${TASK} finished at $(date) ==="
