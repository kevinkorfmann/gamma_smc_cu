#!/bin/bash
#SBATCH --job-name=tmrca-repro
#SBATCH --partition=dgx-b200
#SBATCH --qos=dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=analysis/genome_wide/reproducibility/logs/fast_%j.log

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/${PIXI_ENV}"
export PYTHONPATH="$(pwd)/python:$(pwd)"
export LD_LIBRARY_PATH="$(pwd)/python/tmrca_cu:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

SCRIPT=analysis/genome_wide/reproducibility/infer_chromosome_repro.py

echo "=== $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
echo "Start: $(date)"

# B. Extended coverage: chr15 with 4 diverse pops (was 6 — drop LWK, GIH to fit 30-min slot)
python $SCRIPT --chr 15 --populations YRI CEU CHB PEL --pair-chunk 1000 --subdir default

# C. Chunk-size invariance on chr22
python $SCRIPT --chr 22 --populations YRI CEU CHB --pair-chunk 500  --subdir chunk500
python $SCRIPT --chr 22 --populations YRI CEU CHB --pair-chunk 2000 --subdir chunk2000

echo "=== all done at $(date) ==="
