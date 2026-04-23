#!/bin/bash
#SBATCH --job-name=v41-bench
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:B200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=benchmarks/pairwise_scaling/bench_%j.log

set -uo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/${PIXI_ENV}"
export PYTHONPATH="$(pwd)/python:$(pwd)"
NVCUDA_LIB="$(pwd)/${PIXI_ENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
export LD_LIBRARY_PATH="${NVCUDA_LIB}:$(pwd)/python/gamma_smc_cu:$(pwd)/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

echo "=== Pairwise scaling benchmark ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

python benchmarks/pairwise_scaling/run_benchmark.py

echo "=== Done at $(date) ==="
