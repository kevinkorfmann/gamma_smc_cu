#!/bin/bash
#SBATCH --job-name=perf-bench
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:B200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/pairwise_scaling/perf_bench_%j.log

# Benchmark only — on GPU node after build
set -euo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export PYTHONPATH="$(pwd)/python:$(pwd)"
NVCUDA_LIB="$(pwd)/${PIXI_ENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
export LD_LIBRARY_PATH="${NVCUDA_LIB}:$(pwd)/python/gamma_smc_cu:$(pwd)/${PIXI_ENV}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python benchmarks/pairwise_scaling/bench_chunking.py
