#!/bin/bash
set -uo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export PYTHONPATH="$(pwd)/python:$(pwd)"
NVCUDA_LIB="$(pwd)/${PIXI_ENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
export LD_LIBRARY_PATH="${NVCUDA_LIB}:$(pwd)/python/gamma_smc_cu:$(pwd)/${PIXI_ENV}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
python benchmarks/pairwise_scaling/quick_test.py
