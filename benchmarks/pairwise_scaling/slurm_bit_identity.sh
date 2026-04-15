#!/bin/bash
#SBATCH --job-name=bit-test
#SBATCH --partition=dgx-b200
#SBATCH --gres=gpu:B200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/pairwise_scaling/bit_identity_%j.log

set -euo pipefail

REPO=/vast/projects/smathi/cohort/kkor/tmrca.cu
cd $REPO
PIXI_ENV="${REPO}/.pixi/envs/default"
export PATH="${PIXI_ENV}/bin:${PATH}"
export CONDA_PREFIX="${PIXI_ENV}"
export CUDACXX="${PIXI_ENV}/bin/nvcc"
export CUDAHOSTCXX="${PIXI_ENV}/bin/x86_64-conda-linux-gnu-g++"
export CC="${PIXI_ENV}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${PIXI_ENV}/bin/x86_64-conda-linux-gnu-g++"
export CMAKE_PREFIX_PATH="${PIXI_ENV}"
NVCUDA_LIB="${PIXI_ENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
export CUDA_VISIBLE_DEVICES=0

OUT=/tmp/bit_identity

# ── Step 1: Build MAIN branch in temp dir ──
echo "=== Step 1: Build MAIN branch ==="
MAIN_DIR=/tmp/tmrca_main_src
rm -rf $MAIN_DIR
mkdir -p $MAIN_DIR

# Extract main branch source tree
git archive main | tar -x -C $MAIN_DIR
cd $MAIN_DIR

# Build using the perf branch's pixi env (same compiler, same deps)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CUDA_HOST_COMPILER="${CUDAHOSTCXX}" 2>&1 | tail -3
cmake --build build -j8 2>&1 | tail -5

echo "Main build done."

# Run inference with main build
export PYTHONPATH="${MAIN_DIR}/python:${MAIN_DIR}"
export LD_LIBRARY_PATH="${NVCUDA_LIB}:${MAIN_DIR}/build:${PIXI_ENV}/lib:${LD_LIBRARY_PATH:-}"
# Copy .so to the python package dir
cp build/_core*.so python/tmrca_cu/
cp build/libtmrcacu_kernels.so python/tmrca_cu/

python ${REPO}/benchmarks/pairwise_scaling/test_bit_identity.py --save ${OUT}_main.npy
cd $REPO

# ── Step 2: Build PERF branch (current) ──
echo ""
echo "=== Step 2: Build PERF branch ==="
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CUDA_HOST_COMPILER="${CUDAHOSTCXX}" 2>&1 | tail -3
cmake --build build -j8 2>&1 | tail -5
cp build/_core*.so python/tmrca_cu/
cp build/libtmrcacu_kernels.so python/tmrca_cu/

echo "Perf build done."

export PYTHONPATH="${REPO}/python:${REPO}"
export LD_LIBRARY_PATH="${NVCUDA_LIB}:${REPO}/python/tmrca_cu:${PIXI_ENV}/lib:${LD_LIBRARY_PATH:-}"
python benchmarks/pairwise_scaling/test_bit_identity.py --save ${OUT}_perf.npy --compare ${OUT}_main.npy

# ── Cleanup ──
rm -rf $MAIN_DIR
