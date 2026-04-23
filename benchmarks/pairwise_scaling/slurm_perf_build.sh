#!/bin/bash
#SBATCH --job-name=perf-build
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=benchmarks/pairwise_scaling/perf_build_%j.log

# Build only — on CPU node where nvcc/cmake are available
set -euo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/${PIXI_ENV}"
export CMAKE_PREFIX_PATH="$(pwd)/${PIXI_ENV}"
export CUDACXX="$(pwd)/${PIXI_ENV}/bin/nvcc"
export CUDAHOSTCXX="$(pwd)/${PIXI_ENV}/bin/x86_64-conda-linux-gnu-g++"
export CC="$(pwd)/${PIXI_ENV}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$(pwd)/${PIXI_ENV}/bin/x86_64-conda-linux-gnu-g++"
export LD_LIBRARY_PATH="$(pwd)/${PIXI_ENV}/lib:${LD_LIBRARY_PATH:-}"

echo "=== Build with fast_math + pinned memory ==="
echo "nvcc: $(which nvcc)"
echo "g++: $CXX"
echo "cmake: $(which cmake)"
rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" 2>&1 | tail -10
cmake --build build -j8 2>&1 | tail -10
cp build/_core*.so python/gamma_smc_cu/
cp build/libgamma_smc_cu_kernels.so python/gamma_smc_cu/
echo "=== Build done ==="
