#!/bin/bash
#SBATCH --job-name=tmrca-build
#SBATCH --partition=b200-mig45
#SBATCH --gres=gpu:45gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/genome_wide/logs/build_%j.log

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

# Activate pixi env manually (no pixi binary on compute nodes)
PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/${PIXI_ENV}"
export PYTHONPATH="$(pwd)/python:$(pwd)"
export LD_LIBRARY_PATH="$(pwd)/python/gamma_smc_cu:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0
export CC="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="${CXX}"

echo "Python: $(which python)"
echo "CC: ${CC} ($(${CC} --version | head -1))"
echo "CMake: $(which cmake)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Clean previous build
rm -rf build python/gamma_smc_cu/_core.cpython-*.so python/gamma_smc_cu/libgamma_smc_cu_kernels.so

# Configure
cmake -S . -B build -G Ninja \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="${CONDA_PREFIX}" \
    -DPYTHON_EXECUTABLE="${CONDA_PREFIX}/bin/python"

# Build
cmake --build build -j

# Install shared objects
cp build/_core.cpython-*.so python/gamma_smc_cu/
cp build/libgamma_smc_cu_kernels.so python/gamma_smc_cu/

# Verify
python -c "import gamma_smc_cu; print('gamma_smc_cu imported OK')"
python -c "
import gamma_smc_cu
import numpy as np
import msprime

ts = msprime.sim_ancestry(4, sequence_length=50_000, recombination_rate=1e-8,
                          population_size=10000, random_seed=42)
ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=43)
r = gamma_smc_cu.infer(ts, mean_only=True)
print(f'Smoke test: {r[\"mean\"].shape}, min={r[\"mean\"].min():.0f}, max={r[\"mean\"].max():.0f}')
print('Build verified OK')
"
