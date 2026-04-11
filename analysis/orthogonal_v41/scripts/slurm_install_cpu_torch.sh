#!/bin/bash
#SBATCH --job-name=v41-cputorch
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=analysis/orthogonal_v41/logs/install_cputorch_%j.log

set -uo pipefail
set -x

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/${PIXI_ENV}"

# Install CPU-only torch into ~/.local so it doesn't touch the pixi env
python -m pip install --user --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.0,<3" 2>&1 | tail -20

# Verify
PYTHONPATH=/vast/home/k/korfmann/.local/lib/python3.12/site-packages \
    python -c "
import torch
print('torch version:', torch.__version__)
print('torch.cuda.is_available():', torch.cuda.is_available())
print('torch from:', torch.__file__)
"

# Now verify cxt imports under that PYTHONPATH
PYTHONPATH=/vast/home/k/korfmann/.local/lib/python3.12/site-packages:$(pwd)/python:$(pwd) \
    python -c "
import cxt
print('cxt OK from:', cxt.__file__)
"

echo "=== install complete ==="
