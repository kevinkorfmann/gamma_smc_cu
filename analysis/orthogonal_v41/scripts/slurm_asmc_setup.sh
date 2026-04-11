#!/bin/bash
#SBATCH --job-name=v41-asmc-setup
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/orthogonal_v41/logs/asmc_setup_%j.log

set -uo pipefail
set -x

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export PYTHONPATH="$(pwd)/python:$(pwd)"

# Newer libcudart for torch (cxt importing transitively pulls torch in some
# preparedecoding paths). Not strictly required for prepare_decoding but
# harmless.
NVCUDA_LIB="$(pwd)/${PIXI_ENV}/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
export LD_LIBRARY_PATH="${NVCUDA_LIB}:${LD_LIBRARY_PATH:-}"

ASMC_DATA=/vast/projects/smathi/cohort/kkor/asmc_data
mkdir -p ${ASMC_DATA}

# 1. Download CEU.demo from PalamaraLab/ASMC_data
if [ ! -f ${ASMC_DATA}/CEU.demo ]; then
    wget -q https://raw.githubusercontent.com/PalamaraLab/ASMC_data/main/demographies/CEU.demo \
        -O ${ASMC_DATA}/CEU.demo
fi
ls -l ${ASMC_DATA}/CEU.demo
head -3 ${ASMC_DATA}/CEU.demo

# 2. Compute decoding quantities for CEU with 50 samples (matching cxt)
python - <<'PY'
import os
import sys

ASMC_DATA = "/vast/projects/smathi/cohort/kkor/asmc_data"
DEMO_FILE = os.path.join(ASMC_DATA, "CEU.demo")
OUT_PREFIX = os.path.join(ASMC_DATA, "CEU_50")

# Skip if already computed
if os.path.exists(OUT_PREFIX + ".decodingQuantities.gz"):
    print(f"already exists: {OUT_PREFIX}.decodingQuantities.gz")
    sys.exit(0)

print("Preparing decoding quantities (CEU, 50 samples, UKBB freqs)...")
from asmc.preparedecoding import prepare_decoding
dq = prepare_decoding(
    demography=DEMO_FILE,
    discretization=[[30.0, 15], [100.0, 15], 39],
    frequencies="UKBB",
    samples=50,
    mutation_rate=1.65e-8,
)
dq.save_decoding_quantities(OUT_PREFIX)
print(f"Wrote {OUT_PREFIX}.decodingQuantities.gz")
PY

ls -l ${ASMC_DATA}/CEU_50.decodingQuantities.gz
echo "=== ASMC setup done ==="
