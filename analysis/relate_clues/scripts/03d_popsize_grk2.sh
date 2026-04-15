#!/bin/bash
#SBATCH --job-name=rc-popsize-grk2
#SBATCH --partition=genoa-lrg-mem
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=analysis/relate_clues/logs/03d_popsize_grk2_%j.log

# Step 3: PopSize for chr11 (GRK2) only.

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4

TREEDIR=${BASE}/trees/chr11_EURSAS
INPUT_PREFIX=${TREEDIR}/chr11_relate
OUTPUT_PREFIX=${TREEDIR}/chr11_popsize
POPLABELS=${BASE}/data/chr11/chr11_eursas.poplabels

echo "=== Step 3: EstimatePopulationSize (GRK2, chr11, GIH) ==="
echo "Node: $(hostname), Start: $(date)"

${RELATE}/scripts/EstimatePopulationSize/EstimatePopulationSize.sh \
    -i ${INPUT_PREFIX} \
    -o ${OUTPUT_PREFIX} \
    -m 1.25e-8 \
    --poplabels ${POPLABELS} \
    --pop_of_interest GIH \
    --num_iter 5 --threads 16 \
    --years_per_gen 28 --seed 42 || true

echo "Output:"
ls -lh ${OUTPUT_PREFIX}.*
echo "=== PopSize done at $(date) ==="
