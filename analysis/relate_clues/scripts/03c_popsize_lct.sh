#!/bin/bash
#SBATCH --job-name=rc-popsize-lct
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=analysis/relate_clues/logs/03c_popsize_lct_%j.log

# Step 3: PopSize for chr2 (LCT) only — chr2 Relate trees are done.

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4

TREEDIR=${BASE}/trees/chr2_EUR
INPUT_PREFIX=${TREEDIR}/chr2_relate
OUTPUT_PREFIX=${TREEDIR}/chr2_popsize
POPLABELS=${BASE}/data/chr2/chr2_eur.poplabels

echo "=== Step 3: EstimatePopulationSize (LCT, chr2, CEU) ==="
echo "Node: $(hostname), Start: $(date)"
echo "  Trees: ${INPUT_PREFIX}"
echo "  Poplabels: ${POPLABELS}"

${RELATE}/scripts/EstimatePopulationSize/EstimatePopulationSize.sh \
    -i ${INPUT_PREFIX} \
    -o ${OUTPUT_PREFIX} \
    -m 1.25e-8 \
    --poplabels ${POPLABELS} \
    --pop_of_interest CEU \
    --num_iter 5 --threads 16 \
    --years_per_gen 28 --seed 42

echo "  Output:"
ls -lh ${OUTPUT_PREFIX}.*
echo "=== PopSize done at $(date) ==="
