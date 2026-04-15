#!/bin/bash
#SBATCH --job-name=rc-popsize
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=1-2
#SBATCH --output=analysis/relate_clues/logs/03_popsize_%A_%a.log

# Step 3: Estimate population size history from Relate trees.
#
# Array task 1: chr11 EUR+SAS trees, pop_of_interest=GIH (for GRK2)
# Array task 2: chr2  EUR trees,     pop_of_interest=CEU (for LCT)
#
# This re-estimates branch lengths using MCMC, incorporating population
# size changes. Output: .coal file (coalescence rates over time) and
# updated .anc.gz/.mut.gz files.

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4

echo "=== Step 3: EstimatePopulationSize (task ${SLURM_ARRAY_TASK_ID}) ==="
echo "Node: $(hostname), Start: $(date)"

if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]; then
    CHR=11
    TREEDIR=${BASE}/trees/chr11_EURSAS
    INPUT_PREFIX=${TREEDIR}/chr11_relate
    OUTPUT_PREFIX=${TREEDIR}/chr11_popsize
    POPLABELS=${BASE}/data/chr11/chr11_eursas.poplabels
    POP="GIH"
elif [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]; then
    CHR=2
    TREEDIR=${BASE}/trees/chr2_EUR
    INPUT_PREFIX=${TREEDIR}/chr2_relate
    OUTPUT_PREFIX=${TREEDIR}/chr2_popsize
    POPLABELS=${BASE}/data/chr2/chr2_eur.poplabels
    POP="CEU"
fi

# If poplabels wasn't created by PrepareInputFiles, use the global one
if [ ! -f ${POPLABELS} ]; then
    POPLABELS=${BASE}/data/all.poplabels
fi

echo "  Trees: ${INPUT_PREFIX}"
echo "  Pop: ${POP}"
echo "  Poplabels: ${POPLABELS}"

${RELATE}/scripts/EstimatePopulationSize/EstimatePopulationSize.sh \
    -i ${INPUT_PREFIX} \
    -o ${OUTPUT_PREFIX} \
    -m 1.25e-8 \
    --poplabels ${POPLABELS} \
    --pop_of_interest ${POP} \
    --num_iter 5 --threads 16 \
    --years_per_gen 28 --seed 42

echo "  Output:"
ls -lh ${OUTPUT_PREFIX}.*
echo "=== PopSize done at $(date) ==="
