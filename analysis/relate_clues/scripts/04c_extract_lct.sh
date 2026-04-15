#!/bin/bash
#SBATCH --job-name=rc-extract-lct
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=analysis/relate_clues/logs/04c_extract_lct_%j.log

# Step 4: Extract CEU subtree + sample branch lengths at LCT.

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4
POPLABELS=${BASE}/data/chr2/chr2_eur.poplabels
TREEDIR=${BASE}/trees/chr2_EUR

echo "=== Step 4: Extract + sample (LCT, chr2, CEU) ==="
echo "Start: $(date)"

# Extract CEU subtree from popsize-adjusted trees
${RELATE}/bin/RelateExtract --mode SubTreesForSubpopulation \
    --anc ${TREEDIR}/chr2_popsize.anc.gz \
    --mut ${TREEDIR}/chr2_popsize.mut.gz \
    --poplabels ${POPLABELS} \
    --pop_of_interest CEU \
    -o ${TREEDIR}/chr2_CEU

echo "  CEU subtree extracted."

# Sample branch lengths at LCT window (300 kb around gene)
${RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
    -i ${TREEDIR}/chr2_CEU \
    -o ${BASE}/clues/LCT/lct_sampled \
    -m 1.25e-8 \
    --coal ${TREEDIR}/chr2_popsize.coal \
    --num_samples 200 \
    --first_bp 135700000 --last_bp 136000000 \
    --format n --seed 42

echo "  LCT branch lengths sampled:"
ls -lh ${BASE}/clues/LCT/lct_sampled.*
echo "=== Extract done at $(date) ==="
