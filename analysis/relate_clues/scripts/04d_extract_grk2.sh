#!/bin/bash
#SBATCH --job-name=rc-extract-grk2
#SBATCH --partition=genoa-lrg-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=analysis/relate_clues/logs/04d_extract_grk2_%j.log

# Step 4: Extract GIH subtree + sample branch lengths at GRK2 + neutral control.

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4
POPLABELS=${BASE}/data/chr11/chr11_eursas.poplabels
TREEDIR=${BASE}/trees/chr11_EURSAS

echo "=== Step 4: Extract + sample (GRK2, chr11, GIH) ==="
echo "Start: $(date)"

# Extract GIH subtree
${RELATE}/bin/RelateExtract --mode SubTreesForSubpopulation \
    --anc ${TREEDIR}/chr11_popsize.anc.gz \
    --mut ${TREEDIR}/chr11_popsize.mut.gz \
    --poplabels ${POPLABELS} \
    --pop_of_interest GIH \
    -o ${TREEDIR}/chr11_GIH

echo "  GIH subtree extracted."

# GRK2 window
${RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
    -i ${TREEDIR}/chr11_GIH \
    -o ${BASE}/clues/GRK2/grk2_sampled \
    -m 1.25e-8 \
    --coal ${TREEDIR}/chr11_popsize.coal \
    --num_samples 200 \
    --first_bp 67200000 --last_bp 67500000 \
    --format n --seed 42

echo "  GRK2 branch lengths sampled:"
ls -lh ${BASE}/clues/GRK2/grk2_sampled.*

# Neutral control (KCNQ1, chr11:2.4 Mb)
echo ""
echo "--- Neutral control ---"
${RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
    -i ${TREEDIR}/chr11_GIH \
    -o ${BASE}/clues/neutral/neutral_sampled \
    -m 1.25e-8 \
    --coal ${TREEDIR}/chr11_popsize.coal \
    --num_samples 200 \
    --first_bp 2400000 --last_bp 2700000 \
    --format n --seed 42

echo "  Neutral branch lengths sampled:"
ls -lh ${BASE}/clues/neutral/neutral_sampled.*
echo "=== Extract done at $(date) ==="
