#!/bin/bash
#SBATCH --job-name=rc-extract-grk2
#SBATCH --partition=genoa-lrg-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=analysis/relate_clues/logs/04d_extract_grk2_%j.log

# Step 4: Sample branch lengths at GRK2 + neutral.
# PopSize already extracted GIH subtree.

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4
TREEDIR=${BASE}/trees/chr11_EURSAS

echo "=== Step 4: SampleBranchLengths (GRK2 + neutral, chr11, GIH) ==="
echo "Start: $(date)"

# GRK2
${RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
    -i ${TREEDIR}/chr11_popsize \
    -o ${BASE}/clues/GRK2/grk2_sampled \
    -m 1.25e-8 \
    --coal ${TREEDIR}/chr11_popsize.coal \
    --num_samples 200 \
    --first_bp 67200000 --last_bp 67500000 \
    --format n --seed 42

echo "GRK2 done:"
ls -lh ${BASE}/clues/GRK2/grk2_sampled.*

# Neutral control (KCNQ1)
${RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
    -i ${TREEDIR}/chr11_popsize \
    -o ${BASE}/clues/neutral/neutral_sampled \
    -m 1.25e-8 \
    --coal ${TREEDIR}/chr11_popsize.coal \
    --num_samples 200 \
    --first_bp 2400000 --last_bp 2700000 \
    --format n --seed 42

echo "Neutral done:"
ls -lh ${BASE}/clues/neutral/neutral_sampled.*
echo "=== Done at $(date) ==="
