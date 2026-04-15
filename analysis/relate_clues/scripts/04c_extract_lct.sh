#!/bin/bash
#SBATCH --job-name=rc-extract-lct
#SBATCH --partition=genoa-lrg-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=analysis/relate_clues/logs/04c_extract_lct_%j.log

# Step 4: Sample branch lengths at LCT.
# PopSize already extracted the CEU subtree, so skip RelateExtract.

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4
TREEDIR=${BASE}/trees/chr2_EUR

echo "=== Step 4: SampleBranchLengths (LCT, chr2, CEU) ==="
echo "Start: $(date)"

# PopSize output is already CEU-only, use directly
${RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
    -i ${TREEDIR}/chr2_popsize \
    -o ${BASE}/clues/LCT/lct_sampled \
    -m 1.25e-8 \
    --coal ${TREEDIR}/chr2_popsize.coal \
    --num_samples 200 \
    --first_bp 135700000 --last_bp 136000000 \
    --format n --seed 42

echo "LCT branch lengths sampled:"
ls -lh ${BASE}/clues/LCT/lct_sampled.*
echo "=== Done at $(date) ==="
