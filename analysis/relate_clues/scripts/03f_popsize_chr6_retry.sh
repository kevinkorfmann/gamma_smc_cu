#!/bin/bash
#SBATCH --job-name=rc-popsize-chr6-v2
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=20
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=analysis/relate_clues/logs/03f_popsize_chr6_v2_%j.log

# Re-run chr6 popsize to convergence. Prior job (5479964) timed out at 4h
# after completing only 1 of 10 requested iterations, leaving the in-place
# chr6_popsize.* files representing iter-1 rather than converged output.
# Writes to a fresh prefix (chr6_popsize_v2) to avoid disturbing the running
# CLUES job 5484941 that is still reading the iter-1 files.

set -euo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4
TREEDIR=${BASE}/trees/chr6_EUREAS

cd ${TREEDIR}
REPO_ABS=/vast/projects/smathi/cohort/kkor/tmrca.cu
${REPO_ABS}/${RELATE}/scripts/EstimatePopulationSize/EstimatePopulationSize.sh \
    -i chr6_relate \
    -m 1.25e-8 \
    --poplabels ${REPO_ABS}/${BASE}/data/all.poplabels \
    --years_per_gen 28 \
    --num_iter 10 \
    --threads 20 \
    --seed 42 \
    -o chr6_popsize_v2

echo "=== chr6 popsize v2 done at $(date) ==="
ls -lh chr6_popsize_v2.*
