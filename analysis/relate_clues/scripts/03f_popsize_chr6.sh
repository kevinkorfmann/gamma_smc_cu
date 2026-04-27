#!/bin/bash
#SBATCH --job-name=rc-popsize-chr6
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=analysis/relate_clues/logs/03f_popsize_chr6_%j.log

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
    --threads 4 \
    --seed 42 \
    -o chr6_popsize

echo "=== chr6 popsize done at $(date) ==="
ls -lh chr6_popsize.*
