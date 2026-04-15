#!/bin/bash
#SBATCH --job-name=rc-popsize-ext
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --array=1-2
#SBATCH --output=analysis/relate_clues/logs/03b_popsize_%A_%a.log

# PopSize re-estimation for chr12 (EAS) and chr20 (EUR+SAS)

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4

echo "=== PopSize (task ${SLURM_ARRAY_TASK_ID}) ==="
echo "Node: $(hostname), Start: $(date)"

if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]; then
    CHR=12; TREEDIR=${BASE}/trees/chr12_EAS; POP=CHS; POPLABELS=${BASE}/data/chr12/chr12_eas.poplabels
elif [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]; then
    CHR=20; TREEDIR=${BASE}/trees/chr20_EURSAS; POP=GIH; POPLABELS=${BASE}/data/chr20/chr20_eursas.poplabels
fi

MAP=${BASE}/input_files/Relate_input_files/GRCh38/recomb_map/genetic_map_chr${CHR}.txt

cd ${TREEDIR}
REPO=/vast/projects/smathi/cohort/kkor/tmrca.cu

${REPO}/${RELATE}/scripts/EstimatePopulationSize/EstimatePopulationSize.sh \
    -i chr${CHR}_relate \
    -o chr${CHR}_popsize \
    -m 1.25e-8 \
    --poplabels ${REPO}/${POPLABELS} \
    --pop_of_interest ${POP} \
    --num_iter 5 --threads 16 \
    --years_per_gen 28 \
    --threshold 0 || true

echo "Output:"
ls -lh chr${CHR}_popsize.*
echo "=== Done at $(date) ==="
