#!/bin/bash
#SBATCH --job-name=clues-traj-rest
#SBATCH --partition=genoa-lrg-mem
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/relate_clues/logs/clues_traj_rest_%A_%a.log

# CLUES2 trajectory inference for the 6 remaining swept-block variants in
# the TREML1/TREM2 cluster (TREM2_OoA_v6). pos41137356 was already done
# in job 5506610; pos41129151 has no inference (skipped earlier in v6).
#
# Each array task reuses the existing times_times.txt from the v6
# selection-inference run and adds --popFreq matched to the empirical
# IBS derived-allele frequency computed from each pos's derived.txt.

set -e
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
export PATH=$(pwd)/.pixi/envs/default/bin:$PATH

POSITIONS=(41121942 41166068 41176920 41189316 41189932 41191484)
POPFREQS=(0.6624   0.5732   0.6720   0.6783   0.3376   0.6624)

POS=${POSITIONS[$SLURM_ARRAY_TASK_ID]}
FREQ=${POPFREQS[$SLURM_ARRAY_TASK_ID]}
W=analysis/relate_clues/clues/TREM2_OoA_v6/pos${POS}

echo "=== inference WITH trajectory at chr6:${POS} (IBS derived AF ${FREQ}) ==="
date
hostname

python -u analysis/relate_clues/tools/CLUES2/inference.py \
    --times ${W}/times_times.txt \
    --coal analysis/relate_clues/trees/chr6_EUREAS/chr6_popsize_v2.coal \
    --popFreq ${FREQ} --tCutoff 2000 \
    --df 200 --CI 0.95 \
    --out ${W}/result_traj

echo "=== DONE ==="
date
ls -la ${W}/result_traj*
cat ${W}/result_traj_inference.txt
