#!/bin/bash
#SBATCH --job-name=rc-clues-grk2
#SBATCH --partition=genoa-lrg-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=analysis/relate_clues/logs/05d_clues_grk2_%j.log

set -euo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
CLUES2=${BASE}/tools/CLUES2
export PATH="$(pwd)/.pixi/envs/default/bin:${PATH}"
export PYTHONPATH="${CLUES2}:${PYTHONPATH:-}"

echo "=== CLUES2 (GRK2 + neutral) ==="

# GRK2
echo "--- GRK2 ---"
DIR=${BASE}/clues/GRK2
python3 ${CLUES2}/RelateToCLUES.py --RelateSamples ${DIR}/grk2_sampled.newick --DerivedFile ${DIR}/grk2_derived.txt --out ${DIR}/grk2
python3 ${CLUES2}/inference.py --times ${DIR}/grk2_times.txt --coal ${BASE}/trees/chr11_EURSAS/chr11_popsize.coal --popFreq 0.981 --tCutoff 2000 --df 200 --CI 0.95 --out ${DIR}/grk2_result
cat ${DIR}/grk2_result_inference.txt
python3 ${CLUES2}/plot_traj.py --freqs ${DIR}/grk2_result_freqs.txt --post ${DIR}/grk2_result_post.txt --figure ${DIR}/grk2_trajectory.png --generation_time 28 || true

# Neutral
echo "--- Neutral ---"
DIR=${BASE}/clues/neutral
python3 ${CLUES2}/RelateToCLUES.py --RelateSamples ${DIR}/neutral_sampled.newick --DerivedFile ${DIR}/neutral_derived.txt --out ${DIR}/neutral
python3 ${CLUES2}/inference.py --times ${DIR}/neutral_times.txt --coal ${BASE}/trees/chr11_EURSAS/chr11_popsize.coal --popFreq 0.50 --tCutoff 2000 --df 200 --CI 0.95 --out ${DIR}/neutral_result
cat ${DIR}/neutral_result_inference.txt || true

echo "=== Done at $(date) ==="
