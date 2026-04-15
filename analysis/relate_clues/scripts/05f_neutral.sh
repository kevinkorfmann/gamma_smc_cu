#!/bin/bash
#SBATCH --job-name=rc-neutral
#SBATCH --partition=genoa-lrg-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=analysis/relate_clues/logs/05f_neutral_%j.log

set -euo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
CLUES2=${BASE}/tools/CLUES2
export PATH="$(pwd)/.pixi/envs/default/bin:${PATH}"
export PYTHONPATH="${CLUES2}:${PYTHONPATH:-}"

DIR=${BASE}/clues/neutral
python3 ${CLUES2}/RelateToCLUES.py --RelateSamples ${DIR}/neutral_sampled.newick --DerivedFile ${DIR}/neutral_derived.txt --out ${DIR}/neutral
python3 ${CLUES2}/inference.py --times ${DIR}/neutral_times.txt --coal ${BASE}/trees/chr11_EURSAS/chr11_popsize.coal --popFreq 0.50 --tCutoff 2000 --df 200 --CI 0.95 --out ${DIR}/neutral_result
cat ${DIR}/neutral_result_inference.txt
echo "Done at $(date)"
