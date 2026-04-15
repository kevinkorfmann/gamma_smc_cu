#!/bin/bash
#SBATCH --job-name=rc-ccdc92
#SBATCH --partition=genoa-lrg-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=analysis/relate_clues/logs/05f_ccdc92_%j.log

set -euo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
CLUES2=${BASE}/tools/CLUES2
export PATH="$(pwd)/.pixi/envs/default/bin:${PATH}"
export PYTHONPATH="${CLUES2}:${PYTHONPATH:-}"

DIR=${BASE}/clues/CCDC92
python3 ${CLUES2}/RelateToCLUES.py --RelateSamples ${DIR}/ccdc92_sampled.newick --DerivedFile ${DIR}/ccdc92_derived.txt --out ${DIR}/ccdc92
python3 ${CLUES2}/inference.py --times ${DIR}/ccdc92_times.txt --coal ${BASE}/trees/chr12_EAS/chr12_popsize.coal --popFreq 0.710 --tCutoff 2000 --df 200 --CI 0.95 --out ${DIR}/ccdc92_result
cat ${DIR}/ccdc92_result_inference.txt
echo "Done at $(date)"
