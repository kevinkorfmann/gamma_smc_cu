#!/bin/bash
#SBATCH --job-name=rc-clec6a
#SBATCH --partition=genoa-lrg-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=analysis/relate_clues/logs/05f_clec6a_%j.log

set -euo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
CLUES2=${BASE}/tools/CLUES2
export PATH="$(pwd)/.pixi/envs/default/bin:${PATH}"
export PYTHONPATH="${CLUES2}:${PYTHONPATH:-}"

DIR=${BASE}/clues/CLEC6A
python3 ${CLUES2}/RelateToCLUES.py --RelateSamples ${DIR}/clec6a_sampled.newick --DerivedFile ${DIR}/clec6a_derived.txt --out ${DIR}/clec6a
python3 ${CLUES2}/inference.py --times ${DIR}/clec6a_times.txt --coal ${BASE}/trees/chr12_EAS/chr12_popsize.coal --popFreq 0.629 --tCutoff 2000 --df 200 --CI 0.95 --out ${DIR}/clec6a_result
cat ${DIR}/clec6a_result_inference.txt
echo "Done at $(date)"
