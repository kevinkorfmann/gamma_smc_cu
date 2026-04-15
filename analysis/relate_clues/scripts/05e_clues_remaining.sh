#!/bin/bash
#SBATCH --job-name=rc-clues-rem
#SBATCH --partition=genoa-lrg-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=analysis/relate_clues/logs/05e_clues_%j.log

set -euo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
CLUES2=${BASE}/tools/CLUES2
export PATH="$(pwd)/.pixi/envs/default/bin:${PATH}"
export PYTHONPATH="${CLUES2}:${PYTHONPATH:-}"

run_clues2() {
    local NAME=$1 DIR=$2 SAMPLED=$3 POP_FREQ=$4 T_CUTOFF=$5 COAL=$6
    echo "--- ${NAME} (popFreq=${POP_FREQ}) ---"
    python3 ${CLUES2}/RelateToCLUES.py --RelateSamples ${SAMPLED}.newick --DerivedFile ${DIR}/${NAME,,}_derived.txt --out ${DIR}/${NAME,,}
    python3 ${CLUES2}/inference.py --times ${DIR}/${NAME,,}_times.txt --coal ${COAL} --popFreq ${POP_FREQ} --tCutoff ${T_CUTOFF} --df 200 --CI 0.95 --out ${DIR}/${NAME,,}_result
    cat ${DIR}/${NAME,,}_result_inference.txt || true
}

# CCDC92 — need to redo to get trajectory files
run_clues2 ccdc92 ${BASE}/clues/CCDC92 ${BASE}/clues/CCDC92/ccdc92_sampled 0.710 2000 ${BASE}/trees/chr12_EAS/chr12_popsize.coal

# CLEC6A
run_clues2 clec6a ${BASE}/clues/CLEC6A ${BASE}/clues/CLEC6A/clec6a_sampled 0.629 2000 ${BASE}/trees/chr12_EAS/chr12_popsize.coal

# Neutral
run_clues2 neutral ${BASE}/clues/neutral ${BASE}/clues/neutral/neutral_sampled 0.50 2000 ${BASE}/trees/chr11_EURSAS/chr11_popsize.coal

echo "=== Done at $(date) ==="
