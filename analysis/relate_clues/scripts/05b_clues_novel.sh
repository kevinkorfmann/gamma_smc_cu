#!/bin/bash
#SBATCH --job-name=rc-clues-novel
#SBATCH --partition=genoa-lrg-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=analysis/relate_clues/logs/05b_clues_%j.log

# CLUES2 inference for 4 novel candidates

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
CLUES2=${BASE}/tools/CLUES2
export PATH="$(pwd)/.pixi/envs/default/bin:${PATH}"
export PYTHONPATH="${CLUES2}:${PYTHONPATH:-}"

echo "=== CLUES2 for novel candidates ==="
echo "Start: $(date)"

make_derived() {
    local SITES=$1 TARGET_POS=$2 OUT=$3
    python3 -c "
import numpy as np
from collections import Counter
with open('${SITES}') as f:
    lines = f.readlines()
positions, alleles = [], []
for line in lines[2:]:
    parts = line.strip().split('\t')
    positions.append(int(parts[0]))
    alleles.append(parts[1])
positions = np.array(positions)
idx = np.argmin(np.abs(positions - ${TARGET_POS}))
allele_str = alleles[idx]
counts = Counter(allele_str)
major = counts.most_common(1)[0][0]
derived = np.array([1 if c == major else 0 for c in allele_str])
np.savetxt('${OUT}', derived, fmt='%d')
print(f'  pos={positions[idx]} freq={derived.mean():.3f} n={len(derived)}')
"
}

run_clues2() {
    local NAME=$1 DIR=$2 SAMPLED=$3 TARGET_POS=$4 POP_FREQ=$5 T_CUTOFF=$6 COAL=$7
    echo ""
    echo "--- ${NAME} (popFreq=${POP_FREQ}, tCutoff=${T_CUTOFF}) ---"

    # Make derived file from sites
    make_derived ${SAMPLED}.sites ${TARGET_POS} ${DIR}/${NAME,,}_derived.txt

    # Convert Relate format to CLUES2
    python3 ${CLUES2}/RelateToCLUES.py \
        --RelateSamples ${SAMPLED}.newick \
        --DerivedFile ${DIR}/${NAME,,}_derived.txt \
        --out ${DIR}/${NAME,,}

    # Run inference
    python3 ${CLUES2}/inference.py \
        --times ${DIR}/${NAME,,}_times.txt \
        --coal ${COAL} \
        --popFreq ${POP_FREQ} \
        --tCutoff ${T_CUTOFF} \
        --df 200 --CI 0.95 \
        --out ${DIR}/${NAME,,}_result

    echo "  Result:"
    cat ${DIR}/${NAME,,}_result_inference.txt || true

    # Plot
    python3 ${CLUES2}/plot_traj.py \
        --freqs ${DIR}/${NAME,,}_result_freqs.txt \
        --post ${DIR}/${NAME,,}_result_post.txt \
        --figure ${DIR}/${NAME,,}_trajectory.png \
        --generation_time 28 || true
}

# BPIFA2: chr20, GIH, target ~31,617,000
run_clues2 bpifa2 ${BASE}/clues/BPIFA2 ${BASE}/clues/BPIFA2/bpifa2_sampled \
    31617000 0.95 2000 ${BASE}/trees/chr20_EURSAS/chr20_popsize.coal

# SLC6A15: chr12, CHS, target ~84,887,000
run_clues2 slc6a15 ${BASE}/clues/SLC6A15 ${BASE}/clues/SLC6A15/slc6a15_sampled \
    84887000 0.80 2000 ${BASE}/trees/chr12_EAS/chr12_popsize.coal

# CCDC92: chr12, CDX, target ~124,400,000
run_clues2 ccdc92 ${BASE}/clues/CCDC92 ${BASE}/clues/CCDC92/ccdc92_sampled \
    124400000 0.90 2000 ${BASE}/trees/chr12_EAS/chr12_popsize.coal

# CLEC6A: chr12, CDX, target ~9,730,000
run_clues2 clec6a ${BASE}/clues/CLEC6A ${BASE}/clues/CLEC6A/clec6a_sampled \
    9730000 0.85 2000 ${BASE}/trees/chr12_EAS/chr12_popsize.coal

echo ""
echo "=== All CLUES2 runs complete at $(date) ==="
