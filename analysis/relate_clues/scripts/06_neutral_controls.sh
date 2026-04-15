#!/bin/bash
#SBATCH --job-name=rc-neutral-controls
#SBATCH --partition=genoa-lrg-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --array=1-5
#SBATCH --output=analysis/relate_clues/logs/06_neutral_%A_%a.log

# Proper neutral controls: genes at median TMRCA rank in focal pop,
# never < 20th pct in any other pop, not in sweep catalog.
#
# 1: TACR1   (CEU, chr2:75.0-75.2 Mb)   — LCT neutral
# 2: C11orf65 (GIH, chr11:108.3-108.5)  — GRK2 neutral
# 3: NFATC2  (GIH, chr20:51.4-51.6)     — BPIFA2 neutral
# 4: ATF7IP  (CHS, chr12:14.4-14.5)     — SLC6A15 neutral
# 5: C12orf75 (CDX, chr12:105.2-105.4)  — CCDC92/CLEC6A neutral

set -euo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4
CLUES2=${BASE}/tools/CLUES2
export PATH="$(pwd)/.pixi/envs/default/bin:${PATH}"
export PYTHONPATH="${CLUES2}:${PYTHONPATH:-}"

mkdir -p ${BASE}/clues/neutrals

case ${SLURM_ARRAY_TASK_ID} in
  1) NAME=tacr1;    INPUT=${BASE}/trees/chr2_EUR/chr2_popsize;       COAL=${BASE}/trees/chr2_EUR/chr2_popsize.coal;         START=75046463;  END=75199520;  TARGET=75123000; ;;
  2) NAME=c11orf65; INPUT=${BASE}/trees/chr11_EURSAS/chr11_popsize;  COAL=${BASE}/trees/chr11_EURSAS/chr11_popsize.coal;    START=108308519; END=108467531; TARGET=108388000; ;;
  3) NAME=nfatc2;   INPUT=${BASE}/trees/chr20_EURSAS/chr20_popsize;  COAL=${BASE}/trees/chr20_EURSAS/chr20_popsize.coal;    START=51386957;  END=51562831;  TARGET=51475000; ;;
  4) NAME=atf7ip;   INPUT=${BASE}/trees/chr12_EAS/chr12_popsize;     COAL=${BASE}/trees/chr12_EAS/chr12_popsize.coal;       START=14365676;  END=14502931;  TARGET=14434000; ;;
  5) NAME=c12orf75; INPUT=${BASE}/trees/chr12_EAS/chr12_CDX;         COAL=${BASE}/trees/chr12_EAS/chr12_popsize.coal;       START=105235290; END=105396097; TARGET=105315000; ;;
esac

DIR=${BASE}/clues/neutrals/${NAME}
mkdir -p ${DIR}

echo "=== Neutral control: ${NAME} ==="
echo "Node: $(hostname), Start: $(date)"

# 1. Sample branch lengths
${RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
    -i ${INPUT} \
    -o ${DIR}/${NAME}_sampled \
    -m 1.25e-8 \
    --coal ${COAL} \
    --num_samples 200 \
    --first_bp ${START} --last_bp ${END} \
    --format n --seed 42

# 2. Make derived file at target SNP (closest to midpoint)
python3 -c "
import numpy as np
from collections import Counter
with open('${DIR}/${NAME}_sampled.sites') as f: lines = f.readlines()
positions, alleles = [], []
for line in lines[2:]:
    parts = line.strip().split('\t')
    positions.append(int(parts[0])); alleles.append(parts[1])
positions = np.array(positions)
idx = np.argmin(np.abs(positions - ${TARGET}))
allele_str = alleles[idx]
counts = Counter(allele_str)
major = counts.most_common(1)[0][0]
derived = np.array([1 if c == major else 0 for c in allele_str])
np.savetxt('${DIR}/${NAME}_derived.txt', derived, fmt='%d')
print(f'pos={positions[idx]} freq={derived.mean():.3f} n={len(derived)}')
with open('${DIR}/${NAME}_popfreq.txt','w') as f: f.write(f'{derived.mean():.3f}')
"
POPFREQ=$(cat ${DIR}/${NAME}_popfreq.txt)
echo "Pop freq: ${POPFREQ}"

# 3. Convert to CLUES2 format
python3 ${CLUES2}/RelateToCLUES.py \
    --RelateSamples ${DIR}/${NAME}_sampled.newick \
    --DerivedFile ${DIR}/${NAME}_derived.txt \
    --out ${DIR}/${NAME}

# 4. CLUES2 inference
python3 ${CLUES2}/inference.py \
    --times ${DIR}/${NAME}_times.txt \
    --coal ${COAL} \
    --popFreq ${POPFREQ} \
    --tCutoff 2000 \
    --df 200 --CI 0.95 \
    --out ${DIR}/${NAME}_result

cat ${DIR}/${NAME}_result_inference.txt
ls -lh ${DIR}/${NAME}_result_*
echo "=== Done at $(date) ==="
