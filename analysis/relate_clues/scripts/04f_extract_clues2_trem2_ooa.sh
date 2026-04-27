#!/bin/bash
#SBATCH --job-name=rc-trem2-ooa
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=analysis/relate_clues/logs/04f_trem2_ooa_%j.log

# CLUES2 on the TRUE pan-OoA sweep variant near TREM2.
# Target: chr6:41,166,068 (G>A) — AFR AF ~0.95, non-AFR AF ~0.43. Classical
# OoA-differentiated variant, 3 kb downstream of TREM2 gene body, inside the
# H12 peak block (~41.15 Mb) shared across all non-AFR pops.
#
# Reuses the same IBS Relate tree that `04f_extract_clues2_trem2.sh` built
# (trees/chr6_EUREAS); only the focal SNP changes.
#
# Derived/ancestral check: Akbari-neighboring variants in 41.166 Mb have
# ANC=G,REF=G,ALT=A — derived allele is A. Derived AF in IBS from 1KG NYGC
# = 0.429 (ALT frequency). That is the input for CLUES `--popFreq`.

set -euo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4
CLUES2=${BASE}/tools/CLUES2
PIXI_ENV=.pixi/envs/default
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export PYTHONPATH="${CLUES2}:${PYTHONPATH:-}"

FOCAL_POS=41166068
FOCAL_FREQ=0.429                          # derived (ALT=A) AF in IBS
FIRST=$((FOCAL_POS - 150000))
LAST=$((FOCAL_POS + 150000))

TREEDIR=${BASE}/trees/chr6_EUREAS
OUTDIR=${BASE}/clues/TREM2_OoA
mkdir -p ${OUTDIR}

# Re-use the IBS subtree built for the prior TREM2 run; don't rebuild.
if [ ! -f "${TREEDIR}/chr6_IBS.anc" ] && [ ! -f "${TREEDIR}/chr6_IBS.anc.gz" ]; then
    ${RELATE}/bin/RelateExtract --mode SubTreesForSubpopulation \
        --anc ${TREEDIR}/chr6_popsize.anc.gz \
        --mut ${TREEDIR}/chr6_popsize.mut.gz \
        --poplabels ${BASE}/data/all.poplabels \
        --pop_of_interest IBS \
        -o ${TREEDIR}/chr6_IBS
fi

${RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
    -i ${TREEDIR}/chr6_IBS \
    -o ${OUTDIR}/trem2_ooa_sampled \
    -m 1.25e-8 --coal ${TREEDIR}/chr6_popsize.coal \
    --num_samples 200 --first_bp ${FIRST} --last_bp ${LAST} \
    --format n --seed 42

# Find the nearest SNP in the .mut.gz to FOCAL_POS (within 1 kb).
python3 << PY
import gzip
mut_gz = '${TREEDIR}/chr6_popsize.mut.gz'
pos = ${FOCAL_POS}
best = None
with gzip.open(mut_gz, 'rt') as f:
    for i, line in enumerate(f):
        if i == 0: continue
        parts = line.rstrip().split(';')
        try: bp = int(parts[1])
        except: continue
        d = abs(bp - pos)
        if best is None or d < best[1]:
            best = (bp, d, parts[0])
        if d > 10000 and best is not None and best[1] < 10000:
            break
bp, d, rsid = best
rsid = rsid if rsid else f'chr6_{bp}'
print(f'Chosen focal SNP: chr6:{bp} ({rsid}), {d} bp from {pos}')
with open('${OUTDIR}/trem2_ooa_focal.sites', 'w') as out:
    out.write(f'{rsid}\t{bp}\t1\n')
PY

python ${CLUES2}/RelateToCLUES.py \
    --RelateSamples ${OUTDIR}/trem2_ooa_sampled.newick \
    --DerivedFile ${OUTDIR}/trem2_ooa_focal.sites \
    --out ${OUTDIR}/trem2_ooa

python ${CLUES2}/inference.py \
    --times ${OUTDIR}/trem2_ooa_times.txt \
    --coal ${TREEDIR}/chr6_popsize.coal \
    --popFreq ${FOCAL_FREQ} --tCutoff 2000 \
    --df 450 --CI 0.95 \
    --out ${OUTDIR}/trem2_ooa_result

echo "=== trem2_ooa CLUES2 inference ==="
cat ${OUTDIR}/trem2_ooa_result_inference.txt
echo "=== done at $(date) ==="
