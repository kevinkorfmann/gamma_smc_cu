#!/bin/bash
#SBATCH --job-name=rc-trem2
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=analysis/relate_clues/logs/04f_trem2_%j.log

set -euo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4
CLUES2=${BASE}/tools/CLUES2
PIXI_ENV=.pixi/envs/default
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export PYTHONPATH="${CLUES2}:${PYTHONPATH:-}"

VE=analysis/orthogonal_v41/variant_evidence/TREM2_IBS.json
if [ ! -f "${VE}" ]; then echo "ERR: ${VE} missing"; exit 1; fi
FOCAL_POS=$(python3 -c "import json; print(json.load(open('${VE}'))['most_diff_variant_pos'])")
FOCAL_FREQ=$(python3 -c "import json; print(json.load(open('${VE}'))['most_diff_variant_focal_af'])")
FIRST=$((FOCAL_POS - 150000))
LAST=$((FOCAL_POS + 150000))

TREEDIR=${BASE}/trees/chr6_EUREAS
OUTDIR=${BASE}/clues/TREM2
mkdir -p ${OUTDIR}

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
    -o ${OUTDIR}/trem2_sampled \
    -m 1.25e-8 --coal ${TREEDIR}/chr6_popsize.coal \
    --num_samples 200 --first_bp ${FIRST} --last_bp ${LAST} \
    --format n --seed 42

python3 << PY
import gzip
mut_gz = '${TREEDIR}/chr6_popsize.mut.gz'
pos = ${FOCAL_POS}
with gzip.open(mut_gz, 'rt') as f:
    for i, line in enumerate(f):
        if i == 0: continue
        parts = line.rstrip().split(';')
        try: bp = int(parts[1])
        except: continue
        if bp == pos:
            rsid = parts[0] if parts[0] else f'chr6_{bp}'
            with open('${OUTDIR}/trem2_focal.sites', 'w') as out:
                out.write(f'{rsid}\t{bp}\t1\n')
            print(f'Found focal SNP chr6:{bp}')
            break
    else:
        print(f'WARN: focal chr6:{pos} not in .mut.gz')
PY

python ${CLUES2}/RelateToCLUES.py \
    --RelateSamples ${OUTDIR}/trem2_sampled.newick \
    --DerivedFile ${OUTDIR}/trem2_focal.sites \
    --out ${OUTDIR}/trem2

python ${CLUES2}/inference.py \
    --times ${OUTDIR}/trem2_times.txt \
    --coal ${TREEDIR}/chr6_popsize.coal \
    --popFreq ${FOCAL_FREQ} --tCutoff 2000 \
    --df 450 --CI 0.95 \
    --out ${OUTDIR}/trem2_result

cat ${OUTDIR}/trem2_result_inference.txt
echo "=== TREM2 CLUES2 done at $(date) ==="
