#!/bin/bash
#SBATCH --job-name=rc-ifih1
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=analysis/relate_clues/logs/04f_ifih1_%j.log

# Phase E for IFIH1: iterates through top CEU-informative candidates until one works.
# Uses existing chr2_popsize trees (CEU-filtered).

set -uo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4
CLUES2=${BASE}/tools/CLUES2
PIXI_ENV=.pixi/envs/default
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export PYTHONPATH="${CLUES2}:${PYTHONPATH:-}"

echo "=== IFIH1 extract + CLUES2 (retry loop) at $(date) ==="

VE=analysis/orthogonal_v41/variant_evidence/IFIH1_IBS.json
TARGET_POS=$(python3 -c "import json; print(json.load(open('${VE}'))['most_diff_variant_pos'])")

TREEDIR=${BASE}/trees/chr2_EUR
OUTDIR=${BASE}/clues/IFIH1
mkdir -p ${OUTDIR}

# Build ranked list of candidate focal SNPs (top CEU vs AFR dAF, present in mut.gz)
python3 << 'PY'
import gzip, subprocess, pandas as pd, json
ve = json.load(open('analysis/orthogonal_v41/variant_evidence/IFIH1_IBS.json'))
TARGET = ve['most_diff_variant_pos']
WIN = 300_000
mut_gz = 'analysis/relate_clues/trees/chr2_EUR/chr2_popsize.mut.gz'
mut_bp = set()
rsid_for = {}
with gzip.open(mut_gz, 'rt') as f:
    for i, line in enumerate(f):
        if i == 0: continue
        parts = line.rstrip().split(';')
        try: bp = int(parts[1])
        except: continue
        if abs(bp - TARGET) <= WIN:
            mut_bp.add(bp)
            rsid_for[bp] = parts[0] if parts[0] else f'chr2_{bp}'

samples = pd.read_csv('analysis/genome_wide/data/samples.txt', sep=r'\s+')
ceu_ids = set(samples[samples.Population == 'CEU'].SampleID)
afr_ids = set(samples[samples.Superpopulation == 'AFR'].SampleID)

VCF = 'analysis/genome_wide/data/chr2.vcf.gz'
cmd = ['tabix', VCF, f'chr2:{TARGET - WIN}-{TARGET + WIN}']
out = subprocess.check_output(cmd).decode()
hd = subprocess.check_output(['tabix', '-H', VCF]).decode().splitlines()
for line in hd:
    if line.startswith('#CHROM'):
        sample_order = line.strip().split('\t')[9:]
        break

candidates = []
for row in out.strip().split('\n'):
    fields = row.split('\t')
    if len(fields) < 10: continue
    bp = int(fields[1])
    if bp not in mut_bp: continue
    ref, alt = fields[3], fields[4]
    if len(ref) != 1 or len(alt) != 1: continue
    gts = fields[9:]
    ceu_n0 = ceu_n1 = afr_n0 = afr_n1 = 0
    for sid, gt in zip(sample_order, gts):
        if sid in ceu_ids:
            for a in gt.split('|'):
                if a == '0': ceu_n0 += 1
                elif a == '1': ceu_n1 += 1
        elif sid in afr_ids:
            for a in gt.split('|'):
                if a == '0': afr_n0 += 1
                elif a == '1': afr_n1 += 1
    if ceu_n0 + ceu_n1 == 0 or afr_n0 + afr_n1 == 0: continue
    ceu_freq = ceu_n0 / (ceu_n0 + ceu_n1)
    afr_freq = afr_n0 / (afr_n0 + afr_n1)
    # Sweep allele = the one high in CEU (code '0' for REF high CEU, '1' for ALT high)
    if ceu_freq > 0.5 and afr_freq < 0.5 and (ceu_freq - afr_freq) > 0.3:
        # REF is sweep allele (derived=0)
        candidates.append((ceu_freq - afr_freq, bp, ceu_freq, 'REF'))
    elif (1 - ceu_freq) > 0.5 and (1 - afr_freq) < 0.5 and ((1 - ceu_freq) - (1 - afr_freq)) > 0.3:
        # ALT is sweep allele (derived=1)
        candidates.append(((1 - ceu_freq) - (1 - afr_freq), bp, 1 - ceu_freq, 'ALT'))

# Sort by |dAF| descending — pick top 20 for retry loop
candidates.sort(reverse=True)
with open('/tmp/ifih1_candidates.txt', 'w') as f:
    for daf, bp, freq, which in candidates[:20]:
        rsid = rsid_for[bp]
        derived_code = '0' if which == 'REF' else '1'
        f.write(f'{rsid}\t{bp}\t{derived_code}\t{freq}\t{daf}\n')
print(f'Wrote {len(candidates[:20])} candidate SNPs to /tmp/ifih1_candidates.txt')
PY

# Try each candidate in ranked order; stop at first one where CLUES2 succeeds
SUCCESS=0
while read -r LINE; do
    RSID=$(echo "$LINE" | awk '{print $1}')
    POS=$(echo "$LINE" | awk '{print $2}')
    DER=$(echo "$LINE" | awk '{print $3}')
    FREQ=$(echo "$LINE" | awk '{print $4}')
    echo "--- Trying focal SNP chr2:${POS} rsid=${RSID} derived=${DER} freq=${FREQ} ---"
    
    # Write .sites
    printf '%s\t%s\t%s\n' "${RSID}" "${POS}" "${DER}" > ${OUTDIR}/ifih1_focal.sites
    FIRST=$((POS - 150000))
    LAST=$((POS + 150000))
    
    # Sample branch lengths (~10 min)
    rm -f ${OUTDIR}/ifih1_sampled.* 2>/dev/null
    ${RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
        -i ${TREEDIR}/chr2_popsize \
        -o ${OUTDIR}/ifih1_sampled \
        -m 1.25e-8 --coal ${TREEDIR}/chr2_popsize.coal \
        --num_samples 200 --first_bp ${FIRST} --last_bp ${LAST} \
        --format n --seed 42 2>&1 | tail -3

    # RelateToCLUES
    if python ${CLUES2}/RelateToCLUES.py \
        --RelateSamples ${OUTDIR}/ifih1_sampled.newick \
        --DerivedFile ${OUTDIR}/ifih1_focal.sites \
        --out ${OUTDIR}/ifih1 2>&1 | tee /tmp/rtc_out.log; then
        # Check whether it actually produced times.txt
        if [ -f ${OUTDIR}/ifih1_times.txt ]; then
            # Run CLUES2 inference
            if python ${CLUES2}/inference.py \
                --times ${OUTDIR}/ifih1_times.txt \
                --coal ${TREEDIR}/chr2_popsize.coal \
                --popFreq ${FREQ} --tCutoff 2000 \
                --df 450 --CI 0.95 \
                --out ${OUTDIR}/ifih1_result 2>&1; then
                echo "=== IFIH1 CLUES2 SUCCESS at chr2:${POS} ==="
                cat ${OUTDIR}/ifih1_result_inference.txt
                SUCCESS=1
                echo "${POS} ${FREQ}" > ${OUTDIR}/ifih1_focal_used.txt
                break
            fi
        fi
    fi
    echo "  FAILED at chr2:${POS}, trying next candidate"
done < /tmp/ifih1_candidates.txt

if [ ${SUCCESS} -eq 0 ]; then
    echo "ERR: no candidate worked"
    exit 1
fi

echo "=== IFIH1 CLUES2 done at $(date) ==="
