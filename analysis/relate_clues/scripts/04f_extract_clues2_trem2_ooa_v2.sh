#!/bin/bash
#SBATCH --job-name=rc-trem2-ooa-v2
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=analysis/relate_clues/logs/04f_trem2_ooa_v2_%j.log

# CLUES2 on the pan-OoA TREM2 sweep, using the converged popsize (v2).
# Tries a ranked list of focal variants (all high |dAF(IBS vs AFR)| in the
# TREM2 H12 peak) and uses the first one that passes CLUES's infinite-sites
# check. Prior run (job 5484941) failed at chr6:41,166,068 on that check.
#
# Candidates ranked by |dAF| and retaining MAF in IBS >= 0.3 for CLUES power.

set -euo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4
CLUES2=${BASE}/tools/CLUES2
PIXI_ENV=.pixi/envs/default
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export PYTHONPATH="${CLUES2}:${PYTHONPATH:-}"

TREEDIR=${BASE}/trees/chr6_EUREAS
OUTDIR=${BASE}/clues/TREM2_OoA_v2
mkdir -p ${OUTDIR}

# Ranked focal candidates: pos, derived-AF-in-IBS (for --popFreq).
# From analysis/trem2_deep_dive/find_true_sweep_variant.py (OoA sweep cands).
# Format: "pos freq" pairs.
CANDIDATES=(
    "41176920 0.324"
    "41166068 0.429"
    "41121942 0.338"
    "41129151 0.333"
    "41189316 0.362"
    "41189932 0.362"
    "41191484 0.362"
    "41137356 0.443"
    "41115613 0.557"
)

# Re-extract IBS subtree from the CONVERGED popsize v2 (not the iter-1 v1).
if [ ! -f "${TREEDIR}/chr6_IBS_v2.anc" ] && [ ! -f "${TREEDIR}/chr6_IBS_v2.anc.gz" ]; then
    ${RELATE}/bin/RelateExtract --mode SubTreesForSubpopulation \
        --anc ${TREEDIR}/chr6_popsize_v2.anc.gz \
        --mut ${TREEDIR}/chr6_popsize_v2.mut.gz \
        --poplabels ${BASE}/data/all.poplabels \
        --pop_of_interest IBS \
        -o ${TREEDIR}/chr6_IBS_v2
fi

try_candidate () {
    local POS=$1
    local FREQ=$2
    local TAG=pos${POS}
    echo ""
    echo "=================================================================="
    echo "=== Trying candidate chr6:${POS} (derived AF = ${FREQ}) ==="
    echo "=================================================================="
    local FIRST=$((POS - 150000))
    local LAST=$((POS + 150000))

    local WORK=${OUTDIR}/${TAG}
    mkdir -p ${WORK}

    # Sample branch lengths around candidate
    ${RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
        -i ${TREEDIR}/chr6_IBS_v2 \
        -o ${WORK}/sampled \
        -m 1.25e-8 --coal ${TREEDIR}/chr6_popsize_v2.coal \
        --num_samples 200 --first_bp ${FIRST} --last_bp ${LAST} \
        --format n --seed 42 \
        >${WORK}/sample.log 2>&1 || {
            echo "  [SKIP] SampleBranchLengths failed at ${POS}"
            return 1
        }

    # Find the SNP in the mut file
    python3 << PY
import gzip
mut_gz = '${TREEDIR}/chr6_popsize_v2.mut.gz'
pos = ${POS}
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
        if d > 5000 and best is not None and best[1] < 5000:
            break
if best is None or best[1] > 1000:
    print(f'NO_CLOSE_SNP {pos}')
    raise SystemExit(2)
bp, d, rsid = best
rsid = rsid if rsid else f'chr6_{bp}'
print(f'Chosen SNP: chr6:{bp} ({rsid}), {d} bp from {pos}')
with open('${WORK}/focal.sites', 'w') as out:
    out.write(f'{rsid}\t{bp}\t1\n')
PY
    local PY_RC=$?
    if [ $PY_RC -ne 0 ]; then
        echo "  [SKIP] No SNP within 1 kb of ${POS}"
        return 1
    fi

    # RelateToCLUES — this is where the prior run failed on infinite-sites
    if ! python ${CLUES2}/RelateToCLUES.py \
        --RelateSamples ${WORK}/sampled.newick \
        --DerivedFile ${WORK}/focal.sites \
        --out ${WORK}/times \
        >${WORK}/relate2clues.log 2>&1 ; then
        echo "  [SKIP] RelateToCLUES failed at ${POS} (likely infinite-sites violation)"
        tail -5 ${WORK}/relate2clues.log | sed 's/^/    /'
        return 1
    fi

    # CLUES inference
    if ! python ${CLUES2}/inference.py \
        --times ${WORK}/times_times.txt \
        --coal ${TREEDIR}/chr6_popsize_v2.coal \
        --popFreq ${FREQ} --tCutoff 2000 \
        --df 450 --CI 0.95 \
        --out ${WORK}/result \
        >${WORK}/inference.log 2>&1 ; then
        echo "  [SKIP] CLUES inference failed at ${POS}"
        tail -5 ${WORK}/inference.log | sed 's/^/    /'
        return 1
    fi

    # Success!
    echo "  [SUCCESS] chr6:${POS} passed all stages"
    cat ${WORK}/result_inference.txt
    # Mark as final
    ln -sfn ${TAG} ${OUTDIR}/final
    echo "final_focal=${POS}" > ${OUTDIR}/FINAL_RESULT.txt
    echo "final_freq=${FREQ}" >> ${OUTDIR}/FINAL_RESULT.txt
    cat ${WORK}/result_inference.txt >> ${OUTDIR}/FINAL_RESULT.txt
    return 0
}

# Try each candidate until one succeeds
SUCCESS=0
for cand in "${CANDIDATES[@]}"; do
    POS=$(echo $cand | awk '{print $1}')
    FREQ=$(echo $cand | awk '{print $2}')
    if try_candidate "$POS" "$FREQ"; then
        SUCCESS=1
        break
    fi
done

if [ $SUCCESS -eq 0 ]; then
    echo ""
    echo "=== ALL CANDIDATES FAILED ==="
    echo "None of the ranked focal variants passed RelateToCLUES + inference."
    echo "Per-candidate logs in ${OUTDIR}/pos*/"
    exit 1
fi

echo ""
echo "=== TREM2 OoA CLUES v2 done at $(date) ==="
echo "Final result:"
cat ${OUTDIR}/FINAL_RESULT.txt
