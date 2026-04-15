#!/bin/bash
#SBATCH --job-name=rc-clues2
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/relate_clues/logs/05_clues2_%j.log

# Step 5: Run CLUES2 allele frequency trajectory inference at each locus.
#
# For each locus:
#   1. Convert Relate sampled branch lengths to CLUES2 format
#   2. Run CLUES2 inference (selection coefficient + trajectory)
#   3. Plot the trajectory
#
# References:
#   Vaughn & Nielsen 2024, Mol Biol Evol (CLUES2)
#   Stern et al. 2019, PLoS Genet (original CLUES)

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
CLUES2=${BASE}/tools/CLUES2
PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export PYTHONPATH="${CLUES2}:${PYTHONPATH:-}"

echo "=== Step 5: CLUES2 inference ==="
echo "Start: $(date)"

run_clues2() {
    local LABEL=$1
    local NEWICK=$2
    local SITES=$3
    local COAL=$4
    local POP_FREQ=$5
    local T_CUTOFF=$6
    local OUTDIR=$7

    echo ""
    echo "--- ${LABEL} (popFreq=${POP_FREQ}, tCutoff=${T_CUTOFF}) ---"

    # Convert Relate → CLUES2 format
    python ${CLUES2}/RelateToCLUES.py \
        --RelateSamples ${NEWICK} \
        --DerivedFile ${SITES} \
        --out ${OUTDIR}/${LABEL,,}

    # Run CLUES2 inference
    python ${CLUES2}/inference.py \
        --times ${OUTDIR}/${LABEL,,}_times.txt \
        --coal ${COAL} \
        --popFreq ${POP_FREQ} \
        --tCutoff ${T_CUTOFF} \
        --df 450 --CI 0.95 \
        --out ${OUTDIR}/${LABEL,,}_result

    echo "  Inference result:"
    cat ${OUTDIR}/${LABEL,,}_result_inference.txt

    # Plot trajectory
    python ${CLUES2}/plot_traj.py \
        --freqs ${OUTDIR}/${LABEL,,}_result_freqs.txt \
        --post ${OUTDIR}/${LABEL,,}_result_post.txt \
        --figure ${OUTDIR}/${LABEL,,}_trajectory.png \
        --generation_time 28

    echo "  Trajectory saved: ${OUTDIR}/${LABEL,,}_trajectory.png"
}

# ── GRK2: GIH, high-frequency sweep allele ──
# popFreq needs to be determined from the Relate .sites file
# (the derived allele frequency at the focal SNP in GIH)
# For now, use the value from our variant evidence: the swept allele
# is at ~98% in GIH.
run_clues2 "GRK2" \
    "${BASE}/clues/GRK2/grk2_sampled.newick" \
    "${BASE}/clues/GRK2/grk2_sampled.sites" \
    "${BASE}/trees/chr11_EURSAS/chr11_popsize.coal" \
    0.98 2000 \
    "${BASE}/clues/GRK2"

# ── LCT: CEU, ~70% lactase persistence allele ──
run_clues2 "LCT" \
    "${BASE}/clues/LCT/lct_sampled.newick" \
    "${BASE}/clues/LCT/lct_sampled.sites" \
    "${BASE}/trees/chr2_EUR/chr2_popsize.coal" \
    0.70 1000 \
    "${BASE}/clues/LCT"

# ── Neutral control ──
run_clues2 "NEUTRAL" \
    "${BASE}/clues/neutral/neutral_sampled.newick" \
    "${BASE}/clues/neutral/neutral_sampled.sites" \
    "${BASE}/trees/chr11_EURSAS/chr11_popsize.coal" \
    0.50 2000 \
    "${BASE}/clues/neutral"

echo ""
echo "=== CLUES2 complete at $(date) ==="
echo ""
echo "Summary:"
for d in GRK2 LCT neutral; do
    echo "  ${d}:"
    if [ -f ${BASE}/clues/${d}/*_result_inference.txt ]; then
        cat ${BASE}/clues/${d}/*_result_inference.txt | sed 's/^/    /'
    else
        echo "    (no result)"
    fi
done
