#!/bin/bash
#SBATCH --job-name=rc-clues-grk2
#SBATCH --partition=genoa-lrg-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/relate_clues/logs/05d_clues_grk2_%j.log

# Step 5: CLUES2 at GRK2 + neutral control.

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
CLUES2=${BASE}/tools/CLUES2
PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export PYTHONPATH="${CLUES2}:${PYTHONPATH:-}"

echo "=== Step 5: CLUES2 (GRK2 + neutral) ==="
echo "Start: $(date)"

run_clues2() {
    local LABEL=$1 NEWICK=$2 SITES=$3 COAL=$4 POP_FREQ=$5 T_CUTOFF=$6 OUTDIR=$7
    echo ""
    echo "--- ${LABEL} (popFreq=${POP_FREQ}, tCutoff=${T_CUTOFF}) ---"
    python ${CLUES2}/RelateToCLUES.py \
        --RelateSamples ${NEWICK} --DerivedFile ${SITES} --out ${OUTDIR}/${LABEL,,}
    python ${CLUES2}/inference.py \
        --times ${OUTDIR}/${LABEL,,}_times.txt --coal ${COAL} \
        --popFreq ${POP_FREQ} --tCutoff ${T_CUTOFF} --df 450 --CI 0.95 \
        --out ${OUTDIR}/${LABEL,,}_result
    echo "  Inference:"
    cat ${OUTDIR}/${LABEL,,}_result_inference.txt
    python ${CLUES2}/plot_traj.py \
        --freqs ${OUTDIR}/${LABEL,,}_result_freqs.txt \
        --post ${OUTDIR}/${LABEL,,}_result_post.txt \
        --figure ${OUTDIR}/${LABEL,,}_trajectory.png --generation_time 28
}

run_clues2 "GRK2" \
    "${BASE}/clues/GRK2/grk2_sampled.newick" \
    "${BASE}/clues/GRK2/grk2_sampled.sites" \
    "${BASE}/trees/chr11_EURSAS/chr11_popsize.coal" \
    0.98 2000 "${BASE}/clues/GRK2"

run_clues2 "NEUTRAL" \
    "${BASE}/clues/neutral/neutral_sampled.newick" \
    "${BASE}/clues/neutral/neutral_sampled.sites" \
    "${BASE}/trees/chr11_EURSAS/chr11_popsize.coal" \
    0.50 2000 "${BASE}/clues/neutral"

echo ""
echo "=== CLUES2 done at $(date) ==="
