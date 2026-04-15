#!/bin/bash
#SBATCH --job-name=rc-clues-lct
#SBATCH --partition=genoa-lrg-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=analysis/relate_clues/logs/05c_clues_lct_%j.log

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
CLUES2=${BASE}/tools/CLUES2
export PATH="$(pwd)/.pixi/envs/default/bin:${PATH}"
export PYTHONPATH="${CLUES2}:${PYTHONPATH:-}"

OUTDIR=${BASE}/clues/LCT

echo "=== CLUES2 (LCT, CEU) ==="
echo "Start: $(date)"

# Convert — use pre-built derived file
python ${CLUES2}/RelateToCLUES.py \
    --RelateSamples ${OUTDIR}/lct_sampled.newick \
    --DerivedFile ${OUTDIR}/lct_derived.txt \
    --out ${OUTDIR}/lct

# Inference
python ${CLUES2}/inference.py \
    --times ${OUTDIR}/lct_times.txt \
    --coal ${BASE}/trees/chr2_EUR/chr2_popsize.coal \
    --popFreq 0.746 \
    --tCutoff 1000 \
    --df 200 --CI 0.95 \
    --out ${OUTDIR}/lct_result

echo "Inference:"
cat ${OUTDIR}/lct_result_inference.txt

# Plot
python ${CLUES2}/plot_traj.py \
    --freqs ${OUTDIR}/lct_result_freqs.txt \
    --post ${OUTDIR}/lct_result_post.txt \
    --figure ${OUTDIR}/lct_trajectory.png \
    --generation_time 28

echo "Done:"
ls -lh ${OUTDIR}/lct_*
echo "=== CLUES2 done at $(date) ==="
