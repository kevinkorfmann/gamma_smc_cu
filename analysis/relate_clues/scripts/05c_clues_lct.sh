#!/bin/bash
#SBATCH --job-name=rc-clues-lct
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/relate_clues/logs/05c_clues_lct_%j.log

# Step 5: CLUES2 inference at LCT only.

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
CLUES2=${BASE}/tools/CLUES2
PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export PYTHONPATH="${CLUES2}:${PYTHONPATH:-}"

echo "=== Step 5: CLUES2 (LCT, CEU) ==="
echo "Start: $(date)"

OUTDIR=${BASE}/clues/LCT

# Convert Relate sampled trees to CLUES2 format
python ${CLUES2}/RelateToCLUES.py \
    --RelateSamples ${OUTDIR}/lct_sampled.newick \
    --DerivedFile ${OUTDIR}/lct_sampled.sites \
    --out ${OUTDIR}/lct

# Run inference — LCT lactase persistence allele ~70% in CEU
python ${CLUES2}/inference.py \
    --times ${OUTDIR}/lct_times.txt \
    --coal ${BASE}/trees/chr2_EUR/chr2_popsize.coal \
    --popFreq 0.70 \
    --tCutoff 1000 \
    --df 450 --CI 0.95 \
    --out ${OUTDIR}/lct_result

echo "  Inference result:"
cat ${OUTDIR}/lct_result_inference.txt

# Plot trajectory
python ${CLUES2}/plot_traj.py \
    --freqs ${OUTDIR}/lct_result_freqs.txt \
    --post ${OUTDIR}/lct_result_post.txt \
    --figure ${OUTDIR}/lct_trajectory.png \
    --generation_time 28

echo "  Trajectory saved."
ls -lh ${OUTDIR}/lct_*
echo "=== CLUES2 done at $(date) ==="
