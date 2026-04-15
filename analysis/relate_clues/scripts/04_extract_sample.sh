#!/bin/bash
#SBATCH --job-name=rc-extract
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=analysis/relate_clues/logs/04_extract_%j.log

# Step 4: Extract population subtrees and sample branch lengths at focal SNPs.
#
# For each locus:
#   1. Extract subtrees for the focal population
#   2. Sample branch lengths in a window around the focal SNP
#   3. Output in newick format for CLUES2
#
# Focal loci:
#   GRK2:    chr11:67,200,000-67,500,000 (midpoint 67,276,514; most-diff 67,407,126)
#   LCT:     chr2:135,700,000-136,000,000 (rs4988235 at 135,851,076)
#   Neutral:  use a gene at TMRCA ~50th percentile on chr11

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4

echo "=== Step 4: Extract subtrees + sample branch lengths ==="
echo "Start: $(date)"

# ── GRK2 (chr11, GIH from EUR+SAS trees) ──
echo ""
echo "--- GRK2 ---"
TREEDIR=${BASE}/trees/chr11_EURSAS
POPLABELS=${BASE}/data/all.poplabels

# Extract GIH subtree
${RELATE}/bin/RelateExtract --mode SubTreesForSubpopulation \
    --anc ${TREEDIR}/chr11_popsize.anc.gz \
    --mut ${TREEDIR}/chr11_popsize.mut.gz \
    --poplabels ${POPLABELS} \
    --pop_of_interest GIH \
    -o ${TREEDIR}/chr11_GIH

echo "  GIH subtree extracted."

# Sample branch lengths at GRK2 window
${RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
    -i ${TREEDIR}/chr11_GIH \
    -o ${BASE}/clues/GRK2/grk2_sampled \
    -m 1.25e-8 \
    --coal ${TREEDIR}/chr11_popsize.coal \
    --num_samples 200 \
    --first_bp 67200000 --last_bp 67500000 \
    --format n --seed 42

echo "  GRK2 branch lengths sampled:"
ls -lh ${BASE}/clues/GRK2/grk2_sampled.*

# ── LCT (chr2, CEU from EUR trees) ──
echo ""
echo "--- LCT ---"
TREEDIR2=${BASE}/trees/chr2_EUR

${RELATE}/bin/RelateExtract --mode SubTreesForSubpopulation \
    --anc ${TREEDIR2}/chr2_popsize.anc.gz \
    --mut ${TREEDIR2}/chr2_popsize.mut.gz \
    --poplabels ${POPLABELS} \
    --pop_of_interest CEU \
    -o ${TREEDIR2}/chr2_CEU

echo "  CEU subtree extracted."

${RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
    -i ${TREEDIR2}/chr2_CEU \
    -o ${BASE}/clues/LCT/lct_sampled \
    -m 1.25e-8 \
    --coal ${TREEDIR2}/chr2_popsize.coal \
    --num_samples 200 \
    --first_bp 135700000 --last_bp 136000000 \
    --format n --seed 42

echo "  LCT branch lengths sampled:"
ls -lh ${BASE}/clues/LCT/lct_sampled.*

# ── Neutral control (chr11, GIH — use a gene at ~50th TMRCA percentile) ──
# AQP3 on chr9 was the original neutral control but we need chr11 trees.
# Use ANKRD13D (chr11:67,289,300-67,302,485) which is right next to GRK2
# but should be neutral if GRK2's sweep is localized.
# Better: pick a gene far from GRK2 on chr11 — e.g., KCNQ1 at chr11:2.4 Mb
echo ""
echo "--- Neutral control (KCNQ1, chr11:2.4 Mb) ---"

${RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
    -i ${TREEDIR}/chr11_GIH \
    -o ${BASE}/clues/neutral/neutral_sampled \
    -m 1.25e-8 \
    --coal ${TREEDIR}/chr11_popsize.coal \
    --num_samples 200 \
    --first_bp 2400000 --last_bp 2700000 \
    --format n --seed 42

echo "  Neutral branch lengths sampled:"
ls -lh ${BASE}/clues/neutral/neutral_sampled.*

echo ""
echo "=== Extract complete at $(date) ==="
