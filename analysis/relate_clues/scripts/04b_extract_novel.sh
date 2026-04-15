#!/bin/bash
#SBATCH --job-name=rc-extract-novel
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=analysis/relate_clues/logs/04b_extract_%j.log

# Step 4: Sample branch lengths for 4 novel candidates.
# PopSize already extracted pop subtrees.

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4

echo "=== Extract + sample for 4 novel candidates ==="
echo "Start: $(date)"

# BPIFA2 (chr20, GIH) — popsize already extracted GIH
echo "--- BPIFA2 (chr20, GIH) ---"
mkdir -p ${BASE}/clues/BPIFA2
${RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
    -i ${BASE}/trees/chr20_EURSAS/chr20_popsize \
    -o ${BASE}/clues/BPIFA2/bpifa2_sampled \
    -m 1.25e-8 \
    --coal ${BASE}/trees/chr20_EURSAS/chr20_popsize.coal \
    --num_samples 200 \
    --first_bp 31590000 --last_bp 31640000 \
    --format n --seed 42
echo "  Done:"; ls -lh ${BASE}/clues/BPIFA2/bpifa2_sampled.*

# SLC6A15 (chr12, CHS) — popsize already extracted CHS
echo "--- SLC6A15 (chr12, CHS) ---"
mkdir -p ${BASE}/clues/SLC6A15
${RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
    -i ${BASE}/trees/chr12_EAS/chr12_popsize \
    -o ${BASE}/clues/SLC6A15/slc6a15_sampled \
    -m 1.25e-8 \
    --coal ${BASE}/trees/chr12_EAS/chr12_popsize.coal \
    --num_samples 200 \
    --first_bp 84800000 --last_bp 85000000 \
    --format n --seed 42
echo "  Done:"; ls -lh ${BASE}/clues/SLC6A15/slc6a15_sampled.*

# CCDC92 (chr12, CDX) — need CDX subtree, popsize extracted CHS
# Must extract CDX from the original (pre-popsize) trees
echo "--- CCDC92 (chr12, CDX) ---"
mkdir -p ${BASE}/clues/CCDC92
POPLABELS12=${BASE}/data/chr12/chr12_eas.poplabels

${RELATE}/bin/RelateExtract --mode SubTreesForSubpopulation \
    --anc ${BASE}/trees/chr12_EAS/chr12_relate.anc \
    --mut ${BASE}/trees/chr12_EAS/chr12_relate.mut \
    --poplabels ${POPLABELS12} \
    --pop_of_interest CDX \
    -o ${BASE}/trees/chr12_EAS/chr12_CDX

${RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
    -i ${BASE}/trees/chr12_EAS/chr12_CDX \
    -o ${BASE}/clues/CCDC92/ccdc92_sampled \
    -m 1.25e-8 \
    --coal ${BASE}/trees/chr12_EAS/chr12_popsize.coal \
    --num_samples 200 \
    --first_bp 124300000 --last_bp 124500000 \
    --format n --seed 42
echo "  Done:"; ls -lh ${BASE}/clues/CCDC92/ccdc92_sampled.*

# CLEC6A (chr12, CDX) — reuse CDX subtree
echo "--- CLEC6A (chr12, CDX) ---"
mkdir -p ${BASE}/clues/CLEC6A
${RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
    -i ${BASE}/trees/chr12_EAS/chr12_CDX \
    -o ${BASE}/clues/CLEC6A/clec6a_sampled \
    -m 1.25e-8 \
    --coal ${BASE}/trees/chr12_EAS/chr12_popsize.coal \
    --num_samples 200 \
    --first_bp 9680000 --last_bp 9780000 \
    --format n --seed 42
echo "  Done:"; ls -lh ${BASE}/clues/CLEC6A/clec6a_sampled.*

echo "=== All extractions complete at $(date) ==="
