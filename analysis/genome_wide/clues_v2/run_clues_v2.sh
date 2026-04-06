#!/bin/bash
#SBATCH --job-name=clues_v2
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues_v2/logs/clues_v2_%j.log

# Run CLUES on population-specific Relate trees.
# Assumes trees have been built by run_popspecific_trees.sh

set -euo pipefail

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues_v2
RELATE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues/relate_src
CLUES=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues/clues
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12
TREES=${BASE}/trees
RESULTS=${BASE}/results
MU=1.25e-8

mkdir -p ${RESULTS} ${BASE}/logs

declare -A LOCI_POP LOCI_CHR LOCI_POS LOCI_DESC
LOCI_POP[LCT]=CEU;       LOCI_CHR[LCT]=2;   LOCI_POS[LCT]=135851076;   LOCI_DESC[LCT]="Lactase persistence"
LOCI_POP[SLC24A5]=GBR;   LOCI_CHR[SLC24A5]=15; LOCI_POS[SLC24A5]=48426492; LOCI_DESC[SLC24A5]="Skin pigmentation"
LOCI_POP[EDAR]=CHB;      LOCI_CHR[EDAR]=2;   LOCI_POS[EDAR]=108894481;  LOCI_DESC[EDAR]="Hair/sweat"
LOCI_POP[GRK2]=BEB;      LOCI_CHR[GRK2]=11;  LOCI_POS[GRK2]=67407126;   LOCI_DESC[GRK2]="Cardiovascular"

GENES=(LCT SLC24A5 EDAR GRK2)

for GENE in "${GENES[@]}"; do
    POP=${LOCI_POP[$GENE]}
    CHR=${LOCI_CHR[$GENE]}
    POS=${LOCI_POS[$GENE]}
    TREE_PREFIX=${TREES}/${GENE}_${POP}
    COAL=${TREES}/${GENE}_${POP}_popsize.coal

    echo "===== ${GENE} (${POP}, chr${CHR}:${POS}) — ${LOCI_DESC[$GENE]} ====="

    if [ ! -f "${TREE_PREFIX}.anc" ]; then
        echo "  ERROR: ${TREE_PREFIX}.anc not found, skipping"
        continue
    fi

    # Use existing coal file from v1 or default
    DEFAULT_COAL=${BASE}/../clues/1kg_trees/popsizes/1000GP_CHBGBRYRI_mask_ne.coal
    if [ ! -f "${COAL}" ]; then
        echo "  Using default coal file from v1"
        COAL=${DEFAULT_COAL}
    fi

    # SampleBranchLengths
    if [ -f "${RESULTS}/${GENE}_resample.timeb" ]; then
        echo "  .timeb already exists"
    else
        echo "  Step 1: SampleBranchLengths..."
        bash ${RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
            -i ${TREE_PREFIX} \
            -o ${RESULTS}/${GENE}_resample \
            -m ${MU} \
            --coal ${COAL} \
            --num_samples 100 \
            --first_bp ${POS} \
            --last_bp ${POS} \
            --format b \
            --seed 42
    fi

    # CLUES
    if [ -f "${RESULTS}/${GENE}_clues.post.npy" ]; then
        echo "  CLUES output already exists"
    else
        echo "  Step 2: CLUES inference..."
        cd ${CLUES}
        ${PYTHON} inference.py \
            --times ${RESULTS}/${GENE}_resample \
            --coal ${COAL} \
            --out ${RESULTS}/${GENE}_clues
        cd ${BASE}
    fi

    if [ -f "${RESULTS}/${GENE}_clues.post.npy" ]; then
        echo "  SUCCESS: ${GENE}"
    else
        echo "  WARNING: CLUES may have failed for ${GENE}"
    fi
    echo ""
done

echo "===== All loci done ====="

# Plot
${PYTHON} /vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues/plot_trajectories.py \
    ${RESULTS} ${RESULTS}/clues_v2_4panel.pdf
