#!/bin/bash
#SBATCH --job-name=clues
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=clues_%j.log

set -euo pipefail

# Paths
BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues
RELATE=${BASE}/relate_src
CLUES=${BASE}/clues
TREES=${BASE}/1kg_trees/trees
COAL=${BASE}/1kg_trees/popsizes/1000GP_CHBGBRYRI_mask_ne.coal
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python
OUTDIR=${BASE}/results
MU=1.25e-8

mkdir -p ${OUTDIR}

# ============================================================
# Loci to analyze
# ============================================================
declare -A LOCI_CHR LOCI_POS LOCI_DESC

LOCI_CHR[LCT]=2;        LOCI_POS[LCT]=135851076;     LOCI_DESC[LCT]="Lactase persistence (EUR)"
LOCI_CHR[SLC24A5]=15;   LOCI_POS[SLC24A5]=48426492;   LOCI_DESC[SLC24A5]="Skin pigmentation (EUR)"
LOCI_CHR[EDAR]=2;       LOCI_POS[EDAR]=108894481;     LOCI_DESC[EDAR]="Hair/sweat morphology (EAS)"
LOCI_CHR[GRK2]=11;      LOCI_POS[GRK2]=67407126;      LOCI_DESC[GRK2]="Cardiovascular (SAS+EUR) NOVEL"

GENES=(LCT SLC24A5 EDAR GRK2)

for GENE in "${GENES[@]}"; do
    CHR=${LOCI_CHR[$GENE]}
    POS=${LOCI_POS[$GENE]}
    TREE_PREFIX=${TREES}/1000GP_CHBGBRYRI_mask_chr${CHR}

    echo "===== ${GENE} (chr${CHR}:${POS}) — ${LOCI_DESC[$GENE]} ====="

    if [ ! -f "${TREE_PREFIX}.anc" ]; then
        echo "  ERROR: ${TREE_PREFIX}.anc not found, skipping"
        continue
    fi

    # Step 1: SampleBranchLengths (skip if .timeb already exists)
    if [ -f "${OUTDIR}/${GENE}_resample.timeb" ]; then
        echo "  .timeb already exists, skipping SampleBranchLengths"
    else
        echo "  Step 1: SampleBranchLengths (100 samples at focal SNP)..."
        bash ${RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
            -i ${TREE_PREFIX} \
            -o ${OUTDIR}/${GENE}_resample \
            -m ${MU} \
            --coal ${COAL} \
            --num_samples 100 \
            --first_bp ${POS} \
            --last_bp ${POS} \
            --format b \
            --seed 42

        if [ ! -f "${OUTDIR}/${GENE}_resample.timeb" ]; then
            echo "  ERROR: .timeb not created for ${GENE}"
            continue
        fi
        echo "  .timeb created."
    fi

    # Step 2: Run CLUES inference (skip if already done)
    if [ -f "${OUTDIR}/${GENE}_clues.post.npy" ]; then
        echo "  CLUES output already exists, skipping inference"
    else
        echo "  Step 2: CLUES inference..."
        cd ${CLUES}
        ${PYTHON} inference.py \
            --times ${OUTDIR}/${GENE}_resample \
            --coal ${COAL} \
            --out ${OUTDIR}/${GENE}_clues
        cd ${BASE}
    fi

    if [ -f "${OUTDIR}/${GENE}_clues.post.npy" ]; then
        echo "  SUCCESS: ${GENE} complete"
    else
        echo "  WARNING: CLUES may have failed for ${GENE}"
    fi
    echo ""
done

echo "===== All loci done ====="
ls -la ${OUTDIR}/

# Step 3: Plot trajectories
echo ""
echo "===== Plotting trajectories ====="
${PYTHON} ${BASE}/plot_trajectories.py ${OUTDIR} ${OUTDIR}/clues_trajectories.pdf
echo "Done."
