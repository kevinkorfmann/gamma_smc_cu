#!/bin/bash
#SBATCH --job-name=clues2
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=clues2_%j.log

set -euo pipefail

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues
RELATE=${BASE}/relate_src
CLUES=${BASE}/clues
TREES=${BASE}/1kg_trees/trees
COAL=${BASE}/1kg_trees/popsizes/1000GP_CHBGBRYRI_mask_ne.coal
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python
OUTDIR=${BASE}/results
MU=1.25e-8

mkdir -p ${OUTDIR}

declare -A LOCI_CHR LOCI_POS LOCI_DESC

# Already have chr11 trees
LOCI_CHR[FADS1]=11;      LOCI_POS[FADS1]=61795508;      LOCI_DESC[FADS1]="Fatty acid desaturation (multi-pop)"
LOCI_CHR[TRAF6]=11;      LOCI_POS[TRAF6]=36511382;      LOCI_DESC[TRAF6]="Mucosal immunity NF-kB (EAS)"

# Newly extracted chr12
LOCI_CHR[CLEC6A]=12;     LOCI_POS[CLEC6A]=7679508;      LOCI_DESC[CLEC6A]="Mucosal immunity fungal receptor (EAS)"
LOCI_CHR[SLC6A15]=12;    LOCI_POS[SLC6A15]=85331206;    LOCI_DESC[SLC6A15]="Brain transporter (EAS)"
LOCI_CHR[CCDC92]=12;     LOCI_POS[CCDC92]=123959059;    LOCI_DESC[CCDC92]="Metabolic/obesity (EAS)"

# Newly extracted chr4, chr22, chr7
LOCI_CHR[JCHAIN]=4;      LOCI_POS[JCHAIN]=70673306;     LOCI_DESC[JCHAIN]="IgA transport mucosal (EAS)"
LOCI_CHR[TNFRSF13C]=22;  LOCI_POS[TNFRSF13C]=42279628;  LOCI_DESC[TNFRSF13C]="BAFF receptor mucosal (EAS)"
LOCI_CHR[CYP3A5]=7;      LOCI_POS[CYP3A5]=99647039;     LOCI_DESC[CYP3A5]="Salt retention (EUR)"

GENES=(FADS1 TRAF6 CLEC6A SLC6A15 CCDC92 JCHAIN TNFRSF13C CYP3A5)

for GENE in "${GENES[@]}"; do
    CHR=${LOCI_CHR[$GENE]}
    POS=${LOCI_POS[$GENE]}
    TREE_PREFIX=${TREES}/1000GP_CHBGBRYRI_mask_chr${CHR}

    echo "===== ${GENE} (chr${CHR}:${POS}) — ${LOCI_DESC[$GENE]} ====="

    if [ ! -f "${TREE_PREFIX}.anc" ]; then
        echo "  ERROR: ${TREE_PREFIX}.anc not found, skipping"
        continue
    fi

    # Step 1: SampleBranchLengths (skip if exists)
    if [ -f "${OUTDIR}/${GENE}_resample.timeb" ]; then
        echo "  .timeb already exists, skipping SampleBranchLengths"
    else
        echo "  Step 1: SampleBranchLengths (100 samples)..."
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

    # Step 2: CLUES inference (skip if exists)
    if [ -f "${OUTDIR}/${GENE}_clues.post.npy" ]; then
        echo "  CLUES output already exists, skipping"
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

echo "===== Batch 2 complete ====="
ls -la ${OUTDIR}/*clues*
