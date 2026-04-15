#!/bin/bash
#SBATCH --job-name=rc-prep-ext
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=analysis/relate_clues/logs/01b_prepare_%j.log

# Prepare Relate inputs for chr12 (EAS) and chr20 (EUR+SAS)
# chr12: SLC6A15 (CHS), CCDC92 (CDX), CLEC6A (CDX) — all EAS
# chr20: BPIFA2 (GIH) — SAS, keep EUR+SAS like chr11

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4/bin
INPUT=${BASE}/input_files
DATA_DIR=analysis/genome_wide/data
PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${RELATE}:${PATH}"

echo "=== Prepare chr12 + chr20 inputs ==="
echo "Start: $(date)"

POPLABELS=${BASE}/data/all.poplabels

# Create output dirs
mkdir -p ${BASE}/data/chr12 ${BASE}/data/chr20

# ── Build remove_ids for chr12 (EAS only) ──
python3 -c "
with open('${POPLABELS}') as f:
    next(f)
    remove = []
    for line in f:
        parts = line.strip().split()
        if parts[2] != 'EAS':
            remove.append(parts[0])
with open('${BASE}/data/chr12/remove_ids.txt', 'w') as f:
    for s in remove:
        f.write(s + '\n')
print(f'chr12: removing {len(remove)} samples (keeping EAS)')
"

# ── Build remove_ids for chr20 (EUR+SAS) ──
python3 -c "
with open('${POPLABELS}') as f:
    next(f)
    remove = []
    for line in f:
        parts = line.strip().split()
        if parts[2] not in ('EUR', 'SAS'):
            remove.append(parts[0])
with open('${BASE}/data/chr20/remove_ids.txt', 'w') as f:
    for s in remove:
        f.write(s + '\n')
print(f'chr20: removing {len(remove)} samples (keeping EUR+SAS)')
"

# ── Convert and prepare each chromosome ──
for CHR in 12 20; do
    echo ""
    echo "--- chr${CHR} ---"
    OUTDIR=${BASE}/data/chr${CHR}
    VCF_PREFIX=${DATA_DIR}/chr${CHR}

    echo "  Converting VCF to haps/sample..."
    RelateFileFormats --mode ConvertFromVcf \
        --haps ${OUTDIR}/chr${CHR}.haps \
        --sample ${OUTDIR}/chr${CHR}.sample \
        -i ${VCF_PREFIX}

    echo "  Haps: $(wc -l < ${OUTDIR}/chr${CHR}.haps) sites"

    GRCH38=${INPUT}/Relate_input_files/GRCh38
    ANCESTOR=${GRCH38}/human_ancestor_GRCh38/homo_sapiens_ancestor_${CHR}.fa.gz
    if [ -f "${ANCESTOR}" ] && [ ! -f "${ANCESTOR%.gz}" ]; then
        gunzip -k "${ANCESTOR}"
    fi
    ANCESTOR=${ANCESTOR%.gz}
    MASK=${GRCH38}/20160622_genome_mask_GRCh38/StrictMask/20160622.chr${CHR}.mask.fasta.gz
    MAP=${GRCH38}/recomb_map/genetic_map_chr${CHR}.txt

    if [ ${CHR} -eq 12 ]; then
        PREFIX="chr12_eas"
    else
        PREFIX="chr20_eursas"
    fi

    echo "  Running PrepareInputFiles -> ${PREFIX}..."
    PREP_CMD="${BASE}/tools/relate_v1.2.4/scripts/PrepareInputFiles/PrepareInputFiles.sh"
    PREP_CMD="${PREP_CMD} --haps ${OUTDIR}/chr${CHR}.haps --sample ${OUTDIR}/chr${CHR}.sample"
    [ -f "${ANCESTOR}" ] && PREP_CMD="${PREP_CMD} --ancestor ${ANCESTOR}"
    [ -f "${MASK}" ] && PREP_CMD="${PREP_CMD} --mask ${MASK}"
    PREP_CMD="${PREP_CMD} --remove_ids ${OUTDIR}/remove_ids.txt"
    PREP_CMD="${PREP_CMD} --poplabels ${POPLABELS}"
    PREP_CMD="${PREP_CMD} -o ${OUTDIR}/${PREFIX}"

    ${PREP_CMD}
    echo "  Prepared:"
    ls -lh ${OUTDIR}/${PREFIX}.*
done

echo "=== Prepare complete at $(date) ==="
