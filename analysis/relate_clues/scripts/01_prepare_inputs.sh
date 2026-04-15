#!/bin/bash
#SBATCH --job-name=rc-prepare
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=analysis/relate_clues/logs/01_prepare_%j.log

# Step 1: Convert 1KG VCFs to Relate input format and subset to target populations.
#
# For chr11 (GRK2): keep EUR + SAS (~1900 haplotypes)
# For chr2 (LCT):   keep EUR only (~900 haplotypes)
#
# Produces:
#   data/chr{N}/{prefix}.haps, .sample, .dist, .annot, .poplabels

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4/bin
INPUT=${BASE}/input_files
DATA_DIR=analysis/genome_wide/data
PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${RELATE}:${PATH}"

echo "=== Step 1: Prepare Relate inputs ==="
echo "Start: $(date)"

# ── Build poplabels file from samples.txt ──
echo "Building poplabels..."
POPLABELS=${BASE}/data/all.poplabels
python3 -c "
import csv
with open('${DATA_DIR}/samples.txt') as f:
    next(f)
    rows = []
    for line in f:
        p = line.strip().split()
        if len(p) >= 7:
            # sample, population, group, sex (1=male, 2=female)
            rows.append((p[1], p[5], p[6], '0'))
with open('${POPLABELS}', 'w') as f:
    f.write('sample population group sex\n')
    for r in rows:
        f.write(' '.join(r) + '\n')
print(f'Wrote {len(rows)} samples to poplabels')
"

# ── Build remove_ids files ──
# For chr11 (GRK2): remove AFR, EAS, AMR → keep EUR + SAS
python3 -c "
with open('${POPLABELS}') as f:
    next(f)
    remove = []
    for line in f:
        parts = line.strip().split()
        if parts[2] not in ('EUR', 'SAS'):
            remove.append(parts[0])
with open('${BASE}/data/chr11/remove_ids.txt', 'w') as f:
    for s in remove:
        f.write(s + '\n')
print(f'chr11: removing {len(remove)} samples (keeping EUR+SAS)')
"

# For chr2 (LCT): remove AFR, EAS, SAS, AMR → keep EUR only
python3 -c "
with open('${POPLABELS}') as f:
    next(f)
    remove = []
    for line in f:
        parts = line.strip().split()
        if parts[2] != 'EUR':
            remove.append(parts[0])
with open('${BASE}/data/chr2/remove_ids.txt', 'w') as f:
    for s in remove:
        f.write(s + '\n')
print(f'chr2: removing {len(remove)} samples (keeping EUR)')
"

# ── Convert VCF → haps/sample and PrepareInputFiles ──
for CHR in 2 11; do
    echo ""
    echo "--- chr${CHR} ---"
    OUTDIR=${BASE}/data/chr${CHR}

    # Convert VCF to haps/sample
    # RelateFileFormats appends .vcf.gz to -i, so strip it from our path
    VCF_PREFIX=${DATA_DIR}/chr${CHR}
    echo "  Converting VCF to haps/sample (from ${VCF_PREFIX}.vcf.gz)..."
    RelateFileFormats --mode ConvertFromVcf \
        --haps ${OUTDIR}/chr${CHR}.haps \
        --sample ${OUTDIR}/chr${CHR}.sample \
        -i ${VCF_PREFIX}

    echo "  Haps: $(wc -l < ${OUTDIR}/chr${CHR}.haps) sites"
    echo "  Samples: $(wc -l < ${OUTDIR}/chr${CHR}.sample) lines"

    # Input files from Zenodo Relate_input_files archive
    GRCH38=${INPUT}/Relate_input_files/GRCh38
    ANCESTOR=${GRCH38}/human_ancestor_GRCh38/homo_sapiens_ancestor_${CHR}.fa.gz
    # Decompress ancestor if gzipped (Relate needs uncompressed)
    if [ -f "${ANCESTOR}" ] && [ ! -f "${ANCESTOR%.gz}" ]; then
        gunzip -k "${ANCESTOR}"
    fi
    ANCESTOR=${ANCESTOR%.gz}
    MASK_DIR=${GRCH38}/20160622_genome_mask_GRCh38/StrictMask
    MASK=${MASK_DIR}/20160622.chr${CHR}.mask.fasta.gz
    MAP=${GRCH38}/recomb_map/genetic_map_chr${CHR}.txt

    echo "  Ancestor: ${ANCESTOR:-NOT FOUND}"
    echo "  Mask: ${MASK:-NOT FOUND}"
    echo "  Map: ${MAP:-NOT FOUND}"

    # PrepareInputFiles
    if [ ${CHR} -eq 11 ]; then
        PREFIX="chr11_eursas"
    else
        PREFIX="chr2_eur"
    fi

    echo "  Running PrepareInputFiles -> ${PREFIX}..."
    PREP_CMD="${BASE}/tools/relate_v1.2.4/scripts/PrepareInputFiles/PrepareInputFiles.sh"
    PREP_CMD="${PREP_CMD} --haps ${OUTDIR}/chr${CHR}.haps --sample ${OUTDIR}/chr${CHR}.sample"
    if [ -f "${ANCESTOR}" ]; then
        PREP_CMD="${PREP_CMD} --ancestor ${ANCESTOR}"
    else
        echo "  WARNING: no ancestor file, skipping ancestral annotation"
    fi
    if [ -n "${MASK}" ] && [ -f "${MASK}" ]; then
        PREP_CMD="${PREP_CMD} --mask ${MASK}"
    else
        echo "  WARNING: no mask file, skipping masking"
    fi
    PREP_CMD="${PREP_CMD} --remove_ids ${OUTDIR}/remove_ids.txt"
    PREP_CMD="${PREP_CMD} --poplabels ${POPLABELS}"
    PREP_CMD="${PREP_CMD} -o ${OUTDIR}/${PREFIX}"

    echo "  CMD: ${PREP_CMD}"
    ${PREP_CMD}

    echo "  Prepared files:"
    ls -lh ${OUTDIR}/${PREFIX}.*
done

echo ""
echo "=== Prepare complete at $(date) ==="
