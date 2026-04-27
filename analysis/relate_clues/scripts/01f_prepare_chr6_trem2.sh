#!/bin/bash
#SBATCH --job-name=rc-prep-trem2
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=analysis/relate_clues/logs/01f_prep_trem2_%j.log

# Prepare chr6 Relate inputs for TREM2 analysis.
# Keep EUR + EAS (matches TREM2's 10/10 replication pattern, ~1500 haps).

set -euo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4/bin
INPUT=${BASE}/input_files
DATA_DIR=analysis/genome_wide/data
PIXI_ENV=.pixi/envs/default
export PATH="$(pwd)/${PIXI_ENV}/bin:${RELATE}:${PATH}"

CHR=6
PREFIX="chr6_eureas"
OUTDIR=${BASE}/data/chr${CHR}
mkdir -p ${OUTDIR}

echo "=== Prepare chr6 Relate inputs (EUR+EAS) at $(date) ==="
POPLABELS=${BASE}/data/all.poplabels

# remove_ids: drop AFR, SAS, AMR → keep EUR + EAS
python3 -c "
with open('${POPLABELS}') as f:
    next(f); remove = []
    for line in f:
        p = line.strip().split()
        if p[2] not in ('EUR', 'EAS'):
            remove.append(p[0])
with open('${OUTDIR}/remove_ids_trem2.txt', 'w') as f:
    for s in remove:
        f.write(s + '\\n')
print(f'chr6 TREM2: removing {len(remove)} samples (keeping EUR+EAS)')
"

# Convert VCF -> haps/sample if not done
if [ ! -f "${OUTDIR}/chr6.haps" ]; then
    echo "Converting chr6 VCF -> haps/sample..."
    RelateFileFormats --mode ConvertFromVcf \
        --haps ${OUTDIR}/chr${CHR}.haps \
        --sample ${OUTDIR}/chr${CHR}.sample \
        -i ${DATA_DIR}/chr${CHR}
fi
echo "  Haps: $(wc -l < ${OUTDIR}/chr${CHR}.haps) sites"

# PrepareInputFiles
GRCH38=${INPUT}/Relate_input_files/GRCh38
ANCESTOR=${GRCH38}/human_ancestor_GRCh38/homo_sapiens_ancestor_${CHR}.fa.gz
if [ -f "${ANCESTOR}" ] && [ ! -f "${ANCESTOR%.gz}" ]; then gunzip -k "${ANCESTOR}"; fi
ANCESTOR=${ANCESTOR%.gz}
MASK=${GRCH38}/20160622_genome_mask_GRCh38/StrictMask/20160622.chr${CHR}.mask.fasta.gz

PREP=${BASE}/tools/relate_v1.2.4/scripts/PrepareInputFiles/PrepareInputFiles.sh
${PREP} --haps ${OUTDIR}/chr${CHR}.haps --sample ${OUTDIR}/chr${CHR}.sample \
    --ancestor ${ANCESTOR} --mask ${MASK} \
    --remove_ids ${OUTDIR}/remove_ids_trem2.txt --poplabels ${POPLABELS} \
    -o ${OUTDIR}/${PREFIX}

echo "  Prepared files:"
ls -lh ${OUTDIR}/${PREFIX}.*
echo "=== Done at $(date) ==="
