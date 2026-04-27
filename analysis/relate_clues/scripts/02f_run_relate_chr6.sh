#!/bin/bash
#SBATCH --job-name=rc-relate-trem2
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --output=analysis/relate_clues/logs/02f_relate_chr6_%j.log

set -euo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4/bin
INPUT=${BASE}/input_files

CHR=6
PREFIX="chr6_eureas"
OUTDIR=${BASE}/trees/chr6_EUREAS
Ne=25000
mkdir -p ${OUTDIR}

echo "=== Relate chr6 EUR+EAS at $(date) ==="

# Decompress prepared files if gzipped
for ext in haps sample dist; do
    GZ=${BASE}/data/chr${CHR}/${PREFIX}.${ext}.gz
    PLAIN=${BASE}/data/chr${CHR}/${PREFIX}.${ext}
    if [ -f "${GZ}" ] && [ ! -f "${PLAIN}" ]; then gunzip -k "${GZ}"; fi
done

HAPS=${BASE}/data/chr${CHR}/${PREFIX}.haps
SAMPLE=${BASE}/data/chr${CHR}/${PREFIX}.sample
ANNOT=${BASE}/data/chr${CHR}/${PREFIX}.annot
DIST=${BASE}/data/chr${CHR}/${PREFIX}.dist
MAP=${INPUT}/Relate_input_files/GRCh38/recomb_map/genetic_map_chr${CHR}.txt

cd ${OUTDIR}
REPO_ABS=/vast/projects/smathi/cohort/kkor/tmrca.cu
${REPO_ABS}/${RELATE}/Relate --mode All \
    -m 1.25e-8 -N ${Ne} \
    --haps ${REPO_ABS}/${HAPS} --sample ${REPO_ABS}/${SAMPLE} \
    --map ${REPO_ABS}/${MAP} \
    --annot ${REPO_ABS}/${ANNOT} --dist ${REPO_ABS}/${DIST} \
    --memory 128 --seed 42 \
    -o chr${CHR}_relate

ls -lh ${OUTDIR}/chr${CHR}_relate.*
echo "=== Relate done at $(date) ==="
