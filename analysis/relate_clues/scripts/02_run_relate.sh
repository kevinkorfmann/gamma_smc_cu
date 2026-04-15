#!/bin/bash
#SBATCH --job-name=rc-relate
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --array=1-2
#SBATCH --output=analysis/relate_clues/logs/02_relate_%A_%a.log

# Step 2: Run Relate genealogy inference.
#
# Array task 1: chr11 (GRK2) with EUR+SAS (~1900 haplotypes)
# Array task 2: chr2  (LCT)  with EUR only (~900 haplotypes)
#
# This is the computationally expensive step. Expected: 6-24 hrs per chr.

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4/bin
INPUT=${BASE}/input_files

echo "=== Step 2: Relate inference (task ${SLURM_ARRAY_TASK_ID}) ==="
echo "Node: $(hostname), Start: $(date)"

if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]; then
    CHR=11
    PREFIX="chr11_eursas"
    OUTDIR=${BASE}/trees/chr11_EURSAS
    Ne=30000
elif [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]; then
    CHR=2
    PREFIX="chr2_eur"
    OUTDIR=${BASE}/trees/chr2_EUR
    Ne=20000
fi

mkdir -p ${OUTDIR}

# Decompress prepared files if gzipped (Relate needs uncompressed)
for ext in haps sample dist; do
    GZ=${BASE}/data/chr${CHR}/${PREFIX}.${ext}.gz
    PLAIN=${BASE}/data/chr${CHR}/${PREFIX}.${ext}
    if [ -f "${GZ}" ] && [ ! -f "${PLAIN}" ]; then
        echo "  Decompressing ${GZ}..."
        gunzip -k "${GZ}"
    fi
done

HAPS=${BASE}/data/chr${CHR}/${PREFIX}.haps
SAMPLE=${BASE}/data/chr${CHR}/${PREFIX}.sample
ANNOT=${BASE}/data/chr${CHR}/${PREFIX}.annot
DIST=${BASE}/data/chr${CHR}/${PREFIX}.dist
MAP=${INPUT}/Relate_input_files/GRCh38/recomb_map/genetic_map_chr${CHR}.txt

echo "  CHR=${CHR}, PREFIX=${PREFIX}"
echo "  HAPS=$(wc -l < ${HAPS}) sites"
echo "  MAP=${MAP}"
echo "  Ne=${Ne}, mu=1.25e-8"

# Relate outputs to cwd — cd into output dir and use absolute paths
cd ${OUTDIR}
REPO_ABS=/vast/projects/smathi/cohort/kkor/tmrca.cu

${REPO_ABS}/${RELATE}/Relate --mode All \
    -m 1.25e-8 -N ${Ne} \
    --haps ${REPO_ABS}/${HAPS} --sample ${REPO_ABS}/${SAMPLE} \
    --map ${REPO_ABS}/${MAP} \
    --annot ${REPO_ABS}/${ANNOT} --dist ${REPO_ABS}/${DIST} \
    --memory 128 --seed 42 \
    -o chr${CHR}_relate

echo "  Output files:"
ls -lh ${OUTDIR}/chr${CHR}_relate.*
echo "=== Relate done at $(date) ==="
