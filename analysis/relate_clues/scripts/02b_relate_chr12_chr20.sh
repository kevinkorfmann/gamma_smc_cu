#!/bin/bash
#SBATCH --job-name=rc-relate-ext
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --array=1-2
#SBATCH --output=analysis/relate_clues/logs/02b_relate_%A_%a.log

# Relate inference for chr12 (EAS) and chr20 (EUR+SAS)
# Array 1: chr12 EAS (~970 haplotypes: CDX 93 + CHB 103 + CHS 163 + JPT 104 + KHV 122 = 585 samples = 1170 haps)
# Array 2: chr20 EUR+SAS (~1900 haplotypes, same as chr11)

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues
RELATE=${BASE}/tools/relate_v1.2.4/bin
INPUT=${BASE}/input_files

echo "=== Relate inference (task ${SLURM_ARRAY_TASK_ID}) ==="
echo "Node: $(hostname), Start: $(date)"

if [ ${SLURM_ARRAY_TASK_ID} -eq 1 ]; then
    CHR=12
    PREFIX="chr12_eas"
    OUTDIR=${BASE}/trees/chr12_EAS
    Ne=20000
elif [ ${SLURM_ARRAY_TASK_ID} -eq 2 ]; then
    CHR=20
    PREFIX="chr20_eursas"
    OUTDIR=${BASE}/trees/chr20_EURSAS
    Ne=30000
fi

mkdir -p ${OUTDIR}

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

echo "  CHR=${CHR}, PREFIX=${PREFIX}, Ne=${Ne}"
echo "  HAPS=$(wc -l < ${HAPS}) sites"

cd ${OUTDIR}
REPO_ABS=/vast/projects/smathi/cohort/kkor/tmrca.cu

${REPO_ABS}/${RELATE}/Relate --mode All \
    -m 1.25e-8 -N ${Ne} \
    --haps ${REPO_ABS}/${HAPS} --sample ${REPO_ABS}/${SAMPLE} \
    --map ${REPO_ABS}/${MAP} \
    --annot ${REPO_ABS}/${ANNOT} --dist ${REPO_ABS}/${DIST} \
    --memory 128 --seed 42 \
    -o chr${CHR}_relate

echo "  Output:"
ls -lh chr${CHR}_relate.*
echo "=== Done at $(date) ==="
