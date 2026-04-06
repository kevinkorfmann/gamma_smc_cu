#!/bin/bash
#SBATCH --job-name=relate_chr2
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=1-2
#SBATCH --output=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues_v2/logs/relate_chr2_%a_%j.log

set -euo pipefail

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues_v2
RELATE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues/relate_src
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12
MAPDIR=${BASE}/data/recomb_maps
MU=1.25e-8
NE=20000
PAD=2000000

declare -a POPS=(CEU CHB)
declare -a GENES=(LCT EDAR)
declare -a POSITIONS=(135851076 108894481)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
POP=${POPS[$IDX]}
GENE=${GENES[$IDX]}
POS=${POSITIONS[$IDX]}
CHR=2
TAG="${GENE}_${POP}"
START=$((POS - PAD))
END=$((POS + PAD))

echo "=== ${TAG} chr${CHR}:${START}-${END} ==="

# Prepare
${PYTHON} ${BASE}/prepare_haps.py --pop ${POP} --chr ${CHR} --start ${START} --end ${END} --suffix "_${GENE}" --outdir ${BASE}/data
mv ${BASE}/data/${POP}_chr${CHR}_${GENE}.haps.gz ${BASE}/data/${TAG}.haps.gz
mv ${BASE}/data/${POP}_chr${CHR}_${GENE}.sample.gz ${BASE}/data/${TAG}.sample.gz
mv ${BASE}/data/${POP}_chr${CHR}_${GENE}.poplabels ${BASE}/data/${TAG}.poplabels
mv ${BASE}/data/${POP}_chr${CHR}_${GENE}.dist.gz ${BASE}/data/${TAG}.dist.gz 2>/dev/null || true

# Build tree
cd ${BASE}/trees
rm -rf ${TAG}
${RELATE}/scripts/RelateParallel/RelateParallel.sh \
    --threads 8 \
    --haps ${BASE}/data/${TAG}.haps.gz \
    --sample ${BASE}/data/${TAG}.sample.gz \
    --map ${MAPDIR}/relate_map_chr${CHR}.txt \
    -m ${MU} -N ${NE} \
    -o ${TAG}

ls -lh ${BASE}/trees/${TAG}.* 2>/dev/null && echo "SUCCESS" || echo "FAILED"
