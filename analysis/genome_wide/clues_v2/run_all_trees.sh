#!/bin/bash
#SBATCH --job-name=relate_v2
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-4
#SBATCH --output=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues_v2/logs/relate_v2_%a_%j.log

set -euo pipefail

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues_v2
RELATE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues/relate_src
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12
MAPDIR=${BASE}/data/recomb_maps
MU=1.25e-8
NE=20000
PAD=2000000

mkdir -p ${BASE}/trees ${BASE}/logs

# Gene regions: ±2Mb around each focal SNP
declare -a POPS=(CEU GBR CHB BEB)
declare -a CHRS=(2 15 2 11)
declare -a GENES=(LCT SLC24A5 EDAR GRK2)
declare -a POSITIONS=(135851076 48426492 108894481 67407126)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
POP=${POPS[$IDX]}
CHR=${CHRS[$IDX]}
GENE=${GENES[$IDX]}
POS=${POSITIONS[$IDX]}

START=$((POS - PAD))
END=$((POS + PAD))
if [ ${START} -lt 0 ]; then START=0; fi

TAG="${GENE}_${POP}"

echo "=== ${GENE}: ${POP} chr${CHR}:${START}-${END} (±${PAD}bp) ==="

# Step 1: Prepare region haps
HAPS=${BASE}/data/${TAG}.haps.gz
if [ ! -f "${HAPS}" ]; then
    echo "Preparing haplotype files..."
    ${PYTHON} ${BASE}/prepare_haps.py \
        --pop ${POP} --chr ${CHR} \
        --start ${START} --end ${END} \
        --suffix "_${GENE}" \
        --outdir ${BASE}/data
    # Rename to TAG format
    mv ${BASE}/data/${POP}_chr${CHR}_${GENE}.haps.gz ${BASE}/data/${TAG}.haps.gz
    mv ${BASE}/data/${POP}_chr${CHR}_${GENE}.sample.gz ${BASE}/data/${TAG}.sample.gz
    mv ${BASE}/data/${POP}_chr${CHR}_${GENE}.poplabels ${BASE}/data/${TAG}.poplabels
    mv ${BASE}/data/${POP}_chr${CHR}_${GENE}.dist.gz ${BASE}/data/${TAG}.dist.gz 2>/dev/null || true
fi

# Step 2: Build tree
cd ${BASE}/trees
rm -rf ${TAG}
echo "Running RelateParallel..."
${RELATE}/scripts/RelateParallel/RelateParallel.sh \
    --threads 8 \
    --haps ${BASE}/data/${TAG}.haps.gz \
    --sample ${BASE}/data/${TAG}.sample.gz \
    --map ${MAPDIR}/relate_map_chr${CHR}.txt \
    -m ${MU} \
    -N ${NE} \
    -o ${TAG}

# Step 3: Check
if [ -f "${BASE}/trees/${TAG}.anc" ]; then
    echo "SUCCESS: tree built"
    ls -lh ${BASE}/trees/${TAG}.*
else
    echo "FAILED"
fi
