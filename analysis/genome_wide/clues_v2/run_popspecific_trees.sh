#!/bin/bash
#SBATCH --job-name=relate_pop
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=1-4
#SBATCH --output=logs/relate_%a_%j.log

# Build population-specific Relate trees for CLUES analysis.
# Array index maps to: 1=LCT(CEU,chr2), 2=SLC24A5(GBR,chr15), 3=EDAR(CHB,chr2), 4=GRK2(SAS,chr11)
#
# Requires: haplotype files pre-filtered to focal population only.
# These must be prepared from 1KG VCFs before running this script.

set -euo pipefail

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues_v2
RELATE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues/relate_src
DATA=${BASE}/data
TREES=${BASE}/trees
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12
MU=1.25e-8
NE=20000

mkdir -p ${TREES} ${BASE}/logs

# Define loci
declare -a POPS=(CEU GBR CHB BEB)
declare -a CHRS=(2 15 2 11)
declare -a GENES=(LCT SLC24A5 EDAR GRK2)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
POP=${POPS[$IDX]}
CHR=${CHRS[$IDX]}
GENE=${GENES[$IDX]}

echo "Building Relate tree: ${GENE} (${POP}, chr${CHR})"

# Step 1: Prepare population-specific haplotype files from 1KG VCF
HAPS=${DATA}/${POP}_chr${CHR}.haps.gz
SAMPLE=${DATA}/${POP}_chr${CHR}.sample.gz
DIST=${DATA}/${POP}_chr${CHR}.dist.gz

if [ ! -f "${HAPS}" ]; then
    echo "Preparing haplotype files for ${POP} chr${CHR}..."
    ${PYTHON} ${BASE}/prepare_haps.py --pop ${POP} --chr ${CHR} --outdir ${DATA}
fi

if [ ! -f "${HAPS}" ]; then
    echo "ERROR: ${HAPS} not found after preparation"
    exit 1
fi

# Step 2: Build Relate tree
echo "Running Relate..."
cd ${TREES}
rm -rf ${POP}_chr${CHR}
${RELATE}/scripts/RelateParallel/RelateParallel.sh \
    --threads 8 \
    --haps ${HAPS} \
    --sample ${SAMPLE} \
    --map ${BASE}/data/recomb_maps/relate_map_chr${CHR}.txt \
    -m ${MU} \
    -N ${NE} \
    -o ${POP}_chr${CHR}

echo "Relate tree built: ${TREES}/${POP}_chr${CHR}"

# Step 3: Estimate population size
echo "Estimating population size..."
${RELATE}/scripts/EstimatePopulationSize/EstimatePopulationSize.sh \
    -i ${TREES}/${POP}_chr${CHR} \
    -m ${MU} \
    --poplabels ${DATA}/${POP}_chr${CHR}.poplabels \
    -o ${TREES}/${POP}_chr${CHR}_popsize \
    --threshold 0 \
    --num_iter 5

echo "Done: ${GENE} (${POP}, chr${CHR})"
