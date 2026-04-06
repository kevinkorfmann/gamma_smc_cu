#!/bin/bash
# Submit population-specific CLUES jobs — one per gene, all parallel

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python
SCRIPT=${BASE}/run_popspecific_clues.py

submit() {
    GENE=$1; CHR=$2; POS=$3; POP=$4; REGION=$5
    sbatch --job-name="ps_${GENE}" \
           --partition=genoa-std-mem \
           --cpus-per-task=4 \
           --mem=16G \
           --time=04:00:00 \
           --output="${BASE}/results_popspecific/${GENE}_${POP}_%j.log" \
           --wrap="${PYTHON} -u ${SCRIPT} ${GENE} ${CHR} ${POS} ${POP} ${REGION} 2>&1"
}

# EAS genes — run on CHB
submit CLEC6A    12 7679508    CHB 2000
submit TRAF6     11 36511382   CHB 2000
submit TNFRSF13C 22 42279628   CHB 2000
submit JCHAIN    4  70673306   CHB 2000
submit CCDC92    12 123959059  CHB 2000
submit SLC6A15   12 85331206   CHB 2000

# EUR genes — run on GBR
submit CYP3A5    7  99647039   GBR 2000
submit FADS1     11 61795508   GBR 2000
