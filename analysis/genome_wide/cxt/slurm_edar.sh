#!/bin/bash
#SBATCH --job-name=cxt_edar
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/cxt_edar_%j.log

set -euo pipefail
export CXT_CHECKPOINT_CACHE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt/.cxt_cache
mkdir -p ${CXT_CHECKPOINT_CACHE}

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12

mkdir -p ${BASE}/logs ${BASE}/results

${PYTHON} ${BASE}/run_cxt_region.py \
    --region EDAR \
    --chr 2 --start 108000000 --end 110000000 \
    --pop CHB \
    --n-pairs 100 \
    --outdir ${BASE}/results
