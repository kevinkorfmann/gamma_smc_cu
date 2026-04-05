#!/bin/bash
#SBATCH --job-name=cxt_slc6a15
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/cxt_slc6a15_%j.log

set -euo pipefail
export CXT_CHECKPOINT_CACHE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt/.cxt_cache
mkdir -p ${CXT_CHECKPOINT_CACHE}

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12

mkdir -p ${BASE}/logs ${BASE}/results

${PYTHON} ${BASE}/run_cxt_region.py \
    --region SLC6A15 \
    --chr 12 --start 84000000 --end 86000000 \
    --pop CDX \
    --n-pairs 100 \
    --outdir ${BASE}/results
