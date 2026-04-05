#!/bin/bash
#SBATCH --job-name=cxt_fads1
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/cxt_fads1_%j.log

set -euo pipefail
export CXT_CHECKPOINT_CACHE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt/.cxt_cache
mkdir -p ${CXT_CHECKPOINT_CACHE}

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12

mkdir -p ${BASE}/logs ${BASE}/results

# S3 panel (c): FADS1, multi-pop sweep, chr11:60.5-62 Mb
# Run focal pop ITU (strongest SAS signal)
${PYTHON} ${BASE}/run_cxt_region.py \
    --region FADS1 \
    --chr 11 --start 60500000 --end 62000000 \
    --pop ITU \
    --n-pairs 100 \
    --outdir ${BASE}/results
