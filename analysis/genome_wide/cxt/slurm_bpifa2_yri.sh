#!/bin/bash
#SBATCH --job-name=cxt_bpifa2_yri
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/cxt_bpifa2_yri_%j.log

set -euo pipefail
export CXT_CHECKPOINT_CACHE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt/.cxt_cache
mkdir -p ${CXT_CHECKPOINT_CACHE}

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12

mkdir -p ${BASE}/logs ${BASE}/results

${PYTHON} ${BASE}/run_cxt_region.py \
    --region BPIFA2 \
    --chr 20 --start 32000000 --end 34000000 \
    --pop YRI \
    --n-pairs 100 \
    --outdir ${BASE}/results
