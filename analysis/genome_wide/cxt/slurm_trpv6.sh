#!/bin/bash
#SBATCH --job-name=cxt_trpv6
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/cxt_trpv6_%j.log

set -euo pipefail
export CXT_CHECKPOINT_CACHE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt/.cxt_cache
mkdir -p ${CXT_CHECKPOINT_CACHE}

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12

mkdir -p ${BASE}/logs ${BASE}/results

# S3 panel (f): TRPV6-KEL, EAS+SAS calcium/taste sweep, chr7:142-144 Mb
${PYTHON} ${BASE}/run_cxt_region.py \
    --region TRPV6_KEL \
    --chr 7 --start 142000000 --end 144000000 \
    --pop CHB \
    --n-pairs 100 \
    --outdir ${BASE}/results
