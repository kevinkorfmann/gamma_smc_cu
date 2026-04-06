#!/bin/bash
#SBATCH --job-name=cxt_grk2_fire
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/cxt_grk2_fire_%A_%a.log
#SBATCH --array=0-1

set -euo pipefail
export CXT_CHECKPOINT_CACHE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt/.cxt_cache
mkdir -p ${CXT_CHECKPOINT_CACHE}

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12

mkdir -p ${BASE}/logs ${BASE}/results/fire_cxt

POPS=(BEB YRI)
POP=${POPS[$SLURM_ARRAY_TASK_ID]}

echo "=== cxt fire plot: ${POP} (all 1225 pairs) ==="

${PYTHON} ${BASE}/run_cxt_region.py \
    --region GRK2_fire \
    --chr 11 --start 66000000 --end 68000000 \
    --pop ${POP} \
    --n-pairs 1225 \
    --outdir ${BASE}/results/fire_cxt

echo "=== Done: ${POP} ==="
