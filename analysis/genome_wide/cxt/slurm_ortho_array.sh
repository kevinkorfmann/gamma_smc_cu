#!/bin/bash
#SBATCH --job-name=ortho
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=1-22
#SBATCH --output=logs/ortho_chr%a_%j.log

set -euo pipefail

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12

mkdir -p ${BASE}/logs ${BASE}/results/orthogonal

echo "Processing chromosome ${SLURM_ARRAY_TASK_ID}"
${PYTHON} ${BASE}/run_ortho_genomewide.py ${SLURM_ARRAY_TASK_ID}
