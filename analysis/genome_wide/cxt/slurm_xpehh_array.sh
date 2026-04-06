#!/bin/bash
#SBATCH --job-name=xpehh
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=1-22
#SBATCH --output=logs/xpehh_chr%a_%j.log

set -euo pipefail
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12
${PYTHON} /vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt/run_xpehh_genomewide.py ${SLURM_ARRAY_TASK_ID} EAS
