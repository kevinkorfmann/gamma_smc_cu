#!/bin/bash
#SBATCH --job-name=ihs_neutral
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/ihs_neutral_%j.log

set -euo pipefail
BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12
mkdir -p ${BASE}/logs ${BASE}/results/orthogonal
${PYTHON} ${BASE}/run_ihs_neutral.py
