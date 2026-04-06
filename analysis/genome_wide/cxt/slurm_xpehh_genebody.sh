#!/bin/bash
#SBATCH --job-name=xpehh_gb
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/xpehh_genebody_%j.log

set -euo pipefail
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12
${PYTHON} /vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt/run_xpehh_pergene.py
