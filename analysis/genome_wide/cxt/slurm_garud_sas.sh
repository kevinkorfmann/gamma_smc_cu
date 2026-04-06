#!/bin/bash
#SBATCH --job-name=garud_sas
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=logs/garud_sas_%j.log

set -euo pipefail
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12
${PYTHON} /vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt/run_garud_sas.py
