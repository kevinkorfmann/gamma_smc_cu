#!/bin/bash
#SBATCH --job-name=tmrca-post
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/genome_wide/logs/postprocess_%j.log

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export PYTHONPATH="$(pwd)/python:$(pwd)"

# Make sure gseapy + statsmodels are available
python -c "import gseapy" 2>/dev/null || pip install --user gseapy 2>&1 | tail -3
python -c "import statsmodels" 2>/dev/null || pip install --user statsmodels 2>&1 | tail -3

python analysis/genome_wide/postprocess.py
