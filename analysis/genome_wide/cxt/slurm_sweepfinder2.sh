#!/bin/bash
#SBATCH --job-name=sf2
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=logs/sweepfinder2_%j.log

set -euo pipefail

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12

mkdir -p ${BASE}/logs ${BASE}/results/orthogonal/sweepfinder2

# Step 1: Compile SweepFinder2 if needed
bash ${BASE}/setup_sweepfinder2.sh

# Step 2: Run targeted analysis
${PYTHON} ${BASE}/run_sweepfinder2.py
