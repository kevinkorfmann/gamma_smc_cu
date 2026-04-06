#!/bin/bash
#SBATCH --job-name=hgdp_valid
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/hgdp_validation_%j.log

set -euo pipefail

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12

mkdir -p ${BASE}/hgdp/vcf ${BASE}/cxt/logs

echo "=== HGDP Validation Pipeline ==="
echo "Started: $(date)"

${PYTHON} ${BASE}/cxt/hgdp_validation.py

echo "=== Done: $(date) ==="
