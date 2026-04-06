#!/bin/bash
#SBATCH --job-name=hgdp_val
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/hgdp_val_%A_%a.log
#SBATCH --array=0-7

set -euo pipefail

PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12
SCRIPT=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt/hgdp_validate_single.py

GENES=(GRK2 CLEC6A TRAF6 TNFRSF13C JCHAIN BPIFA2 CCDC92 SLC6A15)
GENE=${GENES[$SLURM_ARRAY_TASK_ID]}

echo "=== HGDP validation: ${GENE} (array task ${SLURM_ARRAY_TASK_ID}) ==="
${PYTHON} ${SCRIPT} --gene ${GENE}
echo "=== Done: ${GENE} ==="
