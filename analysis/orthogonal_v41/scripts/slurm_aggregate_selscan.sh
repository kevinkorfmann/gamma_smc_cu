#!/bin/bash
#SBATCH --job-name=v41-sel-agg
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=analysis/orthogonal_v41/logs/sel_agg_%j.log

set -uo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/${PIXI_ENV}"
export PYTHONPATH="$(pwd)/python:$(pwd)"

echo "=== Aggregating selscan to per-gene tables ==="
python analysis/orthogonal_v41/scripts/aggregate_selscan_per_gene.py --all

echo "=== Comparing against gamma_smc_cu candidates ==="
python analysis/orthogonal_v41/scripts/compare_selscan_vs_tmrca.py

echo "=== Done at $(date) ==="
