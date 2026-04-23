#!/bin/bash
#SBATCH --job-name=xpehh-s3
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=analysis/orthogonal_v41/logs/xpehh_s3_%j.log

set -uo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
export PATH="$(pwd)/.pixi/envs/default/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/.pixi/envs/default"
export PYTHONPATH="$(pwd)/python:$(pwd)"

echo "Node: $(hostname); Start: $(date)"
python analysis/orthogonal_v41/scripts/run_xpehh_s3.py
echo "Done: $(date)"
