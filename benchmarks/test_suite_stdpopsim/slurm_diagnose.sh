#!/bin/bash
#SBATCH --job-name=tmrca_diag
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=/vast/projects/smathi/cohort/kkor/tmrca.cu/benchmarks/test_suite_stdpopsim/logs/diagnose_%j.log

set -euo pipefail

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/benchmarks/test_suite_stdpopsim
PIXI_ENV=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default

export PATH=${PIXI_ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${PIXI_ENV}/lib:${LD_LIBRARY_PATH:-}
export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=0
export STDPOPSIM_CACHE_DIR=/vast/projects/smathi/cohort/kkor/stdpopsim_cache

cd "${BASE}"
"${PIXI_ENV}/bin/python3.12" diagnose_canfam.py
