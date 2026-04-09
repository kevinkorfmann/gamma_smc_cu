#!/bin/bash
#SBATCH --job-name=tmrca_agg
#SBATCH --partition=b200-mig90
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:10:00
#SBATCH --output=/vast/projects/smathi/cohort/kkor/tmrca.cu/benchmarks/test_suite_stdpopsim/logs/aggregate_%j.log

set -euo pipefail
BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/benchmarks/test_suite_stdpopsim
PIXI_ENV=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default
export PATH=${PIXI_ENV}/bin:${PATH}
export MPLBACKEND=Agg

cd "${BASE}"
"${PIXI_ENV}/bin/python3.12" aggregate_and_plot.py
