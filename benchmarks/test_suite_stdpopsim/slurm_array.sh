#!/bin/bash
#SBATCH --job-name=tmrca_suite
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --requeue
#SBATCH --output=/vast/projects/smathi/cohort/kkor/tmrca.cu/benchmarks/test_suite_stdpopsim/logs/config_%a_%j.log
# Submit as (from repo root):
#   N=$(python -c 'import json; print(len(json.load(open("benchmarks/test_suite_stdpopsim/configs.json"))))')
#   sbatch --array=0-$((N-1))%8 benchmarks/test_suite_stdpopsim/slurm_array.sh

set -euo pipefail

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/benchmarks/test_suite_stdpopsim
PIXI_ENV=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default
PYTHON=${PIXI_ENV}/bin/python3.12

# bgzip, tabix, zstd live inside the pixi env; make sure they're on PATH
# so run_one.py's subprocess calls to the gamma_smc pipeline succeed.
export PATH=${PIXI_ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${PIXI_ENV}/lib:${LD_LIBRARY_PATH:-}

export MPLBACKEND=Agg
export CUDA_VISIBLE_DEVICES=0
export STDPOPSIM_CACHE_DIR=/vast/projects/smathi/cohort/kkor/stdpopsim_cache

mkdir -p "${BASE}/results" "${BASE}/logs" "${STDPOPSIM_CACHE_DIR}"

cd "${BASE}"
echo "host=$(hostname)  idx=${SLURM_ARRAY_TASK_ID}  job=${SLURM_JOB_ID}"
"${PYTHON}" "${BASE}/run_one.py" --config-idx "${SLURM_ARRAY_TASK_ID}"
