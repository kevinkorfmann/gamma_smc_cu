#!/bin/bash
# Per-(chr, pop) Akbari-window TMRCA inference.
# The SLURM array indexes into a (chr, pop) table defined below so every task
# runs one chromosome for one population -- maximizing parallelism on MIG slices.
#
# Prototype submission (2 chr x 5 pops = 10 jobs):
#   sbatch --array=0-9 analysis/akbari_479_tmrca/slurm_infer_akbari_pop.sh
#
# Full submission (22 chr x 26 pops = 572 jobs):
#   EDIT CHRS and POPS below, then:
#   sbatch --array=0-571 analysis/akbari_479_tmrca/slurm_infer_akbari_pop.sh
#
#SBATCH --job-name=akbari-tmrca
#SBATCH --partition=b200-mig45
#SBATCH --gres=gpu:45gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=analysis/akbari_479_tmrca/logs/task%a_%A.log

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/${PIXI_ENV}"
export PYTHONPATH="$(pwd)/python:$(pwd)"
export LD_LIBRARY_PATH="$(pwd)/python/tmrca_cu:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

# --- task dispatch table ---
# Edit these two arrays to change the prototype scope.
# Prototype: chr2 + chr11 (includes LCT-region + GRK2), 5 pops = 1 per superpop.
CHRS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)
POPS=(ACB ASW BEB CDX CEU CHB CHS CLM ESN FIN GBR GIH GWD IBS ITU JPT KHV LWK MSL MXL PEL PJL PUR STU TSI YRI)

N_CHR=${#CHRS[@]}
N_POP=${#POPS[@]}
N_TOTAL=$((N_CHR * N_POP))

IDX=${SLURM_ARRAY_TASK_ID}
if [[ ${IDX} -ge ${N_TOTAL} ]]; then
    echo "Task ${IDX} out of range (N_TOTAL=${N_TOTAL})"; exit 1
fi

CHR_IDX=$((IDX / N_POP))
POP_IDX=$((IDX % N_POP))
CHR=${CHRS[${CHR_IDX}]}
POP=${POPS[${POP_IDX}]}

echo "=== Job ${SLURM_JOB_ID} task ${IDX}: chr${CHR} pop ${POP} ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

python analysis/akbari_479_tmrca/infer_akbari_windows.py --chr ${CHR} --populations ${POP}

echo "End: $(date)"
