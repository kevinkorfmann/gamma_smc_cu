#!/bin/bash
# Full-B200 version of per-(chr, pop) Akbari-window TMRCA inference.
# Uses dgx-b200 partition so each task gets a full B200 (not a MIG slice).
#
# Submit:
#   sbatch --array=0-571 analysis/akbari_479_tmrca/slurm_infer_akbari_full.sh
#
#SBATCH --job-name=akbari-tmrca
#SBATCH --partition=b200-mig90
#SBATCH --gres=gpu:90gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=analysis/akbari_479_tmrca/logs/task%a_%A.log

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/${PIXI_ENV}"
export PYTHONPATH="$(pwd)/python:$(pwd)"
export LD_LIBRARY_PATH="$(pwd)/python/gamma_smc_cu:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=0

CHRS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22)
POPS=(ACB ASW BEB CDX CEU CHB CHS CLM ESN FIN GBR GIH GWD IBS ITU JPT KHV LWK MSL MXL PEL PJL PUR STU TSI YRI)

N_CHR=${#CHRS[@]}
N_POP=${#POPS[@]}
IDX=${SLURM_ARRAY_TASK_ID}

CHR_IDX=$((IDX / N_POP))
POP_IDX=$((IDX % N_POP))
CHR=${CHRS[${CHR_IDX}]}
POP=${POPS[${POP_IDX}]}

# Skip if already done (restart-friendly).
OUT_CSV="/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/akbari_479_tmrca/results/chr${CHR}/${POP}.csv"
if [[ -s "${OUT_CSV}" ]]; then
    echo "${OUT_CSV} already exists, skipping."
    exit 0
fi

echo "=== Job ${SLURM_JOB_ID} task ${IDX}: chr${CHR} pop ${POP} ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start: $(date)"

python analysis/akbari_479_tmrca/infer_akbari_windows.py --chr ${CHR} --populations ${POP}

echo "End: $(date)"
