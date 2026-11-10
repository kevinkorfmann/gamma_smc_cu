#!/usr/bin/env bash
#SBATCH --job-name=rank-normalize
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5632M
#SBATCH --time=00:30:00
set -euo pipefail
ROOT=/vast/projects/smathi/cohort/kkor/tmrca.cu/gamma_smc_cu
cd "$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
POPS=(ACB ASW LWK GWD MSL ESN YRI CEU TSI FIN GBR IBS CHB JPT CHS CDX KHV GIH PJL BEB STU ITU MXL PUR CLM PEL)
"$ROOT/.pixi/envs/default/bin/python" analysis/rank_diagnostics/normalize_haplotypes.py \
  --input analysis/rank_diagnostics/inputs --haplotypes analysis/rank_diagnostics/outputs/haplotypes \
  --output analysis/rank_diagnostics/outputs/normalized --pop "${POPS[$SLURM_ARRAY_TASK_ID]}"
