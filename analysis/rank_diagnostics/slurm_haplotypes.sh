#!/usr/bin/env bash
#SBATCH --job-name=rank-haplotypes
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=5632M
#SBATCH --time=02:00:00
set -euo pipefail
ROOT=/vast/projects/smathi/cohort/kkor/tmrca.cu/gamma_smc_cu
RUN=/vast/projects/smathi/cohort/kkor/tmrca.cu/reruns/review-20260925
cd "$ROOT"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LD_LIBRARY_PATH="$ROOT/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"
read -r CHROM POP < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" analysis/rank_diagnostics/inputs/tasks.tsv)
"$ROOT/.pixi/envs/default/bin/python" analysis/rank_diagnostics/haplotype_job.py \
  --input-root analysis/rank_diagnostics/inputs \
  --cache "$RUN/inputs/cache/parsed" --samples "$RUN/inputs/samples.txt" \
  --ihs-root analysis/orthogonal_v41/selscan \
  --output analysis/rank_diagnostics/outputs/haplotypes \
  --chr "$CHROM" --pop "$POP" --resume
