#!/bin/bash
#SBATCH --job-name=bgs-annotate
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5632M
#SBATCH --time=00:20:00
#SBATCH --output=analysis/bgs/logs/annotate_%j.log
set -euo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu/gamma_smc_cu
PYTHON=.pixi/envs/default/bin/python3.12
"${PYTHON}" analysis/bgs/bgs.py download
"${PYTHON}" analysis/bgs/bgs.py annotate \
  --genes analysis/bgs/inputs/genome_wide_ranks.csv \
  --sd analysis/bgs/inputs/sd_flag.csv \
  --gene-coordinates one-based-closed
"${PYTHON}" analysis/bgs/plot_bgs.py \
  --annotations analysis/bgs/results/gene_bgs_annotations.csv.gz \
  --case-data analysis/bgs/inputs/case_studies \
  --output analysis/bgs/results/figures
