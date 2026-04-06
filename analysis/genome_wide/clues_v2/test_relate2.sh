#!/bin/bash
#SBATCH --job-name=relate_test
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues_v2/logs/relate_test_%j.log

set -euxo pipefail

PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12
BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues_v2
RELATE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues/relate_src
MAPDIR=${BASE}/data/recomb_maps

# Re-prepare with dedup
echo "Re-preparing BEB chr11 with dedup..."
rm -f ${BASE}/data/BEB_chr11.haps.gz ${BASE}/data/BEB_chr11.sample.gz ${BASE}/data/BEB_chr11.dist.gz
${PYTHON} ${BASE}/prepare_haps.py --pop BEB --chr 11 --outdir ${BASE}/data

# Clean and run Relate (single thread to avoid dir conflict)
cd ${BASE}/trees
rm -rf BEB_chr11 BEB_chr11.anc BEB_chr11.mut

echo "Running Relate (single thread)..."
${RELATE}/bin/Relate \
    --mode All \
    --haps ${BASE}/data/BEB_chr11.haps.gz \
    --sample ${BASE}/data/BEB_chr11.sample.gz \
    --map ${MAPDIR}/relate_map_chr11.txt \
    -m 1.25e-8 -N 20000 \
    -o BEB_chr11 2>&1 | tail -20

echo "=== Result ==="
ls -lh BEB_chr11.anc BEB_chr11.mut 2>/dev/null && echo "SUCCESS" || echo "FAILED"
