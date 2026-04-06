#!/bin/bash
set -euo pipefail

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues_v2
RELATE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues/relate_src
MAPDIR=${BASE}/data/recomb_maps

cd ${BASE}/trees
rm -rf BEB_chr11

echo "Testing Relate with GRCh38 map..."
${RELATE}/scripts/RelateParallel/RelateParallel.sh \
    --threads 4 \
    --haps ${BASE}/data/BEB_chr11.haps.gz \
    --sample ${BASE}/data/BEB_chr11.sample.gz \
    --map ${MAPDIR}/genetic_map_GRCh38_chr11.txt \
    -m 1.25e-8 -N 20000 \
    -o BEB_chr11

echo "=== Result ==="
ls -lh BEB_chr11.anc BEB_chr11.mut 2>/dev/null && echo "SUCCESS" || echo "FAILED"
