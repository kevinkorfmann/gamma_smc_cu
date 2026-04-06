#!/bin/bash
set -euo pipefail

MAPDIR=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues_v2/data/recomb_maps

for chr in 2 11 15; do
    IN=${MAPDIR}/genetic_map_GRCh38_chr${chr}.txt
    OUT=${MAPDIR}/relate_map_chr${chr}.txt
    echo "pos COMBINED_rate.cM.Mb. Genetic_Map.cM." > ${OUT}
    tail -n+2 ${IN} | awk -F'\t' '{print $2, $3, $4}' >> ${OUT}
    echo "chr${chr}: $(wc -l < ${OUT}) lines"
    head -3 ${OUT}
    echo "---"
done
