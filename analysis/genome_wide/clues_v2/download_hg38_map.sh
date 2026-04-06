#!/bin/bash
set -euo pipefail

MAPDIR=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues_v2/data/recomb_maps
cd ${MAPDIR}

# GRCh38 genetic map from Eagle (Broad)
echo "Downloading GRCh38 genetic map..."
wget -q https://storage.googleapis.com/broad-alkesgroup-public/Eagle/downloads/tables/genetic_map_hg38_withX.txt.gz -O genetic_map_hg38.txt.gz

echo "Splitting by chromosome..."
PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12
${PYTHON} -c "
import gzip

with gzip.open('genetic_map_hg38.txt.gz', 'rt') as f:
    header = f.readline().strip()
    # Format: chr position COMBINED_rate(cM/Mb) Genetic_Map(cM)
    files = {}
    for line in f:
        parts = line.strip().split()
        chrn = parts[0]
        if chrn not in files:
            files[chrn] = open(f'genetic_map_GRCh38_chr{chrn}.txt', 'w')
            # Relate expects: Chromosome Position(bp) Rate(cM/Mb) Map(cM)
            files[chrn].write('Chromosome\tPosition(bp)\tRate(cM/Mb)\tMap(cM)\n')
        files[chrn].write(f'chr{chrn}\t{parts[1]}\t{parts[2]}\t{parts[3]}\n')
    for fh in files.values():
        fh.close()
    print(f'Split into {len(files)} chromosomes')
"

echo "Available maps:"
ls genetic_map_GRCh38_chr*.txt | head -5
echo "Done"
