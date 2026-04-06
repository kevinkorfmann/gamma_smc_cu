#!/bin/bash
set -euo pipefail

PYTHON=/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python3.12
BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues_v2
RELATE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues/relate_src

# Download HapMap recombination maps (GRCh37) if needed
MAPDIR=${BASE}/data/recomb_maps
if [ ! -d "${MAPDIR}" ]; then
    echo "Downloading recombination maps..."
    mkdir -p ${MAPDIR}
    cd ${MAPDIR}
    wget -q https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/working/20110106_recombination_hotspots/HapmapII_GRCh37_RecombinationHotspots.tar.gz -O maps.tar.gz
    tar xzf maps.tar.gz 2>/dev/null || true
    # Also try the genetic map format Relate expects
    for chr in 2 11 15; do
        wget -q "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/technical/working/20130507_omni_recombination_rates/CEU_omni_recombination_20130507.tar" -O ceu_maps.tar 2>/dev/null && tar xf ceu_maps.tar 2>/dev/null || true
    done
    ls ${MAPDIR}/
    cd ${BASE}
fi

# Generate a simple recombination map from the dist file if no proper map
echo "=== GENERATING RECOMB MAP FOR CHR11 ==="
${PYTHON} -c "
import numpy as np, gzip
npz = np.load('/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cache/parsed/chr11.npz')
pos = npz['positions']
# 1 cM/Mb uniform rate
with open('${BASE}/data/genetic_map_chr11.txt', 'w') as f:
    f.write('Chromosome\tPosition(bp)\tRate(cM/Mb)\tMap(cM)\n')
    cm = 0.0
    prev_pos = pos[0]
    for i, p in enumerate(pos[::1000]):  # subsample for speed
        if i > 0:
            cm += (p - prev_pos) * 1e-6
        f.write(f'11\t{int(p)}\t1.0\t{cm:.6f}\n')
        prev_pos = p
print('Map written')
"

# Test Relate
echo "=== TEST RELATE ==="
cd ${BASE}/trees
rm -rf BEB_chr11
${RELATE}/scripts/RelateParallel/RelateParallel.sh \
    --threads 4 \
    --haps ${BASE}/data/BEB_chr11.haps.gz \
    --sample ${BASE}/data/BEB_chr11.sample.gz \
    --map ${MAPDIR}/genetic_map_GRCh37_chr11.txt \
    --dist ${BASE}/data/BEB_chr11.dist.gz \
    -m 1.25e-8 -N 20000 \
    -o BEB_chr11 2>&1 | tail -20

echo "=== DONE ==="
