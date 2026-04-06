#!/bin/bash
# Install SweepFinder2 on betty
set -euo pipefail

BASE=/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt
SF2_DIR=${BASE}/sweepfinder2

mkdir -p ${SF2_DIR}
cd ${SF2_DIR}

# Download from DeGiorgio lab
if [ ! -f SF2 ]; then
    wget -q https://github.com/pjmartinez/SweepFinder2/archive/refs/heads/master.zip -O sf2.zip
    unzip -q sf2.zip
    cd SweepFinder2-master
    gcc -O3 -o SF2 SweepFinder2.c -lm
    cp SF2 ${SF2_DIR}/
    cd ${SF2_DIR}
    rm -rf SweepFinder2-master sf2.zip
    echo "SweepFinder2 compiled: ${SF2_DIR}/SF2"
else
    echo "SF2 already exists"
fi

${SF2_DIR}/SF2 2>&1 | head -5 || true
