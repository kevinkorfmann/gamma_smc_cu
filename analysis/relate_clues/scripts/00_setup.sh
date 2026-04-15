#!/bin/bash
#SBATCH --job-name=rc-setup
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=analysis/relate_clues/logs/00_setup_%j.log

# Step 0: Download and install Relate v1.2.4, CLUES2, and all auxiliary files.
#
# Downloads:
# 1. Relate v1.2.4 static binary (Linux x86_64)
# 2. CLUES2 from GitHub (Vaughn & Nielsen 2024)
# 3. Relate Input Files from Zenodo (ancestral genomes, masks, genetic maps for GRCh38)
# 4. 1KG phased VCFs for chr2 and chr11 (if not already present)

set -euo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
BASE=analysis/relate_clues

mkdir -p ${BASE}/{tools,input_files,data/chr2,data/chr11,trees/chr2_EUR,trees/chr11_EURSAS,clues/GRK2,clues/LCT,clues/neutral,figures,logs}

echo "=== Step 0: Setup ==="
echo "Start: $(date)"

# ── 1. Relate v1.2.4 (copy from archive — gated download, can't wget) ──
ARCHIVE_RELATE=archive_2026_04_09/clues/relate_src/docs/download/relate_v1.2.4_x86_64_static
if [ ! -f ${BASE}/tools/relate_v1.2.4/bin/Relate ]; then
    echo "Copying Relate v1.2.4 from archive..."
    mkdir -p ${BASE}/tools/relate_v1.2.4
    cp -r ${ARCHIVE_RELATE}/* ${BASE}/tools/relate_v1.2.4/
    # Also copy scripts from the source tree
    cp -r archive_2026_04_09/clues/relate_src/scripts ${BASE}/tools/relate_v1.2.4/
    echo "  Relate: $(${BASE}/tools/relate_v1.2.4/bin/Relate --help 2>&1 | head -1)"
else
    echo "  Relate already installed."
fi

# ── 2. CLUES2 ──
if [ ! -d ${BASE}/tools/CLUES2 ]; then
    echo "Cloning CLUES2..."
    git clone --depth 1 https://github.com/avaughn271/CLUES2.git ${BASE}/tools/CLUES2
    echo "  CLUES2 cloned."
else
    echo "  CLUES2 already present."
fi

# Install CLUES2 Python deps into pixi env
PIXI_ENV=".pixi/envs/default"
${PIXI_ENV}/bin/pip install numba scipy biopython 2>/dev/null || \
    ${PIXI_ENV}/bin/python -m pip install numba scipy biopython 2>/dev/null || \
    echo "  CLUES2 deps: pip not available, will use pixi add later"

# ── 3. Relate Input Files ──
# Genetic maps: copy from archive
ARCHIVE_MAPS=archive_2026_04_09/clues_v2/data
mkdir -p ${BASE}/input_files/genetic_maps
for CHR in 2 11; do
    if [ -f ${ARCHIVE_MAPS}/genetic_map_chr${CHR}.txt ]; then
        cp ${ARCHIVE_MAPS}/genetic_map_chr${CHR}.txt ${BASE}/input_files/genetic_maps/
        echo "  Genetic map chr${CHR}: copied from archive"
    elif [ -d ${ARCHIVE_MAPS}/recomb_maps ]; then
        cp ${ARCHIVE_MAPS}/recomb_maps/*chr${CHR}* ${BASE}/input_files/genetic_maps/ 2>/dev/null
        echo "  Genetic map chr${CHR}: copied from archive/recomb_maps"
    fi
done

# Ancestral genome + masks: download from Zenodo
if [ ! -d ${BASE}/input_files/hg38 ]; then
    echo "Downloading Relate input files from Zenodo (2.4 GB)..."
    cd ${BASE}/input_files
    wget https://zenodo.org/api/records/15801307/files/Relate_input_files.tgz/content -O Relate_input_files.tgz
    tar xzf Relate_input_files.tgz
    rm Relate_input_files.tgz
    echo "  Input files extracted:"
    ls -d */ 2>/dev/null || ls
    cd /vast/projects/smathi/cohort/kkor/tmrca.cu
else
    echo "  Relate input files already present."
fi

# ── 4. 1KG phased VCFs for chr2 and chr11 ──
VCF_BASE="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20201028_3202_phased"
DATA_DIR=analysis/genome_wide/data

for CHR in 2 11; do
    VCF_FILE="CCDG_14151_B01_GRM_WGS_2020-08-05_chr${CHR}.filtered.shapeit2-duohmm-phased.vcf.gz"
    if [ ! -f ${DATA_DIR}/chr${CHR}.vcf.gz ]; then
        echo "Downloading chr${CHR} VCF..."
        wget -q -O ${DATA_DIR}/chr${CHR}.vcf.gz "${VCF_BASE}/${VCF_FILE}"
        wget -q -O ${DATA_DIR}/chr${CHR}.vcf.gz.tbi "${VCF_BASE}/${VCF_FILE}.tbi"
        echo "  chr${CHR}: $(ls -lh ${DATA_DIR}/chr${CHR}.vcf.gz | awk '{print $5}')"
    else
        echo "  chr${CHR} VCF already present."
    fi
done

echo ""
echo "=== Setup complete at $(date) ==="
echo "Directory layout:"
find ${BASE} -maxdepth 2 -type d | sort
