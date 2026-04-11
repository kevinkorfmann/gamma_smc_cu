#!/bin/bash
#SBATCH --job-name=v41-install
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=analysis/orthogonal_v41/logs/install_%j.log

set -euo pipefail
set -x

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

# Activate the same pixi env layer as the inference scripts use
PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/${PIXI_ENV}"
export PYTHONPATH="$(pwd)/python:$(pwd)"

TOOLS_DIR=/vast/projects/smathi/cohort/kkor/tools
mkdir -p ${TOOLS_DIR}
cd ${TOOLS_DIR}

# ─── 1. selscan ─────────────────────────────────────────────────
echo "=== Installing selscan ==="
if [ ! -f selscan/bin/linux/selscan ]; then
    git clone https://github.com/szpiech/selscan.git
    cd selscan/src
    make || true   # selscan's Makefile may complain about libgsl, ok if binary still builds
    cd ${TOOLS_DIR}
fi
ls selscan/bin/linux/selscan && echo "selscan OK" || echo "selscan BUILD FAILED"

# ─── 2. Relate (binary release, no compile) ─────────────────────
echo "=== Installing Relate ==="
if [ ! -d relate ]; then
    # Use prebuilt linux release to avoid CMake hassles
    wget -q https://myersgroup.github.io/relate/binaries/relate_v1.2.2_x86_64_static.tgz
    tar xzf relate_v1.2.2_x86_64_static.tgz
    mv relate_v1.2.2_x86_64_static relate
    rm -f relate_v1.2.2_x86_64_static.tgz
fi
ls relate/bin/Relate && echo "Relate OK" || echo "Relate INSTALL FAILED"

# ─── 3. CLUES (Stern et al., installed into pixi env) ──────────
echo "=== Installing CLUES ==="
if ! python -c "import clues" 2>/dev/null; then
    if [ ! -d clues_repo ]; then
        git clone https://github.com/standard-aaron/clues.git clues_repo
    fi
    cd clues_repo
    python -m pip install --user . || python -m pip install --user -e .
    cd ${TOOLS_DIR}
fi
python -c "import clues; print('clues OK', clues.__file__)" 2>&1 || echo "clues INSTALL FAILED"

# ─── 4. Download hg38 genetic maps ─────────────────────────────
echo "=== Downloading hg38 genetic maps ==="
mkdir -p genetic_maps_hg38
cd genetic_maps_hg38
if [ ! -f genetic_map_chr22_combined_b38.txt ]; then
    # 1KG OMNI b38 genetic map (mirror provided by Beagle authors)
    for c in $(seq 1 22); do
        if [ ! -f genetic_map_chr${c}_combined_b38.txt ]; then
            wget -q https://bochet.gcc.biostat.washington.edu/beagle/genetic_maps/plink.chr${c}.GRCh38.map \
                -O plink.chr${c}.GRCh38.map || true
        fi
    done
fi
ls plink.chr22.GRCh38.map && echo "genetic maps OK" || echo "GENETIC MAPS DOWNLOAD FAILED"
cd ${TOOLS_DIR}

# ─── 5. Smoke test cxt and asmc python bindings ────────────────
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
python - <<'PY'
print("=== cxt smoke test ===")
try:
    import cxt
    print("cxt imported:", cxt.__file__)
    # check for the inference function
    if hasattr(cxt, "translate"):
        print("  cxt.translate found")
except Exception as e:
    print("cxt FAIL:", e)

print("=== asmc smoke test ===")
try:
    from asmc import asmc as asmc_mod
    print("asmc imported:", asmc_mod.__file__)
except Exception as e:
    print("asmc FAIL:", e)
PY

# ─── 6. Write tools_ready.txt ──────────────────────────────────
cd /vast/projects/smathi/cohort/kkor/tmrca.cu
{
echo "Tool installation status (run on $(hostname) at $(date))"
echo "============================================================"
echo "selscan: ${TOOLS_DIR}/selscan/bin/linux/selscan"
ls ${TOOLS_DIR}/selscan/bin/linux/selscan 2>/dev/null && echo "  STATUS: OK" || echo "  STATUS: FAILED"
echo
echo "Relate: ${TOOLS_DIR}/relate/bin/Relate"
ls ${TOOLS_DIR}/relate/bin/Relate 2>/dev/null && echo "  STATUS: OK" || echo "  STATUS: FAILED"
echo
echo "clues: pip-installed in pixi env"
python -c "import clues" 2>/dev/null && echo "  STATUS: OK" || echo "  STATUS: FAILED"
echo
echo "cxt: pixi env"
python -c "import cxt" 2>/dev/null && echo "  STATUS: OK" || echo "  STATUS: FAILED"
echo
echo "asmc: pixi env"
python -c "from asmc import asmc as a" 2>/dev/null && echo "  STATUS: OK" || echo "  STATUS: FAILED"
echo
echo "Genetic maps: ${TOOLS_DIR}/genetic_maps_hg38/"
ls ${TOOLS_DIR}/genetic_maps_hg38/plink.chr22.GRCh38.map 2>/dev/null && echo "  STATUS: OK" || echo "  STATUS: FAILED"
} > analysis/orthogonal_v41/tools_ready.txt

cat analysis/orthogonal_v41/tools_ready.txt
echo "=== install job complete ==="
