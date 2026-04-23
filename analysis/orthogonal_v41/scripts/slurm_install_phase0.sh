#!/bin/bash
#SBATCH --job-name=v41-phase0
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=analysis/orthogonal_v41/logs/phase0_%j.log

set -uo pipefail
set -x

cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI_ENV=".pixi/envs/default"
export PATH="$(pwd)/${PIXI_ENV}/bin:${PATH}"
export CONDA_PREFIX="$(pwd)/${PIXI_ENV}"
export PYTHONPATH="$(pwd)/python:$(pwd)"

TOOLS_DIR=/vast/projects/smathi/cohort/kkor/tools

echo "=== Phase 0 (minimal): verify selscan + smoke test cxt and ASMC ==="
echo

# 1. selscan: already built in earlier attempt
if [ -x "${TOOLS_DIR}/selscan/bin/linux/selscan" ]; then
    echo "selscan: OK at ${TOOLS_DIR}/selscan/bin/linux/selscan"
    "${TOOLS_DIR}/selscan/bin/linux/selscan" --help 2>&1 | head -3
else
    echo "selscan: NOT FOUND -- defer to Phase 2 install"
fi
echo

# 2. cxt smoke test: tiny synthetic data
python - <<'PY'
import numpy as np
print("=== cxt smoke test ===")
try:
    import cxt
    print("cxt imported from:", cxt.__file__)
    print("cxt module attrs:", [x for x in dir(cxt) if not x.startswith("_")][:30])
    # Try the translate signature seen in archive scripts
    if hasattr(cxt, "translate"):
        print("  cxt.translate exists")
    if hasattr(cxt, "infer"):
        print("  cxt.infer exists")
    if hasattr(cxt, "Model"):
        print("  cxt.Model exists")
except Exception as e:
    print("cxt FAIL:", e)
PY
echo

# 3. ASMC smoke test: try a minimal prepareDecoding call to find what
#    demographic models / files ship with the package.
python - <<'PY'
print("=== ASMC smoke test ===")
try:
    from asmc import asmc as a
    print("ASMC version available, classes:")
    print(" ", [x for x in dir(a) if not x.startswith("_")][:20])
    # Look for built-in demographic model files
    import os, asmc
    asmc_dir = os.path.dirname(asmc.__file__)
    print("asmc package dir:", asmc_dir)
    for root, dirs, files in os.walk(asmc_dir):
        for f in files:
            if any(k in f.lower() for k in [".dq", "decoding", "csfs", ".demo", ".map"]):
                print(" ", os.path.join(root, f))
    # Try preparedecoding entry point
    from asmc.preparedecoding import preparedecoding_python_bindings as pdb
    print("preparedecoding callable:", hasattr(pdb, "prepareDecoding"))
except Exception as e:
    print("ASMC FAIL:", e)
    import traceback; traceback.print_exc()
PY
echo

# 4. gamma_smc_cu smoke test on a tiny msprime simulation
python - <<'PY'
print("=== gamma_smc_cu smoke test (single pair, 50kb) ===")
try:
    import msprime
    import gamma_smc_cu
    import numpy as np
    ts = msprime.sim_ancestry(4, sequence_length=50_000, recombination_rate=1e-8,
                               population_size=10000, random_seed=42)
    ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=43)
    G = ts.genotype_matrix().T.astype(np.uint8)
    pos = np.array([v.position for v in ts.variants()], dtype=np.float64)
    r = gamma_smc_cu.infer(G, pos, mean_only=True, pairs=[(0, 1)])
    print(f"  gamma_smc_cu OK: mean shape {r['mean'].shape}, "
          f"min={r['mean'].min():.0f}, max={r['mean'].max():.0f}")
except Exception as e:
    print("gamma_smc_cu FAIL:", e)
    import traceback; traceback.print_exc()
PY

# 5. Status summary
{
echo "Phase 0 minimal status (run on $(hostname) at $(date))"
echo "============================================================"
[ -x "${TOOLS_DIR}/selscan/bin/linux/selscan" ] && echo "selscan: OK" || echo "selscan: MISSING"
python -c "import cxt" 2>/dev/null && echo "cxt: OK" || echo "cxt: MISSING"
python -c "from asmc import asmc as a; a.ASMC" 2>/dev/null && echo "asmc: OK" || echo "asmc: MISSING"
python -c "import gamma_smc_cu; gamma_smc_cu.infer" 2>/dev/null && echo "gamma_smc_cu: OK" || echo "gamma_smc_cu: MISSING"
echo
echo "Deferred to later phases:"
echo "  Relate, CLUES (Phase 4)"
echo "  Genetic maps (Phase 2)"
echo "  ASMC decoding quantities (computed in Phase 1 task itself)"
} > analysis/orthogonal_v41/tools_ready.txt

cat analysis/orthogonal_v41/tools_ready.txt
echo "=== Phase 0 complete ==="
