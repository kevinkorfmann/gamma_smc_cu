#!/bin/bash
#SBATCH --job-name=gsmc-verify
#SBATCH --partition=genoa-std-mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=benchmarks/pairwise_scaling/gsmc_verify_%j.log

set -uo pipefail

cd /vast/projects/smathi/cohort/kkor/tmrca.cu
export PATH="$(pwd)/.pixi/envs/default/bin:${PATH}"
export LD_LIBRARY_PATH="$(pwd)/.pixi/envs/default/lib:${LD_LIBRARY_PATH:-}"

GSMC=benchmarks/test_suite_stdpopsim/gamma_smc/bin/gamma_smc
FF=default_flow_field.txt
OUT=benchmarks/pairwise_scaling/gsmc_verify
mkdir -p $OUT

echo "=== Run gamma_smc on full chr22 VCF (all 3202 samples = 63,903 pairs) ==="
echo "Start: $(date)"
time $GSMC -i analysis/genome_wide/data/chr22.vcf.gz -o $OUT/chr22_full -t 0.8 -f $FF -h

echo ""
echo "=== Output files ==="
ls -lh $OUT/chr22_full*

echo ""
echo "=== Meta file content ==="
cat $OUT/chr22_full.meta 2>/dev/null || echo "no .meta file"

echo ""
echo "=== Parse meta for pair count ==="
python3 -c "
import json, os, sys
meta = '$OUT/chr22_full.meta'
if os.path.exists(meta):
    d = json.load(open(meta))
    for k,v in d.items():
        print(f'  {k}: {v}')
else:
    # Check binary output size
    for f in os.listdir('$OUT'):
        path = os.path.join('$OUT', f)
        print(f'  {f}: {os.path.getsize(path)} bytes')
"

echo ""
echo "=== Done at $(date) ==="
