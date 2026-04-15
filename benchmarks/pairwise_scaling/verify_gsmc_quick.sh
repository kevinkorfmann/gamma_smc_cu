#!/bin/bash
set -uo pipefail
cd /vast/projects/smathi/cohort/kkor/tmrca.cu

PIXI=.pixi/envs/default
export PATH="$(pwd)/${PIXI}/bin:${PATH}"
export LD_LIBRARY_PATH="$(pwd)/${PIXI}/lib:${LD_LIBRARY_PATH:-}"

GSMC=benchmarks/test_suite_stdpopsim/gamma_smc/bin/gamma_smc
FF=default_flow_field.txt
OUT=benchmarks/pairwise_scaling/gsmc_verify
mkdir -p $OUT

echo "=== gamma_smc on FULL chr22 VCF (all 3202 samples) ==="
echo "Start: $(date)"
time $GSMC -i analysis/genome_wide/data/chr22.vcf.gz -o $OUT/chr22_full -t 0.8 -f $FF -h
echo "End: $(date)"

echo ""
echo "=== Output files ==="
ls -lh $OUT/chr22_full* 2>/dev/null || echo "no output files"

echo ""
echo "=== Meta file ==="
cat $OUT/chr22_full.meta 2>/dev/null || echo "no meta"

echo ""
echo "=== Done ==="
