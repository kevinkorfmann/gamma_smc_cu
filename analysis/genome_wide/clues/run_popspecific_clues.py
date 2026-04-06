#!/usr/bin/env python3
"""
Build population-specific Relate trees for a small region around a gene,
then run SampleBranchLengths + CLUES to get allele frequency trajectories.

Usage: python run_popspecific_clues.py GENE CHR POS POPULATION REGION_KB
Example: python run_popspecific_clues.py CLEC6A 12 7679508 CHB 2000
"""

import sys
import os
import subprocess
import numpy as np
from cyvcf2 import VCF

GENE = sys.argv[1]
CHR = int(sys.argv[2])
POS = int(sys.argv[3])
POP = sys.argv[4]
REGION_KB = int(sys.argv[5]) if len(sys.argv) > 5 else 2000

BASE = '/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues'
VCF_DIR = '/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/data'
SAMPLES_FILE = f'{VCF_DIR}/samples.txt'
RELATE = f'{BASE}/relate_src'
CLUES = f'{BASE}/clues'
COAL = f'{BASE}/1kg_trees/popsizes/1000GP_CHBGBRYRI_mask_ne.coal'
PYTHON = '/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin/python'
MU = 1.25e-8
OUTDIR = f'{BASE}/results_popspecific'
os.makedirs(OUTDIR, exist_ok=True)

PREFIX_BASE = f'{GENE}_{POP}'
PREFIX = f'{OUTDIR}/{PREFIX_BASE}'
REGION_START = max(1, POS - REGION_KB * 500)  # half on each side
REGION_END = POS + REGION_KB * 500

# Relate requires output in working directory
os.chdir(OUTDIR)

print(f"===== {GENE} (chr{CHR}:{POS}) in {POP} =====")
print(f"Region: {REGION_START}-{REGION_END} ({REGION_KB}kb)")

# ── Step 1: Get population sample IDs ──────────────────────────
print("Step 1: Getting population samples...")
pop_samples = []
with open(SAMPLES_FILE) as f:
    header = f.readline()
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 6 and parts[5] == POP:
            pop_samples.append(parts[1])

print(f"  {POP}: {len(pop_samples)} samples ({len(pop_samples)*2} haplotypes)")

# ── Step 2: Extract region VCF and convert to haps/sample ──────
print("Step 2: Extracting region and converting to haps/sample...")

vcf_file = f'{VCF_DIR}/chr{CHR}.vcf.gz'
region = f'chr{CHR}:{REGION_START}-{REGION_END}'

# Read VCF, subset to population, write as Oxford haps format
vcf = VCF(vcf_file, samples=pop_samples)
n_haps = len(pop_samples) * 2

haps_lines = []
n_snps = 0
seen_pos = set()

for variant in vcf(region):
    if not variant.is_snp or len(variant.ALT) != 1:
        continue
    if variant.num_het + variant.num_hom_alt + variant.num_hom_ref != len(pop_samples):
        continue

    pos_v = variant.POS
    if pos_v in seen_pos:
        continue  # skip duplicate positions (Relate requirement)
    seen_pos.add(pos_v)

    chrom = variant.CHROM.replace('chr', '')
    ref = variant.REF
    alt = variant.ALT[0]

    # Get phased haplotypes
    gts = variant.genotypes  # list of [a1, a2, phased]
    hap_str = []
    for gt in gts:
        hap_str.append(str(gt[0]))
        hap_str.append(str(gt[1]))

    # Oxford haps format: chr rsid pos ref alt hap1 hap2 ...
    line = f"{chrom} {chrom}:{pos_v} {pos_v} {ref} {alt} {' '.join(hap_str)}"
    haps_lines.append(line)
    n_snps += 1

vcf.close()
print(f"  Extracted {n_snps} SNPs in region")

# Write haps file (use relative name since Relate needs it in cwd)
haps_file = f'{PREFIX_BASE}.haps'
with open(haps_file, 'w') as f:
    for line in haps_lines:
        f.write(line + '\n')

# Write sample file
sample_file = f'{PREFIX_BASE}.sample'
with open(sample_file, 'w') as f:
    f.write('ID_1 ID_2 missing\n')
    f.write('0 0 0\n')
    for sid in pop_samples:
        f.write(f'{sid} {sid} 0\n')

print(f"  Wrote {haps_file} ({n_snps} SNPs, {n_haps} haplotypes) in {OUTDIR}")

# ── Step 3: Create genetic map (uniform 1 cM/Mb) ──────────────
print("Step 3: Creating genetic map...")
map_file = f'{PREFIX_BASE}.map'
with open(map_file, 'w') as f:
    f.write('pos COMBINED_rate Genetic_Map\n')
    f.write(f'{REGION_START} 1.0 {REGION_START / 1e6}\n')
    f.write(f'{REGION_END} 1.0 {REGION_END / 1e6}\n')

# ── Step 4: Run Relate ─────────────────────────────────────────
print("Step 4: Running Relate...")
relate_out = f'{PREFIX_BASE}_relate'
# Clean up any previous failed run directory
import shutil
if os.path.isdir(relate_out):
    shutil.rmtree(relate_out)
cmd = [
    f'{RELATE}/bin/Relate',
    '--mode', 'All',
    '--haps', haps_file,
    '--sample', sample_file,
    '--map', map_file,
    '-m', str(MU),
    '-N', '20000',
    '-o', relate_out,
]
result = subprocess.run(cmd, capture_output=True, text=True)
# Relate may exit with code 1 even on success (temp dir cleanup failure)
# Check for actual output files instead of return code
if not os.path.exists(f'{relate_out}.anc') or not os.path.exists(f'{relate_out}.mut'):
    print(f"  ERROR: Relate failed — no output files")
    print(result.stdout[-500:] if result.stdout else "")
    print(result.stderr[-500:] if result.stderr else "")
    sys.exit(1)
print(f"  Relate done: {relate_out}.anc, {relate_out}.mut")

# ── Step 5: Estimate population size ───────────────────────────
print("Step 5: Estimating population size...")
coal_out = f'{PREFIX_BASE}'
cmd_popsize = [
    f'{RELATE}/bin/RelateCoalescentRate',
    '--mode', 'EstimatePopulationSize',
    '-m', str(MU),
    '-i', relate_out,
    '-o', coal_out,
]
result = subprocess.run(cmd_popsize, capture_output=True, text=True)
if result.returncode != 0 or not os.path.exists(f'{coal_out}.coal'):
    print(f"  PopSize estimation failed, using default coal file")
    coal_file = COAL  # absolute path, works from any cwd
else:
    coal_file = f'{OUTDIR}/{coal_out}.coal'
    print(f"  Population size estimated: {coal_file}")

# ── Step 6: SampleBranchLengths at focal SNP ───────────────────
print("Step 6: SampleBranchLengths...")
resample_out = f'{PREFIX_BASE}_resample'
cmd_sbl = f"""bash {RELATE}/scripts/SampleBranchLengths/SampleBranchLengths.sh \
    -i {relate_out} \
    -o {resample_out} \
    -m {MU} \
    --coal {coal_file} \
    --num_samples 100 \
    --first_bp {POS} \
    --last_bp {POS} \
    --format b \
    --seed 42"""

result = subprocess.run(cmd_sbl, shell=True, capture_output=True, text=True)
if not os.path.exists(f'{resample_out}.timeb'):
    # Try nearby positions
    print(f"  .timeb not found at exact position, searching nearby...")
    # Find closest SNP in mut file
    best_pos = None
    best_dist = float('inf')
    with open(f'{relate_out}.mut') as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(';')
            snp_pos = int(parts[1])
            dist = abs(snp_pos - POS)
            if dist < best_dist:
                best_dist = dist
                best_pos = snp_pos
    if best_pos and best_dist < 5000:
        print(f"  Using nearest SNP at {best_pos} (dist={best_dist}bp)")
        cmd_sbl2 = cmd_sbl.replace(f'--first_bp {POS}', f'--first_bp {best_pos}')
        cmd_sbl2 = cmd_sbl2.replace(f'--last_bp {POS}', f'--last_bp {best_pos}')
        subprocess.run(cmd_sbl2, shell=True, capture_output=True, text=True)

if not os.path.exists(f'{resample_out}.timeb'):
    print(f"  ERROR: .timeb not created")
    sys.exit(1)
print(f"  .timeb created")

# ── Step 7: CLUES inference ────────────────────────────────────
print("Step 7: CLUES inference...")
clues_out = f'{PREFIX_BASE}_clues'
os.chdir(CLUES)
resample_full = f'{OUTDIR}/{resample_out}'
clues_full = f'{OUTDIR}/{clues_out}'
cmd_clues = [PYTHON, 'inference.py',
             '--times', resample_full,
             '--coal', coal_file,
             '--out', clues_full]
result = subprocess.run(cmd_clues, capture_output=True, text=True)
if not os.path.exists(f'{clues_full}.post.npy'):
    print(f"  ERROR: CLUES failed\n{result.stderr[:500]}")
    sys.exit(1)

# Print summary
epochs = np.load(f'{clues_full}.epochs.npy')
freqs = np.load(f'{clues_full}.freqs.npy')
post = np.load(f'{clues_full}.post.npy')
if post.shape[0] == len(freqs):
    post = post.T
posterior = np.exp(post - np.max(post, axis=1, keepdims=True))
posterior = posterior / posterior.sum(axis=1, keepdims=True)
mean = np.sum(posterior * freqs[np.newaxis, :], axis=1)
print(f"\n  SUCCESS: {GENE} in {POP}")
print(f"  Present freq: {mean[0]:.3f}")
print(f"  10kya freq: {mean[int(10000/29)]:.3f}")
print(f"  Output: {clues_out}.post.npy")
