#!/usr/bin/env python
"""
Run SweepFinder2 at 8 sweep candidates + 5 neutral controls.
SweepFinder2 uses the composite likelihood ratio (CLR) test based on
the site frequency spectrum to detect hard sweeps.

Input format for SF2:
  frequency file: position x n folded (one line per site)
  spectrum file:  precomputed background SFS

Output: CLR values at grid positions — high CLR = sweep signal.
"""

import numpy as np
import allel
import subprocess
import os
import tempfile

BETTY_BASE = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide"
PARSED_DIR = os.path.join(BETTY_BASE, "cache/parsed")
SAMPLES_FILE = os.path.join(BETTY_BASE, "data/samples.txt")
SF2_BIN = os.path.join(BETTY_BASE, "cxt/sweepfinder2/SF2")
OUTDIR = os.path.join(BETTY_BASE, "cxt/results/orthogonal/sweepfinder2")
os.makedirs(OUTDIR, exist_ok=True)

GENES = [
    ("CLEC6A",    12, 8455962,    8478330,    "EAS"),
    ("TRAF6",     11, 36483769,   36510272,   "EAS"),
    ("TNFRSF13C", 22, 41922032,   41926806,   "EAS"),
    ("JCHAIN",     4, 70655541,   70681817,   "EAS"),
    ("GRK2",      11, 67266473,   67286556,   "SAS"),
    ("BPIFA2",    20, 33161768,   33181412,   "SAS"),
    ("CCDC92",    12, 123918660, 123972831,   "EAS"),
    ("SLC6A15",   12, 84859491,   84913629,   "EAS"),
    ("TTBK1",      6, 43243481,   43288258,   "EAS"),
    ("CCDC70",    13, 51861969,   51866232,   "EAS"),
    ("CZIB",       1, 53214099,   53220634,   "EAS"),
    ("PER1",      17, 8140472,    8156506,    "EAS"),
    ("CUL1",       7, 148697914, 148801110,   "EAS"),
]

SUPERPOP_MAP = {
    "EAS": ["CHB", "JPT", "CHS", "CDX", "KHV"],
    "SAS": ["GIH", "PJL", "BEB", "STU", "ITU"],
}
WINDOW = 500_000


def load_pop_indices(samples_file):
    pops = {}
    with open(samples_file) as f:
        header = f.readline().strip().split()
        pop_col = next(j for j, c in enumerate(header) if c.lower() in ("pop", "population"))
        for i, line in enumerate(f):
            fields = line.strip().split()
            pops.setdefault(fields[pop_col], []).append(i)
    return pops


def get_hap_idx(pop_indices, pop_list):
    idx = []
    for pop in pop_list:
        for si in pop_indices[pop]:
            idx.extend([2 * si, 2 * si + 1])
    return np.array(idx)


def make_sf2_input(pos, haps, n_haplotypes):
    """Create SweepFinder2 frequency file content.
    Format: position x n folded
    x = derived allele count, n = total alleles, folded = 0 (unfolded) or 1 (folded)
    """
    ac = np.sum(haps, axis=0)  # derived allele count per site
    lines = ["position\tx\tn\tfolded\n"]
    for i in range(len(pos)):
        x = int(ac[i])
        # Skip monomorphic
        if x == 0 or x == n_haplotypes:
            continue
        lines.append(f"{int(pos[i])}\t{x}\t{n_haplotypes}\t0\n")
    return "".join(lines)


def compute_background_sfs(haps, n_haplotypes):
    """Compute background SFS for SweepFinder2 -s option.
    Format: n+1 lines, each with count of sites with that many derived alleles.
    """
    ac = np.sum(haps, axis=0)
    sfs = np.bincount(ac.astype(int), minlength=n_haplotypes + 1)
    # Remove 0 and n (monomorphic)
    sfs[0] = 0
    sfs[n_haplotypes] = 0
    lines = [f"{n_haplotypes}\n"]
    for i in range(n_haplotypes + 1):
        lines.append(f"{sfs[i]}\n")
    return "".join(lines)


pop_indices = load_pop_indices(SAMPLES_FILE)

from collections import defaultdict
import gc

chr_groups = defaultdict(list)
for g in GENES:
    chr_groups[g[1]].append(g)

results = []

for chrn, gene_list in sorted(chr_groups.items()):
    print(f"\nLoading chr{chrn}...", flush=True)
    npz = np.load(os.path.join(PARSED_DIR, f"chr{chrn}.npz"))
    G = npz["G"]
    pos = npz["positions"]
    del npz
    print(f"  {G.shape}", flush=True)

    superpops_needed = set(g[4] for g in gene_list)

    for superpop in superpops_needed:
        focal_pops = SUPERPOP_MAP[superpop]
        focal_hap_idx = get_hap_idx(pop_indices, focal_pops)
        n_haps = len(focal_hap_idx)

        for gene_name, _, gs, ge, sp in gene_list:
            if sp != superpop:
                continue

            rs = max(0, gs - WINDOW)
            re = ge + WINDOW
            mask = (pos >= rs) & (pos <= re)
            pos_r = pos[mask]
            haps_r = G[focal_hap_idx][:, mask]  # (n_haps, n_sites)

            print(f"  {gene_name}: {mask.sum()} sites in ±{WINDOW/1e3:.0f}kb", flush=True)

            if mask.sum() < 50:
                results.append({"gene": gene_name, "chr": chrn, "superpop": superpop,
                                "max_CLR": np.nan, "mean_CLR": np.nan})
                continue

            # Write input files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.freq', delete=False) as f:
                freq_file = f.name
                f.write(make_sf2_input(pos_r, haps_r, n_haps))

            with tempfile.NamedTemporaryFile(mode='w', suffix='.sfs', delete=False) as f:
                sfs_file = f.name
                f.write(compute_background_sfs(haps_r, n_haps))

            out_file = os.path.join(OUTDIR, f"{gene_name}_{superpop}_sf2.out")

            # Run SweepFinder2: -lu = use precomputed SFS + grid at every SNP position
            cmd = [SF2_BIN, "-lu", "100", freq_file, sfs_file, out_file]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    print(f"    SF2 error: {result.stderr[:200]}")
                    results.append({"gene": gene_name, "chr": chrn, "superpop": superpop,
                                    "max_CLR": np.nan, "mean_CLR": np.nan})
                else:
                    # Parse output: position LR alpha
                    clr_vals = []
                    with open(out_file) as of:
                        header = of.readline()
                        for line in of:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                clr = float(parts[1])
                                clr_vals.append(clr)
                    clr_vals = np.array(clr_vals)
                    max_clr = float(np.max(clr_vals)) if len(clr_vals) else np.nan
                    mean_clr = float(np.mean(clr_vals)) if len(clr_vals) else np.nan
                    print(f"    max CLR = {max_clr:.1f}, mean CLR = {mean_clr:.1f}")
                    results.append({"gene": gene_name, "chr": chrn, "superpop": superpop,
                                    "max_CLR": max_clr, "mean_CLR": mean_clr})
            except Exception as e:
                print(f"    SF2 failed: {e}")
                results.append({"gene": gene_name, "chr": chrn, "superpop": superpop,
                                "max_CLR": np.nan, "mean_CLR": np.nan})

            os.unlink(freq_file)
            os.unlink(sfs_file)

    del G, pos
    gc.collect()

import pandas as pd
df = pd.DataFrame(results)
outf = os.path.join(OUTDIR, "sweepfinder2_targeted.csv")
df.to_csv(outf, index=False)

sweep = ["CLEC6A", "TRAF6", "TNFRSF13C", "JCHAIN", "GRK2", "BPIFA2", "CCDC92", "SLC6A15"]
neutral = ["TTBK1", "CCDC70", "CZIB", "PER1", "CUL1"]

print(f"\n{'='*55}")
print(f"{'Gene':<14} {'type':<8} {'max CLR':>10} {'mean CLR':>10}")
print(f"{'='*55}")
for gl, label in [(sweep, "SWEEP"), (neutral, "NEUTRAL")]:
    for _, r in df[df["gene"].isin(gl)].iterrows():
        mx = f"{r['max_CLR']:.1f}" if not np.isnan(r['max_CLR']) else "NA"
        mn = f"{r['mean_CLR']:.1f}" if not np.isnan(r['mean_CLR']) else "NA"
        print(f"{r['gene']:<14} {label:<8} {mx:>10} {mn:>10}")
    print()
