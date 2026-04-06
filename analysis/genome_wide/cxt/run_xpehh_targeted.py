#!/usr/bin/env python
"""
Compute XP-EHH at 8 sweep candidates + 5 neutral controls.
Focal population vs YRI (African reference).
"""

import numpy as np
import allel
import os
import time

BETTY_BASE = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide"
PARSED_DIR = os.path.join(BETTY_BASE, "cache/parsed")
SAMPLES_FILE = os.path.join(BETTY_BASE, "data/samples.txt")
OUTDIR = os.path.join(BETTY_BASE, "cxt/results/orthogonal")
os.makedirs(OUTDIR, exist_ok=True)

# (gene, chr, start, end, focal_superpop)
GENES = [
    # Mucosal immunity
    ("CLEC6A",    12, 8455962,    8478330,    "EAS"),
    ("TRAF6",     11, 36483769,   36510272,   "EAS"),
    ("TNFRSF13C", 22, 41922032,   41926806,   "EAS"),
    ("JCHAIN",     4, 70655541,   70681817,   "EAS"),
    # Novel
    ("GRK2",      11, 67266473,   67286556,   "SAS"),
    ("BPIFA2",    20, 33161768,   33181412,   "SAS"),
    ("CCDC92",    12, 123918660, 123972831,   "EAS"),
    ("SLC6A15",   12, 84859491,   84913629,   "EAS"),
    # Neutral controls (~50th TMRCA percentile)
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
REF_POPS = ["YRI"]
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


pop_indices = load_pop_indices(SAMPLES_FILE)
ref_hap_idx = get_hap_idx(pop_indices, REF_POPS)

# Group genes by (chromosome, focal_superpop) to avoid reloading
from collections import defaultdict
chr_groups = defaultdict(list)
for gene_info in GENES:
    chrn = gene_info[1]
    chr_groups[chrn].append(gene_info)

results = []

for chrn, gene_list in sorted(chr_groups.items()):
    print(f"\nLoading chr{chrn}...", flush=True)
    t0 = time.time()
    npz = np.load(os.path.join(PARSED_DIR, f"chr{chrn}.npz"))
    G = npz["G"]
    pos = npz["positions"]
    print(f"  {G.shape}, {time.time()-t0:.1f}s", flush=True)
    del npz

    # Compute XP-EHH per focal superpop needed on this chr
    superpops_needed = set(g[4] for g in gene_list)

    for superpop in superpops_needed:
        focal_pops = SUPERPOP_MAP[superpop]
        focal_hap_idx = get_hap_idx(pop_indices, focal_pops)

        # Region: union of all gene windows on this chr for this superpop
        gene_starts = [g[2] - WINDOW for g in gene_list if g[4] == superpop]
        gene_ends = [g[3] + WINDOW for g in gene_list if g[4] == superpop]
        region_start = max(0, min(gene_starts))
        region_end = max(gene_ends)

        # Generous padding for EHH integration
        pad = 2_000_000
        region_start = max(0, region_start - pad)
        region_end = min(pos[-1], region_end + pad)

        mask = (pos >= region_start) & (pos <= region_end)
        pos_r = pos[mask]
        h_focal = allel.HaplotypeArray(G[focal_hap_idx][:, mask].T)
        h_ref = allel.HaplotypeArray(G[ref_hap_idx][:, mask].T)

        ac_f = h_focal.count_alleles()
        ac_r = h_ref.count_alleles()
        is_ok = (ac_f.is_segregating() | ac_r.is_segregating()) & ac_f.is_biallelic() & ac_r.is_biallelic()

        pos_filt = pos_r[is_ok]
        h_focal_filt = h_focal[is_ok]
        h_ref_filt = h_ref[is_ok]

        print(f"  XP-EHH {superpop} vs YRI on chr{chrn} ({is_ok.sum()} variants)...", flush=True)
        t1 = time.time()
        try:
            xpehh = allel.xpehh(h_focal_filt, h_ref_filt, pos_filt, use_threads=False)
            print(f"  Done in {time.time()-t1:.0f}s", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}")
            for g in gene_list:
                if g[4] == superpop:
                    results.append({"gene": g[0], "chr": chrn, "superpop": superpop,
                                    "max_xpehh": np.nan, "mean_xpehh": np.nan})
            continue

        # Extract per-gene values
        for gene_name, _, gs, ge, sp in gene_list:
            if sp != superpop:
                continue
            gmask = (pos_filt >= gs - WINDOW) & (pos_filt <= ge + WINDOW)
            vals = xpehh[gmask]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                results.append({"gene": gene_name, "chr": chrn, "superpop": superpop,
                                "max_xpehh": np.nan, "mean_xpehh": np.nan})
            else:
                results.append({
                    "gene": gene_name, "chr": chrn, "superpop": superpop,
                    "max_xpehh": float(np.nanmax(vals)),
                    "mean_xpehh": float(np.nanmean(vals)),
                })

    del G, pos
    import gc; gc.collect()

import pandas as pd
df = pd.DataFrame(results)
outf = os.path.join(OUTDIR, "xpehh_targeted.csv")
df.to_csv(outf, index=False)

sweep_genes = ["CLEC6A", "TRAF6", "TNFRSF13C", "JCHAIN", "GRK2", "BPIFA2", "CCDC92", "SLC6A15"]
neutral_genes = ["TTBK1", "CCDC70", "CZIB", "PER1", "CUL1"]

print(f"\n{'='*65}")
print(f"{'Gene':<14} {'type':<8} {'max XP-EHH':>12} {'mean XP-EHH':>12}")
print(f"{'='*65}")
for gl, label in [(sweep_genes, "SWEEP"), (neutral_genes, "NEUTRAL")]:
    for _, r in df[df["gene"].isin(gl)].iterrows():
        mx = f"{r['max_xpehh']:.2f}" if not np.isnan(r['max_xpehh']) else "NA"
        mn = f"{r['mean_xpehh']:.3f}" if not np.isnan(r['mean_xpehh']) else "NA"
        print(f"{r['gene']:<14} {label:<8} {mx:>12} {mn:>12}")
    print()
