"""Compute XP-EHH vs YRI at Table-S3 genes (5 candidates + 5 SI neutrals).
Each focal gene is labelled with its focal-pop superpop (EAS or SAS).
Mean and max XP-EHH over variants in [gene_start - 500kb, gene_end + 500kb].

Output: analysis/orthogonal_v41/xpehh_s3.csv
"""
from __future__ import annotations
import os, time, gc
from collections import defaultdict
import numpy as np
import pandas as pd
import allel

REPO = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
PARSED = os.path.join(REPO, "analysis/genome_wide/cache/parsed")
SAMPLES = os.path.join(REPO, "analysis/genome_wide/data/samples.txt")
OUT = os.path.join(REPO, "analysis/orthogonal_v41/xpehh_s3.csv")

# (gene, chr, start, end, superpop)
GENES = [
    ("CCDC92",  12, 123918660, 123972831, "EAS"),
    ("CLEC6A",  12, 8455962,   8478330,   "EAS"),
    ("SLC6A15", 12, 84859491,  84913629,  "EAS"),
    ("GRK2",    11, 67266473,  67286556,  "SAS"),
    ("BPIFA2",  20, 33161768,  33181412,  "SAS"),
    # S3 neutrals (TMRCA ≈ 50%) — superpop chosen to match S3 table's EAS/SAS pop
    ("ADAM22",  7,  87934143,  88202889,  "EAS"),
    ("CCDC70",  13, 51861969,  51866232,  "SAS"),
    ("TMEM30A", 6,  75252924,  75284948,  "EAS"),
    ("ZNF420",  19, 37007857,  37130368,  "EAS"),
    ("RAB11FIP3",16,  425649,    523011,  "SAS"),
]

SUPERPOP_MAP = {
    "EAS": ["CHB", "JPT", "CHS", "CDX", "KHV"],
    "SAS": ["GIH", "PJL", "BEB", "STU", "ITU"],
}
REF_POPS = ["YRI"]
WINDOW = 500_000  # ±500 kb for gene-body extent
PAD = 2_000_000   # extra padding for EHH integration

def load_pop_indices():
    pops = {}
    with open(SAMPLES) as f:
        header = f.readline().strip().split()
        pop_col = next(i for i, c in enumerate(header) if c.lower() in ("pop", "population"))
        for i, line in enumerate(f):
            fields = line.strip().split()
            pops.setdefault(fields[pop_col], []).append(i)
    return pops

def hap_idx(pop_indices, pops_list):
    idx = []
    for p in pops_list:
        for si in pop_indices[p]:
            idx.extend([2 * si, 2 * si + 1])
    return np.array(idx)

pop_indices = load_pop_indices()
ref_hap = hap_idx(pop_indices, REF_POPS)

chr_groups = defaultdict(list)
for g in GENES:
    chr_groups[g[1]].append(g)

results = []
for chrn, gl in sorted(chr_groups.items()):
    print(f"=== chr{chrn}: {[g[0] for g in gl]} ===", flush=True)
    t0 = time.time()
    npz = np.load(os.path.join(PARSED, f"chr{chrn}.npz"), mmap_mode="r")
    G = npz["G"]; pos = npz["positions"]
    print(f"  loaded G {G.shape} in {time.time()-t0:.0f}s", flush=True)

    for superpop in set(g[4] for g in gl):
        focal_hap = hap_idx(pop_indices, SUPERPOP_MAP[superpop])
        starts = [g[2] - WINDOW - PAD for g in gl if g[4] == superpop]
        ends   = [g[3] + WINDOW + PAD for g in gl if g[4] == superpop]
        region_s = max(0, min(starts))
        region_e = max(ends)
        mask = (pos >= region_s) & (pos <= region_e)
        pos_r = pos[mask]
        h_f = allel.HaplotypeArray(G[focal_hap][:, mask].T)
        h_r = allel.HaplotypeArray(G[ref_hap][:, mask].T)
        ac_f = h_f.count_alleles(); ac_r = h_r.count_alleles()
        ok = (ac_f.is_segregating() | ac_r.is_segregating()) & ac_f.is_biallelic() & ac_r.is_biallelic()
        pos_f = pos_r[ok]; h_f_f = h_f[ok]; h_r_f = h_r[ok]
        print(f"  {superpop} vs YRI chr{chrn}: {ok.sum()} variants", flush=True)
        t1 = time.time()
        try:
            xp = allel.xpehh(h_f_f, h_r_f, pos_f, use_threads=False)
            print(f"    xpehh done in {time.time()-t1:.0f}s", flush=True)
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
            for g in gl:
                if g[4] == superpop:
                    results.append({"gene": g[0], "chr": chrn, "superpop": superpop,
                                    "max_xpehh": np.nan, "mean_xpehh": np.nan, "n_variants": 0})
            continue
        for gene, _, gs, ge, sp in gl:
            if sp != superpop: continue
            gm = (pos_f >= gs - WINDOW) & (pos_f <= ge + WINDOW)
            vals = xp[gm]
            vals = vals[~np.isnan(vals)]
            results.append({
                "gene": gene, "chr": chrn, "superpop": superpop,
                "max_xpehh": float(np.nanmax(vals)) if len(vals) else np.nan,
                "mean_xpehh": float(np.nanmean(vals)) if len(vals) else np.nan,
                "n_variants": int(len(vals)),
            })
    del G, pos; gc.collect()

df = pd.DataFrame(results)
df.to_csv(OUT, index=False)
print(df.to_string(index=False))
print(f"\nwrote {OUT}")
