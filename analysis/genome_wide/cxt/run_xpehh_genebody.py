#!/usr/bin/env python
"""XP-EHH: mean over gene body only (not max over ±500kb window)."""

import numpy as np
import allel
import os
import gc
from collections import defaultdict

BETTY_BASE = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide"
PARSED_DIR = os.path.join(BETTY_BASE, "cache/parsed")
SAMPLES_FILE = os.path.join(BETTY_BASE, "data/samples.txt")

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
REF = ["YRI"]
PAD = 2_000_000


def load_pop_indices(f):
    pops = {}
    with open(f) as fh:
        header = fh.readline().strip().split()
        pc = next(j for j, c in enumerate(header) if c.lower() in ("pop", "population"))
        for i, line in enumerate(fh):
            pops.setdefault(line.strip().split()[pc], []).append(i)
    return pops


def get_hap(pi, plist):
    idx = []
    for p in plist:
        for s in pi[p]:
            idx.extend([2 * s, 2 * s + 1])
    return np.array(idx)


pi = load_pop_indices(SAMPLES_FILE)
ref_idx = get_hap(pi, REF)

chr_groups = defaultdict(list)
for g in GENES:
    chr_groups[g[1]].append(g)

results = []
for chrn, glist in sorted(chr_groups.items()):
    print(f"chr{chrn}...", flush=True)
    npz = np.load(os.path.join(PARSED_DIR, f"chr{chrn}.npz"))
    G, pos = npz["G"], npz["positions"]
    del npz

    for superpop in set(g[4] for g in glist):
        foc_idx = get_hap(pi, SUPERPOP_MAP[superpop])
        starts = [g[2] - PAD for g in glist if g[4] == superpop]
        ends = [g[3] + PAD for g in glist if g[4] == superpop]
        rmask = (pos >= max(0, min(starts))) & (pos <= max(ends))
        pos_r = pos[rmask]
        hf = allel.HaplotypeArray(G[foc_idx][:, rmask].T)
        hr = allel.HaplotypeArray(G[ref_idx][:, rmask].T)
        acf, acr = hf.count_alleles(), hr.count_alleles()
        ok = (acf.is_segregating() | acr.is_segregating()) & acf.is_biallelic() & acr.is_biallelic()
        pos_f = pos_r[ok]
        print(f"  XP-EHH {superpop} vs YRI ({ok.sum()} variants)...", flush=True)
        xp = allel.xpehh(hf[ok], hr[ok], pos_f, use_threads=False)

        for gn, _, gs, ge, sp in glist:
            if sp != superpop:
                continue
            gmask = (pos_f >= gs) & (pos_f <= ge)
            vals = xp[gmask]
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                results.append((gn, sp, np.nan, np.nan, 0))
            else:
                results.append((gn, sp, float(np.mean(vals)), float(np.median(vals)), len(vals)))

    del G, pos
    gc.collect()

sweep = ["CLEC6A", "TRAF6", "TNFRSF13C", "JCHAIN", "GRK2", "BPIFA2", "CCDC92", "SLC6A15"]
neutral = ["TTBK1", "CCDC70", "CZIB", "PER1", "CUL1"]

print()
print(f"{'Gene':<14} {'type':<8} {'mean':>8} {'median':>8} {'n':>6}")
print("-" * 46)
for gl, label in [(sweep, "SWEEP"), (neutral, "NEUTRAL")]:
    for gn, sp, mn, md, n in results:
        if gn in gl:
            m = f"{mn:.3f}" if not np.isnan(mn) else "NA"
            d = f"{md:.3f}" if not np.isnan(md) else "NA"
            print(f"{gn:<14} {label:<8} {m:>8} {d:>8} {n:>6}")
    print()
