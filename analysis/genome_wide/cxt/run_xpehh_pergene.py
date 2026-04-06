#!/usr/bin/env python
"""
XP-EHH per gene: load gene ±2Mb for EHH context, compute XP-EHH,
report mean over gene body. One computation per gene, no giant unions.
"""

import numpy as np
import allel
import os
import gc

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

# Cache chromosome data to avoid reloading
chr_cache = {}

results = []
for gn, chrn, gs, ge, sp in GENES:
    # Load chromosome if not cached (or different from last)
    if chrn not in chr_cache:
        # Clear previous cache to save memory
        chr_cache.clear()
        gc.collect()
        print(f"Loading chr{chrn}...", flush=True)
        npz = np.load(os.path.join(PARSED_DIR, f"chr{chrn}.npz"))
        chr_cache[chrn] = (npz["G"], npz["positions"])
        del npz

    G, pos = chr_cache[chrn]
    foc_idx = get_hap(pi, SUPERPOP_MAP[sp])

    # Gene ± PAD
    rs, re = max(0, gs - PAD), ge + PAD
    rmask = (pos >= rs) & (pos <= re)
    pos_r = pos[rmask]

    hf = allel.HaplotypeArray(G[foc_idx][:, rmask].T)
    hr = allel.HaplotypeArray(G[ref_idx][:, rmask].T)
    acf, acr = hf.count_alleles(), hr.count_alleles()
    ok = (acf.is_segregating() | acr.is_segregating()) & acf.is_biallelic() & acr.is_biallelic()
    pos_f = pos_r[ok]

    print(f"  {gn} ({sp} vs YRI): {ok.sum()} variants in ±{PAD/1e6:.0f}Mb...", flush=True)
    try:
        xp = allel.xpehh(hf[ok], hr[ok], pos_f, use_threads=False)
    except Exception as e:
        print(f"    FAILED: {e}")
        results.append((gn, sp, np.nan, np.nan, 0))
        continue

    # Gene body only
    gmask = (pos_f >= gs) & (pos_f <= ge)
    vals = xp[gmask]
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        results.append((gn, sp, np.nan, np.nan, 0))
    else:
        results.append((gn, sp, float(np.mean(vals)), float(np.median(vals)), len(vals)))
    print(f"    mean={results[-1][2]:.3f}, n={results[-1][4]}" if not np.isnan(results[-1][2]) else "    no data")

sweep = ["CLEC6A", "TRAF6", "TNFRSF13C", "JCHAIN", "GRK2", "BPIFA2", "CCDC92", "SLC6A15"]
neutral = ["TTBK1", "CCDC70", "CZIB", "PER1", "CUL1"]

print()
hdr = f"{'Gene':<14} {'type':<8} {'mean':>8} {'median':>8} {'n':>6}"
print(hdr)
print("-" * len(hdr))
for gl, label in [(sweep, "SWEEP"), (neutral, "NEUTRAL")]:
    for gn, sp, mn, md, n in results:
        if gn in gl:
            m = f"{mn:.3f}" if not np.isnan(mn) else "NA"
            d = f"{md:.3f}" if not np.isnan(md) else "NA"
            print(f"{gn:<14} {label:<8} {m:>8} {d:>8} {n:>6}")
    print()
