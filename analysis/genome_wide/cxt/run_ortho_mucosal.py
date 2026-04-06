#!/usr/bin/env python
"""Compute depletion/enrichment and max FST for all 4 mucosal immunity genes."""
import numpy as np
import os

PARSED_DIR = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cache/parsed"
SAMPLES_FILE = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/data/samples.txt"

GENES = [
    ("CLEC6A",    12, 8295819,  8314573),
    ("TRAF6",     11, 36488025, 36512297),
    ("TNFRSF13C", 22, 41901811, 41912652),
    ("JCHAIN",     4, 70574239, 70591222),
]

def load_pops():
    pops = {}
    with open(SAMPLES_FILE) as f:
        hdr = f.readline().split()
        pc = next(j for j, c in enumerate(hdr) if c.lower() in ("pop", "population"))
        for i, line in enumerate(f):
            p = line.split()[pc]
            pops.setdefault(p, []).append(i)
    return pops

def hap_idx(pops, pop_list):
    idx = []
    for p in pop_list:
        for s in pops.get(p, []):
            idx.extend([2 * s, 2 * s + 1])
    return np.array(idx)

pops = load_pops()
eas = hap_idx(pops, ["CHB", "JPT", "CHS", "CDX", "KHV"])
sas = hap_idx(pops, ["BEB", "GIH", "ITU", "PJL", "STU"])
eur = hap_idx(pops, ["CEU", "TSI", "FIN", "GBR", "IBS"])
afr = hap_idx(pops, ["YRI", "LWK", "GWD", "MSL", "ESN", "ACB", "ASW"])
amr = hap_idx(pops, ["MXL", "PUR", "CLM", "PEL"])
nf = np.concatenate([sas, eur, afr, amr])

for gene, chrn, start, end in GENES:
    path = os.path.join(PARSED_DIR, f"chr{chrn}.npz")
    d = np.load(path)
    G, pos = d["G"], d["positions"]
    # Immediately subset to gene region and free full chromosome
    mask_gene = (pos >= start) & (pos <= end)
    Gg = G[:, mask_gene]
    n = mask_gene.sum()
    del G, pos, d
    import gc; gc.collect()

    ea = Gg[eas].mean(0)
    sa = Gg[sas].mean(0)
    eu = Gg[eur].mean(0)
    af = Gg[afr].mean(0)
    am = Gg[amr].mean(0)

    dep = int(((ea < 0.1) & ((sa > 0.3) | (eu > 0.3) | (af > 0.3) | (am > 0.3))).sum())
    enr = int(((ea > 0.5) & (sa < 0.3) & (eu < 0.3) & (af < 0.3) & (am < 0.3)).sum())

    fst_vals = []
    for s in range(n):
        p1 = Gg[eas, s].mean()
        p2 = Gg[nf, s].mean()
        den = p1 * (1 - p2) + p2 * (1 - p1)
        if den > 0:
            num = (p1 - p2)**2 - p1*(1-p1)/(len(eas)-1) - p2*(1-p2)/(len(nf)-1)
            fst_vals.append(max(0, num / den))
        else:
            fst_vals.append(0)

    mx = max(fst_vals) if fst_vals else 0
    print(f"{gene} (chr{chrn}): {n} sites, Depleted={dep}, Enriched={enr}, MaxFST={mx:.3f}")
