#!/usr/bin/env python
"""Extract a minimal data subset around GRK2 for the demo notebook."""
import numpy as np
import os

REPO = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
PARSED = os.path.join(REPO, "analysis/genome_wide/cache/parsed/chr11.npz")
SAMPLES = os.path.join(REPO, "analysis/genome_wide/data/samples.txt")
TM_DIR = os.path.join(REPO, "analysis/orthogonal_v41/three_method")
OUT = os.path.join(REPO, "analysis/orthogonal_v41/grk2_demo.npz")

d = np.load(PARSED, allow_pickle=True, mmap_mode="r")
G, pos, sids = d["G"], d["positions"], d["sample_ids"]

pops = {}
with open(SAMPLES) as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 7:
            pops[p[1]] = (p[5], p[6])

def get_haps(pop):
    idx = []
    for i, s in enumerate(sids):
        if s in pops and pops[s][0] == pop:
            idx.extend([2 * i, 2 * i + 1])
    return sorted(idx)

gih = get_haps("GIH")
yri = get_haps("YRI")
ceu = get_haps("CEU")
print(f"GIH: {len(gih)} haps, YRI: {len(yri)}, CEU: {len(ceu)}")

mask = (pos >= 66_500_000) & (pos <= 68_000_000)
pos_w = np.array(pos[mask])
G_gih = np.ascontiguousarray(G[np.array(gih)][:, mask])
G_yri = np.ascontiguousarray(G[np.array(yri)][:, mask])
G_ceu = np.ascontiguousarray(G[np.array(ceu)][:, mask])
print(f"Window: {mask.sum()} sites, G_gih {G_gih.shape}")

focal = np.load(os.path.join(TM_DIR, "GRK2_GIH_novel.npz"), allow_pickle=True)
ctrl = np.load(os.path.join(TM_DIR, "GRK2_YRI_control.npz"), allow_pickle=True)

out = {
    "G_gih": G_gih, "G_yri": G_yri, "G_ceu": G_ceu,
    "positions": pos_w,
    "gene_start": 67266473, "gene_end": 67286556,
}
for key in focal.files:
    out[f"focal_{key}"] = focal[key]
for key in ctrl.files:
    out[f"ctrl_{key}"] = ctrl[key]

np.savez_compressed(OUT, **out)
print(f"Wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
