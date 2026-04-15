#!/usr/bin/env python
"""Extract minimal data subsets for the multi-stat demo figure.

For each gene: ±500 kb haplotype window for focal pop + YRI, plus
precomputed TMRCA traces from the three_method NPZ.
"""
import numpy as np
import os

REPO = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
PARSED_DIR = os.path.join(REPO, "analysis/genome_wide/cache/parsed")
GENES_DIR = os.path.join(REPO, "analysis/genome_wide/cache/genes")
SAMPLES_PATH = os.path.join(REPO, "analysis/genome_wide/data/samples.txt")
TM_DIR = os.path.join(REPO, "analysis/orthogonal_v41/three_method")
OUT = os.path.join(REPO, "analysis/orthogonal_v41/multistat_demo.npz")

GENES = [
    ("GRK2",    11, "GIH", "novel"),
    ("BPIFA2",  20, "GIH", "novel"),
    ("SLC6A15", 12, "CHS", "novel"),
    ("CCDC92",  12, "CDX", "novel"),
    ("CLEC6A",  12, "CDX", "novel"),
    ("LCT",      2, "CEU", "positive"),
]

WINDOW_BP = 500_000

pops = {}
with open(SAMPLES_PATH) as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 7:
            pops[p[1]] = (p[5], p[6])

def get_haps(sample_ids, pop):
    idx = []
    for i, s in enumerate(sample_ids):
        if s in pops and pops[s][0] == pop:
            idx.extend([2 * i, 2 * i + 1])
    return sorted(idx)

import pandas as pd

out = {}
chr_cache = {}

for gene, chr_num, focal_pop, group in GENES:
    print(f"=== {gene} chr{chr_num} {focal_pop} ===", flush=True)

    if chr_num not in chr_cache:
        path = os.path.join(PARSED_DIR, f"chr{chr_num}.npz")
        d = np.load(path, allow_pickle=True, mmap_mode="r")
        chr_cache[chr_num] = (d["G"], d["positions"], d["sample_ids"])

    G, positions, sample_ids = chr_cache[chr_num]

    genes_df = pd.read_csv(os.path.join(GENES_DIR, f"chr{chr_num}_genes.tsv"), sep="\t")
    row = genes_df[genes_df["gene_name"] == gene].iloc[0]
    gstart, gend = int(row["start"]), int(row["end"])
    midpoint = (gstart + gend) // 2

    win_lo = midpoint - WINDOW_BP
    win_hi = midpoint + WINDOW_BP
    mask = (positions >= win_lo) & (positions <= win_hi)
    pos_w = np.array(positions[mask])

    focal_idx = get_haps(sample_ids, focal_pop)
    yri_idx = get_haps(sample_ids, "YRI")

    G_focal = np.ascontiguousarray(G[np.array(focal_idx)][:, mask])
    G_yri = np.ascontiguousarray(G[np.array(yri_idx)][:, mask])
    print(f"  G_focal {G_focal.shape}, G_yri {G_yri.shape}, sites {mask.sum()}")

    prefix = f"{gene}_{focal_pop}"
    out[f"{prefix}_G_focal"] = G_focal
    out[f"{prefix}_G_yri"] = G_yri
    out[f"{prefix}_positions"] = pos_w
    out[f"{prefix}_gene_start"] = gstart
    out[f"{prefix}_gene_end"] = gend
    out[f"{prefix}_focal_pop"] = focal_pop
    out[f"{prefix}_group"] = group

    # Load precomputed TMRCA
    focal_npz = os.path.join(TM_DIR, f"{gene}_{focal_pop}_{group}.npz")
    ctrl_npz = os.path.join(TM_DIR, f"{gene}_YRI_control.npz")
    if os.path.exists(focal_npz):
        fd = np.load(focal_npz, allow_pickle=True)
        for k in ["tmrca_cu_mean", "tmrca_cu_positions"]:
            if k in fd.files:
                out[f"{prefix}_focal_{k}"] = fd[k]
    if os.path.exists(ctrl_npz):
        cd = np.load(ctrl_npz, allow_pickle=True)
        for k in ["tmrca_cu_mean", "tmrca_cu_positions"]:
            if k in cd.files:
                out[f"{prefix}_ctrl_{k}"] = cd[k]

np.savez_compressed(OUT, **out)
print(f"Wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
