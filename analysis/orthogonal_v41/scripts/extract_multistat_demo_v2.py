#!/usr/bin/env python
"""Extract minimal data subsets for the multi-stat demo figure (v2).

Same as v1 but also extracts pre-normalized iHS/nSL from the selscan
output (chromosome-wide normalization), so the notebook doesn't need
to do window-local normalization which is unreliable.
"""
import numpy as np
import os
import pandas as pd

REPO = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
PARSED_DIR = os.path.join(REPO, "analysis/genome_wide/cache/parsed")
GENES_DIR = os.path.join(REPO, "analysis/genome_wide/cache/genes")
SAMPLES_PATH = os.path.join(REPO, "analysis/genome_wide/data/samples.txt")
TM_DIR = os.path.join(REPO, "analysis/orthogonal_v41/three_method")
SELSCAN_DIR = os.path.join(REPO, "analysis/orthogonal_v41/selscan")
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

def load_selscan_window(chr_num, pop, win_lo, win_hi):
    """Load selscan iHS output for a (chr, pop) and slice to window.

    Returns (positions, raw_ihs) for sites in [win_lo, win_hi].
    The raw iHS from selscan is already chromosome-wide but NOT
    frequency-bin normalized. We do the normalization here using
    the FULL chromosome's data for proper bin stats.
    """
    task_dir = os.path.join(SELSCAN_DIR, f"chr{chr_num}_{pop}")
    ihs_path = os.path.join(task_dir, "ihs.ihs.out")
    if not os.path.exists(ihs_path):
        return np.array([]), np.array([])

    df = pd.read_csv(ihs_path, sep="\t")
    # Columns: chr, id, pos, freq, ihh1, ihh0, ihs

    # Frequency-bin normalize across FULL chromosome
    ihs_col = df.columns[-1]  # 'ihs'
    freq_col = 'freq'
    raw = df[ihs_col].values.astype(np.float64)
    freq = df[freq_col].values.astype(np.float64)

    bins = np.linspace(0, 1, 21)
    bin_idx = np.digitize(freq, bins) - 1
    bin_idx = np.clip(bin_idx, 0, 19)

    normed = np.full_like(raw, np.nan)
    for b in range(20):
        mask = (bin_idx == b) & np.isfinite(raw)
        if mask.sum() < 50:
            continue
        vals = raw[mask]
        mu, sd = vals.mean(), vals.std(ddof=0)
        if sd > 0:
            normed[mask] = (vals - mu) / sd

    # Slice to window
    pos_all = df['pos'].values.astype(np.int64)
    win_mask = (pos_all >= win_lo) & (pos_all <= win_hi)

    return pos_all[win_mask], normed[win_mask]

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
    print(f"  G_focal {G_focal.shape}, G_yri {G_yri.shape}", flush=True)

    prefix = f"{gene}_{focal_pop}"
    out[f"{prefix}_G_focal"] = G_focal
    out[f"{prefix}_G_yri"] = G_yri
    out[f"{prefix}_positions"] = pos_w
    out[f"{prefix}_gene_start"] = gstart
    out[f"{prefix}_gene_end"] = gend
    out[f"{prefix}_focal_pop"] = focal_pop
    out[f"{prefix}_group"] = group

    # Precomputed TMRCA
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

    # Pre-normalized iHS from selscan (chromosome-wide normalization)
    ihs_pos, ihs_norm = load_selscan_window(chr_num, focal_pop, win_lo, win_hi)
    out[f"{prefix}_ihs_pos"] = ihs_pos
    out[f"{prefix}_ihs_norm"] = ihs_norm
    n_valid = np.isfinite(ihs_norm).sum()
    in_gene = (ihs_pos >= gstart) & (ihs_pos <= gend)
    n_in_gene = np.isfinite(ihs_norm[in_gene]).sum() if in_gene.any() else 0
    print(f"  selscan iHS: {len(ihs_pos)} sites in window, {n_valid} valid, "
          f"{n_in_gene} in gene body", flush=True)

np.savez_compressed(OUT, **out)
print(f"Wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
