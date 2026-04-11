#!/usr/bin/env python
"""Compute Garud's H12 over sliding windows for one (chr, pop), then aggregate
to per-gene H12 max scores.

Garud's H12: take a window of N consecutive SNPs, count distinct haplotypes
and their frequencies p1 >= p2 >= ... >= pK; H12 = (p1 + p2)^2 + sum_{i>=3} pi^2.
This is the "two-haplotype combined" homozygosity from
Garud et al. 2015 (PLOS Genet) — the standard hard+soft sweep statistic.

For each (chr, pop):
  1. Load chr NPZ, subset to pop haplotypes, drop monomorphic sites.
  2. Slide a window of WIN_SNPS=400 SNPs along the chromosome with STEP=50 SNPs.
  3. For each window, hash haplotype rows (as bytes) -> haplotype counts ->
     compute H12.
  4. Per gene: take max H12 over all windows whose midpoint falls in
     [gstart, gend].
  5. Compute within-population per-gene rank percentiles.
  6. Write analysis/orthogonal_v41/h12/{chr}_{pop}.csv with columns:
     gene_name, n_windows, max_h12, max_h12_rank.

This is the new H12 baseline that matches the v4.1 gene set. v3 used the
old genome_wide gene boundaries; v4.1 has 19,119 protein-coding genes from
the new GENCODE TSV at analysis/genome_wide/cache/genes/.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
DATA_DIR = os.path.join(REPO, "analysis/genome_wide")
PARSED_DIR = os.path.join(DATA_DIR, "cache/parsed")
GENES_DIR = os.path.join(DATA_DIR, "cache/genes")
SAMPLES_PATH = os.path.join(DATA_DIR, "data/samples.txt")
OUT_DIR = os.path.join(REPO, "analysis/orthogonal_v41/h12")

WIN_SNPS = 400
STEP_SNPS = 50


def load_samples():
    pops = {}
    with open(SAMPLES_PATH) as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                pops[parts[1]] = (parts[5], parts[6])
    return pops


def load_chr_npz(chr_num):
    path = os.path.join(PARSED_DIR, f"chr{chr_num}.npz")
    d = np.load(path, allow_pickle=True, mmap_mode="r")
    return d["G"], d["positions"], d["sample_ids"]


def get_pop_haps(sample_ids, pop_map, population):
    indices = []
    for i, sid in enumerate(sample_ids):
        if sid in pop_map and pop_map[sid][0] == population:
            indices.extend([2 * i, 2 * i + 1])
    return sorted(indices)


def garud_h12_for_window(G_window: np.ndarray) -> float:
    """G_window: (n_haps, win_snps) uint8 0/1.

    H12 = (p1 + p2)^2 + sum_{i>=3} p_i^2
    where p_i are sorted haplotype frequencies.
    """
    n_haps = G_window.shape[0]
    # Hash each haplotype row to bytes for unique-counting
    rows = G_window.tobytes()
    row_len = G_window.shape[1]
    counts = {}
    for h in range(n_haps):
        key = G_window[h].tobytes()
        counts[key] = counts.get(key, 0) + 1
    freqs = sorted(counts.values(), reverse=True)
    p = np.array(freqs, dtype=np.float64) / n_haps
    if len(p) == 1:
        return 1.0
    h12 = (p[0] + p[1]) ** 2
    if len(p) > 2:
        h12 += float((p[2:] ** 2).sum())
    return h12


def slide_h12(G_pop: np.ndarray, positions: np.ndarray) -> tuple:
    """Slide a window of WIN_SNPS SNPs across the chromosome.

    Returns (window_midpoints_bp, h12_values).
    """
    n_sites = G_pop.shape[1]
    if n_sites < WIN_SNPS:
        return np.array([]), np.array([])
    starts = np.arange(0, n_sites - WIN_SNPS + 1, STEP_SNPS)
    n_win = len(starts)
    mids = np.empty(n_win, dtype=np.int64)
    h12s = np.empty(n_win, dtype=np.float64)
    for i, s in enumerate(starts):
        e = s + WIN_SNPS
        h12s[i] = garud_h12_for_window(G_pop[:, s:e])
        mids[i] = int(positions[(s + e) // 2])
    return mids, h12s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chr", type=int, required=True)
    parser.add_argument("--pop", required=True)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"=== H12: chr{args.chr} {args.pop} ===", flush=True)

    G, positions, sample_ids = load_chr_npz(args.chr)
    pop_map = load_samples()
    hap_idx = get_pop_haps(sample_ids, pop_map, args.pop)
    print(f"  {len(hap_idx)} haplotypes", flush=True)
    if len(hap_idx) < 4:
        print("  too few, skipping", flush=True)
        return

    G_pop = np.ascontiguousarray(G[np.array(hap_idx), :])
    af = G_pop.sum(axis=0) / G_pop.shape[0]
    poly = (af > 0) & (af < 1)
    G_pop = G_pop[:, poly]
    positions_poly = np.asarray(positions)[poly]
    print(f"  {G_pop.shape[1]} polymorphic sites", flush=True)

    t0 = time.time()
    mids, h12s = slide_h12(G_pop, positions_poly)
    print(f"  {len(mids)} windows ({time.time()-t0:.1f}s)", flush=True)

    # Per-gene aggregation: max H12 over windows whose midpoint is in the gene
    genes = pd.read_csv(os.path.join(GENES_DIR, f"chr{args.chr}_genes.tsv"), sep="\t")
    rows = []
    for _, gene in genes.iterrows():
        gstart = int(gene["start"])
        gend = int(gene["end"])
        in_gene = (mids >= gstart) & (mids <= gend)
        if not in_gene.any():
            continue
        rows.append({
            "chr": args.chr,
            "gene_name": gene["gene_name"],
            "gstart": gstart,
            "gend": gend,
            "n_windows": int(in_gene.sum()),
            "max_h12": float(h12s[in_gene].max()),
        })
    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT_DIR, f"chr{args.chr}_{args.pop}.csv")
    df.to_csv(out_path, index=False)
    print(f"  wrote {out_path} ({len(df)} genes)", flush=True)


if __name__ == "__main__":
    main()
