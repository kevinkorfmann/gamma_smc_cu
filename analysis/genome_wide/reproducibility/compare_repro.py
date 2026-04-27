#!/usr/bin/env python
"""Compare reproducibility rerun vs original TMRCA results.

For each (chr, pop), computes:
  - TMRCA agreement: N genes, Pearson & Spearman r on geom_mean_tmrca,
    max |relative diff|, 99th pct |relative diff|, exact-match fraction.
  - Rank agreement: Spearman on per-chr gene rank, top-N overlap (top 20, top 100).
  - Genome-wide rank: since repro is only 2 chr, rank within each chr.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

BASE = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide"
ORIG = os.path.join(BASE, "results")
REPRO = os.path.join(BASE, "reproducibility", "results")

CHRS = [21, 22]
POPS = ["YRI", "CEU", "CHB"]


def load_csv(path):
    df = pd.read_csv(path)
    # Drop rows where geom_mean_tmrca is NaN/empty
    df = df[df["geom_mean_tmrca"].notna()].copy()
    return df


def compare(chr_num, pop):
    f_orig = os.path.join(ORIG, f"chr{chr_num}", f"{pop}.csv")
    f_repro = os.path.join(REPRO, f"chr{chr_num}", f"{pop}.csv")
    if not (os.path.exists(f_orig) and os.path.exists(f_repro)):
        return None

    a = load_csv(f_orig)
    b = load_csv(f_repro)
    m = a.merge(b, on="gene_id", suffixes=("_orig", "_repro"))
    if len(m) == 0:
        return None

    x = m["geom_mean_tmrca_orig"].values
    y = m["geom_mean_tmrca_repro"].values
    rel = np.abs(x - y) / np.maximum(np.abs(x), 1e-9)

    pear = pearsonr(x, y)[0]
    spear = spearmanr(x, y)[0]

    # Ranks within chromosome (lower TMRCA = more selection-like = rank 1)
    r_orig = pd.Series(x).rank(method="min").values
    r_repro = pd.Series(y).rank(method="min").values
    rank_spear = spearmanr(r_orig, r_repro)[0]

    # Top-N overlap by lowest TMRCA
    def topn_overlap(n):
        o = set(np.argsort(x)[:n])
        r = set(np.argsort(y)[:n])
        return len(o & r) / n

    out = {
        "chr": chr_num,
        "pop": pop,
        "n_genes": len(m),
        "pearson_r": pear,
        "spearman_r": spear,
        "max_rel_diff": rel.max(),
        "p99_rel_diff": np.percentile(rel, 99),
        "median_rel_diff": np.median(rel),
        "exact_frac": float((x == y).mean()),
        "rank_spearman": rank_spear,
        "top20_overlap": topn_overlap(min(20, len(m))),
        "top100_overlap": topn_overlap(min(100, len(m))),
    }
    return out


def main():
    rows = []
    for c in CHRS:
        for p in POPS:
            r = compare(c, p)
            if r is not None:
                rows.append(r)
    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v:.6f}")
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print()
    print(f"Populations tested: {sorted(set(df['pop']))}")
    print(f"Chromosomes tested: {sorted(set(df['chr']))}")
    print(f"Mean Pearson r        : {df['pearson_r'].mean():.6f}")
    print(f"Mean Spearman r       : {df['spearman_r'].mean():.6f}")
    print(f"Mean rank Spearman    : {df['rank_spearman'].mean():.6f}")
    print(f"Mean top-20 overlap   : {df['top20_overlap'].mean():.4f}")
    print(f"Mean top-100 overlap  : {df['top100_overlap'].mean():.4f}")
    print(f"Max  max_rel_diff     : {df['max_rel_diff'].max():.2e}")
    print(f"Max  p99_rel_diff     : {df['p99_rel_diff'].max():.2e}")
    print(f"Min  exact_frac       : {df['exact_frac'].min():.4f}")

    out_csv = os.path.join(BASE, "reproducibility", "comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nWritten: {out_csv}")


if __name__ == "__main__":
    main()
