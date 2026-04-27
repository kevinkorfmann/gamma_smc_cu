#!/usr/bin/env python
"""Extended reproducibility comparison.

Covers three test categories against the canonical genome-wide results:
  A. Same-params independent rerun   (original repro dir: default/chr21, chr22 OR raw chr21/chr22)
  B. Extended chr + population coverage (default/chr15 with 6 pops, default/chr7 with 5 pops)
  C. Chunk-size invariance           (chunk500/chr22 and chunk2000/chr22, default was 1000)

For every (repro_csv, orig_csv) pair: Pearson, Spearman, max/p99/median |rel diff|,
exact-match fraction, within-chr rank Spearman, top-20/top-100 overlap.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

BASE = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide"
ORIG = os.path.join(BASE, "results")
REPRO = os.path.join(BASE, "reproducibility", "results")


def load_csv(path):
    df = pd.read_csv(path)
    df = df[df["geom_mean_tmrca"].notna()].copy()
    return df


def compare_one(f_orig, f_repro, label, chr_num, pop):
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

    r_orig = pd.Series(x).rank(method="min").values
    r_repro = pd.Series(y).rank(method="min").values

    def topn_overlap(n):
        n = min(n, len(m))
        o = set(np.argsort(x)[:n])
        r = set(np.argsort(y)[:n])
        return len(o & r) / n

    return {
        "test": label,
        "chr": chr_num,
        "pop": pop,
        "n_genes": len(m),
        "pearson_r": pearsonr(x, y)[0],
        "spearman_r": spearmanr(x, y)[0],
        "max_rel_diff": rel.max(),
        "p99_rel_diff": np.percentile(rel, 99),
        "median_rel_diff": np.median(rel),
        "exact_frac": float((x == y).mean()),
        "rank_spearman": spearmanr(r_orig, r_repro)[0],
        "top20_overlap": topn_overlap(20),
        "top100_overlap": topn_overlap(100),
    }


def collect_tests():
    tests = []
    # A. Same-params rerun (results from first slurm job, stored at REPRO/chr{N}/)
    for chr_num in [21, 22]:
        for pop in ["YRI", "CEU", "CHB"]:
            tests.append((
                "A_same_params",
                os.path.join(ORIG, f"chr{chr_num}", f"{pop}.csv"),
                os.path.join(REPRO, f"chr{chr_num}", f"{pop}.csv"),
                chr_num, pop,
            ))
    # B. Extended coverage (default/chr15, default/chr7)
    for chr_num, pops in [(15, ["YRI", "CEU", "CHB", "GIH", "PEL", "LWK"]),
                          (7,  ["YRI", "CEU", "CHB", "GIH", "PEL"])]:
        for pop in pops:
            tests.append((
                "B_extended_coverage",
                os.path.join(ORIG, f"chr{chr_num}", f"{pop}.csv"),
                os.path.join(REPRO, "default", f"chr{chr_num}", f"{pop}.csv"),
                chr_num, pop,
            ))
    # C. Chunk-size invariance (chunk500, chunk2000 vs original which used 1000)
    for subdir in ["chunk500", "chunk2000"]:
        for pop in ["YRI", "CEU", "CHB"]:
            tests.append((
                f"C_{subdir}",
                os.path.join(ORIG, "chr22", f"{pop}.csv"),
                os.path.join(REPRO, subdir, "chr22", f"{pop}.csv"),
                22, pop,
            ))
    return tests


def main():
    rows = []
    for label, fo, fr, c, p in collect_tests():
        r = compare_one(fo, fr, label, c, p)
        if r is not None:
            rows.append(r)
        else:
            print(f"[skip] {label} chr{c} {p} (missing file)")

    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v:.2e}" if abs(v) < 1 and v != 0 else f"{v:.6f}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print()
    print(df.to_string(index=False))
    print()
    print("=" * 80)
    print("SUMMARY BY TEST CATEGORY")
    print("=" * 80)
    for label, g in df.groupby("test"):
        print(f"\n[{label}]   n_cells={len(g)}   total_genes={g['n_genes'].sum()}")
        print(f"  Pearson r         : min={g['pearson_r'].min():.10f}   mean={g['pearson_r'].mean():.10f}")
        print(f"  Spearman r        : min={g['spearman_r'].min():.10f}   mean={g['spearman_r'].mean():.10f}")
        print(f"  rank Spearman     : min={g['rank_spearman'].min():.10f}   mean={g['rank_spearman'].mean():.10f}")
        print(f"  top-20 overlap    : min={g['top20_overlap'].min():.4f}   mean={g['top20_overlap'].mean():.4f}")
        print(f"  top-100 overlap   : min={g['top100_overlap'].min():.4f}   mean={g['top100_overlap'].mean():.4f}")
        print(f"  max |rel diff|    : overall max = {g['max_rel_diff'].max():.2e}")
        print(f"  p99 |rel diff|    : overall max = {g['p99_rel_diff'].max():.2e}")
        print(f"  median |rel diff| : overall max = {g['median_rel_diff'].max():.2e}")
        print(f"  exact-match frac  : min={g['exact_frac'].min():.4f}   mean={g['exact_frac'].mean():.4f}")

    out_csv = os.path.join(BASE, "reproducibility", "comparison_extended.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nWritten: {out_csv}")


if __name__ == "__main__":
    main()
