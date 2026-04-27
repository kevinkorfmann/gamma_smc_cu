#!/usr/bin/env python3
"""Summarize our own iHS/nSL ranks for TREM2 and TREML1 across populations.

Key question: if the H12 peak at 41.15 Mb is pan-non-AFR, do our own iHS/nSL
scans flag TREM2/TREML1 as extreme in any EUR/EAS/SAS population?
"""
from __future__ import annotations

import os
import pandas as pd

DIR = "/Users/kevinkorfmann/Projects/tmrca.cu/analysis/orthogonal_v41/selscan_genelevel"

POPS = ["ACB","ASW","ESN","GWD","LWK","MSL","YRI",
        "CEU","FIN","GBR","IBS","TSI",
        "CDX","CHB","CHS","JPT","KHV",
        "BEB","GIH","ITU","PJL","STU",
        "CLM","MXL","PEL","PUR"]

rows = []
for p in POPS:
    fn = os.path.join(DIR, f"{p}.csv")
    if not os.path.exists(fn):
        continue
    df = pd.read_csv(fn)
    for gene in ["TREM2", "TREML1", "TREM1", "TREML2", "TREML4", "NCR2", "FOXP4"]:
        g = df[df.gene_name == gene]
        if len(g) == 0:
            rows.append({"pop": p, "gene": gene, "n_ihs": 0,
                         "max_ihs": None, "frac_ext_ihs": None,
                         "rank_max_ihs": None, "rank_frac_ihs": None})
            continue
        r = g.iloc[0]
        rows.append({
            "pop": p,
            "gene": gene,
            "n_ihs": int(r["n_ihs_sites"]),
            "max_ihs": round(r["max_abs_ihs_norm"], 3),
            "frac_ext_ihs": round(r["frac_ihs_extreme"], 3),
            "rank_max_ihs": round(r["max_abs_ihs_norm_rank"], 4),
            "rank_frac_ihs": round(r["frac_ihs_extreme_rank"], 4),
        })

df = pd.DataFrame(rows)

# For each gene, print pop-by-pop ranks
print("Rank values: LOWER = MORE EXTREME (0.001 = top 0.1%)\n")
for gene in ["TREM2", "TREML1"]:
    print(f"=== {gene} ===")
    g = df[df.gene == gene].copy()
    # Sort by rank_frac_ihs ascending (most extreme first)
    g = g.sort_values("rank_frac_ihs", na_position="last")
    print(g[["pop", "n_ihs", "max_ihs", "frac_ext_ihs",
             "rank_max_ihs", "rank_frac_ihs"]].to_string(index=False))
    print()

# Flag populations where TREM2 or TREML1 rank in top 1% or 5%
print("\n=== Populations with iHS top-5% hit at TREM2 or TREML1 ===")
for gene in ["TREM2", "TREML1"]:
    g = df[(df.gene == gene) & (df.rank_frac_ihs < 0.05)]
    if len(g) == 0:
        continue
    for _, r in g.iterrows():
        print(f"  {gene:>8} {r['pop']}: rank_frac_ihs={r['rank_frac_ihs']:.4f} (n_sites={r['n_ihs']})")

print("\n=== Summary across TREM cluster + FOXP4 ===")
for gene in ["TREM2", "TREML1", "TREML2", "TREML4", "TREM1", "NCR2", "FOXP4"]:
    g = df[df.gene == gene]
    valid = g.dropna(subset=["rank_frac_ihs"])
    if len(valid) == 0:
        print(f"  {gene:>8}: no selscan data (likely insufficient SNPs)")
        continue
    best = valid.nsmallest(3, "rank_frac_ihs")
    best_str = "; ".join(f"{r['pop']}={r['rank_frac_ihs']:.3f}" for _, r in best.iterrows())
    print(f"  {gene:>8}: top-3 most-extreme pops: {best_str}")
