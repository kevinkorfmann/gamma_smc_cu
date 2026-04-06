#!/usr/bin/env python
"""Merge per-chromosome orthogonal stats and compute genome-wide percentiles."""

import pandas as pd
import numpy as np
import os

OUTDIR = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt/results/orthogonal"

dfs = []
for chrn in range(1, 23):
    f = os.path.join(OUTDIR, f"ortho_chr{chrn}.csv")
    if os.path.exists(f):
        dfs.append(pd.read_csv(f))
        print(f"chr{chrn}: {len(dfs[-1])} genes (ortho)")
    else:
        print(f"chr{chrn}: MISSING (ortho)")

df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal ortho: {len(df)} genes")

# Merge Garud's H if available
garud_dfs = []
for chrn in range(1, 23):
    f = os.path.join(OUTDIR, f"garud_chr{chrn}.csv")
    if os.path.exists(f):
        garud_dfs.append(pd.read_csv(f))
if garud_dfs:
    garud = pd.concat(garud_dfs, ignore_index=True)
    print(f"Total garud: {len(garud)} genes")
    df = df.merge(garud[["gene", "chr", "H1", "H12", "H123", "H2_H1"]],
                  on=["gene", "chr"], how="left")

# Compute percentiles (low value = low percentile for all stats)
for stat in ["tajima_d", "pi_ratio", "max_ihs", "max_fst"]:
    df[f"{stat}_pctl"] = df[stat].rank(pct=True, na_option="keep")
# H12: high value = sweep, so high percentile = sweep signal
for stat in ["H1", "H12", "H123"]:
    if stat in df.columns:
        df[f"{stat}_pctl"] = df[stat].rank(pct=True, na_option="keep")
# H2/H1: low = hard sweep
if "H2_H1" in df.columns:
    df["H2_H1_pctl"] = df["H2_H1"].rank(pct=True, na_option="keep")

outf = os.path.join(OUTDIR, "ortho_genomewide.csv")
df.to_csv(outf, index=False)
print(f"Saved: {outf}")

# Print candidates vs controls
candidates = ["CLEC6A", "TRAF6", "TNFRSF13C", "JCHAIN",
               "GRK2", "BPIFA2", "CCDC92", "SLC6A15"]
controls = ["TTBK1", "CCDC70", "CZIB", "PER1", "CUL1"]

has_garud = "H12" in df.columns

hdr = f"{'Gene':<14} {'type':<8} {'TajD':>7} {'pctl':>6} {'piR':>7} {'pctl':>6} {'iHS':>7} {'pctl':>6} {'FST':>7} {'pctl':>6}"
if has_garud:
    hdr += f" {'H12':>7} {'pctl':>6} {'H2/H1':>7} {'pctl':>6}"
print(f"\n{'='*len(hdr)}")
print(hdr)
print(f"{'='*len(hdr)}")

def fmt(val, f=".2f"):
    return f"{val:{f}}" if not np.isnan(val) else "NA"
def fmtp(val):
    return f"{val:.1%}" if not np.isnan(val) else "NA"

for gene_list, label in [(candidates, "SWEEP"), (controls, "NEUTRAL")]:
    for _, row in df[df["gene"].isin(gene_list)].iterrows():
        line = (f"{row['gene']:<14} {label:<8} "
                f"{fmt(row['tajima_d']):>7} {fmtp(row['tajima_d_pctl']):>6} "
                f"{fmt(row['pi_ratio'], '.3f'):>7} {fmtp(row['pi_ratio_pctl']):>6} "
                f"{fmt(row['max_ihs']):>7} {fmtp(row['max_ihs_pctl']):>6} "
                f"{fmt(row['max_fst'], '.3f'):>7} {fmtp(row['max_fst_pctl']):>6}")
        if has_garud:
            line += (f" {fmt(row.get('H12', np.nan), '.4f'):>7} {fmtp(row.get('H12_pctl', np.nan)):>6}"
                     f" {fmt(row.get('H2_H1', np.nan), '.3f'):>7} {fmtp(row.get('H2_H1_pctl', np.nan)):>6}")
        print(line)
    print()
