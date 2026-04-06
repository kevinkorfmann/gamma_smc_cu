#!/usr/bin/env python
import pandas as pd, numpy as np, os

OUTDIR = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/cxt/results/orthogonal"

dfs = []
for chrn in range(1, 23):
    dfs.append(pd.read_csv(os.path.join(OUTDIR, f"garud_chr{chrn}.csv")))
df = pd.concat(dfs, ignore_index=True)
print(f"Total: {len(df)} genes")

df["H12_pctl"] = df["H12"].rank(pct=True, na_option="keep")
df["H2_H1_pctl"] = df["H2_H1"].rank(pct=True, na_option="keep")

candidates = ["CLEC6A","TRAF6","TNFRSF13C","JCHAIN","GRK2","BPIFA2","CCDC92","SLC6A15"]
controls = ["TTBK1","CCDC70","CZIB","PER1","CUL1"]

print(f"\n{'Gene':<14} {'type':<8} {'H12':>8} {'pctl':>8} {'H2/H1':>8} {'pctl':>8}")
print("-" * 58)
for gl, label in [(candidates, "SWEEP"), (controls, "NEUTRAL")]:
    for _, r in df[df["gene"].isin(gl)].iterrows():
        print(f"{r['gene']:<14} {label:<8} {r['H12']:.4f}  {r['H12_pctl']:.1%}  {r['H2_H1']:.4f}  {r['H2_H1_pctl']:.1%}")
    print()

print(f"Genome-wide H12: median={df['H12'].median():.4f}, mean={df['H12'].mean():.4f}")
