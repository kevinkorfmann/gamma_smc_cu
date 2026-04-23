#!/usr/bin/env python3
"""Scan Akbari 2026 selection stats for user's candidate gene regions.

For each candidate gene, find the maximum POSTERIOR probability of selection
across all variants in [start-flank, end+flank]. Report whether the gene
clears the paper's thresholds: 0.99 (main 347-loci set), 0.90, 0.50 (FDR~50%).
"""
import pandas as pd
from collections import defaultdict

STATS = "/Users/kevinkorfmann/Projects/gamma_smc_cu/local_memory/akbari2026/selection_stats.tsv"
USER_CSV = "/Users/kevinkorfmann/Projects/gamma_smc_cu/analysis/genome_wide/postprocess/genes_sd_flag.csv"
FLANK = 100_000  # +/- 100 kb around gene boundaries

# Pull the candidate regions from the user's file.
genes_df = pd.read_csv(USER_CSV)
candidates = [
    "GRK2", "BPIFA2", "SLC6A15", "CCDC92", "CLEC6A",
    "SLC24A5", "LCT", "MCM6", "ZRANB3", "DARS1",
    "ABCC11", "MYEF2", "CTXN2", "SHCBP1", "NOTCH2NLR",
    "AMY1A", "AMY1B",
]
cand = genes_df[genes_df["gene_name"].isin(candidates)].copy()
cand = cand.sort_values(["chr", "start"]).reset_index(drop=True)

# Build per-chromosome interval list.
intervals_by_chr = defaultdict(list)
for _, row in cand.iterrows():
    intervals_by_chr[str(int(row["chr"]))].append(
        (int(row["start"]) - FLANK, int(row["end"]) + FLANK,
         row["gene_name"], float(row["min_rank"]))
    )

# Result accumulators.
best = {g: {"max_post": 0.0, "max_row": None, "n_variants": 0,
            "n_99": 0, "n_90": 0, "n_50": 0}
        for g in candidates}

header = None
with open(STATS) as f:
    for line in f:
        if line.startswith("##"):
            continue
        if header is None:
            header = line.rstrip("\n").split("\t")
            idx = {c: i for i, c in enumerate(header)}
            continue
        parts = line.rstrip("\n").split("\t")
        chrom = parts[idx["CHROM"]]
        if chrom not in intervals_by_chr:
            continue
        pos = int(parts[idx["POS"]])
        for (lo, hi, gname, _) in intervals_by_chr[chrom]:
            if lo <= pos <= hi:
                post = parts[idx["POSTERIOR"]]
                try:
                    post = float(post)
                except ValueError:
                    continue
                b = best[gname]
                b["n_variants"] += 1
                if post >= 0.99: b["n_99"] += 1
                if post >= 0.90: b["n_90"] += 1
                if post >= 0.50: b["n_50"] += 1
                if post > b["max_post"]:
                    b["max_post"] = post
                    b["max_row"] = {
                        "pos": pos, "rsid": parts[idx["RSID"]],
                        "S": parts[idx["S"]], "X": parts[idx["X"]],
                        "P_X": parts[idx["P_X"]],
                        "POSTERIOR": parts[idx["POSTERIOR"]],
                        "FDR": parts[idx["FDR"]],
                        "AF": parts[idx["AF"]],
                    }

# Print report.
print()
print(f"{'gene':<12} {'chr':>3}  {'gamma_smc_cu rank':>14}  "
      f"{'n_var':>6} {'max_post':>9} {'#≥.99':>6} {'#≥.90':>6} {'#≥.50':>6}  "
      f"top_variant")
print("-" * 130)
out_rows = []
for g in candidates:
    row = cand[cand["gene_name"] == g]
    if row.empty:
        print(f"{g:<12} (not in user's gene list)")
        continue
    row = row.iloc[0]
    b = best[g]
    top = b["max_row"]
    top_str = ("-" if top is None
               else f"{top['rsid']}@{row['chr']}:{top['pos']}  S={top['S'][:8]}  P_X={top['P_X'][:8]}  FDR={top['FDR'][:6]}")
    print(f"{g:<12} {int(row['chr']):>3}  {row['min_rank']:>14.2e}  "
          f"{b['n_variants']:>6} {b['max_post']:>9.4f} "
          f"{b['n_99']:>6} {b['n_90']:>6} {b['n_50']:>6}  {top_str}")
    out_rows.append({
        "gene": g, "chr": int(row["chr"]),
        "gamma_smc_cu_min_rank": row["min_rank"],
        "gamma_smc_cu_min_pop": row["min_pop"],
        "akbari_n_variants": b["n_variants"],
        "akbari_max_posterior": b["max_post"],
        "akbari_n_post_ge_99": b["n_99"],
        "akbari_n_post_ge_90": b["n_90"],
        "akbari_n_post_ge_50": b["n_50"],
        "akbari_top_rsid": None if top is None else top["rsid"],
        "akbari_top_pos": None if top is None else top["pos"],
        "akbari_top_S":   None if top is None else top["S"],
        "akbari_top_P_X": None if top is None else top["P_X"],
        "akbari_top_FDR": None if top is None else top["FDR"],
    })

pd.DataFrame(out_rows).to_csv(
    "/Users/kevinkorfmann/Projects/gamma_smc_cu/local_memory/akbari2026/candidate_overlap.csv",
    index=False)
print()
print("wrote: local_memory/akbari2026/candidate_overlap.csv")
