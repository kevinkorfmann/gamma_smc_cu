#!/usr/bin/env python
"""Aggregate per-chromosome per-population CSVs into genome-wide ranks.

Reads results/chr{1..22}/{POP}.csv files. The CSVs now carry multiple
per-gene summary statistics (geom_mean_tmrca, arith_mean_tmrca, min_tmrca).
Primary ranking uses geom_mean_tmrca (geometric mean of per-pair TMRCA).

Outputs:
    results/genome_wide_ranks.csv   — full gene x population matrix with
                                      raw TMRCAs and within-pop percentile ranks
    results/genome_wide_stats.csv   — per-gene summary sorted by min_rank
"""

import os

import numpy as np
import pandas as pd

BASE = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide"
RESULTS = os.path.join(BASE, "results")

ALL_POPULATIONS = [
    "ACB", "ASW", "BEB", "CDX", "CEU", "CHB", "CHS", "CLM",
    "ESN", "FIN", "GBR", "GIH", "GWD", "IBS", "ITU", "JPT",
    "KHV", "LWK", "MSL", "MXL", "PEL", "PJL", "PUR", "STU",
    "TSI", "YRI",
]

# Which per-gene statistic to use as the primary TMRCA value for ranking.
# Change this string to "arith_mean_tmrca" or "min_tmrca" to re-aggregate
# with a different statistic; alternatively use reaggregate_from_npz.py
# for histogram-based quantiles.
PRIMARY_STAT = "geom_mean_tmrca"


def main():
    all_genes = []

    for chr_num in range(1, 23):
        chr_dir = os.path.join(RESULTS, f"chr{chr_num}")
        if not os.path.isdir(chr_dir):
            print(f"Skipping chr{chr_num} (no results)")
            continue

        pop_data = {}
        for pop in ALL_POPULATIONS:
            csv_path = os.path.join(chr_dir, f"{pop}.csv")
            if os.path.exists(csv_path):
                pop_data[pop] = pd.read_csv(csv_path)

        if not pop_data:
            print(f"Skipping chr{chr_num} (no population CSVs)")
            continue

        ref = next(iter(pop_data.values()))
        for _, row in ref.iterrows():
            gene_entry = {
                "gene_id": row["gene_id"],
                "gene_name": row["gene_name"],
                "chr": chr_num,
                "start": row["start"],
                "end": row["end"],
            }
            for pop in ALL_POPULATIONS:
                if pop in pop_data:
                    match = pop_data[pop][pop_data[pop]["gene_id"] == row["gene_id"]]
                    if not match.empty and PRIMARY_STAT in match.columns:
                        val = match.iloc[0][PRIMARY_STAT]
                        gene_entry[f"{pop}_tmrca"] = val
                    else:
                        gene_entry[f"{pop}_tmrca"] = np.nan
                else:
                    gene_entry[f"{pop}_tmrca"] = np.nan
            all_genes.append(gene_entry)

    if not all_genes:
        print("No results found!")
        return

    df = pd.DataFrame(all_genes)
    print(f"Total genes: {len(df)}")
    print(f"Primary statistic: {PRIMARY_STAT}")

    for pop in ALL_POPULATIONS:
        col = f"{pop}_tmrca"
        rank_col = f"{pop}_rank"
        if col in df.columns:
            df[rank_col] = df[col].rank(pct=True, na_option="keep")

    rank_cols = [f"{p}_rank" for p in ALL_POPULATIONS if f"{p}_rank" in df.columns]
    df["min_rank"] = df[rank_cols].min(axis=1)
    valid_mask = df[rank_cols].notna().any(axis=1)
    df.loc[valid_mask, "min_pop"] = (
        df.loc[valid_mask, rank_cols].idxmin(axis=1).str.replace("_rank", "", regex=False)
    )
    df["max_rank"] = df[rank_cols].max(axis=1)
    df["rank_range"] = df["max_rank"] - df["min_rank"]

    out_ranks = os.path.join(RESULTS, "genome_wide_ranks.csv")
    df.to_csv(out_ranks, index=False)
    print(f"Wrote {out_ranks}")

    stats_cols = ["gene_id", "gene_name", "chr", "start", "end",
                  "min_rank", "min_pop", "max_rank", "rank_range"]
    df_stats = df[stats_cols].sort_values("min_rank")
    out_stats = os.path.join(RESULTS, "genome_wide_stats.csv")
    df_stats.to_csv(out_stats, index=False)
    print(f"Wrote {out_stats}")

    print("\nTop 20 candidates (lowest min_rank):")
    print(df_stats.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
