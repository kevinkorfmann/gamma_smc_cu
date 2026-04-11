#!/usr/bin/env python
"""Re-aggregate genome-wide ranks from the raw NPZ accumulators.

Reads results/chr{N}/{POP}.npz files (saved by infer_chromosome.py) and
produces a fresh genome_wide_ranks.csv / genome_wide_stats.csv using
whatever per-gene summary statistic you pick. No GPU inference needed.

Supported statistics:
    geom_mean   : exp(log_sum / count)          — geometric mean (default)
    arith_mean  : lin_sum / count               — arithmetic mean
    min         : min_lin                       — youngest per-pair linear TMRCA
    min_log     : exp(min_log)                  — youngest per-pair log TMRCA (== min)
    p1, p5, p10, p25, p50, p75, p90   : histogram-derived percentiles of
                                         per-pair log-TMRCA (exp-transformed)
    frac_below_<N>   : fraction of pairs with log-TMRCA < log(N)
                       (e.g. frac_below_1000 for <1000 gen)

Usage:
    python reaggregate_from_npz.py --stat geom_mean
    python reaggregate_from_npz.py --stat p5
    python reaggregate_from_npz.py --stat frac_below_500
"""

from __future__ import annotations

import argparse
import os
import re

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


def percentile_from_histogram(histogram, bin_edges, q):
    """Compute an approximate percentile from a log-space histogram.

    histogram: (n_genes, n_bins) counts of per-pair log-TMRCA
    bin_edges: (n_bins+1,) natural-log edges
    q: percentile in [0, 1]

    Returns: (n_genes,) linear-TMRCA values at that percentile, NaN for
    genes with zero counts.
    """
    n_genes, n_bins = histogram.shape
    totals = histogram.sum(axis=1)
    out = np.full(n_genes, np.nan, dtype=np.float64)

    # Bin midpoints in log space for linear interpolation of the CDF
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for gi in range(n_genes):
        if totals[gi] == 0:
            continue
        counts = histogram[gi]
        cum = np.cumsum(counts)
        target = q * totals[gi]
        # First bin where cum >= target
        bi = int(np.searchsorted(cum, target, side="left"))
        if bi >= n_bins:
            bi = n_bins - 1
        out[gi] = np.exp(bin_centers[bi])
    return out


def fraction_below(histogram, bin_edges, log_threshold):
    """Fraction of per-pair values below log_threshold, per gene."""
    n_genes, n_bins = histogram.shape
    totals = histogram.sum(axis=1)
    out = np.full(n_genes, np.nan, dtype=np.float64)

    # Count bins entirely below threshold + fractional contribution of
    # the bin that contains the threshold.
    bi_cut = int(np.searchsorted(bin_edges, log_threshold, side="right") - 1)
    for gi in range(n_genes):
        if totals[gi] == 0:
            continue
        counts = histogram[gi]
        if bi_cut < 0:
            out[gi] = 0.0
            continue
        if bi_cut >= n_bins:
            out[gi] = 1.0
            continue
        below = counts[:bi_cut].sum()
        bin_lo = bin_edges[bi_cut]
        bin_hi = bin_edges[bi_cut + 1]
        frac_in_bin = (log_threshold - bin_lo) / (bin_hi - bin_lo)
        frac_in_bin = float(np.clip(frac_in_bin, 0.0, 1.0))
        below += counts[bi_cut] * frac_in_bin
        out[gi] = below / totals[gi]
    return out


def compute_stat(npz, stat):
    """Compute the requested statistic for all genes in an NPZ file."""
    count = npz["count"]
    lin_sum = npz["lin_sum"]
    log_sum = npz["log_sum"]
    min_lin = npz["min_lin"]
    min_log = npz["min_log"]
    histogram = npz["histogram"]
    bin_edges = npz["bin_edges"]

    with np.errstate(divide="ignore", invalid="ignore"):
        if stat == "geom_mean":
            return np.where(count > 0, np.exp(log_sum / count), np.nan)
        if stat == "arith_mean":
            return np.where(count > 0, lin_sum / count, np.nan)
        if stat == "min":
            out = np.where(np.isfinite(min_lin), min_lin, np.nan)
            return out
        if stat == "min_log":
            out = np.where(np.isfinite(min_log), np.exp(min_log), np.nan)
            return out

        # pN percentile from histogram
        m = re.fullmatch(r"p(\d+)", stat)
        if m:
            pct = float(m.group(1)) / 100.0
            return percentile_from_histogram(histogram, bin_edges, pct)

        # frac_below_<N>
        m = re.fullmatch(r"frac_below_(\d+)", stat)
        if m:
            threshold = float(m.group(1))
            return fraction_below(histogram, bin_edges, np.log(threshold))

    raise ValueError(f"Unknown stat: {stat}")


def main():
    parser = argparse.ArgumentParser(description="Re-aggregate from NPZ files")
    parser.add_argument(
        "--stat",
        default="geom_mean",
        help="Summary statistic: geom_mean, arith_mean, min, p1, p5, p10, "
             "p25, p50, p75, p90, frac_below_<N>",
    )
    parser.add_argument(
        "--out-prefix",
        default=None,
        help="Output prefix (default: uses stat name)",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="For frac_below_* stats, flip the rank direction so higher "
             "fractions get lower ranks (better sweep candidates).",
    )
    args = parser.parse_args()

    out_prefix = args.out_prefix or args.stat
    all_genes = []

    for chr_num in range(1, 23):
        chr_dir = os.path.join(RESULTS, f"chr{chr_num}")
        if not os.path.isdir(chr_dir):
            continue

        # Load one NPZ per population for this chromosome
        pop_values = {}
        ref_meta = None
        for pop in ALL_POPULATIONS:
            npz_path = os.path.join(chr_dir, f"{pop}.npz")
            if not os.path.exists(npz_path):
                continue
            with np.load(npz_path, allow_pickle=True) as npz:
                values = compute_stat(npz, args.stat)
                if ref_meta is None:
                    ref_meta = {
                        "gene_id": npz["gene_id"].copy(),
                        "gene_name": npz["gene_name"].copy(),
                        "start": npz["start"].copy(),
                        "end": npz["end"].copy(),
                    }
                pop_values[pop] = values

        if ref_meta is None:
            continue

        n_genes = len(ref_meta["gene_id"])
        for gi in range(n_genes):
            entry = {
                "gene_id": str(ref_meta["gene_id"][gi]),
                "gene_name": str(ref_meta["gene_name"][gi]),
                "chr": chr_num,
                "start": int(ref_meta["start"][gi]),
                "end": int(ref_meta["end"][gi]),
            }
            for pop in ALL_POPULATIONS:
                entry[f"{pop}_tmrca"] = pop_values.get(pop, [np.nan] * n_genes)[gi] \
                    if pop in pop_values else np.nan
            all_genes.append(entry)

    if not all_genes:
        print("No NPZ files found!")
        return

    df = pd.DataFrame(all_genes)
    print(f"Total genes: {len(df)}")
    print(f"Statistic: {args.stat}")

    # Rank direction: for frac_below_*, higher is better → rank ascending False
    rank_ascending = not args.stat.startswith("frac_below_")

    for pop in ALL_POPULATIONS:
        col = f"{pop}_tmrca"
        rank_col = f"{pop}_rank"
        if col in df.columns:
            df[rank_col] = df[col].rank(pct=True, ascending=rank_ascending, na_option="keep")

    rank_cols = [f"{p}_rank" for p in ALL_POPULATIONS if f"{p}_rank" in df.columns]
    df["min_rank"] = df[rank_cols].min(axis=1)
    valid = df[rank_cols].notna().any(axis=1)
    df.loc[valid, "min_pop"] = (
        df.loc[valid, rank_cols].idxmin(axis=1).str.replace("_rank", "", regex=False)
    )
    df["max_rank"] = df[rank_cols].max(axis=1)
    df["rank_range"] = df["max_rank"] - df["min_rank"]

    out_ranks = os.path.join(RESULTS, f"genome_wide_ranks_{out_prefix}.csv")
    df.to_csv(out_ranks, index=False)
    print(f"Wrote {out_ranks}")

    stats_cols = ["gene_id", "gene_name", "chr", "start", "end",
                  "min_rank", "min_pop", "max_rank", "rank_range"]
    df_stats = df[stats_cols].sort_values("min_rank")
    out_stats = os.path.join(RESULTS, f"genome_wide_stats_{out_prefix}.csv")
    df_stats.to_csv(out_stats, index=False)
    print(f"Wrote {out_stats}")

    print(f"\nTop 20 candidates by {args.stat} (lowest min_rank):")
    print(df_stats.head(20).to_string(index=False))

    # Quick known-sweep check
    controls = ["SLC24A5","HERC2","LCT","MCM6","EDAR","TRPV6","KITLG",
                "TYRP1","ADH1B","OCA2","APOL1"]
    hits = df[df["gene_name"].isin(controls)].sort_values("min_rank")
    print(f"\nKnown sweep recovery ({args.stat}):")
    print(hits[["gene_name","chr","min_rank","min_pop"]].to_string(index=False))
    n_below_10 = (hits["min_rank"] < 0.10).sum()
    n_below_05 = (hits["min_rank"] < 0.05).sum()
    print(f"{n_below_10}/{len(hits)} below 10%, {n_below_05}/{len(hits)} below 5%")


if __name__ == "__main__":
    main()
