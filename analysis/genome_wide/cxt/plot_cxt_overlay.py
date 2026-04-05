#!/usr/bin/env python
"""Overlay cxt and tmrca.cu TMRCA profiles at 6 known sweep regions (S3 Fig)."""

import numpy as np
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(os.path.abspath(__file__))
CXT_RESULTS = os.path.join(BASE, "results")
OUTFILE = os.path.join(BASE, "fig_cxt_tmrcacu_overlay.png")

REGIONS = [
    # (region_name, pop, title, chrom, start, end, highlight_genes)
    ("SH2B3_ALDH2", "TSI", "SH2B3-ALDH2 (EUR/TSI)", "chr12", 110e6, 113e6, ["SH2B3", "ALDH2"]),
    ("CYP3A", "FIN", "CYP3A (EUR/FIN)", "chr7", 99e6, 100.5e6, ["CYP3A5", "CYP3A4"]),
    ("FADS1", "ITU", "FADS1 (SAS/ITU)", "chr11", 60.5e6, 62e6, ["FADS1", "FADS2"]),
    ("CLEC6A", "CDX", "CLEC6A (EAS/CDX)", "chr12", 7.5e6, 10e6, ["CLEC6A"]),
    ("ABCC11", "CHB", "ABCC11 (EAS/CHB)", "chr16", 47.5e6, 49e6, ["ABCC11"]),
    ("TRPV6_KEL", "CHB", "TRPV6-KEL (EAS/CHB)", "chr7", 142e6, 144e6, ["TRPV6", "KEL"]),
]


def load_cxt(region, pop):
    data = np.load(os.path.join(CXT_RESULTS, f"cxt_{region}_{pop}.npz"), allow_pickle=True)
    start = int(data["start"])
    end = int(data["end"])

    if "log_tmrca_raw" in data:
        raw = data["log_tmrca_raw"]
        blocks = data["blocks"]
        index_map = data["index_map"]
        raw_t = np.transpose(raw, (1, 0, 2))  # (n_items, n_reps, n_windows)
        profiles, stds, positions = [], [], []
        for b in range(len(blocks)):
            bd = raw_t[index_map[:, 0] == b]
            profiles.append(bd.mean(axis=(0, 1)))
            stds.append(bd.mean(axis=1).std(axis=0))
            b_start, b_end = blocks[b]
            positions.append(np.linspace(b_start, b_end, bd.shape[-1]))
        log_mean = np.concatenate(profiles)
        log_std = np.concatenate(stds)
        pos = np.concatenate(positions)
        # Trim to actual region (last block may extend beyond)
        mask = pos <= end
        pos, log_mean, log_std = pos[mask], log_mean[mask], log_std[mask]
    else:
        log_mean = data["pop_mean_log_tmrca"]
        log_std = data["pop_std_log_tmrca"]
        end = start + 1_000_000
        pos = np.linspace(start, end, len(log_mean))

    return pos, np.exp(log_mean), np.exp(log_mean - log_std), np.exp(log_mean + log_std)


import pandas as pd

GW_RESULTS = "/Users/kevinkorfmann/Projects/tmrca.cu/docs_local/genome_wide_results"


def load_gene_ranks(chr_num, pop, start, end):
    """Load tmrca.cu gene-level rank percentiles for a region."""
    gene_pos_file = os.path.join(GW_RESULTS, "gene_positions", f"chr{chr_num}_genes.tsv")
    genes = pd.read_csv(gene_pos_file, sep="\t")
    genes["midpoint"] = (genes["start"] + genes["end"]) / 2
    mask = (genes["midpoint"] >= start) & (genes["midpoint"] <= end)
    genes_region = genes[mask]

    ranks_file = os.path.join(GW_RESULTS, f"chr{chr_num}", "gene_ranks.csv")
    ranks_df = pd.read_csv(ranks_file, index_col=0)
    if pop not in ranks_df.index:
        return None, None
    pop_ranks = ranks_df.loc[pop]

    positions, rank_values, names = [], [], []
    for _, gene in genes_region.iterrows():
        gname = gene["gene_name"]
        if gname in pop_ranks.index:
            positions.append(gene["midpoint"])
            rank_values.append(pop_ranks[gname] * 100)
            names.append(gname)
    return np.array(positions), np.array(rank_values), names


fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.ravel()

for idx, (region, pop, title, chrom, start, end, highlight) in enumerate(REGIONS):
    ax = axes[idx]
    chr_num = int(chrom.replace("chr", ""))

    # cxt TMRCA (left y-axis, log scale)
    cxt_pos, cxt_tmrca, cxt_lo, cxt_hi = load_cxt(region, pop)
    ax.plot(cxt_pos / 1e6, cxt_tmrca, color="steelblue", linewidth=0.8,
            label="cxt", zorder=2)
    ax.fill_between(cxt_pos / 1e6, cxt_lo, cxt_hi,
                     color="steelblue", alpha=0.12, zorder=1)
    ax.set_yscale("log")
    ax.set_ylabel("cxt TMRCA (gen)", fontsize=8, color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue", labelsize=7)

    # tmrca.cu rank percentile (right y-axis)
    result = load_gene_ranks(chr_num, pop, start, end)
    if result[0] is not None and len(result[0]) > 0:
        gene_pos, gene_ranks, gene_names = result
        sort_idx = np.argsort(gene_pos)
        gene_pos = gene_pos[sort_idx]
        gene_ranks = gene_ranks[sort_idx]
        gene_names = [gene_names[i] for i in sort_idx]
        ax2 = ax.twinx()
        ax2.scatter(gene_pos / 1e6, gene_ranks, color="orangered", s=12,
                    alpha=0.5, zorder=3, edgecolors="none")
        # Highlight target genes
        for i, gn in enumerate(gene_names):
            if gn in highlight:
                ax2.scatter(gene_pos[i] / 1e6, gene_ranks[i], color="orangered",
                            s=50, zorder=5, edgecolors="black", linewidths=0.8)
                ax2.annotate(gn, (gene_pos[i] / 1e6, gene_ranks[i]),
                             textcoords="offset points", xytext=(0, -12),
                             fontsize=7, fontstyle="italic", ha="center",
                             fontweight="bold")
        ax2.set_ylabel("tmrca.cu rank (%)", fontsize=8, color="orangered")
        ax2.tick_params(axis="y", labelcolor="orangered", labelsize=7)
        ax2.set_ylim(0, 100)

    ax.set_title(title, fontsize=10, fontweight="normal", loc="left")
    ax.set_xlabel(f"{chrom} (Mb)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)

    if idx == 0:
        from matplotlib.lines import Line2D
        ax.legend(
            handles=[
                Line2D([0], [0], color="steelblue", lw=1.5, label="cxt (TMRCA)"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="orangered",
                       markersize=5, label="tmrca.cu (rank %)"),
            ],
            fontsize=7, loc="upper right", framealpha=0.8,
        )

plt.tight_layout()
plt.savefig(OUTFILE, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {OUTFILE}")
plt.close()
