#!/usr/bin/env python
"""Overlay cxt + tmrca.cu at mucosal immunity and novel finding loci."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

BASE = os.path.dirname(os.path.abspath(__file__))
CXT_RESULTS = os.path.join(BASE, "results")
GW_RESULTS = "/Users/kevinkorfmann/Projects/tmrca.cu/docs_local/genome_wide_results"

MUCOSAL = [
    ("CLEC6A", "CDX", "CLEC6A (EAS/CDX)", "chr12", 7.5e6, 10e6, ["CLEC6A"]),
    ("TRAF6", "CDX", "TRAF6 (EAS/CDX)", "chr11", 35.5e6, 37.5e6, ["TRAF6"]),
    ("JCHAIN", "CDX", "JCHAIN (EAS/CDX)", "chr4", 69.5e6, 71.5e6, ["JCHAIN"]),
    ("TNFRSF13C", "CDX", "TNFRSF13C (EAS/CDX)", "chr22", 41e6, 43e6, ["TNFRSF13C"]),
]

NOVEL = [
    ("GRK2", "BEB", "GRK2 (SAS/BEB)", "chr11", 66e6, 68e6, ["GRK2"]),
    ("BPIFA2", "ITU", "BPIFA2 (SAS/ITU)", "chr20", 32e6, 34e6, ["BPIFA2"]),
    ("CCDC92_ZNF664", "CDX", "CCDC92/ZNF664 (EAS/CDX)", "chr12", 123e6, 125e6, ["CCDC92", "ZNF664"]),
    ("SLC6A15", "CDX", "SLC6A15 (EAS/CDX)", "chr12", 84e6, 86e6, ["SLC6A15"]),
]


def load_cxt(region, pop):
    data = np.load(os.path.join(CXT_RESULTS, f"cxt_{region}_{pop}.npz"), allow_pickle=True)
    start, end = int(data["start"]), int(data["end"])
    raw = data["log_tmrca_raw"]
    blocks = data["blocks"]
    index_map = data["index_map"]
    raw_t = np.transpose(raw, (1, 0, 2))
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
    mask = pos <= end
    return pos[mask], np.exp(log_mean[mask]), np.exp(log_mean[mask] - log_std[mask]), np.exp(log_mean[mask] + log_std[mask])


def load_gene_ranks(chr_num, pop, start, end):
    gene_pos_file = os.path.join(GW_RESULTS, "gene_positions", f"chr{chr_num}_genes.tsv")
    genes = pd.read_csv(gene_pos_file, sep="\t")
    genes["midpoint"] = (genes["start"] + genes["end"]) / 2
    mask = (genes["midpoint"] >= start) & (genes["midpoint"] <= end)
    genes_region = genes[mask]
    ranks_file = os.path.join(GW_RESULTS, f"chr{chr_num}", "gene_ranks.csv")
    ranks_df = pd.read_csv(ranks_file, index_col=0)
    if pop not in ranks_df.index:
        return None, None, None
    pop_ranks = ranks_df.loc[pop]
    positions, rank_values, names = [], [], []
    for _, gene in genes_region.iterrows():
        gname = gene["gene_name"]
        if gname in pop_ranks.index:
            positions.append(gene["midpoint"])
            rank_values.append(pop_ranks[gname] * 100)
            names.append(gname)
    return np.array(positions), np.array(rank_values), names


def plot_panel(ax, region, pop, title, chrom, start, end, highlight):
    chr_num = int(chrom.replace("chr", ""))

    # cxt
    cxt_pos, cxt_tmrca, cxt_lo, cxt_hi = load_cxt(region, pop)
    ax.plot(cxt_pos / 1e6, cxt_tmrca, color="steelblue", linewidth=0.8, zorder=2)
    ax.fill_between(cxt_pos / 1e6, cxt_lo, cxt_hi, color="steelblue", alpha=0.12, zorder=1)
    ax.set_yscale("log")
    ax.set_ylabel("cxt TMRCA (gen)", fontsize=8, color="steelblue")
    ax.tick_params(axis="y", labelcolor="steelblue", labelsize=7)

    # tmrca.cu ranks
    gene_pos, gene_ranks, gene_names = load_gene_ranks(chr_num, pop, start, end)
    if gene_pos is not None and len(gene_pos) > 0:
        sort_idx = np.argsort(gene_pos)
        gene_pos = gene_pos[sort_idx]
        gene_ranks = gene_ranks[sort_idx]
        gene_names = [gene_names[i] for i in sort_idx]
        ax2 = ax.twinx()
        ax2.scatter(gene_pos / 1e6, gene_ranks, color="orangered", s=12,
                    alpha=0.5, zorder=3, edgecolors="none")
        for i, gn in enumerate(gene_names):
            if gn in highlight:
                ax2.scatter(gene_pos[i] / 1e6, gene_ranks[i], color="orangered",
                            s=50, zorder=5, edgecolors="black", linewidths=0.8)
                ax2.annotate(gn, (gene_pos[i] / 1e6, gene_ranks[i]),
                             textcoords="offset points", xytext=(0, -12),
                             fontsize=7, fontstyle="italic", fontweight="bold", ha="center")
        ax2.set_ylabel("tmrca.cu rank (%)", fontsize=8, color="orangered")
        ax2.tick_params(axis="y", labelcolor="orangered", labelsize=7)
        ax2.set_ylim(0, 100)

    ax.set_title(title, fontsize=10, fontweight="normal", loc="left")
    ax.set_xlabel(f"{chrom} (Mb)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)


legend_elements = [
    Line2D([0], [0], color="steelblue", lw=1.5, label="cxt (TMRCA)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="orangered",
           markersize=5, label="tmrca.cu (rank %)"),
]

# --- Mucosal immunity figure ---
fig1, axes1 = plt.subplots(2, 2, figsize=(12, 7))
for idx, (region, pop, title, chrom, start, end, hl) in enumerate(MUCOSAL):
    plot_panel(axes1.ravel()[idx], region, pop, title, chrom, start, end, hl)
    if idx == 0:
        axes1.ravel()[idx].legend(handles=legend_elements, fontsize=7, loc="upper right", framealpha=0.8)
fig1.suptitle("Mucosal immunity loci", fontsize=12, fontweight="normal", y=1.01)
fig1.tight_layout()
out1 = os.path.join(BASE, "fig_cxt_mucosal_immunity.png")
fig1.savefig(out1, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out1}")
plt.close(fig1)

# --- Novel findings figure ---
fig2, axes2 = plt.subplots(2, 2, figsize=(12, 7))
for idx, (region, pop, title, chrom, start, end, hl) in enumerate(NOVEL):
    plot_panel(axes2.ravel()[idx], region, pop, title, chrom, start, end, hl)
    if idx == 0:
        axes2.ravel()[idx].legend(handles=legend_elements, fontsize=7, loc="upper right", framealpha=0.8)
fig2.suptitle("Novel selection loci", fontsize=12, fontweight="normal", y=1.01)
fig2.tight_layout()
out2 = os.path.join(BASE, "fig_cxt_novel_findings.png")
fig2.savefig(out2, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out2}")
plt.close(fig2)
