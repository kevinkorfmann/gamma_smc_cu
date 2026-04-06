#!/usr/bin/env python
"""
Figure 7: cxt regional TMRCA confirmation of selection signals.

Row 1 (a-d): cxt profiles at 4 known sweep loci (focal pop vs YRI control)
Row 2 (e-h): cxt profiles at 4 novel loci
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os

BASE = os.path.dirname(os.path.abspath(__file__))
CXT_RESULTS = os.path.join(BASE, "results")
OUTFILE = os.path.join(BASE, "fig7_multimethod.png")

GEN_TIME = 29


def load_cxt(region, pop):
    fname = os.path.join(CXT_RESULTS, f"cxt_{region}_{pop}.npz")
    if not os.path.exists(fname):
        return None, None, None, None
    data = np.load(fname, allow_pickle=True)
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
    return (pos[mask], np.exp(log_mean[mask]),
            np.exp(log_mean[mask] - log_std[mask]),
            np.exp(log_mean[mask] + log_std[mask]))


# Row 1: known loci
CXT_KNOWN = [
    ("LCT", "CEU", "LCT (CEU vs YRI)", "chr2", "#2166ac",
     [("LCT", 135812517)]),
    ("SLC24A5", "GBR", "SLC24A5 (GBR vs YRI)", "chr15", "#b2182b",
     [("SLC24A5", 48131831)]),
    ("EDAR", "CHB", "EDAR (CHB vs YRI)", "chr2", "#1b7837",
     [("EDAR", 108941921)]),
    ("FADS1", "ITU", "FADS1 / FADS2 (ITU vs YRI)", "chr11", "#762a83",
     [("FADS1", 61575500), ("FADS2", 61605500)]),
]

# Row 2: novel loci
CXT_NOVEL = [
    ("GRK2", "BEB", "GRK2 (BEB vs YRI)", "chr11", "#e08214",
     [("GRK2", 67276514)]),
    ("CLEC6A", "CDX", "CLEC6A (CDX vs YRI)", "chr12", "#d62728",
     [("CLEC6A", 8467146)]),
    ("BPIFA2", "ITU", "BPIFA2 (ITU vs YRI)", "chr20", "#9467bd",
     [("BPIFA2", 33171590)]),
    ("CCDC92_ZNF664", "CDX", "CCDC92 (CDX vs YRI)", "chr12", "#ff7f0e",
     [("CCDC92", 123945745), ("ZNF664", 123993642)]),
]


def plot_cxt_panel(ax, region, focal_pop, title, chrom, color, gene_positions, label):
    pos, tmrca, lo, hi = load_cxt(region, focal_pop)
    if pos is None:
        ax.text(0.5, 0.5, f"{region}\n(pending)", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
        ax.set_title(title, fontsize=9, loc="left")
        ax.text(-0.12, 1.08, f"$\\bf{{{label}}}$", transform=ax.transAxes,
                fontsize=12, va="bottom")
        return

    ax.plot(pos / 1e6, tmrca, color=color, linewidth=0.9, label=focal_pop)
    ax.fill_between(pos / 1e6, lo, hi, color=color, alpha=0.15)

    yri_pos, yri_tmrca, yri_lo, yri_hi = load_cxt(region, "YRI")
    if yri_pos is not None:
        ax.plot(yri_pos / 1e6, yri_tmrca, color="gray", linewidth=0.6,
                alpha=0.6, label="YRI")
        ax.fill_between(yri_pos / 1e6, yri_lo, yri_hi, color="gray", alpha=0.06)

    for gene_name, gene_mid in gene_positions:
        ax.axvline(gene_mid / 1e6, color="black", linewidth=0.8,
                   linestyle="--", alpha=0.7)
        ax.text(gene_mid / 1e6, 0.97, f" {gene_name}",
                transform=ax.get_xaxis_transform(),
                fontsize=6.5, fontstyle="italic", fontweight="bold",
                va="top", ha="left")

    ax.set_yscale("log")
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel(f"{chrom} (Mb)", fontsize=8)
    ax.set_ylabel("TMRCA (gen)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.text(-0.12, 1.08, f"$\\bf{{{label}}}$", transform=ax.transAxes,
            fontsize=12, va="bottom")


fig, axes = plt.subplots(2, 4, figsize=(16, 7))
labels = "abcdefgh"

for i, (region, fpop, title, chrom, color, gpos) in enumerate(CXT_KNOWN):
    plot_cxt_panel(axes[0, i], region, fpop, title, chrom, color, gpos, labels[i])

for i, (region, fpop, title, chrom, color, gpos) in enumerate(CXT_NOVEL):
    plot_cxt_panel(axes[1, i], region, fpop, title, chrom, color, gpos, labels[4 + i])

# Row labels
fig.text(0.01, 0.73, "Known\nsweeps", fontsize=10, fontweight="bold",
         va="center", rotation=90)
fig.text(0.01, 0.27, "Novel\nsweeps", fontsize=10, fontweight="bold",
         va="center", rotation=90)

legend_elements = [
    Line2D([0], [0], color="black", lw=1.5, label="Focal population"),
    Line2D([0], [0], color="gray", lw=1, alpha=0.6, label="YRI (African control)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9,
           frameon=False, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0.03, 0.03, 1, 1])
plt.savefig(OUTFILE, dpi=200, bbox_inches="tight", facecolor="white")
plt.savefig(OUTFILE.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
print(f"Saved: {OUTFILE}")
plt.close()
