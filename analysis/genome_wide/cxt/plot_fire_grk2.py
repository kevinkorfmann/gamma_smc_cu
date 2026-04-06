#!/usr/bin/env python
"""
Pairwise coalescence 'fire plot' for GRK2 (chr11:66-68 Mb).
Replicates the style of Schweiger & Durbin (2023) Figure 5B:
  X = genomic position, Y = years ago (log scale), Color = # of pairs.

Reads pre-computed 2D histograms from run_fire_grk2.py.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

# GRK2 gene annotation (GRCh38)
GRK2_START_MB = 67.242
GRK2_END_MB   = 67.264


def make_fire_panel(ax, npz_path, title):
    d = np.load(npz_path, allow_pickle=True)
    H = d["histogram"]           # (n_pos_bins, n_time_bins)
    pos_edges = d["pos_edges"]   # (n_pos_bins + 1,) Mb
    time_edges = d["time_edges"] # (n_time_bins + 1,) log10(years)
    n_pairs = int(d["n_pairs"])

    H_plot = H.T.astype(float)   # (n_time_bins, n_pos_bins)

    # Normalize each column to show fraction, then take log-ratio vs
    # genome-wide average to highlight deviations (sweep = excess recent)
    col_sums = H_plot.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    H_frac = H_plot / col_sums

    # Average column profile (expected under neutrality)
    avg_profile = H_frac.mean(axis=1, keepdims=True)
    avg_profile[avg_profile == 0] = 1e-10

    # Enrichment: how many more pairs at this time than expected
    enrichment = H_frac / avg_profile

    # Light smooth to reduce site-level noise but keep fire texture
    enrichment = gaussian_filter(enrichment, sigma=(0.8, 1.0))

    im = ax.pcolormesh(
        pos_edges, time_edges, enrichment,
        cmap="inferno",
        norm=LogNorm(vmin=0.3, vmax=3.0),
        rasterized=True,
    )

    # Gene annotation
    rect = Rectangle(
        (GRK2_START_MB, time_edges[0]),
        GRK2_END_MB - GRK2_START_MB,
        0.12,
        linewidth=1.2, edgecolor="#2ecc71", facecolor="#2ecc71", alpha=0.9,
        zorder=5,
    )
    ax.add_patch(rect)
    ax.text(
        (GRK2_START_MB + GRK2_END_MB) / 2, time_edges[0] + 0.2,
        "GRK2", ha="center", va="bottom", fontsize=9,
        color="#2ecc71", fontweight="bold", zorder=5,
    )

    ax.set_xlabel("Genomic position (Mb)")
    ax.set_ylabel("Years ago")
    ax.set_title(f"{title}  ({n_pairs:,} pairs)", fontweight="normal", loc="left")

    yticks = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"$10^{{{y:.1f}}}$" if y != int(y)
                        else f"$10^{{{int(y)}}}$" for y in yticks])

    return im


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="analysis/genome_wide/cxt/results/fire_quick",
                        help="Directory with fire_GRK2_*.npz files")
    parser.add_argument("--out", default="analysis/genome_wide/cxt/fire_grk2.png")
    args = parser.parse_args()

    sas_path = f"{args.dir}/fire_GRK2_SAS.npz"
    yri_path = f"{args.dir}/fire_GRK2_YRI.npz"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True,
                             constrained_layout=True)

    im1 = make_fire_panel(axes[0], sas_path, "a  SAS (South Asian) — sweep")
    im2 = make_fire_panel(axes[1], yri_path, "b  YRI (Yoruba) — control")
    axes[1].set_ylabel("")

    cbar = fig.colorbar(im1, ax=axes, shrink=0.85, pad=0.02, aspect=30)
    cbar.set_label("Enrichment vs. region average")

    fig.suptitle(
        "Pairwise coalescence times at GRK2 (chr11:66–68 Mb)",
        fontweight="normal", fontsize=13,
    )

    out_png = args.out
    out_pdf = out_png.replace(".png", ".pdf")
    fig.savefig(out_png, dpi=250, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_png}, {out_pdf}")


if __name__ == "__main__":
    main()
