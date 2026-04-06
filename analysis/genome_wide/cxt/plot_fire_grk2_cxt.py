#!/usr/bin/env python
"""GRK2 fire plot from cxt data (1225 pairs)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

GENERATION_TIME = 29
GRK2_START_MB = 67.242
GRK2_END_MB   = 67.264


def load_fire_data(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    raw = d["log_tmrca_raw"]
    idx = d["index_map"]
    blocks = d["blocks"]
    n_windows = raw.shape[2]

    mean_log = raw.mean(axis=0)
    tmrca_years = np.exp(mean_log) * GENERATION_TIME

    all_pos, all_tmrca = [], []
    for b in range(len(blocks)):
        pos_mb = np.linspace(blocks[b][0] / 1e6, blocks[b][1] / 1e6,
                             n_windows, endpoint=False)
        all_pos.append(pos_mb)
        block_items = np.where(idx[:, 0] == b)[0]
        pair_order = np.argsort(idx[block_items, 1])
        all_tmrca.append(tmrca_years[block_items[pair_order]])

    return np.concatenate(all_pos), np.concatenate(all_tmrca, axis=1)


def make_fire_panel(ax, positions_mb, tmrca_years, title,
                    n_pos_bins=250, n_time_bins=180,
                    time_range=(3.5, 6.2)):
    n_pairs, n_windows = tmrca_years.shape
    pos_flat = np.tile(positions_mb, n_pairs)
    time_flat = np.log10(tmrca_years.ravel())

    pos_edges = np.linspace(positions_mb.min(), positions_mb.max(), n_pos_bins + 1)
    time_edges = np.linspace(time_range[0], time_range[1], n_time_bins + 1)

    H, xedges, yedges = np.histogram2d(
        pos_flat, time_flat, bins=[pos_edges, time_edges])
    H = H.T
    H_masked = np.ma.masked_where(H == 0, H)

    im = ax.pcolormesh(xedges, yedges, H_masked, cmap="inferno",
                       norm=LogNorm(vmin=1, vmax=H.max()), rasterized=True)

    rect = Rectangle((GRK2_START_MB, time_range[0]),
                      GRK2_END_MB - GRK2_START_MB, 0.12,
                      linewidth=1.2, edgecolor="#2ecc71", facecolor="#2ecc71",
                      alpha=0.85, zorder=5)
    ax.add_patch(rect)
    ax.text((GRK2_START_MB + GRK2_END_MB) / 2, time_range[0] + 0.18,
            "GRK2", ha="center", va="bottom", fontsize=9,
            color="#2ecc71", fontweight="bold", zorder=5)

    ax.set_xlabel("Genomic position (Mb)")
    ax.set_ylabel("Years ago")
    ax.set_title(title, fontweight="normal", loc="left")

    yticks = [4.0, 4.5, 5.0, 5.5, 6.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"$10^{{{int(y)}}}$" if y == int(y)
                        else f"$10^{{{y}}}$" for y in yticks])
    return im


# Load data
pos_beb, tmrca_beb = load_fire_data(
    "analysis/genome_wide/cxt/results/fire_cxt/cxt_GRK2_fire_BEB.npz")
# Use original 100-pair YRI until 1225-pair finishes
pos_yri, tmrca_yri = load_fire_data(
    "analysis/genome_wide/cxt/results/cxt_GRK2_YRI.npz")

n_beb = tmrca_beb.shape[0]
n_yri = tmrca_yri.shape[0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True,
                         constrained_layout=True)

im1 = make_fire_panel(axes[0], pos_beb, tmrca_beb,
                      f"a  BEB (South Asian) — sweep  ({n_beb:,} pairs)")
im2 = make_fire_panel(axes[1], pos_yri, tmrca_yri,
                      f"b  YRI (Yoruba) — control  ({n_yri:,} pairs)")
axes[1].set_ylabel("")

cbar = fig.colorbar(im1, ax=axes, shrink=0.85, pad=0.02, aspect=30)
cbar.set_label("Number of pairs")

fig.suptitle("Pairwise coalescence times at GRK2 (chr11:66–68 Mb)",
             fontweight="normal", fontsize=13)

out = "analysis/genome_wide/cxt/fire_grk2_1225.png"
fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
