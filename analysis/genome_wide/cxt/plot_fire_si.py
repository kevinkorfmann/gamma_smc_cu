#!/usr/bin/env python
"""
SI fire plots for known sweep loci (cxt-based, Schweiger Fig 5B style).
3 rows: SH2B3/ALDH2, CYP3A, FADS1/FADS2. Each row: sweep pop + YRI control.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

GENERATION_TIME = 29

# Gene annotations (GRCh38, Mb)
# Format: (name, start_mb, end_mb, color, label_offset_y)
# label_offset_y staggers labels to avoid overlap
GENES = {
    "SH2B3_ALDH2": [
        ("ATXN2",  111.452, 111.599, "#95a5a6", 0),
        ("ALDH2",  111.766, 111.817, "#3498db", 0),
        ("SH2B3",  111.843, 111.890, "#e74c3c", 0.25),  # stagger up
    ],
    "CYP3A": [
        ("CYP3A43", 99.536, 99.570, "#95a5a6", 0),
        ("CYP3A7",  99.605, 99.636, "#95a5a6", 0.25),
        ("CYP3A5",  99.648, 99.680, "#e74c3c", 0),
        ("CYP3A4",  99.756, 99.784, "#3498db", 0),
    ],
    "FADS1": [
        ("TMEM258", 61.547, 61.560, "#95a5a6", 0.25),
        ("FADS1",   61.567, 61.584, "#e74c3c", 0),
        ("FADS2",   61.586, 61.625, "#3498db", 0.25),
    ],
}

LOCI = [
    {
        "name": "SH2B3_ALDH2",
        "sweep_file": "cxt_SH2B3_ALDH2_TSI.npz",
        "ctrl_file": "cxt_SH2B3_ALDH2_YRI.npz",
        "sweep_label": "TSI (European) — sweep",
        "ctrl_label": "YRI (Yoruba) — control",
        "title": "SH2B3 / ALDH2 (chr12:110–113 Mb)",
    },
    {
        "name": "CYP3A",
        "sweep_file": "cxt_CYP3A_FIN.npz",
        "ctrl_file": "cxt_CYP3A_YRI.npz",
        "sweep_label": "FIN (Finnish) — sweep",
        "ctrl_label": "YRI (Yoruba) — control",
        "title": "CYP3A cluster (chr7:99–100.5 Mb)",
    },
    {
        "name": "FADS1",
        "sweep_file": "cxt_FADS1_ITU.npz",
        "ctrl_file": "cxt_FADS1_YRI.npz",
        "sweep_label": "ITU (South Asian) — sweep",
        "ctrl_label": "YRI (Yoruba) — control",
        "title": "FADS1 / FADS2 (chr11:60.5–62.5 Mb)",
    },
]


def load_fire_data(npz_path):
    """Load cxt data and return positions + TMRCA in years."""
    d = np.load(npz_path, allow_pickle=True)
    raw = d["log_tmrca_raw"]   # (n_reps, n_items, n_windows)
    idx = d["index_map"]
    blocks = d["blocks"]
    n_windows = raw.shape[2]

    mean_log = raw.mean(axis=0)
    tmrca_years = np.exp(mean_log) * GENERATION_TIME

    all_pos, all_tmrca = [], []
    for b in range(len(blocks)):
        block_start, block_end = blocks[b]
        pos_mb = np.linspace(block_start / 1e6, block_end / 1e6,
                             n_windows, endpoint=False)
        all_pos.append(pos_mb)

        block_mask = idx[:, 0] == b
        block_items = np.where(block_mask)[0]
        pair_order = np.argsort(idx[block_items, 1])
        all_tmrca.append(tmrca_years[block_items[pair_order]])

    return np.concatenate(all_pos), np.concatenate(all_tmrca, axis=1)


def make_fire_panel(ax, positions_mb, tmrca_years, title, genes,
                    n_pos_bins=250, n_time_bins=180,
                    time_range=(3.5, 6.2)):
    n_pairs, n_windows = tmrca_years.shape
    pos_flat = np.tile(positions_mb, n_pairs)
    time_flat = np.log10(tmrca_years.ravel())

    # Only bin within actual data range (no extrapolation beyond inferred region)
    pos_edges = np.linspace(positions_mb.min(), positions_mb.max(), n_pos_bins + 1)
    time_edges = np.linspace(time_range[0], time_range[1], n_time_bins + 1)

    H, xedges, yedges = np.histogram2d(
        pos_flat, time_flat, bins=[pos_edges, time_edges])
    H = H.T
    H_masked = np.ma.masked_where(H == 0, H)

    im = ax.pcolormesh(
        xedges, yedges, H_masked,
        cmap="inferno",
        norm=LogNorm(vmin=1, vmax=H.max()),
        rasterized=True,
    )

    # Restrict x-axis to data range (don't show empty space)
    ax.set_xlim(positions_mb.min(), positions_mb.max())

    # Gene annotations below plot
    gene_base = time_range[0] - 0.05
    bar_height = 0.08
    for gene_name, start, end, color, y_off in genes:
        # Gene bar just below the plot
        rect = Rectangle(
            (start, gene_base - bar_height + y_off * 0),
            end - start, bar_height,
            linewidth=0.8, edgecolor=color, facecolor=color, alpha=0.85,
            zorder=5, clip_on=False,
        )
        ax.add_patch(rect)
        # Label with stagger offset
        label_y = gene_base - bar_height - 0.08 - y_off
        ax.text(
            (start + end) / 2, label_y,
            gene_name, ha="center", va="top", fontsize=6.5,
            color=color, fontweight="bold", zorder=5, clip_on=False,
        )

    ax.set_title(title, fontweight="normal", loc="left", fontsize=9)

    yticks = [4.0, 4.5, 5.0, 5.5, 6.0]
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"$10^{{{int(y)}}}$" if y == int(y)
                        else f"$10^{{{y}}}$" for y in yticks])
    ax.set_ylim(time_range[0], time_range[1])

    return im


def main():
    results_dir = "analysis/genome_wide/cxt/results"

    fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharey=True,
                             constrained_layout=True)

    for row, locus in enumerate(LOCI):
        genes = GENES[locus["name"]]
        panel = chr(ord('a') + row * 2)
        panel2 = chr(ord('a') + row * 2 + 1)

        pos_s, tmrca_s = load_fire_data(f"{results_dir}/{locus['sweep_file']}")
        pos_c, tmrca_c = load_fire_data(f"{results_dir}/{locus['ctrl_file']}")

        make_fire_panel(axes[row, 0], pos_s, tmrca_s,
                        f"{panel}  {locus['sweep_label']}", genes)
        im = make_fire_panel(axes[row, 1], pos_c, tmrca_c,
                             f"{panel2}  {locus['ctrl_label']}", genes)

        axes[row, 0].set_ylabel("Years ago")
        axes[row, 1].set_ylabel("")

    axes[-1, 0].set_xlabel("Genomic position (Mb)")
    axes[-1, 1].set_xlabel("Genomic position (Mb)")

    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02, aspect=40)
    cbar.set_label("Number of pairs")

    out = "analysis/genome_wide/cxt/fire_si_known_sweeps.png"
    fig.savefig(out, dpi=250, bbox_inches="tight", facecolor="white")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
