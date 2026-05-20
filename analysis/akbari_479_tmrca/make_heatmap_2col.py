#!/usr/bin/env python3
"""Two-column variant of the Akbari-479 TMRCA heatmap.

Rows are sorted globally by min-TMRCA across populations. The sorted list is
split into N columns (default 2): left column holds the youngest half, right
column the older half. Columns share the 26-pop x-axis and colorbar; each
column has its own gene labels and Akbari |X| side-bar.

Output: figure_heatmap_2col.{png,pdf}
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE = "/Users/kevinkorfmann/Projects/tmrca.cu/analysis/akbari_479_tmrca"
sys.path.insert(0, BASE)
import make_heatmap as mh

YR_PER_GEN = mh.YR_PER_GEN
SUPERPOP = mh.SUPERPOP
SUPERPOP_OF = mh.SUPERPOP_OF
SUPERPOP_COLOR = mh.SUPERPOP_COLOR

N_COLS = 3   # number of columns in the layout


def label_for(row):
    nearest = row["gene"] if row["gene"] else ""
    known = mh.known_sweep_label(row["chrom"], int(row["center_pos"]))
    if nearest and known and nearest != known.split("/")[0]:
        return f"{nearest} ({known})", "#333333"
    if known and not nearest:
        return f"intergenic ({known})", "#888888"
    if nearest:
        return nearest, "#333333"
    return "intergenic", "#888888"


def plot():
    matrix, meta, pops_have = mh.build_matrix()
    meta = mh.annotate_genes(meta)
    min_tmrca_kya = (matrix.min(axis=1) * YR_PER_GEN / 1000.0)
    order = min_tmrca_kya.sort_values(ascending=True).index
    matrix = matrix.loc[order]
    meta = meta.loc[order]

    # Split index into N_COLS near-equal chunks (fill columns top-down left-right).
    n_total = len(order)
    chunk = (n_total + N_COLS - 1) // N_COLS
    col_indices = [order[i * chunk:(i + 1) * chunk] for i in range(N_COLS)]
    rows_per_col = [len(c) for c in col_indices]

    tmrca_kya = matrix * YR_PER_GEN / 1000.0
    log_kya = np.log10(tmrca_kya.clip(lower=0.1))
    vmin = np.nanpercentile(log_kya.values, 2)
    vmax = np.nanpercentile(log_kya.values, 98)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#d6d6d6")   # NaN cells -> light grey, not pure white

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "savefig.facecolor": "white",
    })

    n_pops = len(pops_have)
    max_col_rows = max(rows_per_col)
    fig_h = max(7.5, 0.10 * max_col_rows + 2.5)
    fig_w = 6.0 + 4.5 * (N_COLS - 1)
    fig = plt.figure(figsize=(fig_w, fig_h))

    # For each column block: [genes | heatmap | |X| bar]
    # Then one shared colorbar column at the very right.
    inner_widths = []
    for _ in range(N_COLS):
        inner_widths += [0.28, 1.0, 0.15]
    inner_widths += [0.04]
    outer = fig.add_gridspec(
        nrows=2, ncols=len(inner_widths),
        width_ratios=inner_widths,
        height_ratios=[0.04, 1.0],
        wspace=0.03, hspace=0.005,
        top=0.955, bottom=0.04, left=0.03, right=0.985,
    )

    im = None
    for ci in range(N_COLS):
        keys = col_indices[ci]
        n = len(keys)
        col_base = ci * 3
        ax_g = fig.add_subplot(outer[1, col_base + 0])
        ax_h = fig.add_subplot(outer[1, col_base + 1])
        ax_b = fig.add_subplot(outer[1, col_base + 2], sharey=ax_h)
        ax_sp = fig.add_subplot(outer[0, col_base + 1])

        sub = log_kya.loc[keys]
        # Pad the shorter column with NaN so all columns have the same image
        # height (keeps gridspec ratios stable). Pads render as empty cells.
        if n < max_col_rows:
            pad = max_col_rows - n
            pad_rows = np.full((pad, n_pops), np.nan)
            img = np.vstack([sub.values, pad_rows])
        else:
            img = sub.values

        im = ax_h.imshow(img, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                         interpolation="nearest")
        ax_h.set_yticks([])
        ax_h.set_xticks(range(n_pops))
        ax_h.set_xticklabels(pops_have, rotation=90, fontsize=7)
        for s in ax_h.spines.values():
            s.set_linewidth(0.6)

        prev_sp = None
        for i, p in enumerate(pops_have):
            sp2 = SUPERPOP_OF[p]
            if prev_sp is not None and sp2 != prev_sp:
                ax_h.axvline(i - 0.5, color="white", lw=1.2)
                ax_sp.axvline(i - 0.5, color="white", lw=1.2)
            prev_sp = sp2

        ax_sp.set_xlim(-0.5, n_pops - 0.5); ax_sp.set_ylim(0, 1)
        for i, p in enumerate(pops_have):
            ax_sp.add_patch(Rectangle(
                (i - 0.5, 0), 1, 1, color=SUPERPOP_COLOR[SUPERPOP_OF[p]],
                linewidth=0))
        for sp in ["AFR", "EUR", "SAS", "EAS", "AMR"]:
            xs = [i for i, p in enumerate(pops_have) if SUPERPOP_OF[p] == sp]
            if xs:
                ax_sp.text(np.mean(xs), 0.5, sp, ha="center", va="center",
                           fontsize=8.5, color="white", fontweight="bold")
        ax_sp.set_xticks([]); ax_sp.set_yticks([])
        for s in ax_sp.spines.values(): s.set_visible(False)

        ax_g.set_xlim(0, 1); ax_g.set_ylim(max_col_rows - 0.5, -0.5)
        ax_g.set_xticks([]); ax_g.set_yticks([])
        for s in ax_g.spines.values(): s.set_visible(False)
        for i, k in enumerate(keys):
            lab, col = label_for(meta.loc[k])
            weight = "bold" if mh.is_flagship_label(lab) else "normal"
            ax_g.text(1.0, i, lab, ha="right", va="center",
                      fontsize=6.5, color=col, fontstyle="italic",
                      fontweight=weight)

        x_abs = meta.loc[keys, "akbari_X"].abs().values
        x_abs_full = np.concatenate([x_abs, np.zeros(max_col_rows - n)])
        ax_b.barh(range(max_col_rows), x_abs_full, color="#444444",
                  height=0.82, edgecolor="none")
        x_cap = max(meta["akbari_X"].abs().max(), 6) * 1.05
        ax_b.set_xlim(0, x_cap)
        ax_b.set_ylim(max_col_rows - 0.5, -0.5)
        ax_b.tick_params(axis="x", labelsize=5.5, pad=1)
        ax_b.tick_params(axis="y", which="both", length=0)
        for s in ["top", "right", "left"]: ax_b.spines[s].set_visible(False)
        ax_b.axvline(5.45, color="#c0395b", lw=0.5, linestyle="--", alpha=0.5)
        ax_b.set_xlabel("Akbari |X|", fontsize=7, labelpad=3)

        # Column letter label
        fig.text(
            ax_g.get_position().x0 - 0.005,
            ax_sp.get_position().y1 + 0.006,
            "abcd"[ci],
            fontsize=11, fontweight="bold",
        )

    # Shared colorbar
    cax = fig.add_subplot(outer[1, -1])
    cbar = fig.colorbar(im, cax=cax)
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(labelsize=6, length=2.4, which="major")
    cbar.ax.tick_params(labelsize=5, length=1.4, which="minor")
    major = [10, 20, 50, 100, 200, 500, 1000]
    minor = [15, 30, 40, 70, 150, 300, 400, 700]
    maj = [np.log10(x) for x in major if vmin <= np.log10(x) <= vmax]
    mn = [np.log10(x) for x in minor if vmin <= np.log10(x) <= vmax]
    if maj:
        cbar.set_ticks(maj)
        cbar.set_ticklabels([f"{int(10**t)}" for t in maj])
    if mn:
        cbar.ax.yaxis.set_ticks(mn, minor=True)
    cbar.set_label("TMRCA (kyr)", fontsize=7, labelpad=4)

    # Top-line suptitle removed (small/illegible per reviewer; manuscript caption carries the
    # same information at full size).

    png = os.path.join(BASE, f"figure_heatmap_{N_COLS}col.png")
    pdf = os.path.join(BASE, f"figure_heatmap_{N_COLS}col.pdf")
    plt.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"wrote: {png}")

    # Mirror into manuscript figures dir when the working copy is checked out.
    manuscript_fig_dir = "/Users/kevinkorfmann/Projects/tmrca.cu/docs_local/manuscript/v4.1/figures"
    if os.path.isdir(manuscript_fig_dir):
        import shutil
        for src, name in [(png, "fig_akbari_heatmap.png"),
                          (pdf, "fig_akbari_heatmap.pdf")]:
            shutil.copy(src, os.path.join(manuscript_fig_dir, name))
        print(f"mirrored into: {manuscript_fig_dir}/fig_akbari_heatmap.{{png,pdf}}")


if __name__ == "__main__":
    plot()
