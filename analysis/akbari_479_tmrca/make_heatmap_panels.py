#!/usr/bin/env python3
"""4-panel variant of the Akbari-479 TMRCA heatmap.

Rows grouped by focal superpopulation = the one with the youngest mean TMRCA
for that locus. Panels stack vertically:
    a  AFR-focal
    b  EUR-focal
    c  SAS-focal
    d  EAS-focal
    e  AMR-focal (optional, usually sparse)

Inside each panel: rows sorted by min-TMRCA ascending. Shared 26-pop x-axis
(labels only on the bottom panel) and shared colorbar.

Output: figure_heatmap_panels.{png,pdf}
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

BASE = "/Users/kevinkorfmann/Projects/tmrca.cu/analysis/akbari_479_tmrca"
sys.path.insert(0, BASE)
import make_heatmap as mh

YR_PER_GEN = mh.YR_PER_GEN
SUPERPOP = mh.SUPERPOP
SUPERPOP_OF = mh.SUPERPOP_OF
SUPERPOP_COLOR = mh.SUPERPOP_COLOR

PANEL_ORDER = ["AFR", "EUR", "SAS", "EAS", "AMR"]


def focal_superpop_per_row(matrix_gens, pops_have):
    """Return superpop label whose mean TMRCA is lowest for each row."""
    sp_mean = {sp: [] for sp in PANEL_ORDER}
    for sp in PANEL_ORDER:
        cols = [p for p in SUPERPOP[sp] if p in pops_have]
        if cols:
            sp_mean[sp] = matrix_gens[cols].mean(axis=1)
        else:
            sp_mean[sp] = None
    per_row = []
    valid_sps = [sp for sp in PANEL_ORDER if sp_mean[sp] is not None]
    for i in range(len(matrix_gens)):
        vals = {sp: sp_mean[sp].iloc[i] for sp in valid_sps
                if not np.isnan(sp_mean[sp].iloc[i])}
        if not vals:
            per_row.append(None)
        else:
            per_row.append(min(vals, key=vals.get))
    return per_row


def build():
    matrix, meta, pops_have = mh.build_matrix()
    meta = mh.annotate_genes(meta)
    min_tmrca_kya = (matrix.min(axis=1) * YR_PER_GEN / 1000.0)

    focal = focal_superpop_per_row(matrix, pops_have)
    meta = meta.copy()
    meta["focal_sp"] = focal
    meta["min_tmrca_kya"] = min_tmrca_kya
    return matrix, meta, pops_have


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
    matrix, meta, pops_have = build()

    # Split rows by focal superpop, sort each group by min-TMRCA.
    groups = {}
    for sp in PANEL_ORDER:
        rows = meta[meta["focal_sp"] == sp].sort_values("min_tmrca_kya").index
        if len(rows):
            groups[sp] = rows
    print(f"panel row counts: {[(sp, len(r)) for sp, r in groups.items()]}")

    n_panels = len(groups)
    if n_panels == 0:
        raise SystemExit("no rows classified")

    # Global colour scale over all panels.
    tmrca_kya_all = matrix * YR_PER_GEN / 1000.0
    log_all = np.log10(tmrca_kya_all.clip(lower=0.1))
    vmin = np.nanpercentile(log_all.values, 2)
    vmax = np.nanpercentile(log_all.values, 98)

    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "savefig.facecolor": "white",
    })

    # Panel heights proportional to row counts. Minimum height so small groups
    # aren't squished.
    row_counts = [len(groups[sp]) for sp in groups]
    total_rows = sum(row_counts)
    fig_h = max(8.5, 0.10 * total_rows + 3.5)
    # Panel height ratio: rows + small offset so tiny groups are still visible.
    height_ratios = [max(r, 6) for r in row_counts] + [0.7]  # last small one for pop axis

    fig = plt.figure(figsize=(9.0, fig_h))
    outer = fig.add_gridspec(
        nrows=len(groups) + 1, ncols=4,
        width_ratios=[0.28, 1.0, 0.15, 0.035],
        height_ratios=height_ratios,
        wspace=0.035, hspace=0.14,
    )

    n_pops = len(pops_have)
    cmap = plt.get_cmap("magma")
    im = None
    panel_keys = list(groups.keys())

    for pi, sp in enumerate(panel_keys):
        rows = groups[sp]
        n = len(rows)

        ax_g = fig.add_subplot(outer[pi, 0])
        ax_h = fig.add_subplot(outer[pi, 1])
        ax_b = fig.add_subplot(outer[pi, 2], sharey=ax_h)

        sub = matrix.loc[rows]
        sub_kya = sub * YR_PER_GEN / 1000.0
        sub_log = np.log10(sub_kya.clip(lower=0.1))

        im = ax_h.imshow(
            sub_log.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )
        ax_h.set_yticks([])
        if pi == len(panel_keys) - 1:
            ax_h.set_xticks(range(n_pops))
            ax_h.set_xticklabels(pops_have, rotation=90, fontsize=7)
        else:
            ax_h.set_xticks([])
        for s in ax_h.spines.values():
            s.set_linewidth(0.6)

        # Superpop separator lines
        prev_sp = None
        for i, p in enumerate(pops_have):
            sp2 = SUPERPOP_OF[p]
            if prev_sp is not None and sp2 != prev_sp:
                ax_h.axvline(i - 0.5, color="white", lw=1.2)
            prev_sp = sp2

        # Panel label + focal superpop chip
        panel_letter = "abcde"[pi]
        ax_h.text(
            -0.18, 1.04, panel_letter,
            transform=ax_h.transAxes,
            fontsize=11, fontweight="bold",
        )
        chip = Rectangle((-0.14, 1.01), 0.04, 0.06,
                         transform=ax_h.transAxes, color=SUPERPOP_COLOR[sp],
                         clip_on=False)
        ax_h.add_patch(chip)
        ax_h.text(
            -0.09, 1.04, f"{sp}-focal  ({n} loci)",
            transform=ax_h.transAxes,
            fontsize=9, color=SUPERPOP_COLOR[sp], fontweight="bold",
            va="center",
        )

        # Gene labels
        ax_g.set_xlim(0, 1); ax_g.set_ylim(n - 0.5, -0.5)
        ax_g.set_xticks([]); ax_g.set_yticks([])
        for s in ax_g.spines.values():
            s.set_visible(False)
        for i, k in enumerate(rows):
            lab, col = label_for(meta.loc[k])
            weight = "bold" if mh.is_flagship_label(lab) else "normal"
            ax_g.text(1.0, i, lab, ha="right", va="center",
                      fontsize=6.5, color=col, fontstyle="italic",
                      fontweight=weight)

        # |X| bar
        x_abs = meta.loc[rows, "akbari_X"].abs().values
        ax_b.barh(range(n), x_abs, color="#444444", height=0.82, edgecolor="none")
        x_cap = max(meta["akbari_X"].abs().max(), 6) * 1.05
        ax_b.set_xlim(0, x_cap)
        ax_b.set_ylim(n - 0.5, -0.5)
        ax_b.set_yticks([])
        ax_b.tick_params(axis="x", labelsize=5.5, pad=1)
        ax_b.tick_params(axis="y", which="both", length=0)
        for s in ["top", "right", "left"]:
            ax_b.spines[s].set_visible(False)
        ax_b.axvline(5.45, color="#c0395b", lw=0.5, linestyle="--", alpha=0.5)
        if pi == len(panel_keys) - 1:
            ax_b.set_xlabel("Akbari |X|", fontsize=7, labelpad=3)

    # Shared colorbar in far-right column, spanning all panels.
    cax = fig.add_subplot(outer[:len(panel_keys), 3])
    cbar = fig.colorbar(im, cax=cax)
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(labelsize=6, length=2.4, which="major")
    cbar.ax.tick_params(labelsize=5, length=1.4, which="minor")
    major = [10, 20, 50, 100, 200, 500, 1000]
    minor = [15, 30, 40, 70, 150, 300, 400, 700]
    maj = [np.log10(x) for x in major if vmin <= np.log10(x) <= vmax]
    min_ = [np.log10(x) for x in minor if vmin <= np.log10(x) <= vmax]
    if maj:
        cbar.set_ticks(maj)
        cbar.set_ticklabels([f"{int(10**t)}" for t in maj])
    if min_:
        cbar.ax.yaxis.set_ticks(min_, minor=True)
    cbar.set_label("TMRCA (kyr)", fontsize=7, labelpad=4)

    fig.suptitle(
        f"gamma_smc_cu coalescent dates at {sum(row_counts)} Akbari 2026 lead variants "
        "(\u00b125 kb), grouped by focal superpopulation",
        fontsize=9.5, x=0.04, y=0.995, ha="left", fontweight="normal",
    )

    png = os.path.join(BASE, "figure_heatmap_panels.png")
    pdf = os.path.join(BASE, "figure_heatmap_panels.pdf")
    plt.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"wrote: {png}")


if __name__ == "__main__":
    plot()
