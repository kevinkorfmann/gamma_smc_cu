#!/usr/bin/env python3
"""Linear-TMRCA variant of make_heatmap.py for comparison against log10.
Writes figure_heatmap_linear.{png,pdf}."""
import runpy, os, sys
import numpy as np

# Monkey-patch: reuse make_heatmap.py but replace log transform with identity.
BASE = "/Users/kevinkorfmann/Projects/tmrca.cu/analysis/akbari_479_tmrca"
sys.path.insert(0, BASE)
import make_heatmap as mh

def plot_linear():
    matrix, meta, pops_have = mh.build_matrix()
    meta = mh.annotate_genes(meta)
    order, min_tmrca_kya = mh.sort_rows(matrix, meta)
    matrix = matrix.loc[order]
    meta = meta.loc[order]
    tmrca_kya = matrix * mh.YR_PER_GEN / 1000.0

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.patches import Rectangle

    n_loci = len(order); n_pops = len(pops_have)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8,
                         "axes.linewidth": 0.6})
    fig_h = max(7.0, 0.11 * n_loci + 2.2)
    fig = plt.figure(figsize=(9.0, fig_h))
    gs = fig.add_gridspec(2, 4, width_ratios=[0.28, 1.0, 0.18, 0.04],
                          height_ratios=[0.06, 1.0], wspace=0.04, hspace=0.02)
    ax_heat = fig.add_subplot(gs[1, 1])
    ax_genes = fig.add_subplot(gs[1, 0], sharey=ax_heat)
    ax_bar = fig.add_subplot(gs[1, 2], sharey=ax_heat)
    ax_sp = fig.add_subplot(gs[0, 1])
    cax = fig.add_subplot(gs[1, 3])

    cmap = plt.get_cmap("magma")
    vmin = np.nanpercentile(tmrca_kya.values, 2)
    vmax = np.nanpercentile(tmrca_kya.values, 98)
    im = ax_heat.imshow(tmrca_kya.values, aspect="auto", cmap=cmap,
                        vmin=vmin, vmax=vmax, interpolation="nearest")
    ax_heat.set_xticks(range(n_pops))
    ax_heat.set_xticklabels(pops_have, rotation=90, fontsize=7)
    ax_heat.set_yticks([])

    prev_sp = None
    for i, p in enumerate(pops_have):
        sp = mh.SUPERPOP_OF[p]
        if prev_sp is not None and sp != prev_sp:
            ax_heat.axvline(i - 0.5, color="white", lw=1.4)
            ax_sp.axvline(i - 0.5, color="white", lw=1.4)
        prev_sp = sp

    ax_sp.set_xlim(-0.5, n_pops - 0.5); ax_sp.set_ylim(0, 1)
    for i, p in enumerate(pops_have):
        ax_sp.add_patch(Rectangle((i - 0.5, 0), 1, 1,
                        color=mh.SUPERPOP_COLOR[mh.SUPERPOP_OF[p]], linewidth=0))
    for sp in ["AFR", "EUR", "SAS", "EAS", "AMR"]:
        xs = [i for i, p in enumerate(pops_have) if mh.SUPERPOP_OF[p] == sp]
        if xs:
            ax_sp.text(np.mean(xs), 0.5, sp, ha="center", va="center",
                       fontsize=8.5, color="white", fontweight="bold")
    ax_sp.set_xticks([]); ax_sp.set_yticks([])
    for s in ax_sp.spines.values(): s.set_visible(False)

    ax_genes.set_xlim(0, 1); ax_genes.set_ylim(n_loci - 0.5, -0.5)
    ax_genes.set_xticks([]); ax_genes.set_yticks([])
    for s in ax_genes.spines.values(): s.set_visible(False)
    for i, name in enumerate(meta["gene"].tolist()):
        if not name: continue
        ax_genes.text(1.0, i, name, ha="right", va="center",
                      fontsize=6.5, color="#333333", fontstyle="italic")

    x_abs = meta["akbari_X"].abs().values
    ax_bar.barh(range(n_loci), x_abs, color="#444444", height=0.82, edgecolor="none")
    ax_bar.set_xlim(0, max(x_abs.max(), 6) * 1.05)
    ax_bar.set_ylim(n_loci - 0.5, -0.5)
    ax_bar.set_xlabel("Akbari |X|", fontsize=7, labelpad=4)
    ax_bar.tick_params(axis="x", labelsize=6, pad=2)
    ax_bar.tick_params(axis="y", which="both", length=0)
    for s in ["top", "right"]: ax_bar.spines[s].set_visible(False)
    ax_bar.spines["left"].set_visible(False)
    ax_bar.axvline(5.45, color="#c0395b", lw=0.6, linestyle="--", alpha=0.6)

    cbar = fig.colorbar(im, cax=cax)
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(labelsize=6, length=2.4)
    cbar.set_label("TMRCA (kyr)", fontsize=7, labelpad=4)

    fig.suptitle(
        f"gamma_smc_cu coalescent dates at {n_loci} Akbari 2026 lead variants "
        f"(\u00b125 kb, sorted by min TMRCA) — linear scale",
        fontsize=9.5, x=0.04, y=0.98, ha="left", fontweight="normal")

    png = os.path.join(BASE, "figure_heatmap_linear.png")
    pdf = os.path.join(BASE, "figure_heatmap_linear.pdf")
    plt.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"wrote: {png}")


plot_linear()
