#!/usr/bin/env python3
"""Akbari-479 TMRCA heatmap — publication-quality rendering.

Rows:     lead variants, sorted by minimum TMRCA across populations.
Columns:  26 populations grouped by superpopulation (AFR EUR SAS EAS AMR).
Colour:   log10 TMRCA in years (kyr/Myr scale), ground-truth human generation time 29 y.

Row-side panels:
  - nearest gene name (gene body overlap; else nearest centre within 500 kb)
  - Akbari |X| horizontal bar, same row order.

Top panels:
  - superpopulation colour strip with labels.

Inputs : results/chr{N}/{POP}.csv   (gamma_smc_cu per-variant geom_mean)
Outputs: figure_heatmap.{png,pdf},   heatmap_matrix_kya.csv, heatmap_row_order.csv
"""
from __future__ import annotations

import glob
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

# The 23 canonical human-sweep genes in docs_local/manuscript/v4.1/tables/
# table1_known_sweeps.tex. Rows whose label contains any of these (as a
# standalone token) are drawn in bold italic.
KNOWN_SWEEPS_FLAGSHIP = frozenset({
    "SLC24A5", "ABCC11", "LCT", "EDAR", "FADS1", "KITLG", "TRPV6",
    "ALDH2", "HERC2", "ADH1B", "IRGM", "SH2B3", "ADH4", "SLC45A2",
    "ACKR1", "OAS1", "HMGA2", "CASP12", "TYRP1", "OCA2", "MC1R",
    "EPAS1", "HBB",
})
_GENE_TOKEN_RE = re.compile(r"[A-Z][A-Z0-9]*[0-9A-Z]")


def is_flagship_label(label: str) -> bool:
    """True when the drawn label contains a canonical-sweep gene as a token."""
    if not label:
        return False
    tokens = set(_GENE_TOKEN_RE.findall(label))
    return bool(tokens & KNOWN_SWEEPS_FLAGSHIP)

BASE = "/Users/kevinkorfmann/Projects/tmrca.cu/analysis/akbari_479_tmrca"
RESULTS = os.path.join(BASE, "results")
GENES = os.path.join(BASE, "genes")
YR_PER_GEN = 29.0

SUPERPOP = {
    "AFR": ["YRI", "LWK", "GWD", "MSL", "ESN", "ACB", "ASW"],
    "EUR": ["CEU", "TSI", "FIN", "GBR", "IBS"],
    "SAS": ["GIH", "PJL", "BEB", "STU", "ITU"],
    "EAS": ["CHB", "JPT", "CHS", "CDX", "KHV"],
    "AMR": ["MXL", "PUR", "CLM", "PEL"],
}
POP_ORDER = sum(SUPERPOP.values(), [])
SUPERPOP_OF = {p: sp for sp, ps in SUPERPOP.items() for p in ps}
SUPERPOP_COLOR = {
    "AFR": "#e8b02c",
    "EUR": "#2f6db0",
    "SAS": "#0a9a6a",
    "EAS": "#c0395b",
    "AMR": "#6fa54d",
}

# Curated sweep-locus overrides (GRCh37) — shown in parentheses next to the
# automatically-assigned nearest gene when the Akbari lead SNP falls in one
# of these well-known sweep haplotypes. The tag SNP often sits in a
# downstream/regulatory gene, obscuring the actual target of selection.
KNOWN_SWEEPS = [
    ("2",  135_000_000, 136_500_000, "LCT/MCM6"),
    ("2",  109_000_000, 110_500_000, "EDAR"),
    ("15",  47_500_000,  48_800_000, "SLC24A5"),
    ("15",  28_000_000,  29_000_000, "OCA2/HERC2"),
    ("16",  89_500_000,  90_000_000, "MC1R"),
    ("11",  61_200_000,  61_800_000, "FADS1/2"),
    ("11",  88_800_000,  89_300_000, "TYR"),
    ("5",   33_700_000,  34_100_000, "SLC45A2"),
    ("9",  135_900_000, 136_250_000, "ABO"),
    ("4",  100_000_000, 100_400_000, "ADH1B"),
    ("11",   5_100_000,   5_300_000, "HBB"),
    ("6",   28_000_000,  34_000_000, "HLA"),
    ("3",   46_200_000,  46_600_000, "CCR5"),
    ("2",   45_900_000,  46_700_000, "EPAS1"),
    ("19",  45_200_000,  45_500_000, "APOE"),
]


def known_sweep_label(chrom, pos):
    for c, lo, hi, name in KNOWN_SWEEPS:
        if str(chrom) == c and lo <= pos <= hi:
            return name
    return None


def load_gene_table():
    frames = []
    for path in sorted(glob.glob(os.path.join(GENES, "chr*_genes.tsv"))):
        chrom = os.path.basename(path).replace("chr", "").split("_")[0]
        df = pd.read_csv(path, sep="\t")
        df["chrom"] = str(chrom)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def nearest_gene(chrom, pos, gtab, flank=500_000):
    """Return gene name that overlaps the position, else nearest centre within flank."""
    sub = gtab[gtab.chrom == str(chrom)]
    if sub.empty:
        return ""
    overlap = sub[(sub.start <= pos) & (pos <= sub.end)]
    if not overlap.empty:
        return overlap.iloc[0].gene_name
    center = (sub.start + sub.end) / 2
    d = (center - pos).abs()
    if d.min() > flank:
        return ""
    return sub.iloc[d.idxmin() - sub.index[0]].gene_name


def load_per_pop(pop):
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS, "chr*", f"{pop}.csv"))):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["pop"] = pop
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


MIN_SITES_FOR_VALID = 20  # cells with fewer aggregated sites -> NaN (unreliable)


def build_matrix():
    frames = {p: load_per_pop(p) for p in POP_ORDER}
    frames = {p: df for p, df in frames.items() if not df.empty}
    if not frames:
        raise SystemExit("no results yet")
    pops_have = [p for p in POP_ORDER if p in frames]
    print(f"have {len(pops_have)} populations: {pops_have}")

    matrix = None
    meta = None
    for pop, df in frames.items():
        df = df.copy()
        df["key"] = df.apply(
            lambda r: r["rsid"] if isinstance(r["rsid"], str) and r["rsid"].startswith("rs")
            else f"{r['chrom']}:{r['center_pos']}", axis=1)
        # mask low-site cells: they come from pericentromeric / SD / deserts
        # where aggregates are noise, not signal.
        valid = df["n_sites"] >= MIN_SITES_FOR_VALID
        col = df.set_index("key")["geom_mean_tmrca"].where(
            valid.set_axis(df["key"]).reindex(df.set_index("key").index).values)
        if matrix is None:
            matrix = pd.DataFrame({pop: col})
            meta = df.set_index("key")[["chrom", "center_pos", "akbari_X", "akbari_S"]]
        else:
            matrix[pop] = col
    matrix = matrix.reindex(columns=pops_have)

    # Drop rows where >= 50% of cells are NaN (variant lives in a site-desert
    # or segmental-duplication region for most populations).
    frac_valid = matrix.notna().mean(axis=1)
    keep = frac_valid >= 0.5
    dropped = (~keep).sum()
    if dropped:
        print(f"dropping {dropped} rows with <50% trustworthy cells "
              f"(pericentromeric / SD / site-deserts)")
    matrix = matrix[keep]
    meta = meta.loc[matrix.index]
    return matrix, meta, pops_have


def sort_rows(matrix, meta):
    min_tmrca_kya = (matrix.min(axis=1) * YR_PER_GEN / 1000.0)
    order = min_tmrca_kya.sort_values(ascending=True).index
    return order, min_tmrca_kya


def annotate_genes(meta):
    gtab = load_gene_table()
    names = []
    for k, row in meta.iterrows():
        names.append(nearest_gene(str(int(row["chrom"])), int(row["center_pos"]), gtab))
    meta["gene"] = names
    return meta


def plot(matrix, meta, order, min_tmrca_kya, pops_have):
    matrix = matrix.loc[order]
    meta = meta.loc[order]
    tmrca_kya = matrix * YR_PER_GEN / 1000.0
    log_kya = np.log10(tmrca_kya.clip(lower=0.1))
    n_loci = len(order)
    n_pops = len(pops_have)

    # Typography & canvas
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "savefig.facecolor": "white",
    })

    # Figure size: width fits 8.5" journal page, height scales with row count.
    fig_h = max(7.0, 0.11 * n_loci + 2.2)
    fig = plt.figure(figsize=(9.0, fig_h))
    # Columns: [gene labels | heatmap | |X| bar | colorbar]
    gs = fig.add_gridspec(
        nrows=2, ncols=4,
        width_ratios=[0.28, 1.0, 0.18, 0.04],
        height_ratios=[0.06, 1.0],
        wspace=0.04, hspace=0.02,
    )
    ax_heat = fig.add_subplot(gs[1, 1])
    ax_genes = fig.add_subplot(gs[1, 0], sharey=ax_heat)
    ax_bar = fig.add_subplot(gs[1, 2], sharey=ax_heat)
    ax_sp = fig.add_subplot(gs[0, 1])
    ax_sp.set_xlim(ax_heat.get_xlim())
    cax = fig.add_subplot(gs[1, 3])

    # ---- Heatmap ----
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#d6d6d6")   # NaN cells -> light grey, not pure white
    vmin = np.nanpercentile(log_kya.values, 2)
    vmax = np.nanpercentile(log_kya.values, 98)
    im = ax_heat.imshow(
        log_kya.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )
    ax_heat.set_xticks(range(n_pops))
    ax_heat.set_xticklabels(pops_have, rotation=90, fontsize=7)
    ax_heat.set_yticks([])
    for s in ax_heat.spines.values():
        s.set_linewidth(0.6)

    # Draw superpop separator lines
    prev_sp = None
    for i, p in enumerate(pops_have):
        sp = SUPERPOP_OF[p]
        if prev_sp is not None and sp != prev_sp:
            ax_heat.axvline(i - 0.5, color="white", lw=1.4)
            ax_sp.axvline(i - 0.5, color="white", lw=1.4)
        prev_sp = sp

    # ---- Superpop colour strip ----
    ax_sp.set_xlim(-0.5, n_pops - 0.5)
    ax_sp.set_ylim(0, 1)
    for i, p in enumerate(pops_have):
        ax_sp.add_patch(Rectangle(
            (i - 0.5, 0), 1, 1, color=SUPERPOP_COLOR[SUPERPOP_OF[p]],
            linewidth=0))
    # Superpop label in centre of its block
    for sp in ["AFR", "EUR", "SAS", "EAS", "AMR"]:
        xs = [i for i, p in enumerate(pops_have) if SUPERPOP_OF[p] == sp]
        if xs:
            ax_sp.text(np.mean(xs), 0.5, sp,
                       ha="center", va="center",
                       fontsize=8.5, color="white", fontweight="bold")
    ax_sp.set_xticks([]); ax_sp.set_yticks([])
    for s in ax_sp.spines.values():
        s.set_visible(False)

    # ---- Gene name labels on left ----
    ax_genes.set_xlim(0, 1)
    ax_genes.set_ylim(n_loci - 0.5, -0.5)
    ax_genes.set_xticks([]); ax_genes.set_yticks([])
    for s in ax_genes.spines.values():
        s.set_visible(False)
    # Only label the top-N and those carrying a named gene
    for i, (k, row) in enumerate(meta.iterrows()):
        nearest = row["gene"] if row["gene"] else ""
        known = known_sweep_label(row["chrom"], int(row["center_pos"]))
        if nearest and known and nearest != known.split("/")[0]:
            label = f"{nearest} ({known})"
        elif known and not nearest:
            label = f"intergenic ({known})"
        elif nearest:
            label = nearest
        else:
            label = "intergenic"
        color = "#333333" if nearest else "#888888"
        weight = "bold" if is_flagship_label(label) else "normal"
        ax_genes.text(
            1.0, i, label,
            ha="right", va="center",
            fontsize=6.5, color=color,
            fontstyle="italic", fontweight=weight,
        )

    # ---- Akbari |X| side bar ----
    x_abs = meta["akbari_X"].abs().values
    ax_bar.barh(range(n_loci), x_abs, color="#444444", height=0.82, edgecolor="none")
    ax_bar.set_xlim(0, max(x_abs.max(), 6) * 1.05)
    ax_bar.set_ylim(n_loci - 0.5, -0.5)
    ax_bar.set_xlabel("Akbari |X|", fontsize=7, labelpad=4)
    ax_bar.tick_params(axis="x", labelsize=6, pad=2)
    ax_bar.tick_params(axis="y", which="both", length=0)
    for s in ["top", "right"]:
        ax_bar.spines[s].set_visible(False)
    ax_bar.spines["left"].set_visible(False)
    ax_bar.axvline(5.45, color="#c0395b", lw=0.6, linestyle="--", alpha=0.6)

    # ---- Colorbar ----
    cbar = fig.colorbar(im, cax=cax)
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(labelsize=6, length=2.4, which="major")
    cbar.ax.tick_params(labelsize=5, length=1.4, which="minor")
    major_values = [10, 20, 50, 100, 200, 500, 1000, 2000]
    minor_values = [15, 30, 40, 70, 150, 300, 400, 700, 1500]
    major_ticks = [np.log10(x) for x in major_values if vmin <= np.log10(x) <= vmax]
    minor_ticks = [np.log10(x) for x in minor_values if vmin <= np.log10(x) <= vmax]
    if major_ticks:
        cbar.set_ticks(major_ticks)
        cbar.set_ticklabels([f"{int(10**t)}" for t in major_ticks])
    if minor_ticks:
        cbar.ax.yaxis.set_ticks(minor_ticks, minor=True)
    cbar.set_label("TMRCA (kyr)", fontsize=7, labelpad=4)

    # ---- Title ----
    fig.suptitle(
        f"gamma_smc_cu coalescent dates at {n_loci} Akbari 2026 lead variants "
        f"(\u00b125 kb, sorted by min TMRCA)",
        fontsize=9.5, x=0.04, y=0.98, ha="left", fontweight="normal",
    )

    png = os.path.join(BASE, "figure_heatmap.png")
    pdf = os.path.join(BASE, "figure_heatmap.pdf")
    plt.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"wrote: {png}\nwrote: {pdf}")

    # underlying data
    tmrca_kya.to_csv(os.path.join(BASE, "heatmap_matrix_kya.csv"))
    meta.assign(min_tmrca_kya=min_tmrca_kya.loc[order]).to_csv(
        os.path.join(BASE, "heatmap_row_order.csv"))


def main():
    matrix, meta, pops_have = build_matrix()
    meta = annotate_genes(meta)
    order, min_tmrca_kya = sort_rows(matrix, meta)
    print(f"matrix: {len(order)} loci x {len(pops_have)} pops")
    print(f"min-TMRCA quantiles (kya): "
          f"p01={min_tmrca_kya.quantile(0.01):.1f}  "
          f"p50={min_tmrca_kya.median():.1f}  "
          f"p99={min_tmrca_kya.quantile(0.99):.1f}")
    plot(matrix, meta, order, min_tmrca_kya, pops_have)


if __name__ == "__main__":
    main()
