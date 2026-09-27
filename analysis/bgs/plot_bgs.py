#!/usr/bin/env python3
"""Exploratory BGS plot integration; preserves original TMRCA values and ranks."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bgs import CACHE, MODELS, read_track, sha256

COLORS = {"regulatory": "#0072B2", "phastcons": "#D55E00"}
LABELS = {"regulatory": "CDS + regulatory", "phastcons": "CDS + phastCons"}


def save(fig, output, name):
    fig.savefig(output / f"{name}.pdf", dpi=600)
    fig.savefig(output / f"{name}.png", dpi=220)
    plt.close(fig)


def regional(args):
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.3), sharex="col",
                             gridspec_kw={"height_ratios": [1.4, 1, 1]}, layout="constrained")
    for column, (gene, chrom, pop) in enumerate([("GRK2", 11, "GIH"), ("TREM2", 6, "IBS")]):
        z = np.load(args.case_data / f"{gene}_plot_data.npz")
        genes = pd.read_csv(args.case_data / f"{gene}_gene_track.csv")
        focal_names = ["GRK2"] if gene == "GRK2" else ["TREML1", "TREM2"]
        for tag, color, style, label in [("focal", "#173F66", "-", pop),
                                          ("control", "#7B8B98", "--", "YRI")]:
            x = z[f"{tag}_positions"] / 1e6
            axes[0, column].plot(x, z[f"{tag}_median"], color=color, ls=style, lw=.9, label=label)
            axes[0, column].fill_between(x, z[f"{tag}_q25"], z[f"{tag}_q75"], color=color, alpha=.18, lw=0)
        axes[0, column].set(yscale="log", ylabel="TMRCA (generations)",
                            title="GRK2" if column == 0 else "TREML1 / TREM2")
        axes[0, column].legend(frameon=False, ncol=2, loc="upper right")
        for row, model in enumerate(MODELS, start=1):
            starts, ends, b = read_track(args.cache, model, chrom)
            keep = (ends > z["window"][0]) & (starts < z["window"][1])
            # NaN separator prevents connecting across absent bins.
            x = np.column_stack([starts[keep], ends[keep], np.full(keep.sum(), np.nan)]).ravel() / 1e6
            y = np.column_stack([b[keep], b[keep], np.full(keep.sum(), np.nan)]).ravel()
            ax = axes[row, column]
            ax.plot(x, y, color=COLORS[model], lw=.8)
            ax.axhline(.9, color=".55", lw=.6, ls="--")
            ax.set(ylim=(0, 1.02), ylabel="B (retained diversity)")
            ax.text(.02, .07, LABELS[model], transform=ax.transAxes, fontsize=7)
        for ax in axes[:, column]:
            ax.set_xlim(z["window"] / 1e6)
            for row in genes[genes.gene_name.isin(focal_names)].itertuples():
                ax.axvspan(row.start / 1e6, row.end / 1e6, color="#DBE6EB", zorder=-2)
        axes[-1, column].set_xlabel(f"Chromosome {chrom} position (Mb, GRCh38)")
    fig.suptitle("BGS context for the existing locus plots\nBarroso et al. (2026), preprint · YRI B maps", fontsize=9)
    save(fig, args.output, "bgs_locus_tracks")


def genomewide(args, genes):
    plotted = genes[genes.min_rank.notna()].copy()
    plotted["mid"] = (plotted.start + plotted.end) / 2 / 1e6
    offsets, ticks, ticklabels, end = {}, [], [], 0
    for chrom, length in plotted.groupby("chr").mid.max().items():
        offsets[chrom] = end
        ticks.append(end + length / 2)
        ticklabels.append(str(int(chrom)))
        end += length + 5
    plotted["x"] = plotted.chr.map(offsets) + plotted.mid
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.7), sharex=True, layout="constrained")
    for ax, model in zip(axes, MODELS):
        valid = ~plotted.is_sd & plotted[f"b_{model}_coverage"].ge(.95)
        ax.scatter(plotted.loc[~valid, "x"], -np.log10(plotted.loc[~valid, "min_rank"]),
                   s=2, c=".83", marker="x", lw=.4, rasterized=True)
        points = ax.scatter(plotted.loc[valid, "x"], -np.log10(plotted.loc[valid, "min_rank"]),
                            c=plotted.loc[valid, f"b_{model}_mean"], cmap="viridis", vmin=0, vmax=1,
                            s=3.3, lw=0, rasterized=True)
        for name, dx, dy in [("GRK2", 0, 12), ("TREM2", 0, -21), ("LCT", 10, 29), ("EDAR", -13, 11), ("SLC24A5", 0, 10)]:
            row = plotted[plotted.gene_name == name].iloc[0]
            ax.scatter([row.x], [-np.log10(row.min_rank)], s=24, facecolor="none", edgecolor="black", lw=.6)
            ax.annotate(name, (row.x, -np.log10(row.min_rank)), xytext=(dx, dy), textcoords="offset points",
                        ha="center", fontsize=6.5, arrowprops={"arrowstyle": "-", "lw": .4})
        ax.axhline(2, color=".65", lw=.5, ls="--")
        ax.set(ylabel="−log₁₀(minimum rank)", title=LABELS[model], xlim=(-10, end), ylim=(-.04, 4.8))
        fig.colorbar(points, ax=ax, label="Gene mean B", fraction=.025, pad=.015)
    axes[-1].set_xticks(ticks, ticklabels, rotation=45, fontsize=6)
    axes[-1].set_xlabel("Chromosome")
    fig.suptitle("Existing genome-wide ranks with BGS annotations\nBarroso et al. (2026), preprint · YRI B maps", fontsize=9)
    save(fig, args.output, "bgs_manhattan")


def diagnostics(args, genes):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), layout="constrained")
    for column, model in enumerate(MODELS):
        valid = ~genes.is_sd & genes[f"b_{model}_coverage"].ge(.95) & genes.YRI_rank.notna()
        frame = genes[valid]
        ax = axes[0, column]
        ax.scatter(frame[f"b_{model}_mean"], frame.YRI_rank, s=2, color=COLORS[model], alpha=.18, rasterized=True, lw=0)
        for name, marker in [("GRK2", "D"), ("TREML1", "^"), ("TREM2", "o")]:
            row = frame[frame.gene_name == name].iloc[0]
            ax.scatter([row[f"b_{model}_mean"]], [row.YRI_rank], s=17, edgecolor="black", facecolor="white", lw=.6,
                       marker=marker, label=name)
        ax.legend(loc="upper left", frameon=False, fontsize=6.5)
        ax.set(xlabel="Gene mean B", ylabel="Original YRI TMRCA rank", title=LABELS[model], ylim=(0, 1), xlim=(0, 1))
        decile = f"b_{model}_decile"
        boxes = [group.YRI_rank.to_numpy() for _, group in frame.groupby(decile)]
        axes[1, column].boxplot(boxes, showfliers=False, widths=.6,
                               medianprops={"color": COLORS[model]},
                               boxprops={"linewidth": .6}, whiskerprops={"linewidth": .6}, capprops={"linewidth": .6})
        axes[1, column].set(xlabel="B decile (stronger → weaker predicted BGS)", ylabel="Original YRI TMRCA rank", ylim=(0, 1))
        ax.text(.97, .95, f"n = {len(frame):,}", transform=ax.transAxes, ha="right", va="top", fontsize=7)
    fig.suptitle("BGS and ranking: descriptive checks in YRI\nBarroso et al. (2026), preprint · YRI B maps", fontsize=9)
    save(fig, args.output, "bgs_rank_diagnostics")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--case-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42})
    genes = pd.read_csv(args.annotations)
    regional(args)
    genomewide(args, genes)
    diagnostics(args, genes)
    (args.output / "plot_provenance.json").write_text(json.dumps({
        "annotations_sha256": sha256(args.annotations), "script_sha256": sha256(__file__),
        "case_inputs": {p.name: sha256(p) for p in args.case_data.iterdir() if p.suffix in (".npz", ".csv")},
        "figures": ["bgs_locus_tracks", "bgs_manhattan", "bgs_rank_diagnostics"],
        "label": "Exploratory; Barroso et al. 2026 preprint; YRI B maps applied as shared annotations",
        "regional_shading": "Gene bodies; TMRCA lines and IQRs preserved from archived arrays",
        "manhattan_gray": "SD flagged or map coverage below 95%; ranks unchanged",
        "boxes": "Median, interquartile range and 1.5-IQR whiskers; outliers omitted",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
