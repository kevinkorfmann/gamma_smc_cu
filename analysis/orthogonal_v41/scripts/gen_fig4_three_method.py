#!/usr/bin/env python
"""Build the v4.1 Figure 4: three-method TMRCA comparison.

Reads analysis/orthogonal_v41/three_method/{gene}_{pop}_{group}.npz
files (one per task in the SLURM array) and produces a 15-row x 2-col
figure (cxt + gamma_smc_cu pairwise) with the focal pop traces in color
and the YRI control traces overlaid in gray.

The third column (ASMC) is reserved for a future sub-task and is shown
as a blank panel with a "pending" label until ASMC integration is in
place.

Run from the repo root after all 30 SLURM tasks have completed:

    python analysis/orthogonal_v41/scripts/gen_fig4_three_method.py

Output: docs_local/manuscript/v4.1/figures/fig4_three_method.{png,pdf}
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = "/Users/kevinkorfmann/Projects/tmrca.cu"
THREE_METHOD_DIR = os.path.join(REPO, "analysis/orthogonal_v41/three_method")
OUT_DIR = os.path.join(REPO, "docs_local/manuscript/v4.1/figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Same gene order as the SLURM tasks.txt: novel, positive, neutral.
GENES = [
    ("GRK2",     11, "GIH", "novel",    "purple"),
    ("BPIFA2",   20, "GIH", "novel",    "purple"),
    ("SLC6A15",  12, "CHS", "novel",    "purple"),
    ("CCDC92",   12, "CDX", "novel",    "purple"),
    ("CLEC6A",   12, "CDX", "novel",    "purple"),
    ("SLC24A5",  15, "GBR", "positive", "blue"),
    ("LCT",       2, "CEU", "positive", "blue"),
    ("EDAR",      2, "CHB", "positive", "blue"),
    ("ABCC11",   16, "CHB", "positive", "blue"),
    ("KITLG",    12, "MXL", "positive", "blue"),
    # Neutral controls — picked dynamically by build_tasks.py.
    # Match the order in tasks.txt.
    ("AQP3",      9, "GWD", "neutral",  "gray"),
    ("MYO1F",    19, "CEU", "neutral",  "gray"),
    ("NUP35",     2, "CDX", "neutral",  "gray"),
    ("DSC1",     18, "ITU", "neutral",  "gray"),
    ("SEC13",     3, "PEL", "neutral",  "gray"),
]

GROUP_BG = {
    "novel":    "#fff6e6",
    "positive": "#e8f3ff",
    "neutral":  "#f4f4f4",
}


def load_task(gene, pop, group):
    path = os.path.join(THREE_METHOD_DIR, f"{gene}_{pop}_{group}.npz")
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


def plot_gamma_smc_cu_panel(ax, focal, control, gene, focal_pop, color):
    """Per-pair TMRCA traces from gamma_smc_cu pairwise mode."""
    if focal is None or "gamma_smc_cu_mean" not in focal.files:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
        return
    pos = focal["gamma_smc_cu_positions"]
    mean = focal["gamma_smc_cu_mean"]   # (n_filtered_sites, n_pairs)
    pos_mb = pos / 1e6
    # log10 generations, clipped
    log_gen = np.log10(np.clip(mean, 10, 1e6))

    # Plot focal pop pairs in color, with low alpha
    for j in range(min(20, log_gen.shape[1])):
        ax.plot(pos_mb, log_gen[:, j], color=color, alpha=0.18, linewidth=0.5)

    # YRI control overlay
    if control is not None and "gamma_smc_cu_mean" in control.files:
        cpos = control["gamma_smc_cu_positions"]
        cmean = control["gamma_smc_cu_mean"]
        clog = np.log10(np.clip(cmean, 10, 1e6))
        for j in range(min(20, clog.shape[1])):
            ax.plot(cpos / 1e6, clog[:, j], color="gray", alpha=0.12, linewidth=0.4)

    # Gene body shading
    if focal is not None and "gstart" in focal.files:
        ax.axvspan(int(focal["gstart"]) / 1e6, int(focal["gend"]) / 1e6,
                   alpha=0.18, color=color, lw=0)

    ax.set_ylim(2, 6)
    ax.set_yticks([2, 3, 4, 5, 6])
    ax.set_yticklabels(["100", "1k", "10k", "100k", "1M"], fontsize=6)
    ax.set_xticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_cxt_panel(ax, focal, control, gene, focal_pop, color):
    """cxt regional log-TMRCA per pair, averaged over reps."""
    if focal is None or "cxt_log_tmrca" not in focal.files:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
        return
    log_t = focal["cxt_log_tmrca"]   # (n_pairs, n_reps, n_blocks=1, n_windows)
    win_lo = float(focal["win_lo"])
    win_hi = float(focal["win_hi"])

    n_pairs = log_t.shape[0]
    n_windows = log_t.shape[-1]
    # Average over reps and blocks
    if log_t.ndim == 4:
        per_pair_curve = log_t.mean(axis=(1, 2))   # (n_pairs, n_windows)
    elif log_t.ndim == 3:
        per_pair_curve = log_t.mean(axis=1)
    else:
        per_pair_curve = log_t

    # cxt log_tmrca is in natural log (ln) of generations from cxt source
    # converted via to_log_times. Convert to log10 to match other panels.
    per_pair_curve_log10 = per_pair_curve / np.log(10)

    x = np.linspace(win_lo / 1e6, win_hi / 1e6, n_windows)
    for j in range(n_pairs):
        ax.plot(x, per_pair_curve_log10[j], color=color, alpha=0.45, linewidth=0.6)

    if control is not None and "cxt_log_tmrca" in control.files:
        clog = control["cxt_log_tmrca"]
        if clog.ndim == 4:
            cpc = clog.mean(axis=(1, 2))
        elif clog.ndim == 3:
            cpc = clog.mean(axis=1)
        else:
            cpc = clog
        cpc_log10 = cpc / np.log(10)
        for j in range(cpc.shape[0]):
            ax.plot(x, cpc_log10[j], color="gray", alpha=0.25, linewidth=0.5)

    if focal is not None and "gstart" in focal.files:
        ax.axvspan(int(focal["gstart"]) / 1e6, int(focal["gend"]) / 1e6,
                   alpha=0.18, color=color, lw=0)

    ax.set_ylim(2, 6)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_asmc_panel(ax, focal, control, gene, focal_pop, color):
    """ASMC per-pair posterior mean TMRCA traces."""
    if focal is None or "asmc_mean" not in focal.files:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8, color="gray")
        ax.set_yticks([])
        ax.set_xticks([])
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        return
    pos = focal["asmc_positions"]
    mean = focal["asmc_mean"]   # (n_pairs, n_sites) — note transposed vs gamma_smc_cu
    pos_mb = pos / 1e6
    log_gen = np.log10(np.clip(mean, 10, 1e6))

    # Plot focal pop pairs in color
    for j in range(min(20, log_gen.shape[0])):
        ax.plot(pos_mb, log_gen[j, :], color=color, alpha=0.18, linewidth=0.5)

    # YRI control overlay
    if control is not None and "asmc_mean" in control.files:
        cpos = control["asmc_positions"]
        cmean = control["asmc_mean"]
        clog = np.log10(np.clip(cmean, 10, 1e6))
        for j in range(min(20, clog.shape[0])):
            ax.plot(cpos / 1e6, clog[j, :], color="gray", alpha=0.12, linewidth=0.4)

    if focal is not None and "gstart" in focal.files:
        ax.axvspan(int(focal["gstart"]) / 1e6, int(focal["gend"]) / 1e6,
                   alpha=0.18, color=color, lw=0)

    ax.set_ylim(2, 6)
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def main():
    n_genes = len(GENES)
    fig, axes = plt.subplots(
        n_genes, 4,
        figsize=(11, 0.75 * n_genes + 1.5),
        gridspec_kw={"width_ratios": [0.9, 2, 2, 2], "wspace": 0.08, "hspace": 0.25},
    )

    for ri, (gene, chr_, focal_pop, group, color) in enumerate(GENES):
        # Label column
        ax_label = axes[ri, 0]
        ax_label.set_facecolor(GROUP_BG[group])
        ax_label.text(
            0.5, 0.55, f"$\\it{{{gene}}}$",
            ha="center", va="center", fontsize=11, fontweight="bold",
            transform=ax_label.transAxes,
        )
        ax_label.text(
            0.5, 0.30, f"chr{chr_}",
            ha="center", va="center", fontsize=8, color="#444",
            transform=ax_label.transAxes,
        )
        ax_label.text(
            0.5, 0.15, f"{focal_pop} | {group}",
            ha="center", va="center", fontsize=7, color="#666",
            transform=ax_label.transAxes,
        )
        ax_label.set_xticks([])
        ax_label.set_yticks([])
        for s in ("top", "right", "left", "bottom"):
            ax_label.spines[s].set_visible(False)

        focal = load_task(gene, focal_pop, group)
        control = load_task(gene, "YRI", "control")

        plot_gamma_smc_cu_panel(axes[ri, 1], focal, control, gene, focal_pop, color)
        plot_cxt_panel(axes[ri, 2], focal, control, gene, focal_pop, color)
        plot_asmc_panel(axes[ri, 3], focal, control, gene, focal_pop, color)

        if ri == 0:
            axes[ri, 1].set_title("gamma_smc_cu pairwise", fontsize=10, loc="left")
            axes[ri, 2].set_title("cxt regional", fontsize=10, loc="left")
            axes[ri, 3].set_title("ASMC", fontsize=10, loc="left")
        if ri == n_genes - 1:
            axes[ri, 1].set_xlabel("Chromosome position (Mb)", fontsize=8)
            axes[ri, 1].set_xticks([])
            axes[ri, 2].set_xlabel("Chromosome position (Mb)", fontsize=8)
            axes[ri, 2].set_xticks([])

    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "fig4_three_method.png")
    out_pdf = os.path.join(OUT_DIR, "fig4_three_method.pdf")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
