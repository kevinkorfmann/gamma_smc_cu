"""
Generate publication figure: deep tmrca.cu vs gamma_smc comparison.

Reads bench_gamma_deep.npz and produces a 3-panel figure:
  (a) Wall-clock vs n_haplotypes at fixed seq_len (log-log)
  (b) Wall-clock vs seq_length at fixed n_hap (log-log)
  (c) Speedup heatmap across the full grid

Usage:
    pixi run python benchmarks/make_fig_gamma_deep.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker

# ── Style ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

C = {
    "blue": "#3B7DD8",
    "coral": "#E8634F",
    "gold": "#D4A843",
    "teal": "#4CA69E",
    "grey": "#888888",
    "purple": "#8B6BAE",
}


def main():
    data = np.load("benchmarks/bench_gamma_deep.npz")
    n_haps = data["n_haplotypes"]
    seq_lens = data["seq_lengths"]
    gamma_wall = data["gamma_wall"]
    cu_wall = data["cu_wall"]
    n_snps = data["n_snps"]

    n_pairs = np.array([nh * (nh - 1) // 2 for nh in n_haps])

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))

    # ────────────────────────────────────────────
    # (a) Wall-clock vs n_haplotypes (1 Mb)
    # ────────────────────────────────────────────
    ax = axes[0]
    j = 1  # 1 Mb index
    mask_g = ~np.isnan(gamma_wall[:, j])
    mask_c = ~np.isnan(cu_wall[:, j])

    ax.plot(n_haps[mask_g], gamma_wall[mask_g, j], 'o-',
            color=C["coral"], ms=4, label="gamma_smc", zorder=3)
    ax.plot(n_haps[mask_c], cu_wall[mask_c, j], 's-',
            color=C["blue"], ms=4, label="tmrca.cu", zorder=3)

    # Annotate speedups
    for i in range(len(n_haps)):
        if not np.isnan(gamma_wall[i, j]) and not np.isnan(cu_wall[i, j]):
            sp = gamma_wall[i, j] / cu_wall[i, j]
            y_mid = np.sqrt(gamma_wall[i, j] * cu_wall[i, j])
            ax.annotate(f"{sp:.0f}x", (n_haps[i], y_mid),
                        fontsize=6, ha="center", va="center",
                        color=C["grey"], fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of haplotypes")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("(a) Scaling with samples\n(1 Mb)", fontsize=8, fontweight="bold")
    ax.legend(fontsize=6, loc="upper left", frameon=False)
    ax.set_xticks(n_haps)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlim(1.5, 300)

    # ────────────────────────────────────────────
    # (b) Wall-clock vs seq_length (n_hap=50)
    # ────────────────────────────────────────────
    ax = axes[1]
    # Use n_hap=50 (index 3) for a good middle ground
    i = 3  # n_hap=50

    seq_mb = seq_lens / 1e6
    mask_g = ~np.isnan(gamma_wall[i, :])
    mask_c = ~np.isnan(cu_wall[i, :])

    ax.plot(seq_mb[mask_g], gamma_wall[i, mask_g], 'o-',
            color=C["coral"], ms=4, label="gamma_smc", zorder=3)
    ax.plot(seq_mb[mask_c], cu_wall[i, mask_c], 's-',
            color=C["blue"], ms=4, label="tmrca.cu", zorder=3)

    # Annotate speedups
    for jj in range(len(seq_lens)):
        if not np.isnan(gamma_wall[i, jj]) and not np.isnan(cu_wall[i, jj]):
            sp = gamma_wall[i, jj] / cu_wall[i, jj]
            y_mid = np.sqrt(gamma_wall[i, jj] * cu_wall[i, jj])
            ax.annotate(f"{sp:.0f}x", (seq_mb[jj], y_mid),
                        fontsize=6, ha="center", va="center",
                        color=C["grey"], fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sequence length (Mb)")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(f"(b) Scaling with seq length\n(n = {n_haps[i]} haplotypes)", fontsize=8, fontweight="bold")
    ax.legend(fontsize=6, loc="upper left", frameon=False)
    ax.set_xticks(seq_mb)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    # ────────────────────────────────────────────
    # (c) Speedup heatmap
    # ────────────────────────────────────────────
    ax = axes[2]
    speedup = gamma_wall / cu_wall
    # Mask NaN
    speedup_masked = np.ma.masked_invalid(speedup)

    im = ax.imshow(speedup_masked, aspect="auto",
                   norm=LogNorm(vmin=10, vmax=500),
                   cmap="YlOrRd_r", origin="lower")

    # Annotate each cell
    for i in range(len(n_haps)):
        for j in range(len(seq_lens)):
            val = speedup[i, j]
            if not np.isnan(val):
                color = "white" if val > 100 else "black"
                ax.text(j, i, f"{val:.0f}x", ha="center", va="center",
                        fontsize=6.5, fontweight="bold", color=color)
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=7, color=C["grey"])

    ax.set_xticks(range(len(seq_lens)))
    ax.set_xticklabels([f"{sl/1e6:.0f}" if sl >= 1e6 else f"{sl/1e3:.0f}k"
                        for sl in seq_lens])
    ax.set_yticks(range(len(n_haps)))
    ax.set_yticklabels(n_haps)
    ax.set_xlabel("Sequence length (Mb)")
    ax.set_ylabel("Haplotypes")
    ax.set_title("(c) Speedup (gamma_smc / tmrca.cu)", fontsize=8, fontweight="bold")

    # Re-enable spines for heatmap
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Speedup", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    plt.tight_layout(w_pad=1.5)

    for ext in ("pdf", "png"):
        fig.savefig(f"benchmarks/fig7_gamma_deep.{ext}")
    print("Saved benchmarks/fig7_gamma_deep.{pdf,png}")
    plt.close()

    # ────────────────────────────────────────────
    # Second figure: throughput and per-pair cost
    # ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.4))

    # (a) Throughput (M site·pairs/s) vs n_haplotypes at 1 Mb
    ax = axes[0]
    j = 1  # 1 Mb
    for i in range(len(n_haps)):
        nh = n_haps[i]
        np_i = nh * (nh - 1) // 2
        snps = n_snps[i, j]
        if not np.isnan(gamma_wall[i, j]):
            rate_g = np_i * snps / gamma_wall[i, j] / 1e6
            ax.bar(i - 0.18, rate_g, width=0.35, color=C["coral"],
                   edgecolor="white", linewidth=0.3, zorder=3)
        if not np.isnan(cu_wall[i, j]):
            rate_c = np_i * snps / cu_wall[i, j] / 1e6
            ax.bar(i + 0.18, rate_c, width=0.35, color=C["blue"],
                   edgecolor="white", linewidth=0.3, zorder=3)

    ax.set_xticks(range(len(n_haps)))
    ax.set_xticklabels(n_haps)
    ax.set_xlabel("Number of haplotypes")
    ax.set_ylabel("Throughput (M site·pairs/s)")
    ax.set_title("(a) Throughput at 1 Mb", fontsize=8, fontweight="bold")
    ax.set_yscale("log")

    # Manual legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=C["coral"], label="gamma_smc"),
        Patch(facecolor=C["blue"], label="tmrca.cu"),
    ], fontsize=6, frameon=False, loc="upper left")

    # (b) Per-pair amortized cost vs n_pairs
    ax = axes[1]
    j = 1  # 1 Mb
    pp_gamma = []
    pp_cu = []
    pp_nhaps = []
    for i in range(len(n_haps)):
        nh = n_haps[i]
        np_i = nh * (nh - 1) // 2
        if np_i == 0:
            continue
        pp_nhaps.append(np_i)
        if not np.isnan(gamma_wall[i, j]):
            pp_gamma.append(gamma_wall[i, j] / np_i * 1e3)
        else:
            pp_gamma.append(np.nan)
        if not np.isnan(cu_wall[i, j]):
            pp_cu.append(cu_wall[i, j] / np_i * 1e3)
        else:
            pp_cu.append(np.nan)

    pp_nhaps = np.array(pp_nhaps)
    pp_gamma = np.array(pp_gamma)
    pp_cu = np.array(pp_cu)

    mask_g = ~np.isnan(pp_gamma)
    mask_c = ~np.isnan(pp_cu)

    ax.plot(pp_nhaps[mask_g], pp_gamma[mask_g], 'o-',
            color=C["coral"], ms=4, label="gamma_smc")
    ax.plot(pp_nhaps[mask_c], pp_cu[mask_c], 's-',
            color=C["blue"], ms=4, label="tmrca.cu")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of pairs")
    ax.set_ylabel("Time per pair (ms)")
    ax.set_title("(b) Amortized cost per pair\n(1 Mb)", fontsize=8, fontweight="bold")
    ax.legend(fontsize=6, frameon=False, loc="upper right")

    plt.tight_layout(w_pad=1.0)
    for ext in ("pdf", "png"):
        fig.savefig(f"benchmarks/fig8_gamma_throughput.{ext}")
    print("Saved benchmarks/fig8_gamma_throughput.{pdf,png}")
    plt.close()


if __name__ == "__main__":
    main()
