"""
Aggregate per-config JSON results and produce the test-suite figure + CSV.

Produces:
  figures/test_suite_stdpopsim.{pdf,png}
  figures/test_suite_summary.csv
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(HERE, "figures")

# Style copied from benchmarks/bench_demographics.py
rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "figure.dpi": 300, "savefig.dpi": 300, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})

C_OURS = "#3182bd"     # tmrca.cu
C_SCHW = "#e6550d"     # gamma_smc
C_TRUTH = "#252525"

# One distinctive color per species for the scatter panels.
SPECIES_COLORS = {
    "HomSap": "#1f77b4",
    "PonAbe": "#17becf",
    "PanTro": "#8c564b",
    "DroMel": "#2ca02c",
    "AraTha": "#9467bd",
    "AnoGam": "#d62728",
    "CanFam": "#e377c2",
    "BosTau": "#bcbd22",
}


def load_results():
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, "config_*.json")))
    if not files:
        sys.exit(f"No result JSONs found in {RESULTS_DIR}")
    rows = []
    for fp in files:
        with open(fp) as f:
            rows.append(json.load(f))
    df = pd.DataFrame(rows)
    # Stable ordering: species first, then model name.
    df["species"] = df["species"].astype(str)
    df = df.sort_values(["species", "model_id"]).reset_index(drop=True)
    return df


def _label(row):
    model = row["model_id"]
    if len(model) > 28:
        model = model[:25] + "..."
    return f"{row['species']}  {model}"


def plot(df):
    os.makedirs(FIG_DIR, exist_ok=True)

    fig = plt.figure(figsize=(8.5, 7.5))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.3, 1.0],
        height_ratios=[1, 1],
        hspace=0.45, wspace=0.45,
        left=0.22, right=0.97, top=0.95, bottom=0.07,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])

    # ---- shared y-axis for panels (a) and (b) --------------------------
    labels = [_label(r) for _, r in df.iterrows()]
    y = np.arange(len(df))[::-1]  # top-to-bottom

    # ---------- panel (a): accuracy dots with IQR whiskers --------------
    r_t_med = df["r_gamma_smc_cu_median"].to_numpy(float)
    r_t_lo = df["r_gamma_smc_cu_q25"].to_numpy(float)
    r_t_hi = df["r_gamma_smc_cu_q75"].to_numpy(float)
    r_g_med = df["r_gsmc_median"].to_numpy(float)
    r_g_lo = df["r_gsmc_q25"].to_numpy(float)
    r_g_hi = df["r_gsmc_q75"].to_numpy(float)

    # dotted connector between the two dots for visual grouping
    for yy, a, b in zip(y, r_g_med, r_t_med):
        if np.isfinite(a) and np.isfinite(b):
            ax_a.plot([min(a, b), max(a, b)], [yy, yy],
                      color="#bdbdbd", linewidth=0.4, linestyle=":", zorder=1)

    ax_a.errorbar(r_t_med, y, xerr=[r_t_med - r_t_lo, r_t_hi - r_t_med],
                  fmt="o", color=C_OURS, ecolor=C_OURS, elinewidth=0.6,
                  markersize=3.2, capsize=1.2, label="tmrca.cu", zorder=3)
    ax_a.errorbar(r_g_med, y, xerr=[r_g_med - r_g_lo, r_g_hi - r_g_med],
                  fmt="s", color=C_SCHW, ecolor=C_SCHW, elinewidth=0.6,
                  markersize=2.8, capsize=1.2,
                  label="gamma_smc (Schweiger & Durbin, 2023)", zorder=2)

    ax_a.set_yticks(y)
    ax_a.set_yticklabels(labels)
    ax_a.set_xlabel("Pearson r of log TMRCA (median, IQR across pairs)")
    ax_a.set_xlim(0.3, 1.0)
    ax_a.set_title("a  accuracy per config", loc="left", fontweight="bold", fontsize=8)
    ax_a.grid(axis="x", alpha=0.15, linewidth=0.3)
    ax_a.legend(loc="lower left", fontsize=5.5)

    # species-color strip on the left margin
    for i, (_, row) in enumerate(df.iterrows()):
        yy = y[i]
        c = SPECIES_COLORS.get(row["species"], "#888888")
        ax_a.scatter(0.305, yy, color=c, s=14, marker="s",
                     clip_on=False, zorder=4, edgecolors="none")

    # ---------- panel (b): wall time dots (log x) -----------------------
    t_t = df["t_gamma_smc_cu_total"].to_numpy(float)
    t_g = df["t_gsmc_total"].to_numpy(float)
    t_g_c = df["t_gsmc_compute"].to_numpy(float)

    for yy, a, b in zip(y, t_g, t_t):
        if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
            ax_b.plot([min(a, b), max(a, b)], [yy, yy],
                      color="#bdbdbd", linewidth=0.4, linestyle=":", zorder=1)

    ax_b.scatter(t_t, y, color=C_OURS, s=14, marker="o",
                 label="tmrca.cu", zorder=3)
    ax_b.scatter(t_g, y, color=C_SCHW, s=12, marker="s",
                 label="gamma_smc total", zorder=2)
    ax_b.scatter(t_g_c, y, facecolors="none", edgecolors=C_SCHW, s=12,
                 marker="s", linewidth=0.6, label="gamma_smc compute", zorder=2)

    ax_b.set_xscale("log")
    ax_b.set_yticks(y)
    ax_b.set_yticklabels(labels)
    ax_b.set_xlabel("Wall time (s)")
    ax_b.set_title("b  speed per config", loc="left", fontweight="bold", fontsize=8)
    ax_b.grid(axis="x", which="both", alpha=0.15, linewidth=0.3)
    ax_b.legend(loc="upper right", fontsize=5.5)

    for i, (_, row) in enumerate(df.iterrows()):
        yy = y[i]
        c = SPECIES_COLORS.get(row["species"], "#888888")
        # species strip just left of the axis (use figure coords)
        xlim = ax_b.get_xlim()
        ax_b.scatter(xlim[0], yy, color=c, s=14, marker="s",
                     clip_on=False, zorder=4, edgecolors="none")

    # ---------- panel (c): accuracy parity scatter -----------------------
    for _, row in df.iterrows():
        c = SPECIES_COLORS.get(row["species"], "#888888")
        ax_c.scatter(row["r_gsmc_median"], row["r_gamma_smc_cu_median"],
                     color=c, s=22, edgecolors="white", linewidth=0.4,
                     label=row["species"])
    lo = min(
        df["r_gsmc_median"].min(skipna=True),
        df["r_gamma_smc_cu_median"].min(skipna=True),
    ) - 0.05
    lo = max(lo, 0.3)
    ax_c.plot([lo, 1.0], [lo, 1.0], color="#999999", linewidth=0.5, linestyle="--")
    ax_c.set_xlim(lo, 1.0)
    ax_c.set_ylim(lo, 1.0)
    ax_c.set_xlabel("gamma_smc  r of log TMRCA")
    ax_c.set_ylabel("tmrca.cu  r of log TMRCA")
    ax_c.set_title("c  accuracy parity", loc="left", fontweight="bold", fontsize=8)
    ax_c.set_aspect("equal", adjustable="box")
    ax_c.grid(alpha=0.15, linewidth=0.3)
    # de-duplicate legend
    handles, labs = ax_c.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labs):
        seen.setdefault(l, h)
    ax_c.legend(
        seen.values(), seen.keys(),
        loc="lower right", fontsize=5.0, ncol=1, handletextpad=0.3,
    )

    # ---------- panel (d): speed parity scatter (log-log) ----------------
    mask = np.isfinite(t_t) & np.isfinite(t_g) & (t_t > 0) & (t_g > 0)
    xs = t_g[mask]; ys = t_t[mask]
    if len(xs) > 0:
        lo_xy = min(xs.min(), ys.min()) * 0.5
        hi_xy = max(xs.max(), ys.max()) * 2.0
    else:
        lo_xy, hi_xy = 0.01, 100.0
    for i, (_, row) in enumerate(df.iterrows()):
        if not (mask[i]):
            continue
        c = SPECIES_COLORS.get(row["species"], "#888888")
        ax_d.scatter(t_g[i], t_t[i], color=c, s=22,
                     edgecolors="white", linewidth=0.4)

    ref = np.array([lo_xy, hi_xy])
    ax_d.plot(ref, ref, color="#999999", linewidth=0.5, linestyle="--",
              label="1:1")
    for fold, style in ((10, ":"), (100, "-."), (1000, ":")):
        ax_d.plot(ref, ref / fold, color="#999999", linewidth=0.4,
                  linestyle=style, label=f"{fold}x")
    ax_d.set_xscale("log")
    ax_d.set_yscale("log")
    ax_d.set_xlim(lo_xy, hi_xy)
    ax_d.set_ylim(lo_xy, hi_xy)
    ax_d.set_xlabel("gamma_smc wall time (s)")
    ax_d.set_ylabel("tmrca.cu wall time (s)")
    ax_d.set_title("d  speed parity", loc="left", fontweight="bold", fontsize=8)
    ax_d.set_aspect("equal", adjustable="box")
    ax_d.grid(which="both", alpha=0.15, linewidth=0.3)
    ax_d.legend(loc="upper left", fontsize=5.0, ncol=2, handletextpad=0.3)

    for fmt in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"test_suite_stdpopsim.{fmt}")
        fig.savefig(out, dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"wrote {out}")
    plt.close(fig)


def main():
    df = load_results()
    os.makedirs(FIG_DIR, exist_ok=True)
    csv_path = os.path.join(FIG_DIR, "test_suite_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}  ({len(df)} configs)")

    # Short text summary
    n_ok = int((df["status"] == "ok").sum()) if "status" in df else len(df)
    print(f"\nLoaded {n_ok}/{len(df)} successful configs")
    if len(df):
        r_t = df["r_gamma_smc_cu_median"].dropna()
        r_g = df["r_gsmc_median"].dropna()
        sp = df["speedup_total"].dropna()
        print(f"  r_gamma_smc_cu median across configs: {r_t.median():.3f}  "
              f"(min={r_t.min():.3f}, max={r_t.max():.3f})")
        print(f"  r_gsmc     median across configs: {r_g.median():.3f}  "
              f"(min={r_g.min():.3f}, max={r_g.max():.3f})")
        if len(sp):
            print(f"  speedup_total median: {sp.median():.1f}x  "
                  f"(min={sp.min():.1f}x, max={sp.max():.1f}x)")

    plot(df)


if __name__ == "__main__":
    main()
