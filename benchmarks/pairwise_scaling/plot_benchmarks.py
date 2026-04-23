#!/usr/bin/env python3
"""Plot pairwise scaling: gamma_smc_cu vs gamma_smc vs ASMC on chr22 YRI.

Solid lines + filled markers = measured. Dashed + open markers = extrapolated.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

REPO = "/Users/kevinkorfmann/Projects/gamma_smc_cu"
DATA_DIR = os.path.join(REPO, "benchmarks/pairwise_scaling")
OUT_DIR = os.path.join(REPO, "docs_local/manuscript/v4.1/figures")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
})

df = pd.read_csv(os.path.join(DATA_DIR, "results.csv"))

fig, ax = plt.subplots(figsize=(6, 4.5))

colors = {"gamma_smc_cu": "#2ca02c", "gamma_smc": "#1f77b4", "ASMC": "#e67e22"}
markers = {"gamma_smc_cu": "s", "gamma_smc": "^", "ASMC": "o"}
labels = {"gamma_smc_cu": "gamma_smc_cu (GPU)", "gamma_smc": "gamma_smc (CPU)",
          "ASMC": "ASMC (CPU)"}

for method in ["gamma_smc_cu", "gamma_smc", "ASMC"]:
    sub = df[df["method"] == method].sort_values("n_pairs")
    if len(sub) == 0:
        continue
    measured = sub[sub["type"] == "measured"]
    extrap = sub[sub["type"] == "extrapolated"]

    if len(measured):
        ax.plot(measured["n_pairs"], measured["seconds"],
                f"{markers[method]}-", color=colors[method], ms=6, lw=2,
                label=labels[method])
    if len(extrap):
        bridge = pd.concat([measured.tail(1), extrap]).sort_values("n_pairs")
        ax.plot(bridge["n_pairs"], bridge["seconds"],
                f"{markers[method]}--", color=colors[method], ms=5, lw=1.5,
                alpha=0.5, markerfacecolor="white", markeredgecolor=colors[method],
                markeredgewidth=1.5)

# Speedup annotations at largest measured gamma_smc_cu point
tcu = df[df["method"] == "gamma_smc_cu"].sort_values("n_pairs")
for method in ["gamma_smc", "ASMC"]:
    other = df[df["method"] == method].sort_values("n_pairs")
    # Find the largest pair count where both have data
    common = set(tcu["n_pairs"]) & set(other["n_pairs"])
    if not common:
        continue
    n_max = max(common)
    t_tcu = float(tcu[tcu["n_pairs"] == n_max]["seconds"].values[0])
    t_other = float(other[other["n_pairs"] == n_max]["seconds"].values[0])
    speedup = t_other / t_tcu
    vy = 12 if method == "ASMC" else -12
    ax.annotate(f'{speedup:.0f}×',
                (n_max, t_tcu),
                textcoords="offset points", xytext=(10, vy),
                fontsize=9, color=colors["gamma_smc_cu"], fontweight="bold")

# Time reference lines
for t, label in [(60, "1 min"), (3600, "1 hour")]:
    ax.axhline(t, color="#e0e0e0", lw=0.8, zorder=0)
    ax.text(0.8, t * 1.2, label, fontsize=8, color="#999")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of haplotype pairs")
ax.set_ylabel("Wall time (seconds)")
ax.legend(fontsize=9, frameon=False, loc="upper left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

plt.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(os.path.join(OUT_DIR, f"fig_scaling.{ext}"),
                dpi=250, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(DATA_DIR, f"scaling_plot.{ext}"),
                dpi=250, bbox_inches="tight", facecolor="white")
    print(f"Saved fig_scaling.{ext}")
plt.close()
