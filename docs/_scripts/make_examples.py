"""Generate the docs example figures.

Runs a small reproducible msprime simulation and shows what each of the
three tmrca_cu.infer() modes produces:

  examples_mean.png      -- mean_only=True
  examples_ci.png        -- mean_only=False
  examples_posterior.png -- return_posterior=True (full Gamma per site)

These are deliberately tiny tutorial figures, not scientific results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.stats import gamma

import msprime
import tmrca_cu

# Reproducibility / cosmetic constants
SEED = 42
N_SAMPLES = 8
SEQ_LEN = 500_000
RECOMB_RATE = 1e-8
MUT_RATE = 1.25e-8
NE = 10_000.0
PAIR = (0, 1)


def simulate():
    ts = msprime.sim_ancestry(
        samples=N_SAMPLES,
        sequence_length=SEQ_LEN,
        recombination_rate=RECOMB_RATE,
        population_size=NE,
        random_seed=SEED,
    )
    ts = msprime.sim_mutations(ts, rate=MUT_RATE, random_seed=SEED + 1)
    return ts


def truth_per_site(ts, pair):
    """Per-site ground-truth TMRCA from the msprime tree at each variant site."""
    pos = np.array([v.position for v in ts.variants()], dtype=np.float64)
    truth = np.empty(len(pos))
    tree_iter = ts.trees()
    tree = next(tree_iter)
    for i, p in enumerate(pos):
        while p >= tree.interval.right:
            tree = next(tree_iter)
        truth[i] = tree.tmrca(*pair)
    return pos, truth


def setup_axes(ax, pos_mb, truth, title):
    ax.plot(pos_mb, truth, color="#444", lw=0.7, label="ground truth (msprime)", zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("position (Mb)")
    ax.set_ylabel("TMRCA (generations)")
    ax.set_title(title, loc="left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_mean(out_dir, pos_mb, truth, mean):
    fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=150)
    setup_axes(ax, pos_mb, truth, "infer(..., mean_only=True)")
    ax.plot(pos_mb, mean, color="#3182bd", lw=1.2, label="posterior mean")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "examples_mean.png")
    plt.close(fig)


def plot_ci(out_dir, pos_mb, truth, mean, lower, upper):
    fig, ax = plt.subplots(figsize=(7.0, 3.4), dpi=150)
    setup_axes(ax, pos_mb, truth, "infer(..., mean_only=False)")
    ax.fill_between(
        pos_mb, lower, upper,
        color="#3182bd", alpha=0.20, lw=0,
        label="95% CI (Wilson-Hilferty)",
    )
    ax.plot(pos_mb, mean, color="#3182bd", lw=1.2, label="posterior mean")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "examples_ci.png")
    plt.close(fig)


def plot_posterior(out_dir, pos_mb, truth, mean, alpha, beta):
    """Heatmap of the per-site Gamma posterior P(T | site).

    Top: mean + multiple credible bands derived from (alpha, beta).
    Bottom: log-density heatmap over a TMRCA grid.
    """
    # Custom credible intervals straight from (alpha, beta) — the whole point
    # of return_posterior=True.
    quantile_pairs = [(0.025, 0.975), (0.10, 0.90), (0.25, 0.75)]
    band_labels = ["95%", "80%", "50%"]
    band_alphas = [0.15, 0.20, 0.30]

    bands = []
    for q_lo, q_hi in quantile_pairs:
        lo = gamma(alpha, scale=2.0 * NE / beta).ppf(q_lo)
        hi = gamma(alpha, scale=2.0 * NE / beta).ppf(q_hi)
        bands.append((lo, hi))

    # Build a (T, site) heatmap of log P(T | site).
    t_grid = np.geomspace(
        max(1.0, np.percentile(mean, 1) / 5.0),
        np.percentile(mean, 99) * 5.0,
        300,
    )
    # PDF in real generations: T ~ Gamma(alpha, scale=2*Ne/beta)
    scale = 2.0 * NE / beta  # (n_sites,)
    # Log-density grid: shape (n_T, n_sites)
    log_pdf = np.empty((t_grid.size, alpha.size))
    for i in range(alpha.size):
        log_pdf[:, i] = gamma(alpha[i], scale=scale[i]).logpdf(t_grid)
    pdf = np.exp(log_pdf - log_pdf.max(axis=0, keepdims=True))  # column-normalized

    fig, axes = plt.subplots(
        2, 1, figsize=(7.0, 5.6), dpi=150, sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.4]},
    )
    ax0, ax1 = axes

    # Top panel: mean + custom credible bands
    setup_axes(ax0, pos_mb, truth, "infer(..., return_posterior=True)")
    ax0.set_xlabel("")
    for (lo, hi), label, a in zip(bands, band_labels, band_alphas):
        ax0.fill_between(
            pos_mb, lo, hi, color="#3182bd", alpha=a, lw=0, label=f"{label} CI",
        )
    ax0.plot(pos_mb, mean, color="#3182bd", lw=1.2, label="posterior mean")
    ax0.legend(frameon=False, loc="upper right", fontsize=8, ncol=2)

    # Bottom panel: per-site posterior density heatmap
    extent = [pos_mb.min(), pos_mb.max(), t_grid.min(), t_grid.max()]
    im = ax1.imshow(
        pdf, origin="lower", aspect="auto", extent=extent,
        cmap="Blues", interpolation="nearest", vmin=0.0, vmax=1.0,
    )
    ax1.plot(pos_mb, truth, color="#444", lw=0.7, label="ground truth")
    ax1.plot(pos_mb, mean, color="#e34a33", lw=0.9, label="posterior mean")
    ax1.set_yscale("log")
    ax1.set_xlabel("position (Mb)")
    ax1.set_ylabel("TMRCA (generations)")
    ax1.set_title(
        "per-site posterior density P(T | site)  (column-normalized)",
        loc="left",
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(frameon=False, loc="upper right", fontsize=8)

    cbar = fig.colorbar(im, ax=ax1, fraction=0.025, pad=0.02)
    cbar.set_label("normalized density", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "examples_posterior.png")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--flow-field", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("simulating ...", flush=True)
    ts = simulate()
    pos, truth = truth_per_site(ts, PAIR)
    pos_mb = pos / 1e6
    print(f"  n_sites = {ts.num_sites}", flush=True)

    G = ts.genotype_matrix().T.astype(np.uint8)
    positions = pos.astype(np.float64)
    pairs = [PAIR]

    print("running infer(mean_only=True) ...", flush=True)
    r1 = tmrca_cu.infer(
        G, positions, pairs=pairs, Ne=NE, mu=MUT_RATE, rho=RECOMB_RATE,
        flow_field_path=args.flow_field, mean_only=True,
    )
    plot_mean(out_dir, pos_mb, truth, r1["mean"][:, 0])

    print("running infer(mean_only=False) ...", flush=True)
    r2 = tmrca_cu.infer(
        G, positions, pairs=pairs, Ne=NE, mu=MUT_RATE, rho=RECOMB_RATE,
        flow_field_path=args.flow_field, mean_only=False,
    )
    plot_ci(out_dir, pos_mb, truth, r2["mean"][:, 0], r2["lower"][:, 0], r2["upper"][:, 0])

    print("running infer(return_posterior=True) ...", flush=True)
    r3 = tmrca_cu.infer(
        G, positions, pairs=pairs, Ne=NE, mu=MUT_RATE, rho=RECOMB_RATE,
        flow_field_path=args.flow_field, return_posterior=True,
    )
    plot_posterior(
        out_dir, pos_mb, truth,
        r3["mean"][:, 0],
        r3["posterior_alpha"][:, 0],
        r3["posterior_beta"][:, 0],
    )

    print(f"wrote figures to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
