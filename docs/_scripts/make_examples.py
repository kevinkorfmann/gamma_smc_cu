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
    """Mean + custom credible bands across position, plus the actual
    Gamma PDF at three representative sites picked at low / median / high
    posterior uncertainty.
    """
    # Custom credible bands straight from (alpha, beta) — the whole point
    # of return_posterior=True.
    quantile_pairs = [(0.025, 0.975), (0.10, 0.90), (0.25, 0.75)]
    band_labels = ["95%", "80%", "50%"]
    band_alphas = [0.15, 0.22, 0.32]

    bands = []
    for q_lo, q_hi in quantile_pairs:
        lo = gamma(alpha, scale=2.0 * NE / beta).ppf(q_lo)
        hi = gamma(alpha, scale=2.0 * NE / beta).ppf(q_hi)
        bands.append((lo, hi))

    # Pick three representative sites by 95% CI *relative width*
    # (= width / mean), which is the most intuitive uncertainty measure.
    ci_lo, ci_hi = bands[0]
    rel_width = (ci_hi - ci_lo) / np.maximum(mean, 1.0)
    rank = np.argsort(rel_width)
    pick_idx = [
        int(rank[int(0.10 * len(rank))]),  # most certain (10th percentile)
        int(rank[int(0.50 * len(rank))]),  # median
        int(rank[int(0.90 * len(rank))]),  # least certain (90th percentile)
    ]
    pick_labels = ["tight", "typical", "broad"]
    pick_colors = ["#2ca25f", "#3182bd", "#e34a33"]

    fig = plt.figure(figsize=(7.0, 5.6), dpi=150)
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1.0, 0.9], hspace=0.55, wspace=0.35,
    )
    ax_top = fig.add_subplot(gs[0, :])
    ax_bot = [fig.add_subplot(gs[1, k]) for k in range(3)]

    # ----- top panel: mean + credible bands -----
    setup_axes(ax_top, pos_mb, truth, "infer(..., return_posterior=True)")
    for (lo, hi), label, a in zip(bands, band_labels, band_alphas):
        ax_top.fill_between(
            pos_mb, lo, hi, color="#3182bd", alpha=a, lw=0,
            label=f"{label} CI",
        )
    ax_top.plot(pos_mb, mean, color="#3182bd", lw=1.2, label="posterior mean")
    # Mark the three picked sites
    for idx, color, label in zip(pick_idx, pick_colors, pick_labels):
        ax_top.axvline(
            pos_mb[idx], color=color, lw=0.9, ls="--", alpha=0.85,
            label=f"site '{label}'",
        )
    ax_top.legend(frameon=False, loc="upper right", fontsize=8, ncol=2)

    # ----- bottom panels: actual Gamma PDF at the three picked sites -----
    for ax, idx, color, label in zip(ax_bot, pick_idx, pick_colors, pick_labels):
        a_i = float(alpha[idx])
        b_i = float(beta[idx])
        m_i = float(mean[idx])
        t_i = float(truth[idx])
        scale_i = 2.0 * NE / b_i
        post = gamma(a_i, scale=scale_i)
        # Plot range: ±4 sigma around the mean, clipped at 0
        sd = post.std()
        t_min = max(0.0, m_i - 4 * sd)
        t_max = m_i + 4 * sd
        t_lin = np.linspace(t_min, t_max, 400)
        density = post.pdf(t_lin)

        ax.fill_between(t_lin, 0, density, color=color, alpha=0.25, lw=0)
        ax.plot(t_lin, density, color=color, lw=1.4)
        ax.axvline(m_i, color=color, lw=1.0, ls="-",
                   label=f"mean = {m_i:.0f}")
        ax.axvline(t_i, color="#444", lw=0.9, ls=":",
                   label=f"truth = {t_i:.0f}")
        ax.set_xlabel("TMRCA (generations)", fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.set_title(
            f"site '{label}'  (α={a_i:.1f}, β={b_i:.2f})",
            fontsize=9, loc="left",
        )
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_yticks([])
        ax.legend(frameon=False, fontsize=7, loc="upper right")
        # Format x ticks compactly
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 4))
        ax.xaxis.get_offset_text().set_fontsize(7)

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
