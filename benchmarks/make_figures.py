#!/usr/bin/env python3
"""
Generate minimal, ultra-clean publication figures for tmrca.cu manuscript.

Aesthetic: sparse, high-contrast, Nature/Cell-style. No chartjunk.
Palette: muted blues/corals, thin lines, generous whitespace.

Generates:
  fig1_architecture.pdf   — Schematic of 4-tier pipeline (placeholder)
  fig2_accuracy.pdf       — Scatter + landscape + CI coverage (3 panels)
  fig3_speedup.pdf        — Throughput bar chart + scaling curves (2 panels)
  fig4_competitors.pdf    — tmrca.cu vs PSMC vs gamma_smc (2 panels)
  fig5_prior_recovery.pdf — EM-estimated vs true coalescent prior

Usage:
    pixi run python benchmarks/make_figures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── Global style ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.5,
})

# ── Palette ───────────────────────────────────────────────────
C = {
    "blue":    "#3B7DD8",
    "coral":   "#E8634F",
    "gold":    "#D4A843",
    "teal":    "#3AAFA9",
    "grey":    "#8E8E8E",
    "light":   "#F0F0F0",
    "dark":    "#2D2D2D",
    "blue_l":  "#A8C8F0",
    "coral_l": "#F5B5A8",
    "purple":  "#7B6BA5",
}

OUTDIR = "benchmarks"

# ── Simulation helper ─────────────────────────────────────────
def simulate(n_hap, seq_len, Ne=10_000, mu=1.25e-8, rho=1e-8, seed=42):
    import msprime
    ts = msprime.sim_ancestry(
        samples=n_hap // 2,
        sequence_length=seq_len,
        recombination_rate=rho,
        population_size=Ne,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed + 1)
    return ts


def ts_to_arrays(ts):
    G = ts.genotype_matrix().T.astype(np.uint8)
    positions = np.array([v.position for v in ts.variants()])
    return G, positions


def true_tmrca(ts, i, j):
    t = []
    for v in ts.variants():
        tree = ts.at(v.position)
        t.append(tree.tmrca(i, j))
    return np.array(t)


def panel_label(ax, letter, x=-0.12, y=1.08):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left",
            color=C["dark"])


# ══════════════════════════════════════════════════════════════
# FIGURE 2: Accuracy (scatter + landscape + CI coverage)
# ══════════════════════════════════════════════════════════════
def make_fig2():
    from tmrca_cu import _core
    print("  Simulating data for Fig 2...")
    ts = simulate(n_hap=40, seq_len=1_000_000, seed=123)
    G, positions = ts_to_arrays(ts)
    S = len(positions)

    pairs = [(i, j) for i in range(10) for j in range(i + 1, 10)]
    n_pairs = len(pairs)

    # Run Tier 3 inference
    _, mean_t3, lower_t3, upper_t3, loglik = _core.hmm_posterior_batched(
        G, positions, pairs, 32, 10000.0, 1.25e-8, 1e-8, -1.0)

    # Collect true TMRCAs
    true_all = np.zeros((n_pairs, S))
    for pidx, (i, j) in enumerate(pairs):
        true_all[pidx] = true_tmrca(ts, i, j)

    # ── Figure ──
    fig = plt.figure(figsize=(7.2, 2.4))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1.6, 0.9],
                  wspace=0.4)

    # Panel A: Scatter (all pairs, subsampled)
    ax_a = fig.add_subplot(gs[0, 0])
    panel_label(ax_a, "A")

    rng = np.random.default_rng(42)
    # Subsample for clarity
    idx_pairs = rng.choice(n_pairs, min(10, n_pairs), replace=False)
    idx_sites = rng.choice(S, min(2000, S), replace=False)
    t_true_sub = true_all[np.ix_(idx_pairs, idx_sites)].ravel()
    t_est_sub = mean_t3[np.ix_(idx_pairs, idx_sites)].ravel()

    mask = np.isfinite(t_true_sub) & np.isfinite(t_est_sub)
    t_true_sub = t_true_sub[mask]
    t_est_sub = t_est_sub[mask]

    ax_a.hexbin(t_true_sub, t_est_sub, gridsize=50,
                cmap="Blues", mincnt=1, linewidths=0.1, edgecolors="none",
                bins="log")
    lim = max(t_true_sub.max(), t_est_sub.max()) * 1.05
    ax_a.plot([0, lim], [0, lim], "--", color=C["coral"], lw=0.8, alpha=0.7)
    r = np.corrcoef(t_true_sub, t_est_sub)[0, 1]
    ax_a.text(0.05, 0.92, f"$r = {r:.2f}$", transform=ax_a.transAxes,
              fontsize=8, color=C["dark"],
              bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    ax_a.set_xlabel("True TMRCA (gen)")
    ax_a.set_ylabel("Estimated TMRCA (gen)")
    ax_a.set_xlim(0, lim)
    ax_a.set_ylim(0, lim)
    ax_a.set_aspect("equal")

    # Panel B: Landscape (single pair)
    ax_b = fig.add_subplot(gs[0, 1])
    panel_label(ax_b, "B")

    pidx = 0
    pos_kb = positions / 1000
    step = max(1, S // 800)
    ss = slice(None, None, step)

    ax_b.fill_between(pos_kb[ss], lower_t3[pidx, ss], upper_t3[pidx, ss],
                       color=C["blue_l"], alpha=0.5, lw=0,
                       label="95% CI")
    ax_b.plot(pos_kb[ss], true_all[pidx, ss],
              color=C["coral"], lw=0.6, alpha=0.8, label="True")
    ax_b.plot(pos_kb[ss], mean_t3[pidx, ss],
              color=C["blue"], lw=0.6, alpha=0.8, label="Estimated")
    ax_b.set_xlabel("Position (kb)")
    ax_b.set_ylabel("TMRCA (gen)")
    ax_b.legend(loc="upper right", ncol=3, columnspacing=0.8,
                handlelength=1.2, borderpad=0.2)
    ax_b.set_xlim(pos_kb[0], pos_kb[-1])

    # Panel C: CI coverage histogram
    ax_c = fig.add_subplot(gs[0, 2])
    panel_label(ax_c, "C")

    coverages = []
    for pidx in range(n_pairs):
        inside = (true_all[pidx] >= lower_t3[pidx]) & (true_all[pidx] <= upper_t3[pidx])
        coverages.append(inside.mean())
    coverages = np.array(coverages)

    ax_c.hist(coverages, bins=np.linspace(0.5, 1.0, 16),
              color=C["blue"], alpha=0.75, edgecolor="white", lw=0.5)
    ax_c.axvline(0.95, color=C["coral"], ls="--", lw=0.8, label="Nominal 95%")
    med = np.median(coverages)
    ax_c.axvline(med, color=C["blue"], ls="-", lw=0.8, alpha=0.5)
    ax_c.text(med - 0.01, ax_c.get_ylim()[1] * 0.85,
              f"med={med:.2f}", fontsize=6.5, ha="right", color=C["blue"])
    ax_c.set_xlabel("CI coverage")
    ax_c.set_ylabel("Count (pairs)")
    ax_c.legend(loc="upper left", fontsize=6.5)

    fig.savefig(f"{OUTDIR}/fig2_accuracy.pdf")
    fig.savefig(f"{OUTDIR}/fig2_accuracy.png")
    plt.close(fig)
    print("  → fig2_accuracy.pdf")


# ══════════════════════════════════════════════════════════════
# FIGURE 3: Throughput and speedup
# ══════════════════════════════════════════════════════════════
def make_fig3():
    import time
    from tmrca_cu import _core
    print("  Benchmarking throughput for Fig 3...")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.6), gridspec_kw={"wspace": 0.45})

    # Panel A: Throughput bar chart (optimization phases)
    panel_label(ax_a, "A")

    phases = [
        "Baseline\n(FP64)",
        "Phase 1\nPrecomp+\nfast math",
        "Phase 2\nFP32",
        "Phase 3\nMulti-pair\nblocks",
        "Phase 4\nFused\nsummary",
        "Phase 5\nPersistent\ncontext",
    ]
    rates = [9.1, 32, 64, 73, 78, 93]
    colors = [C["grey"]] + [C["blue"]] * 4 + [C["coral"]]

    bars = ax_a.bar(range(len(phases)), rates, color=colors, width=0.65,
                    edgecolor="white", lw=0.5)
    for bar, rate in zip(bars, rates):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                  f"{rate:.0f}", ha="center", va="bottom", fontsize=6.5,
                  color=C["dark"])

    ax_a.set_xticks(range(len(phases)))
    ax_a.set_xticklabels(phases, fontsize=6)
    ax_a.set_ylabel("Throughput (M site·pairs/s)")
    ax_a.set_ylim(0, 110)
    ax_a.set_title("Cumulative optimization effect", fontsize=8.5, pad=6)

    # Speedup annotation
    ax_a.annotate("", xy=(5, 95), xytext=(0, 95),
                  arrowprops=dict(arrowstyle="->", color=C["coral"],
                                  lw=1.2, connectionstyle="arc3,rad=0.15"))
    ax_a.text(2.5, 99, "10.2× speedup", ha="center", fontsize=7.5,
              color=C["coral"], fontweight="bold")

    # Panel B: Scaling with number of pairs
    panel_label(ax_b, "B")

    pair_counts = [10, 50, 100, 500, 1000, 2000, 5000]
    throughputs = []

    rng = np.random.default_rng(42)
    n = 200
    S = 50000
    G = rng.integers(0, 2, (n, S), dtype=np.uint8)
    positions = np.sort(rng.uniform(0, 249e6, S))
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    for np_ in pair_counts:
        pairs = all_pairs[:np_]
        # Warmup
        _core.hmm_posterior_batched(G, positions, pairs[:min(5, np_)],
                                    32, 10000.0, 1.25e-8, 1e-8, -1.0)
        t0 = time.perf_counter()
        _core.hmm_posterior_batched(G, positions, pairs,
                                    32, 10000.0, 1.25e-8, 1e-8, -1.0)
        t = time.perf_counter() - t0
        rate = np_ * S / t / 1e6
        throughputs.append(rate)
        print(f"    {np_:>5d} pairs: {rate:.1f} M/s ({t:.2f}s)")

    ax_b.semilogx(pair_counts, throughputs, "o-", color=C["blue"],
                   markersize=4, markeredgecolor="white", markeredgewidth=0.5)
    ax_b.axhline(93, color=C["coral"], ls="--", lw=0.7, alpha=0.6)
    ax_b.text(pair_counts[-1] * 0.7, 95, "93 M/s peak", fontsize=6.5,
              color=C["coral"], ha="right")
    ax_b.set_xlabel("Number of pairs")
    ax_b.set_ylabel("Throughput (M site·pairs/s)")
    ax_b.set_title("Throughput scaling ($S = 50$K, $K = 32$)", fontsize=8.5, pad=6)
    ax_b.set_ylim(0, 110)

    fig.savefig(f"{OUTDIR}/fig3_speedup.pdf")
    fig.savefig(f"{OUTDIR}/fig3_speedup.png")
    plt.close(fig)
    print("  → fig3_speedup.pdf")


# ══════════════════════════════════════════════════════════════
# FIGURE 4: Competitor comparison
# ══════════════════════════════════════════════════════════════
def make_fig4():
    print("  Generating Fig 4 (competitor comparison)...")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.2, 2.6),
                                      gridspec_kw={"wspace": 0.4})

    # Panel A: Single-pair time vs sequence length
    panel_label(ax_a, "A")

    seq_lens_mb = [1.0, 5.0, 10.0, 50.0]
    # Data from benchmark run
    t_psmc     = [3.16, 11.77, 22.62, 108.90]
    t_gamma    = [6.103, 6.270, 6.445, 6.892]
    t_tmrca_cu = [0.4055, 0.0439, 0.0630, 0.2068]

    ax_a.semilogy(seq_lens_mb, t_psmc, "s-", color=C["grey"], markersize=4,
                   label="PSMC", markeredgecolor="white", markeredgewidth=0.4)
    ax_a.semilogy(seq_lens_mb, t_gamma, "^-", color=C["gold"], markersize=4.5,
                   label="gamma-SMC", markeredgecolor="white", markeredgewidth=0.4)
    ax_a.semilogy(seq_lens_mb, t_tmrca_cu, "o-", color=C["coral"], markersize=4,
                   label="tmrca.cu", markeredgecolor="white", markeredgewidth=0.4)

    ax_a.set_xlabel("Sequence length (Mb)")
    ax_a.set_ylabel("Wall-clock time (s)")
    ax_a.set_title("Single-pair inference", fontsize=8.5, pad=6)
    ax_a.legend(loc="upper left", fontsize=6.5)
    ax_a.set_xlim(0, 55)

    # Speedup annotations
    for sl, tg, tc in zip(seq_lens_mb, t_gamma, t_tmrca_cu):
        if tg / tc > 10:
            ax_a.annotate(f"{tg/tc:.0f}×",
                          xy=(sl, tc), xytext=(sl + 1, tc * 0.35),
                          fontsize=5.5, color=C["coral"],
                          arrowprops=dict(arrowstyle="-", color=C["coral"],
                                          lw=0.4))

    # Panel B: Multi-pair total time
    panel_label(ax_b, "B")

    n_haps = [10, 50, 100]
    n_pairs_arr = [45, 1225, 4950]
    t_gamma_mp = [6.32, 6.55, 8.38]
    t_cu_mp = [0.035, 0.053, 0.173]

    x = np.arange(len(n_haps))
    w = 0.3
    bars1 = ax_b.bar(x - w/2, t_gamma_mp, w, color=C["gold"], label="gamma-SMC",
                      edgecolor="white", lw=0.5)
    bars2 = ax_b.bar(x + w/2, t_cu_mp, w, color=C["coral"], label="tmrca.cu",
                      edgecolor="white", lw=0.5)

    ax_b.set_yscale("log")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"$n={h}$\n({p} pairs)" for h, p in zip(n_haps, n_pairs_arr)],
                          fontsize=6.5)
    ax_b.set_ylabel("Wall-clock time (s)")
    ax_b.set_title("All-pairs inference (1 Mb)", fontsize=8.5, pad=6)
    ax_b.legend(loc="upper left", fontsize=6.5)

    # Speedup labels
    for i, (tg, tc) in enumerate(zip(t_gamma_mp, t_cu_mp)):
        ax_b.text(i + w/2, tc * 0.5, f"{tg/tc:.0f}×",
                  ha="center", fontsize=6, color=C["coral"], fontweight="bold")

    fig.savefig(f"{OUTDIR}/fig4_competitors.pdf")
    fig.savefig(f"{OUTDIR}/fig4_competitors.png")
    plt.close(fig)
    print("  → fig4_competitors.pdf")


# ══════════════════════════════════════════════════════════════
# FIGURE 5: Prior recovery under bottleneck
# ══════════════════════════════════════════════════════════════
def make_fig5():
    from tmrca_cu import _core
    print("  Running EM for Fig 5 (prior recovery)...")

    # Bottleneck: 10k → 500 at t=2000 → 20k at t=3000
    import msprime
    demography = msprime.Demography()
    demography.add_population(initial_size=10_000)
    demography.add_population_parameters_change(time=2000, population=0, initial_size=500)
    demography.add_population_parameters_change(time=3000, population=0, initial_size=20_000)

    ts = msprime.sim_ancestry(
        samples=25, sequence_length=2_000_000,
        recombination_rate=1e-8, demography=demography,
        random_seed=77,
    )
    ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=78)
    G, positions = ts_to_arrays(ts)
    S = len(positions)
    K = 64

    pairs = [(i, j) for i in range(20) for j in range(i + 1, 20)]

    # Get time midpoints and constant-Ne prior
    midpoints = _core.time_midpoints(K, 10000.0, -1.0)
    prior_const = _core.coalescent_prior(10000.0, K, -1.0)

    # Run adaptive prior inference (Tier 4 EM)
    result = _core.adaptive_prior_infer(
        G, positions, pairs, K, 10000.0, 1.25e-8, 1e-8, -1.0,
        15, 0.7)
    prior_em = result["prior"]

    # True prior: compute from the demographic model
    # Sample true coalescence times and histogram into K bins
    boundaries = _core.time_boundaries(K, 10000.0, -1.0)
    true_times = []
    for tree in ts.trees():
        for i, j in [(0, 1), (2, 3), (4, 5), (6, 7)]:
            true_times.append(tree.tmrca(i, j))
    true_times = np.array(true_times)
    true_hist, _ = np.histogram(true_times, bins=boundaries)
    true_prior = true_hist / true_hist.sum()

    # ── Figure ──
    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.6))
    panel_label(ax, "")

    ax.step(midpoints, true_prior, where="mid", color=C["dark"], lw=1.2,
            label="True (bottleneck)", alpha=0.9)
    ax.step(midpoints, prior_const, where="mid", color=C["grey"], lw=0.9,
            ls="--", label="Assumed (const $N_e$)", alpha=0.7)
    ax.step(midpoints, prior_em, where="mid", color=C["coral"], lw=1.2,
            label="EM-estimated (15 iter)", alpha=0.9)

    ax.set_xscale("log")
    ax.set_xlabel("Coalescence time (generations)")
    ax.set_ylabel("Prior probability")
    ax.set_title("Prior recovery under bottleneck demography", fontsize=8.5, pad=6)
    ax.legend(loc="upper right", fontsize=6.5)
    ax.set_xlim(midpoints[0] * 0.5, midpoints[-1] * 2)

    fig.savefig(f"{OUTDIR}/fig5_prior_recovery.pdf")
    fig.savefig(f"{OUTDIR}/fig5_prior_recovery.png")
    plt.close(fig)
    print("  → fig5_prior_recovery.pdf")


# ══════════════════════════════════════════════════════════════
# FIGURE 6: Projected wall-clock times (chr1 extrapolation)
# ══════════════════════════════════════════════════════════════
def make_fig6():
    print("  Generating Fig 6 (wall-clock projection)...")

    fig, ax = plt.subplots(1, 1, figsize=(3.5, 2.6))

    sample_sizes = np.array([100, 200, 500, 1000, 2000, 5000])
    n_pairs = sample_sizes * (sample_sizes - 1) // 2
    S_chr1 = 4.5e6
    rate_baseline = 9.1e6
    rate_optimized = 90e6

    t_baseline = n_pairs * S_chr1 / rate_baseline / 3600
    t_optimized = n_pairs * S_chr1 / rate_optimized / 3600

    ax.loglog(sample_sizes, t_baseline, "s--", color=C["grey"], markersize=4,
              label="Baseline (9.1 M/s)", markeredgecolor="white", markeredgewidth=0.4)
    ax.loglog(sample_sizes, t_optimized, "o-", color=C["coral"], markersize=4,
              label="tmrca.cu (90 M/s)", markeredgecolor="white", markeredgewidth=0.4)

    # Reference lines
    for hours, label in [(1, "1 hour"), (24, "1 day"), (24*7, "1 week")]:
        ax.axhline(hours, color=C["light"], lw=0.6, zorder=0)
        ax.text(sample_sizes[-1] * 1.3, hours, label,
                fontsize=5.5, color=C["grey"], va="center")

    ax.set_xlabel("Sample size ($2N$ haplotypes)")
    ax.set_ylabel("Projected time (hours)")
    ax.set_title("All-pairs chr1 ($S = 4.5$M SNPs), single A100", fontsize=8.5, pad=6)
    ax.legend(loc="upper left", fontsize=6.5)
    ax.set_xlim(80, 7000)
    ax.set_ylim(0.01, 5000)

    fig.savefig(f"{OUTDIR}/fig6_wallclock.pdf")
    fig.savefig(f"{OUTDIR}/fig6_wallclock.png")
    plt.close(fig)
    print("  → fig6_wallclock.pdf")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating manuscript figures...")
    print()

    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()

    print()
    print("All figures saved to benchmarks/")
