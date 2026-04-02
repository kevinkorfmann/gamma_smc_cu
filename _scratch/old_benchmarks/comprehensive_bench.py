"""
Comprehensive accuracy & scalability benchmarks for tmrca_cu.

Generates 7 figures:
  fig1_scatter_grid.png      – Inferred vs true TMRCA (density scatter), 4 scenarios × 2 tiers
  fig2_landscape_scenarios.png – Per-site TMRCA landscape, one per scenario
  fig3_ll_convergence.png    – Log-likelihood vs EM iteration
  fig4_scaling_n.png         – Accuracy & runtime vs number of haplotypes
  fig5_scaling_seqlen.png    – Accuracy & runtime vs sequence length
  fig6_calibration.png       – 95% CI coverage histograms
  fig7_prior_recovery.png    – Estimated vs true coalescent prior

Usage:
  python comprehensive_bench.py              # full run
  python comprehensive_bench.py --plot-only  # regenerate figures from cache
"""

import argparse
import time
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import msprime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tmrca_cu
import tmrca_cu._core as _core

OUT_DIR = os.path.dirname(__file__)
SEED_ANC = 42
SEED_MUT = 43
MU = 1.25e-8
RHO = 1e-8

EVAL_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]

# ═══════════════════════════════════════════════════════════════════════════
# Demographic scenarios
# ═══════════════════════════════════════════════════════════════════════════

def make_constant():
    d = msprime.Demography()
    d.add_population(initial_size=10_000)
    return d, "Constant Ne=10k"

def make_bottleneck():
    d = msprime.Demography()
    d.add_population(initial_size=10_000)
    d.add_population_parameters_change(time=2000, initial_size=500)
    d.add_population_parameters_change(time=3000, initial_size=20_000)
    return d, "Bottleneck (10k→500→20k)"

def make_exponential_growth():
    d = msprime.Demography()
    d.add_population(initial_size=100_000, growth_rate=0.01)
    d.add_population_parameters_change(time=200, growth_rate=0, initial_size=10_000)
    return d, "Exp. growth (10k→100k)"

def make_structured():
    d = msprime.Demography()
    d.add_population(name="A", initial_size=5_000)
    d.add_population(name="B", initial_size=5_000)
    d.set_symmetric_migration_rate(["A", "B"], 1e-4)
    return d, "Structured (2-pop, m=1e-4)"

SCENARIOS = {
    "constant": make_constant,
    "bottleneck": make_bottleneck,
    "exp_growth": make_exponential_growth,
    "structured": make_structured,
}

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def simulate(demography, n_diploid, seq_len, seed=SEED_ANC):
    if demography.num_populations > 1:
        samples = {p.name: n_diploid // demography.num_populations
                   for p in demography.populations}
    else:
        samples = n_diploid
    ts = msprime.sim_ancestry(
        samples=samples, sequence_length=seq_len,
        recombination_rate=RHO, demography=demography, random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=SEED_MUT)
    G = ts.genotype_matrix().T.astype(np.uint8)
    pos = np.array([v.position for v in ts.variants()], dtype=np.float64)
    return ts, G, pos


def true_tmrca_at_sites(ts, i, j, site_positions):
    t = np.empty(len(site_positions))
    tree_iter = ts.trees()
    tree = next(tree_iter)
    for si, p in enumerate(site_positions):
        while p >= tree.interval.right:
            tree = next(tree_iter)
        t[si] = tree.tmrca(i, j)
    return t


def make_bg_pairs(n_hap, eval_pairs, max_bg=500, seed=0):
    rng = np.random.RandomState(seed)
    n_bg = min(max_bg, n_hap * (n_hap - 1) // 2)
    bg = set()
    while len(bg) < n_bg:
        a, b = sorted(rng.choice(n_hap, 2, replace=False))
        bg.add((int(a), int(b)))
    all_pairs = sorted(set(eval_pairs) | bg)
    eval_idx = {p: all_pairs.index(p) for p in eval_pairs}
    return all_pairs, eval_idx


def run_inference(G, pos, all_pairs, eval_idx, Ne=10_000.0, K=64):
    """Run Tier 1, 3, 4 and return metrics."""
    S = G.shape[1]
    midpoints = np.array(tmrca_cu.time_midpoints(K=K, Ne=Ne))

    # Tier 3 — fused summaries (gamma is scratch in SUMMARY_ONLY mode)
    t0 = time.perf_counter()
    r3 = _core.hmm_posterior_batched(G, pos, all_pairs, K=K, Ne=Ne, mu=MU, rho=RHO)
    t_tier3 = time.perf_counter() - t0
    mean_t3 = np.array(r3[1])
    lower_t3 = np.array(r3[2])
    upper_t3 = np.array(r3[3])

    # Tier 4 — fused summaries + EM
    t0 = time.perf_counter()
    r4 = _core.adaptive_prior_infer(G, pos, all_pairs, K=K, Ne=Ne, mu=MU, rho=RHO,
                                     max_iterations=20, blend_alpha=0.7)
    t_tier4 = time.perf_counter() - t0
    prior_est = np.array(r4["prior"])
    ll_history = np.array(r4["ll_history"])

    # Tier 1
    ws = max(1, int(round(20_000 / ((pos[-1] - pos[0]) / (S - 1)))))
    div_raw = np.array(tmrca_cu.windowed_divergence(G, list(EVAL_PAIRS), ws))
    wbp = ws * (pos[-1] - pos[0]) / (S - 1)

    return {
        "midpoints": midpoints,
        "mean_t3": mean_t3,
        "lower_t3": lower_t3,
        "upper_t3": upper_t3,
        "prior_est": prior_est,
        "ll_history": ll_history,
        "div_raw": div_raw,
        "wbp": wbp,
        "t_tier3": t_tier3,
        "t_tier4": t_tier4,
        "mean_t4": np.array(r4["mean"]),
        "lower_t4": np.array(r4["lower"]),
        "upper_t4": np.array(r4["upper"]),
    }


def compute_metrics(true_dict, inf, eval_idx, S):
    """Compute correlation, RMSE, CI coverage for eval pairs."""
    results = {"t1": [], "t3": [], "t4": [],
               "rmse_t3": [], "rmse_t4": [],
               "cov_t3": [], "cov_t4": []}

    for pi, p in enumerate(EVAL_PAIRS):
        true_t = true_dict[p]
        idx = eval_idx[p]

        # Posterior means (fused in kernel)
        m3 = inf["mean_t3"][idx]
        m4 = inf["mean_t4"][idx]

        # Tier 1
        m1 = inf["div_raw"][pi] / (inf["wbp"] * 2 * MU)

        results["t1"].append(np.corrcoef(true_t, m1)[0, 1])
        results["t3"].append(np.corrcoef(true_t, m3)[0, 1])
        results["t4"].append(np.corrcoef(true_t, m4)[0, 1])
        results["rmse_t3"].append(np.sqrt(np.mean((true_t - m3)**2)))
        results["rmse_t4"].append(np.sqrt(np.mean((true_t - m4)**2)))

        # CI coverage (fused lower/upper from kernel)
        lo4 = inf["lower_t4"][idx]
        hi4 = inf["upper_t4"][idx]
        results["cov_t4"].append(np.mean((true_t >= lo4) & (true_t <= hi4)))

        lo3 = inf["lower_t3"][idx]
        hi3 = inf["upper_t3"][idx]
        results["cov_t3"].append(np.mean((true_t >= lo3) & (true_t <= hi3)))

    return {k: np.array(v) for k, v in results.items()}


def true_coalescent_prior(scenario_name, boundaries, ts=None):
    """Compute analytic true coalescent prior for a given scenario."""
    K = len(boundaries) - 1

    if scenario_name == "constant":
        Ne = 10_000
        q = np.array([np.exp(-boundaries[k] / (2*Ne)) - np.exp(-boundaries[k+1] / (2*Ne))
                       for k in range(K)])

    elif scenario_name == "bottleneck":
        def cum_rate(t):
            if t <= 2000:
                return t / 20_000
            elif t <= 3000:
                return 2000 / 20_000 + (t - 2000) / 1_000
            else:
                return 2000 / 20_000 + 1000 / 1_000 + (t - 3000) / 40_000
        q = np.array([np.exp(-cum_rate(boundaries[k])) - np.exp(-cum_rate(boundaries[k+1]))
                       for k in range(K)])

    elif scenario_name == "exp_growth":
        # Ne(t) = 100_000 * exp(-0.01*t) for t < 200, then 10_000 for t >= 200
        # Rate = 1/(2*Ne(t)), Λ(t) = ∫ 1/(2*Ne(s)) ds
        def cum_rate(t):
            if t <= 200:
                # ∫_0^t 1/(2*100000*exp(-0.01*s)) ds = ∫_0^t exp(0.01*s)/(200000) ds
                # = (exp(0.01*t) - 1) / (0.01 * 200000)
                return (np.exp(0.01 * t) - 1.0) / 2000.0
            else:
                base = (np.exp(0.01 * 200) - 1.0) / 2000.0
                return base + (t - 200) / 20_000
        q = np.array([np.exp(-cum_rate(boundaries[k])) - np.exp(-cum_rate(boundaries[k+1]))
                       for k in range(K)])

    elif scenario_name == "structured":
        # Empirical from tree sequence — no simple closed form
        if ts is None:
            return np.ones(K) / K
        tmrcas = []
        for tree in ts.trees():
            span = tree.interval.right - tree.interval.left
            for i in range(min(10, ts.num_samples)):
                for j in range(i + 1, min(10, ts.num_samples)):
                    tmrcas.append(tree.tmrca(i, j))
        tmrcas = np.array(tmrcas)
        q = np.array([np.mean((tmrcas >= boundaries[k]) & (tmrcas < boundaries[k+1]))
                       for k in range(K)])
        q = np.maximum(q, 1e-30)

    q /= q.sum()
    return q


# ═══════════════════════════════════════════════════════════════════════════
# Data generation
# ═══════════════════════════════════════════════════════════════════════════

def run_all(args):
    results = {}

    # ── Scenario sweep (for fig1, fig2, fig3, fig6, fig7) ─────────────────
    print("=" * 70)
    print("SCENARIO SWEEP (n=50 diploid, seq_len=1Mb, K=64)")
    print("=" * 70)
    for sc_name, sc_fn in SCENARIOS.items():
        demog, label = sc_fn()
        print(f"\n  [{sc_name}] {label}")
        t0 = time.perf_counter()
        ts, G, pos = simulate(demog, n_diploid=50, seq_len=1_000_000)
        n_hap = G.shape[0]
        S = G.shape[1]
        print(f"    n={n_hap}, S={S}")

        true_eval = {p: true_tmrca_at_sites(ts, p[0], p[1], pos) for p in EVAL_PAIRS}
        all_pairs, eval_idx = make_bg_pairs(n_hap, EVAL_PAIRS)

        # Run at K=64
        inf = run_inference(G, pos, all_pairs, eval_idx, K=64)
        metrics = compute_metrics(true_eval, inf, eval_idx, S)

        # Also run at K=32 and K=128 for LL convergence
        inf_k32 = run_inference(G, pos, all_pairs, eval_idx, K=32)
        inf_k128 = run_inference(G, pos, all_pairs, eval_idx, K=128)

        # Scatter data: pool per-site estimates from many pairs
        rng = np.random.RandomState(0)
        n_scatter_pairs = min(80, len(all_pairs))
        scatter_idx = rng.choice(len(all_pairs), n_scatter_pairs, replace=False)
        all_true_s, all_est3_s, all_est4_s = [], [], []
        for si in scatter_idx:
            p = all_pairs[si]
            tt = true_tmrca_at_sites(ts, p[0], p[1], pos)
            all_true_s.append(tt)
            all_est3_s.append(inf["mean_t3"][si])
            all_est4_s.append(inf["mean_t4"][si])
        all_true_s = np.concatenate(all_true_s)
        all_est3_s = np.concatenate(all_est3_s)
        all_est4_s = np.concatenate(all_est4_s)

        # CI coverage for many pairs (fig6)
        cov_t3_all, cov_t4_all = [], []
        for si in scatter_idx:
            p = all_pairs[si]
            tt = true_tmrca_at_sites(ts, p[0], p[1], pos)
            lo4 = inf["lower_t4"][si]
            hi4 = inf["upper_t4"][si]
            cov_t4_all.append(np.mean((tt >= lo4) & (tt <= hi4)))
            lo3 = inf["lower_t3"][si]
            hi3 = inf["upper_t3"][si]
            cov_t3_all.append(np.mean((tt >= lo3) & (tt <= hi3)))

        # True prior
        bnd64 = np.array(tmrca_cu.time_boundaries(K=64, Ne=10_000))
        bnd32 = np.array(tmrca_cu.time_boundaries(K=32, Ne=10_000))
        bnd128 = np.array(tmrca_cu.time_boundaries(K=128, Ne=10_000))
        true_prior_64 = true_coalescent_prior(sc_name, bnd64, ts)
        true_prior_32 = true_coalescent_prior(sc_name, bnd32, ts)

        elapsed = time.perf_counter() - t0
        print(f"    r_t3={np.mean(metrics['t3']):.3f}  r_t4={np.mean(metrics['t4']):.3f}  "
              f"cov_t4={np.mean(metrics['cov_t4']):.1%}  [{elapsed:.1f}s]")

        results[("scenario", sc_name)] = {
            "label": label,
            "ts": ts, "G": G, "pos": pos,
            "true_eval": true_eval,
            "inf_k64": inf, "inf_k32": inf_k32, "inf_k128": inf_k128,
            "metrics": metrics,
            "all_true_scatter": all_true_s,
            "all_est3_scatter": all_est3_s,
            "all_est4_scatter": all_est4_s,
            "cov_t3_all": np.array(cov_t3_all),
            "cov_t4_all": np.array(cov_t4_all),
            "true_prior_64": true_prior_64,
            "true_prior_32": true_prior_32,
            "bnd64": bnd64, "bnd32": bnd32,
        }

    # ── Sample size scaling (fig4) ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAMPLE SIZE SCALING (bottleneck, seq_len=500kb)")
    print("=" * 70)
    n_diploid_values = [5, 10, 25, 50, 100, 250]
    demog_bn, _ = make_bottleneck()

    for n_dip in n_diploid_values:
        t0 = time.perf_counter()
        ts, G, pos = simulate(demog_bn, n_diploid=n_dip, seq_len=500_000)
        n_hap = G.shape[0]
        S = G.shape[1]

        true_eval = {p: true_tmrca_at_sites(ts, p[0], p[1], pos) for p in EVAL_PAIRS}
        all_pairs, eval_idx = make_bg_pairs(n_hap, EVAL_PAIRS)
        inf = run_inference(G, pos, all_pairs, eval_idx, K=64)
        metrics = compute_metrics(true_eval, inf, eval_idx, S)

        elapsed = time.perf_counter() - t0
        print(f"  n_dip={n_dip:>4} (n_hap={n_hap:>4}, S={S:>5}): "
              f"r_t3={np.mean(metrics['t3']):.3f}  r_t4={np.mean(metrics['t4']):.3f}  "
              f"time_t4={inf['t_tier4']:.1f}s  [{elapsed:.1f}s]")

        results[("n_scaling", n_dip)] = {
            "n_hap": n_hap, "S": S,
            "metrics": metrics, "inf": inf,
        }

    # ── Sequence length scaling (fig5) ────────────────────────────────────
    print("\n" + "=" * 70)
    print("SEQUENCE LENGTH SCALING (bottleneck, n=50 diploid)")
    print("=" * 70)
    seq_lengths = [100_000, 250_000, 500_000, 1_000_000, 2_000_000, 5_000_000]

    for seq_len in seq_lengths:
        t0 = time.perf_counter()
        ts, G, pos = simulate(demog_bn, n_diploid=50, seq_len=seq_len)
        n_hap = G.shape[0]
        S = G.shape[1]

        true_eval = {p: true_tmrca_at_sites(ts, p[0], p[1], pos) for p in EVAL_PAIRS}
        all_pairs, eval_idx = make_bg_pairs(n_hap, EVAL_PAIRS)
        inf = run_inference(G, pos, all_pairs, eval_idx, K=64)
        metrics = compute_metrics(true_eval, inf, eval_idx, S)

        elapsed = time.perf_counter() - t0
        print(f"  seq_len={seq_len/1e6:.1f}Mb (S={S:>6}): "
              f"r_t3={np.mean(metrics['t3']):.3f}  r_t4={np.mean(metrics['t4']):.3f}  "
              f"time_t4={inf['t_tier4']:.1f}s  [{elapsed:.1f}s]")

        results[("seq_scaling", seq_len)] = {
            "n_hap": n_hap, "S": S,
            "metrics": metrics, "inf": inf,
        }

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Figure generation
# ═══════════════════════════════════════════════════════════════════════════

SCENARIO_ORDER = ["constant", "bottleneck", "exp_growth", "structured"]
SCENARIO_COLORS = {"constant": "C0", "bottleneck": "C1",
                   "exp_growth": "C2", "structured": "C3"}


def fig1_scatter_grid(results):
    """Density scatter: inferred vs true TMRCA, 4 scenarios × 2 tiers."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Inferred vs True TMRCA (per-site, density)",
                 fontsize=15, fontweight="bold", y=0.98)

    for col, sc in enumerate(SCENARIO_ORDER):
        r = results[("scenario", sc)]
        true_s = r["all_true_scatter"]
        est3_s = r["all_est3_scatter"]
        est4_s = r["all_est4_scatter"]

        # Subsample
        rng = np.random.RandomState(1)
        n_pts = min(10_000, len(true_s))
        idx = rng.choice(len(true_s), n_pts, replace=False)

        for row, (est_s, tier_name) in enumerate([
            (est4_s, "Tier 4 (adaptive)"),
            (est3_s, "Tier 3 (constant Ne)"),
        ]):
            ax = axes[row, col]
            vmax = np.percentile(true_s, 99) * 1.2
            hb = ax.hexbin(true_s[idx], est_s[idx], gridsize=60,
                           cmap="viridis", norm=LogNorm(), mincnt=1,
                           extent=[0, vmax, 0, vmax])
            ax.plot([0, vmax], [0, vmax], "r-", linewidth=1, alpha=0.7)
            corr = np.corrcoef(true_s[idx], est_s[idx])[0, 1]
            rmse = np.sqrt(np.mean((true_s[idx] - est_s[idx])**2))
            ax.text(0.05, 0.92, f"r = {corr:.3f}\nRMSE = {rmse:.0f}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

            ax.set_xlim(0, vmax)
            ax.set_ylim(0, vmax)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.15)

            if row == 0:
                ax.set_title(r["label"], fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{tier_name}\nEstimated TMRCA (gen)", fontsize=10)
            if row == 1:
                ax.set_xlabel("True TMRCA (gen)", fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, "fig1_scatter_grid.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


def fig2_landscape_scenarios(results):
    """Per-site TMRCA landscape for each scenario."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 16))
    fig.suptitle("TMRCA Landscape: Tier 3 vs Tier 4 vs Truth",
                 fontsize=15, fontweight="bold", y=0.99)

    pair = EVAL_PAIRS[0]
    for row, sc in enumerate(SCENARIO_ORDER):
        r = results[("scenario", sc)]
        ax = axes[row]
        pos = r["pos"]
        S = len(pos)
        inf = r["inf_k64"]
        mid = inf["midpoints"]
        idx = list(r["true_eval"].keys()).index(pair)  # eval_idx for scenario sweep

        # Recompute eval_idx properly
        n_hap = r["G"].shape[0]
        _, eval_idx = make_bg_pairs(n_hap, EVAL_PAIRS)
        eidx = eval_idx[pair]

        true_t = r["true_eval"][pair]
        m3 = inf["mean_t3"][eidx]
        m4 = inf["mean_t4"][eidx]
        lo4 = inf["lower_t4"][eidx]
        hi4 = inf["upper_t4"][eidx]

        step = max(1, S // 1000)
        xs = pos[::step] / 1e3

        ax.fill_between(xs, lo4[::step], hi4[::step], alpha=0.15, color="C2")
        ax.plot(xs, true_t[::step], color="C3", linewidth=0.8, alpha=0.9, label="True")
        ax.plot(xs, m3[::step], color="C0", linewidth=0.7, alpha=0.6, label="Tier 3")
        ax.plot(xs, m4[::step], color="C2", linewidth=0.7, alpha=0.8, label="Tier 4")

        r3c = np.corrcoef(true_t, m3)[0, 1]
        r4c = np.corrcoef(true_t, m4)[0, 1]
        ax.set_title(f"{r['label']}  |  Pair {pair}  |  T3 r={r3c:.3f}, T4 r={r4c:.3f}",
                     fontsize=11)
        ax.set_ylabel("TMRCA (gen)")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Position (kb)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(OUT_DIR, "fig2_landscape_scenarios.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


def fig3_ll_convergence(results):
    """Log-likelihood vs EM iteration for each scenario."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Log-Likelihood Convergence (Tier 4 EM)",
                 fontsize=14, fontweight="bold")

    for idx, sc in enumerate(SCENARIO_ORDER):
        ax = axes[idx // 2, idx % 2]
        r = results[("scenario", sc)]

        for K_val, key, color, ls in [
            (32, "inf_k32", "C0", "-"),
            (64, "inf_k64", "C4", "-"),
            (128, "inf_k128", "C2", "-"),
        ]:
            ll = r[key]["ll_history"]
            # Normalize: show ΔLL from initial
            if len(ll) > 0:
                ax.plot(range(1, len(ll) + 1), ll - ll[0],
                        color=color, linestyle=ls, linewidth=1.5,
                        marker="o", markersize=3, label=f"K={K_val}")

        ax.set_title(r["label"], fontsize=11)
        ax.set_xlabel("EM iteration")
        ax.set_ylabel("ΔLL (from initial)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, "fig3_ll_convergence.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


def fig4_scaling_n(results):
    """Accuracy & runtime vs number of haplotypes."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Scalability with Sample Size (bottleneck, 500kb, K=64)",
                 fontsize=14, fontweight="bold")

    n_dips = [5, 10, 25, 50, 100, 250]
    n_haps = []
    r_t1, r_t3, r_t4 = [], [], []
    rmse_t3, rmse_t4 = [], []
    time_t3, time_t4 = [], []

    for nd in n_dips:
        r = results[("n_scaling", nd)]
        m = r["metrics"]
        n_haps.append(r["n_hap"])
        r_t1.append(np.mean(m["t1"]))
        r_t3.append(np.mean(m["t3"]))
        r_t4.append(np.mean(m["t4"]))
        rmse_t3.append(np.mean(m["rmse_t3"]))
        rmse_t4.append(np.mean(m["rmse_t4"]))
        time_t3.append(r["inf"]["t_tier3"])
        time_t4.append(r["inf"]["t_tier4"])

    # (a) Correlation
    ax = axes[0]
    ax.plot(n_haps, r_t1, "s--", color="C1", linewidth=1.5, markersize=6, label="Tier 1 (div/2μ)")
    ax.plot(n_haps, r_t3, "o:", color="C0", linewidth=1.5, markersize=6, label="Tier 3 (const Ne)")
    ax.plot(n_haps, r_t4, "^-", color="C2", linewidth=2, markersize=7, label="Tier 4 (adaptive)")
    ax.set_xlabel("Number of haplotypes")
    ax.set_ylabel("Mean Pearson r")
    ax.set_title("Accuracy vs sample size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) RMSE
    ax = axes[1]
    ax.plot(n_haps, rmse_t3, "o:", color="C0", linewidth=1.5, markersize=6, label="Tier 3")
    ax.plot(n_haps, rmse_t4, "^-", color="C2", linewidth=2, markersize=7, label="Tier 4")
    ax.set_xlabel("Number of haplotypes")
    ax.set_ylabel("Mean RMSE (generations)")
    ax.set_title("RMSE vs sample size")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Runtime
    ax = axes[2]
    ax.plot(n_haps, time_t3, "o:", color="C0", linewidth=1.5, markersize=6, label="Tier 3")
    ax.plot(n_haps, time_t4, "^-", color="C2", linewidth=2, markersize=7, label="Tier 4 (20 EM iters)")
    ax.set_xlabel("Number of haplotypes")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Runtime vs sample size")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT_DIR, "fig4_scaling_n.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


def fig5_scaling_seqlen(results):
    """Accuracy & runtime vs sequence length."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Scalability with Sequence Length (bottleneck, n=100, K=64)",
                 fontsize=14, fontweight="bold")

    seq_lengths = [100_000, 250_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
    S_vals = []
    r_t1, r_t3, r_t4 = [], [], []
    rmse_t3, rmse_t4 = [], []
    time_t3, time_t4 = [], []

    for sl in seq_lengths:
        r = results[("seq_scaling", sl)]
        m = r["metrics"]
        S_vals.append(r["S"])
        r_t1.append(np.mean(m["t1"]))
        r_t3.append(np.mean(m["t3"]))
        r_t4.append(np.mean(m["t4"]))
        rmse_t3.append(np.mean(m["rmse_t3"]))
        rmse_t4.append(np.mean(m["rmse_t4"]))
        time_t3.append(r["inf"]["t_tier3"])
        time_t4.append(r["inf"]["t_tier4"])

    seq_mb = [sl / 1e6 for sl in seq_lengths]

    # (a) Correlation
    ax = axes[0]
    ax.plot(seq_mb, r_t1, "s--", color="C1", linewidth=1.5, markersize=6, label="Tier 1")
    ax.plot(seq_mb, r_t3, "o:", color="C0", linewidth=1.5, markersize=6, label="Tier 3")
    ax.plot(seq_mb, r_t4, "^-", color="C2", linewidth=2, markersize=7, label="Tier 4")
    ax.set_xlabel("Sequence length (Mb)")
    ax.set_ylabel("Mean Pearson r")
    ax.set_title("Accuracy vs sequence length")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) RMSE
    ax = axes[1]
    ax.plot(seq_mb, rmse_t3, "o:", color="C0", linewidth=1.5, markersize=6, label="Tier 3")
    ax.plot(seq_mb, rmse_t4, "^-", color="C2", linewidth=2, markersize=7, label="Tier 4")
    ax.set_xlabel("Sequence length (Mb)")
    ax.set_ylabel("Mean RMSE (generations)")
    ax.set_title("RMSE vs sequence length")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Runtime
    ax = axes[2]
    ax.plot(seq_mb, time_t3, "o:", color="C0", linewidth=1.5, markersize=6, label="Tier 3")
    ax.plot(seq_mb, time_t4, "^-", color="C2", linewidth=2, markersize=7, label="Tier 4")
    ax.set_xlabel("Sequence length (Mb)")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Runtime vs sequence length")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT_DIR, "fig5_scaling_seqlen.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


def fig6_calibration(results):
    """95% CI coverage histograms per scenario."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Credible Interval Calibration (95% CI coverage per pair)",
                 fontsize=14, fontweight="bold")

    for idx, sc in enumerate(SCENARIO_ORDER):
        ax = axes[idx // 2, idx % 2]
        r = results[("scenario", sc)]

        bins = np.linspace(0.4, 1.0, 25)
        ax.hist(r["cov_t3_all"], bins=bins, alpha=0.5, color="C0",
                edgecolor="white", label=f"Tier 3 (mean={np.mean(r['cov_t3_all']):.2f})")
        ax.hist(r["cov_t4_all"], bins=bins, alpha=0.5, color="C2",
                edgecolor="white", label=f"Tier 4 (mean={np.mean(r['cov_t4_all']):.2f})")
        ax.axvline(0.95, color="k", linestyle="--", linewidth=1.5, label="Nominal 95%")
        ax.set_title(r["label"], fontsize=11)
        ax.set_xlabel("Fraction of sites within 95% CI")
        ax.set_ylabel("Number of pairs")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, "fig6_calibration.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


def fig7_prior_recovery(results):
    """Estimated vs true coalescent prior per scenario."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Coalescent Prior Recovery (K=64)",
                 fontsize=14, fontweight="bold")

    for idx, sc in enumerate(SCENARIO_ORDER):
        ax = axes[idx]
        r = results[("scenario", sc)]
        mid64 = np.array(tmrca_cu.time_midpoints(K=64, Ne=10_000))
        const_prior = np.array(tmrca_cu.coalescent_prior(Ne=10_000, K=64))

        ax.plot(mid64, r["true_prior_64"], "C3-", linewidth=2, alpha=0.9, label="True")
        ax.plot(mid64, r["inf_k64"]["prior_est"], "C2o-", linewidth=1.5,
                markersize=3, label="Estimated (T4)")
        ax.plot(mid64, const_prior, "k--", linewidth=1.5, alpha=0.5,
                label="Assumed (const Ne)")

        ax.set_title(r["label"], fontsize=11)
        ax.set_xlabel("Time (generations)")
        ax.set_ylabel("Prior q(t)")
        ax.set_xscale("log")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(OUT_DIR, "fig7_prior_recovery.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(path)}")


# ═══════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(results):
    print("\n" + "=" * 90)
    print("ACCURACY SUMMARY BY SCENARIO (K=64)")
    print("=" * 90)
    print(f"{'Scenario':<25} {'Tier 1 r':>9} {'Tier 3 r':>9} {'Tier 4 r':>9} "
          f"{'RMSE T3':>9} {'RMSE T4':>9} {'CI T3':>7} {'CI T4':>7}")
    print("-" * 90)
    for sc in SCENARIO_ORDER:
        r = results[("scenario", sc)]
        m = r["metrics"]
        print(f"{r['label']:<25} {np.mean(m['t1']):>9.3f} {np.mean(m['t3']):>9.3f} "
              f"{np.mean(m['t4']):>9.3f} {np.mean(m['rmse_t3']):>9.0f} "
              f"{np.mean(m['rmse_t4']):>9.0f} {np.mean(r['cov_t3_all']):>6.1%} "
              f"{np.mean(r['cov_t4_all']):>6.1%}")

    print("\n" + "=" * 70)
    print("SAMPLE SIZE SCALING (bottleneck, 500kb, K=64)")
    print("=" * 70)
    print(f"{'n_hap':>6} {'Tier 1 r':>9} {'Tier 3 r':>9} {'Tier 4 r':>9} "
          f"{'T3 time':>8} {'T4 time':>8}")
    print("-" * 50)
    for nd in [5, 10, 25, 50, 100, 250]:
        r = results[("n_scaling", nd)]
        m = r["metrics"]
        print(f"{r['n_hap']:>6} {np.mean(m['t1']):>9.3f} {np.mean(m['t3']):>9.3f} "
              f"{np.mean(m['t4']):>9.3f} {r['inf']['t_tier3']:>7.2f}s "
              f"{r['inf']['t_tier4']:>7.2f}s")

    print("\n" + "=" * 70)
    print("SEQUENCE LENGTH SCALING (bottleneck, n=100, K=64)")
    print("=" * 70)
    print(f"{'SeqLen':>8} {'S':>7} {'Tier 1 r':>9} {'Tier 3 r':>9} {'Tier 4 r':>9} "
          f"{'T3 time':>8} {'T4 time':>8}")
    print("-" * 60)
    for sl in [100_000, 250_000, 500_000, 1_000_000, 2_000_000, 5_000_000]:
        r = results[("seq_scaling", sl)]
        m = r["metrics"]
        print(f"{sl/1e6:>7.1f}M {r['S']:>7} {np.mean(m['t1']):>9.3f} "
              f"{np.mean(m['t3']):>9.3f} {np.mean(m['t4']):>9.3f} "
              f"{r['inf']['t_tier3']:>7.2f}s {r['inf']['t_tier4']:>7.2f}s")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Comprehensive tmrca_cu benchmarks")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip data generation, regenerate plots from cache")
    args = parser.parse_args()

    t_total = time.perf_counter()

    if args.plot_only:
        print("--plot-only not yet supported (results contain non-serializable objects)")
        print("Running full benchmark instead.")

    results = run_all(args)

    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    fig1_scatter_grid(results)
    fig2_landscape_scenarios(results)
    fig3_ll_convergence(results)
    fig4_scaling_n(results)
    fig5_scaling_seqlen(results)
    fig6_calibration(results)
    fig7_prior_recovery(results)

    print_summary(results)

    elapsed = time.perf_counter() - t_total
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
