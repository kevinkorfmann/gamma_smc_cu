"""
Accuracy with adaptive Ne(t) and variable K: demonstrates Tier 4 improvement
when the true demography differs from the assumed constant Ne.

Simulates a bottleneck + expansion, then shows:
  - Tier 3 (constant Ne assumption) is biased
  - Tier 4 (adaptive Ne from data, kernel-level EM) corrects the bias
  - Higher K (64, 128) gives finer time resolution
  - More pairs → better Ne(t) estimate → better per-pair accuracy
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import msprime
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tmrca_cu
import tmrca_cu._core as _core

out_dir = os.path.dirname(__file__)
mu = 1.25e-8
rho = 1e-8
Ne_assumed = 10_000.0  # what we assume (wrong)


def true_tmrca_at_sites(ts, i, j, sp):
    t = np.empty(len(sp))
    ti = ts.trees(); tree = next(ti)
    for si, p in enumerate(sp):
        while p >= tree.interval.right: tree = next(ti)
        t[si] = tree.tmrca(i, j)
    return t


# ══════════════════════════════════════════════════════════════════════════
# Simulate with bottleneck demography
# ══════════════════════════════════════════════════════════════════════════
print("Simulating with bottleneck demography ...")
print("  Present Ne=10k, bottleneck Ne=500 at 2000-3000 gen, ancestral Ne=20k")

demography = msprime.Demography()
demography.add_population(initial_size=10_000)
demography.add_population_parameters_change(time=2000, initial_size=500)
demography.add_population_parameters_change(time=3000, initial_size=20_000)

sample_sizes_diploid = [10, 25, 50, 100, 200]
eval_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
K_values = [32, 64, 128]

results = {}
for n_dip in sample_sizes_diploid:
    n_hap = n_dip * 2
    print(f"\n--- n = {n_hap} haplotypes ---")

    ts = msprime.sim_ancestry(
        samples=n_dip, sequence_length=500_000,
        recombination_rate=rho, demography=demography, random_seed=42,
    )
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=43)
    G = ts.genotype_matrix().T.astype(np.uint8)
    pos = np.array([v.position for v in ts.variants()], dtype=np.float64)
    S = G.shape[1]
    print(f"  S={S} sites")

    true_eval = {p: true_tmrca_at_sites(ts, p[0], p[1], pos) for p in eval_pairs}

    # Background pairs for Ne estimation
    rng = np.random.RandomState(0)
    n_bg = min(500, n_hap * (n_hap - 1) // 2)
    bg = set()
    while len(bg) < n_bg:
        a, b = sorted(rng.choice(n_hap, 2, replace=False))
        bg.add((int(a), int(b)))
    all_pairs = sorted(set(eval_pairs) | bg)
    eval_idx = {p: all_pairs.index(p) for p in eval_pairs}

    res_n = {}
    for K in K_values:
        midpoints = np.array(tmrca_cu.time_midpoints(K=K, Ne=Ne_assumed))

        # Tier 3: constant Ne HMM
        r3 = _core.hmm_posterior_batched(G, pos, all_pairs, K=K, Ne=Ne_assumed, mu=mu, rho=rho)
        gamma_t3 = np.array(r3[0])

        # Tier 4: adaptive prior (kernel-level EM)
        r4 = _core.adaptive_prior_infer(G, pos, all_pairs, K=K, Ne=Ne_assumed, mu=mu, rho=rho,
                                         max_iterations=5, blend_alpha=0.7)
        gamma_t4 = np.array(r4["gamma"])
        prior_est = np.array(r4["prior"])

        # Tier 1 (only at K=32 to avoid redundancy)
        if K == 32:
            ws = max(1, int(round(20_000 / ((pos[-1]-pos[0])/(S-1)))))
            div_raw = np.array(tmrca_cu.windowed_divergence(G, eval_pairs, ws))
            wbp = ws * (pos[-1]-pos[0])/(S-1)

        # Metrics
        corrs_t3, corrs_t4 = [], []
        rmses_t3, rmses_t4 = [], []
        for p in eval_pairs:
            true_t = true_eval[p]
            idx = eval_idx[p]
            m3 = gamma_t3[idx] @ midpoints
            m4 = gamma_t4[idx] @ midpoints

            corrs_t3.append(np.corrcoef(true_t, m3)[0, 1])
            corrs_t4.append(np.corrcoef(true_t, m4)[0, 1])
            rmses_t3.append(np.sqrt(np.mean((true_t - m3)**2)))
            rmses_t4.append(np.sqrt(np.mean((true_t - m4)**2)))

        res_n[K] = {
            'corrs_t3': np.array(corrs_t3),
            'corrs_t4': np.array(corrs_t4),
            'rmses_t3': np.array(rmses_t3),
            'rmses_t4': np.array(rmses_t4),
            'prior_est': prior_est,
            'gamma_t3': gamma_t3,
            'gamma_t4': gamma_t4,
        }

        delta = np.mean(corrs_t4) - np.mean(corrs_t3)
        rmse_red = 1.0 - np.mean(rmses_t4) / np.mean(rmses_t3)
        print(f"  K={K:>3}: T3 r={np.mean(corrs_t3):.3f}, T4 r={np.mean(corrs_t4):.3f} "
              f"(Δ={delta:+.3f}), RMSE↓={rmse_red:+.1%}")

    # Tier 1 metrics
    corrs_t1 = []
    for p in eval_pairs:
        true_t = true_eval[p]
        m1 = div_raw[eval_pairs.index(p)] / (wbp * 2 * mu)
        corrs_t1.append(np.corrcoef(true_t, m1)[0, 1])

    results[n_hap] = {
        'per_K': res_n,
        'corrs_t1': np.array(corrs_t1),
        'true_eval': true_eval,
        'pos': pos,
        'eval_idx': eval_idx,
        'S': S,
    }


# ══════════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════════
print("\nPlotting ...")
ns = sorted(results.keys())

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.suptitle("Adaptive Ne(t) + Variable K under bottleneck demography\n"
             "(Present Ne=10k → bottleneck Ne=500 at 2-3kya → ancestral Ne=20k)",
             fontsize=13, fontweight="bold")

# (1) Accuracy vs sample size, all K values
ax = axes[0, 0]
m1 = [np.mean(results[n_]['corrs_t1']) for n_ in ns]
ax.plot(ns, m1, "s--", color="C1", linewidth=2, markersize=7, label="Tier 1 (div/2μ)")
colors_K = {"32": "C0", "64": "C4", "128": "C2"}
for K in K_values:
    m3 = [np.mean(results[n_]['per_K'][K]['corrs_t3']) for n_ in ns]
    m4 = [np.mean(results[n_]['per_K'][K]['corrs_t4']) for n_ in ns]
    ax.plot(ns, m3, "o:", color=colors_K[str(K)], linewidth=1, markersize=5,
            alpha=0.5, label=f"T3 K={K}")
    ax.plot(ns, m4, "^-", color=colors_K[str(K)], linewidth=2, markersize=7,
            label=f"T4 K={K}")
ax.set_xlabel("Sample size (n haplotypes)")
ax.set_ylabel("Mean Pearson r")
ax.set_title("Accuracy vs sample size & K")
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)

# (2) RMSE improvement by K
ax = axes[0, 1]
for K in K_values:
    rmse_3 = [np.mean(results[n_]['per_K'][K]['rmses_t3']) for n_ in ns]
    rmse_4 = [np.mean(results[n_]['per_K'][K]['rmses_t4']) for n_ in ns]
    ax.plot(ns, rmse_3, "o:", color=colors_K[str(K)], linewidth=1, markersize=5,
            alpha=0.5, label=f"T3 K={K}")
    ax.plot(ns, rmse_4, "^-", color=colors_K[str(K)], linewidth=2, markersize=7,
            label=f"T4 K={K}")
ax.set_xlabel("Sample size (n haplotypes)")
ax.set_ylabel("Mean RMSE (generations)")
ax.set_title("RMSE vs sample size & K")
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)

# (3) Estimated prior vs true demography prior (K=32)
ax = axes[0, 2]
std_prior = np.array(tmrca_cu.coalescent_prior(Ne=Ne_assumed, K=32))
mid32 = np.array(tmrca_cu.time_midpoints(K=32, Ne=Ne_assumed))
bnd32 = np.array(tmrca_cu.time_boundaries(K=32, Ne=Ne_assumed))

# True coalescent prior under bottleneck: q[k] = exp(-Λ(t_k)) - exp(-Λ(t_{k+1}))
# Demography: Ne=10k for [0,2000), Ne=500 for [2000,3000), Ne=20k for [3000,∞)
# Rate for two lineages under diploid Ne: 1/(2*Ne)
def cum_coal_rate(t):
    """Cumulative coalescent rate Λ(t) = ∫_0^t 1/(2*Ne(s)) ds"""
    if t <= 2000:
        return t / 20_000
    elif t <= 3000:
        return 2000 / 20_000 + (t - 2000) / 1_000
    else:
        return 2000 / 20_000 + 1000 / 1_000 + (t - 3000) / 40_000

true_prior = np.array([np.exp(-cum_coal_rate(bnd32[k])) - np.exp(-cum_coal_rate(bnd32[k+1]))
                        for k in range(32)])
true_prior /= true_prior.sum()

for n_ in [min(ns), max(ns)]:
    ax.plot(mid32, results[n_]['per_K'][32]['prior_est'], "o-", markersize=3,
            label=f"Estimated (n={n_})")
ax.plot(mid32, true_prior, "C3-", linewidth=2, alpha=0.8, label="True (bottleneck)")
ax.plot(mid32, std_prior, "k--", linewidth=2, alpha=0.5,
        label=f"Assumed (const Ne={Ne_assumed:,.0f})")
ax.set_xlabel("Time (generations)")
ax.set_ylabel("Prior q(t)")
ax.set_title("Coalescent prior: assumed vs estimated vs true")
ax.set_xscale("log")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# (4,5) Landscape at smallest and largest n (K=128, Tier 4 vs Tier 3 vs true)
for col, n_ in enumerate([min(ns), max(ns)]):
    ax = axes[1, col]
    r = results[n_]
    K_show = 128
    midpoints = np.array(tmrca_cu.time_midpoints(K=K_show, Ne=Ne_assumed))
    pair = eval_pairs[0]
    idx = r['eval_idx'][pair]
    true_t = r['true_eval'][pair]
    m3 = r['per_K'][K_show]['gamma_t3'][idx] @ midpoints
    m4 = r['per_K'][K_show]['gamma_t4'][idx] @ midpoints
    pos = r['pos']
    step = max(1, r['S'] // 600)
    xs = pos[::step] / 1e3

    ax.plot(xs, true_t[::step], color="C3", linewidth=0.8, alpha=0.8, label="True")
    ax.plot(xs, m3[::step], color="C0", linewidth=0.8, alpha=0.6, label="T3 (const Ne)")
    ax.plot(xs, m4[::step], color="C2", linewidth=0.8, alpha=0.8, label="T4 (adaptive)")
    r3c = np.corrcoef(true_t, m3)[0, 1]
    r4c = np.corrcoef(true_t, m4)[0, 1]
    ax.set_title(f"Pair {pair}, n={n_}, K={K_show}\nT3 r={r3c:.2f}, T4 r={r4c:.2f}")
    ax.set_xlabel("Position (kb)")
    ax.set_ylabel("TMRCA (gen)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2)

# (6) Δr by K and sample size (grouped bar chart)
ax = axes[1, 2]
x = np.arange(len(ns))
width = 0.25
for i, K in enumerate(K_values):
    delta_r = [np.mean(results[n_]['per_K'][K]['corrs_t4']) -
               np.mean(results[n_]['per_K'][K]['corrs_t3']) for n_ in ns]
    ax.bar(x + (i - 1) * width, delta_r, width, label=f"K={K}",
           color=colors_K[str(K)], alpha=0.8, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([str(n_) for n_ in ns], fontsize=9)
ax.set_xlabel("Sample size (n haplotypes)")
ax.set_ylabel("Δr (Tier 4 − Tier 3)")
ax.set_title("Improvement from adaptive Ne(t)")
ax.axhline(0, color="k", linewidth=0.5)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(os.path.join(out_dir, "accuracy_ep_convergence.png"), dpi=150, bbox_inches="tight")
print(f"  Saved accuracy_ep_convergence.png")

# ── summary table ────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("Summary: bottleneck demography, accuracy vs sample size and K")
print("=" * 80)
print(f"{'n':>6}  {'K':>4}  {'Tier 1':>8}  {'Tier 3':>8}  {'Tier 4':>8}  {'Δ(T4-T3)':>10}  {'RMSE↓':>8}")
print("-" * 64)
for n_ in ns:
    r = results[n_]
    r1 = np.mean(r['corrs_t1'])
    for K in K_values:
        rk = r['per_K'][K]
        r3 = np.mean(rk['corrs_t3'])
        r4 = np.mean(rk['corrs_t4'])
        rmse_red = 1.0 - np.mean(rk['rmses_t4']) / np.mean(rk['rmses_t3'])
        t1_str = f"{r1:>8.3f}" if K == K_values[0] else " " * 8
        print(f"{n_:>6}  {K:>4}  {t1_str}  {r3:>8.3f}  {r4:>8.3f}  {r4-r3:>+10.3f}  {rmse_red:>+7.1%}")
