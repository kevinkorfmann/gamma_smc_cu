"""
Accuracy plots: HMM posterior TMRCA estimates vs true TMRCAs from msprime.

Generates:
  - Per-site TMRCA landscape for individual pairs (estimated vs true)
  - Scatter plot of posterior mean vs true TMRCA across many pairs
  - Calibration: do 95% credible intervals cover the truth ~95% of the time?
  - Tier 1 vs Tier 3 comparison
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
from tmrca_cu import CoalescenceEstimator

out_dir = os.path.dirname(__file__)

# ── simulate ─────────────────────────────────────────────────────────────
print("Simulating tree sequence (n=40, 500 kb, constant Ne=10000) ...")
ts = msprime.sim_ancestry(
    samples=20,
    sequence_length=500_000,
    recombination_rate=1e-8,
    population_size=10_000,
    random_seed=42,
)
ts = msprime.sim_mutations(ts, rate=1.25e-8, random_seed=43)
G = ts.genotype_matrix().T.astype(np.uint8)
positions = np.array([v.position for v in ts.variants()], dtype=np.float64)
n = G.shape[0]
S = G.shape[1]
print(f"  n={n} haplotypes, S={S} sites")

Ne = 10_000.0
mu = 1.25e-8
rho = 1e-8
K = 32


def true_tmrca_at_sites(ts, i, j, site_positions):
    """Extract true TMRCA at each variant site."""
    t = np.empty(len(site_positions))
    tree_iter = ts.trees()
    tree = next(tree_iter)
    for s_idx, pos in enumerate(site_positions):
        while pos >= tree.interval.right:
            tree = next(tree_iter)
        t[s_idx] = tree.tmrca(i, j)
    return t


# ── select pairs ─────────────────────────────────────────────────────────
rng = np.random.RandomState(99)
n_pairs_scatter = 80
pair_set = set()
while len(pair_set) < n_pairs_scatter:
    a, b = sorted(rng.choice(n, 2, replace=False))
    pair_set.add((int(a), int(b)))
all_pairs = sorted(pair_set)

# showcase pairs for landscape plots (pick diverse ones)
showcase_pairs = all_pairs[:4]

# ── compute true TMRCAs ──────────────────────────────────────────────────
print("Extracting true TMRCAs ...")
true_tmrca = {}
for pair in all_pairs:
    true_tmrca[pair] = true_tmrca_at_sites(ts, pair[0], pair[1], positions)

# ── run HMM inference ────────────────────────────────────────────────────
print(f"Running HMM forward-backward for {n_pairs_scatter} pairs ...")
midpoints = np.array(tmrca_cu.time_midpoints(K=K, Ne=Ne))
boundaries = np.array(tmrca_cu.time_boundaries(K=K, Ne=Ne))

posteriors = {}
post_means = {}
post_lowers = {}
post_uppers = {}

for pair in all_pairs:
    gamma = np.array(tmrca_cu.hmm_posterior(
        G, positions, pair, K=K, Ne=Ne, mu=mu, rho=rho
    ))
    posteriors[pair] = gamma
    post_means[pair] = gamma @ midpoints
    # credible intervals
    cum = np.cumsum(gamma, axis=1)
    lo_idx = np.array([np.searchsorted(cum[s], 0.025) for s in range(S)])
    hi_idx = np.array([np.searchsorted(cum[s], 0.975) for s in range(S)])
    post_lowers[pair] = midpoints[np.minimum(lo_idx, K - 1)]
    post_uppers[pair] = midpoints[np.minimum(hi_idx, K - 1)]

# ── Tier 1 landscape ────────────────────────────────────────────────────
print("Computing Tier 1 windowed divergence ...")
est = CoalescenceEstimator(G, positions, mu=mu, rho=rho, Ne=Ne)

# ════════════════════════════════════════════════════════════════════════
# Plot 1: Per-site TMRCA landscape for showcase pairs
# ════════════════════════════════════════════════════════════════════════
print("Plotting landscapes ...")
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("TMRCA landscape: HMM posterior vs truth (msprime)", fontsize=14, fontweight="bold")

for idx, pair in enumerate(showcase_pairs):
    ax = axes[idx // 2, idx % 2]
    i, j = pair
    true_t = true_tmrca[pair]
    est_t = post_means[pair]
    lo_t = post_lowers[pair]
    hi_t = post_uppers[pair]

    # Subsample for plotting clarity
    step = max(1, S // 800)
    xs = positions[::step] / 1e3  # kb

    ax.fill_between(xs, lo_t[::step], hi_t[::step], alpha=0.2, color="C0", label="95% CI")
    ax.plot(xs, true_t[::step], color="C3", linewidth=0.8, alpha=0.8, label="True TMRCA")
    ax.plot(xs, est_t[::step], color="C0", linewidth=0.8, alpha=0.8, label="Posterior mean")

    ax.set_xlabel("Position (kb)")
    ax.set_ylabel("TMRCA (generations)")
    ax.set_title(f"Pair ({i}, {j})")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(out_dir, "accuracy_landscape.png"), dpi=150, bbox_inches="tight")
print(f"  Saved accuracy_landscape.png")

# ════════════════════════════════════════════════════════════════════════
# Plot 2: Scatter — posterior mean vs true TMRCA (per-pair average)
# ════════════════════════════════════════════════════════════════════════
print("Plotting scatter ...")
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("HMM accuracy: posterior mean vs true TMRCA", fontsize=14, fontweight="bold")

# (a) Per-pair genome-wide average
ax = axes[0]
true_avg = np.array([true_tmrca[p].mean() for p in all_pairs])
est_avg = np.array([post_means[p].mean() for p in all_pairs])
ax.scatter(true_avg, est_avg, s=20, alpha=0.6, c="C0", edgecolors="none")
lims = [0, max(true_avg.max(), est_avg.max()) * 1.1]
ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="y = x")
corr = np.corrcoef(true_avg, est_avg)[0, 1]
ax.set_xlabel("True mean TMRCA (generations)")
ax.set_ylabel("Estimated mean TMRCA")
ax.set_title(f"Per-pair average (r = {corr:.3f})")
ax.legend(fontsize=9)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)

# (b) Per-site scatter (subsample for clarity)
ax = axes[1]
# Pool all pairs, subsample
all_true = np.concatenate([true_tmrca[p] for p in all_pairs])
all_est = np.concatenate([post_means[p] for p in all_pairs])
sub = rng.choice(len(all_true), size=min(5000, len(all_true)), replace=False)
ax.scatter(all_true[sub], all_est[sub], s=3, alpha=0.15, c="C2", edgecolors="none")
lims2 = [0, np.percentile(all_true, 99) * 1.3]
ax.plot(lims2, lims2, "k--", linewidth=1, alpha=0.5)
corr_site = np.corrcoef(all_true[sub], all_est[sub])[0, 1]
ax.set_xlabel("True TMRCA (generations)")
ax.set_ylabel("Estimated TMRCA")
ax.set_title(f"Per-site (r = {corr_site:.3f}, n={len(sub)})")
ax.set_xlim(lims2)
ax.set_ylim(lims2)
ax.set_aspect("equal")
ax.grid(True, alpha=0.2)

# (c) Calibration: coverage of 95% CI
ax = axes[2]
# For each pair, what fraction of sites have true TMRCA within CI?
coverages = []
for pair in all_pairs:
    true_t = true_tmrca[pair]
    lo = post_lowers[pair]
    hi = post_uppers[pair]
    covered = np.mean((true_t >= lo) & (true_t <= hi))
    coverages.append(covered)
coverages = np.array(coverages)

ax.hist(coverages, bins=20, range=(0, 1), color="C1", edgecolor="white", alpha=0.8)
ax.axvline(0.95, color="k", linestyle="--", linewidth=1.5, label="Nominal 95%")
ax.axvline(coverages.mean(), color="C3", linestyle="-", linewidth=1.5,
           label=f"Mean = {coverages.mean():.2f}")
ax.set_xlabel("Fraction of sites covered by 95% CI")
ax.set_ylabel("Number of pairs")
ax.set_title("Credible interval calibration")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(out_dir, "accuracy_scatter.png"), dpi=150, bbox_inches="tight")
print(f"  Saved accuracy_scatter.png")

# ════════════════════════════════════════════════════════════════════════
# Plot 3: Tier 1 vs Tier 3 comparison
# ════════════════════════════════════════════════════════════════════════
print("Plotting Tier 1 vs Tier 3 ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Tier 1 (windowed divergence) vs Tier 3 (HMM) accuracy",
             fontsize=14, fontweight="bold")

# Tier 1 estimate for all pairs
window_sites = max(1, int(round(20_000 / ((positions[-1] - positions[0]) / (S - 1)))))
div_raw = np.array(tmrca_cu.windowed_divergence(G, all_pairs, window_sites))
mean_spacing = (positions[-1] - positions[0]) / (S - 1)
window_bp = window_sites * mean_spacing
tier1_tmrca = {}
for p_idx, pair in enumerate(all_pairs):
    tier1_tmrca[pair] = div_raw[p_idx] / (window_bp * 2.0 * mu)

# (a) Landscape comparison for one pair
ax = axes[0]
pair = showcase_pairs[0]
i, j = pair
step = max(1, S // 800)
xs = positions[::step] / 1e3
ax.plot(xs, true_tmrca[pair][::step], color="C3", linewidth=0.8, alpha=0.8, label="True")
ax.plot(xs, post_means[pair][::step], color="C0", linewidth=0.8, alpha=0.8, label="Tier 3 (HMM)")
ax.plot(xs, tier1_tmrca[pair][::step], color="C1", linewidth=0.8, alpha=0.6, label="Tier 1 (div/2μ)")
ax.set_xlabel("Position (kb)")
ax.set_ylabel("TMRCA (generations)")
ax.set_title(f"Pair ({i}, {j})")
ax.legend(fontsize=9)
ax.set_ylim(bottom=0)
ax.grid(True, alpha=0.2)

# (b) Per-pair correlation comparison
ax = axes[1]
tier3_corrs = []
tier1_corrs = []
for pair in all_pairs:
    true_t = true_tmrca[pair]
    r3 = np.corrcoef(true_t, post_means[pair])[0, 1]
    r1 = np.corrcoef(true_t, tier1_tmrca[pair])[0, 1]
    tier3_corrs.append(r3)
    tier1_corrs.append(r1)
tier3_corrs = np.array(tier3_corrs)
tier1_corrs = np.array(tier1_corrs)

bins = np.linspace(-0.2, 1.0, 30)
ax.hist(tier1_corrs, bins=bins, alpha=0.6, color="C1", edgecolor="white",
        label=f"Tier 1 (median r={np.median(tier1_corrs):.3f})")
ax.hist(tier3_corrs, bins=bins, alpha=0.6, color="C0", edgecolor="white",
        label=f"Tier 3 (median r={np.median(tier3_corrs):.3f})")
ax.set_xlabel("Per-pair Pearson r (estimated vs true TMRCA)")
ax.set_ylabel("Number of pairs")
ax.set_title("Accuracy distribution across pairs")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(out_dir, "accuracy_tier1_vs_tier3.png"), dpi=150, bbox_inches="tight")
print(f"  Saved accuracy_tier1_vs_tier3.png")

# ── summary stats ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Accuracy summary")
print("=" * 60)
print(f"  Per-pair average TMRCA correlation:  r = {corr:.3f}")
print(f"  Per-site TMRCA correlation:          r = {corr_site:.3f}")
print(f"  95% CI mean coverage:                {coverages.mean():.1%}")
print(f"  Tier 3 median per-pair r:            {np.median(tier3_corrs):.3f}")
print(f"  Tier 1 median per-pair r:            {np.median(tier1_corrs):.3f}")
print(f"  Tier 3 improvement over Tier 1:      +{np.median(tier3_corrs) - np.median(tier1_corrs):.3f}")
