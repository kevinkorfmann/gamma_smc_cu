"""Benchmark: Flow-field FB vs discrete HMM vs moment-match forward-only."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import msprime
import time
import tmrca_cu

Ne = 10_000
mu = 1.25e-8
rho = 1e-8

# Simulate
print("Simulating...")
ts = msprime.sim_ancestry(
    20, sequence_length=2_000_000, recombination_rate=rho,
    population_size=Ne, random_seed=42)
ts = msprime.sim_mutations(ts, rate=mu, random_seed=43)
G = ts.genotype_matrix().T.astype(np.uint8)
pos = np.array([v.position for v in ts.variants()])
print(f"n_haps={G.shape[0]}, S={G.shape[1]}")

# Pick 10 pairs
pairs = [(2*i, 2*i+1) for i in range(10)]

# True TMRCA
print("Computing true TMRCA...")
true_tmrca = np.zeros((len(pos), len(pairs)))
for p_idx, (i, j) in enumerate(pairs):
    for s_idx, var in enumerate(ts.variants()):
        tree = ts.at(var.position)
        true_tmrca[s_idx, p_idx] = tree.tmrca(i, j)

# 1. Flow field FB
print("\nRunning flow field FB...")
t0 = time.perf_counter()
flow_result = tmrca_cu.gamma_smc_flow_fb(G, pos, pairs, Ne=Ne, mu=mu, rho=rho, mean_only=True)
t_flow = time.perf_counter() - t0
flow_mean = flow_result['mean']

# 2. Moment-match forward only
print("Running moment-match forward...")
t0 = time.perf_counter()
mm_result = tmrca_cu.gamma_smc_forward(G, pos, pairs, Ne=Ne, mu=mu, rho=rho, mean_only=True)
t_mm = time.perf_counter() - t0
mm_mean = mm_result['mean']

# 3. HMM forward-backward
print("Running HMM forward-backward...")
try:
    t0 = time.perf_counter()
    ctx = tmrca_cu.HMMContext(G, pos, K=32, Ne=Ne, mu=mu, rho=rho)
    hmm_mean, hmm_lower, hmm_upper, hmm_ll = ctx.run_batch(pairs)
    t_hmm = time.perf_counter() - t0
    hmm_ok = True
except Exception as e:
    print(f"  HMM failed: {e}")
    hmm_ok = False
    hmm_mean = None
    t_hmm = 0

# Compare
print(f"\n{'='*70}")
print(f"{'Method':<25} {'Time (s)':<12} {'r(log) mean':<15} {'r(log) per pair'}")
print(f"{'='*70}")

methods = [
    ("Flow field FB", flow_mean, t_flow),
    ("Moment-match fwd", mm_mean, t_mm),
]
if hmm_ok:
    methods.append(("HMM FB (K=32)", hmm_mean, t_hmm))

for name, estimates, elapsed in methods:
    rs = []
    for p_idx in range(len(pairs)):
        log_true = np.log(true_tmrca[:, p_idx] + 1)
        if estimates.ndim == 2 and estimates.shape[0] == len(pos):
            log_est = np.log(np.abs(estimates[:, p_idx]) + 1)
        else:
            log_est = np.log(np.abs(estimates[p_idx, :]) + 1)
        r = np.corrcoef(log_true, log_est)[0, 1]
        rs.append(r)
    avg_r = np.mean(rs)
    per_pair = " ".join(f"{r:.3f}" for r in rs[:5]) + " ..."
    print(f"{name:<25} {elapsed:<12.3f} {avg_r:<15.4f} {per_pair}")

print(f"\nSite-pairs/s:")
sp = len(pos) * len(pairs)
print(f"  Flow field FB:    {sp/t_flow:,.0f}")
print(f"  Moment-match fwd: {sp/t_mm:,.0f}")
if hmm_ok:
    print(f"  HMM FB (K=32):    {sp/t_hmm:,.0f}")
