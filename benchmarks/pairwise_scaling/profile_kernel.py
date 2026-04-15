#!/usr/bin/env python
"""Profile kernel timing separately from D2H by measuring small vs large pairs."""
import numpy as np, sys, os, time
REPO = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
sys.path.insert(0, os.path.join(REPO, "python"))
import tmrca_cu

d = np.load(os.path.join(REPO, "analysis/genome_wide/cache/parsed/chr22.npz"),
            allow_pickle=True, mmap_mode="r")
G, pos, sids = d["G"], d["positions"], d["sample_ids"]
pops = {}
with open(os.path.join(REPO, "analysis/genome_wide/data/samples.txt")) as f:
    next(f)
    for line in f:
        p = line.strip().split()
        if len(p) >= 7: pops[p[1]] = p[5]
idx = [2*i+j for i, s in enumerate(sids) if s in pops and pops[s] == "YRI" for j in (0,1)]
G_pop = np.ascontiguousarray(G[np.array(idx), :])
pos = np.asarray(pos)
n = G_pop.shape[0]
print(f"{n} haps, {G_pop.shape[1]} sites", flush=True)

# Warmup
tmrca_cu.infer_blockwise(G_pop, pos, mu=1.25e-8, rho=1e-8, Ne=10000,
                          pairs=[(0,1)], mean_only=True, auto_estimate_theta=True)

# Measure at different pair counts to separate kernel scaling from D2H scaling
# D2H scales linearly with n_pairs (output size = sites × pairs × 4 bytes)
# Kernel scales sub-linearly (GPU parallelism)
print("\n=== Pair count scaling (all single call, pair_batch_size=n_pairs) ===", flush=True)
print(f"{'n_pairs':>8} {'time_s':>8} {'output_GB':>10} {'per_pair_ms':>12}", flush=True)

all_pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
np.random.default_rng(42).shuffle(all_pairs)

for np_ in [10, 50, 100, 500, 1000, 5000, 10000, 30000, 63190]:
    if np_ > len(all_pairs):
        break
    pairs = all_pairs[:np_]
    t0 = time.time()
    r = tmrca_cu.infer_blockwise(G_pop, pos, mu=1.25e-8, rho=1e-8, Ne=10000,
                                  pairs=pairs, mean_only=True,
                                  auto_estimate_theta=True,
                                  pair_batch_size=np_)
    elapsed = time.time() - t0
    out_gb = r['mean'].nbytes / 1e9
    per_pair = elapsed / np_ * 1000
    print(f"{np_:>8} {elapsed:>8.3f} {out_gb:>10.2f} {per_pair:>12.4f}", flush=True)
    del r

print("\nDONE", flush=True)
