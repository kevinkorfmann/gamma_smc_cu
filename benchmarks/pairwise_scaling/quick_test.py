#!/usr/bin/env python
"""Quick accuracy + speed test with small pair count."""
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
pairs = [(0,1), (2,3), (4,5), (6,7), (8,9)]

print(f"{G_pop.shape[0]} haps, {G_pop.shape[1]} sites, {len(pairs)} pairs", flush=True)

# Test with 5 pairs
print("\n--- 5 pairs, verbose ---", flush=True)
t0 = time.time()
r = tmrca_cu.infer_blockwise(G_pop, pos, mu=1.25e-8, rho=1e-8, Ne=10000,
                              pairs=pairs, mean_only=True,
                              auto_estimate_theta=True, verbose=True)
print(f"Time: {time.time()-t0:.3f}s", flush=True)
print(f"Output shape: {r['mean'].shape}", flush=True)
print(f"Mean TMRCA at site 0: {r['mean'][0]}", flush=True)
print(f"Mean TMRCA range: [{r['mean'].min():.1f}, {r['mean'].max():.1f}]", flush=True)

# Test with 1000 pairs
pairs_1k = [(i,j) for i in range(20) for j in range(i+1,20)][:1000]
print(f"\n--- {len(pairs_1k)} pairs ---", flush=True)
t0 = time.time()
r = tmrca_cu.infer_blockwise(G_pop, pos, mu=1.25e-8, rho=1e-8, Ne=10000,
                              pairs=pairs_1k, mean_only=True,
                              auto_estimate_theta=True, verbose=True)
print(f"Time: {time.time()-t0:.3f}s", flush=True)
print(f"Mean TMRCA range: [{r['mean'].min():.1f}, {r['mean'].max():.1f}]", flush=True)
print("DONE", flush=True)
