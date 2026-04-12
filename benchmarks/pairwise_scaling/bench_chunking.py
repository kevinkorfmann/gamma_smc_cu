#!/usr/bin/env python
"""Compare chunking strategies for tmrca.cu on chr22 YRI."""
import numpy as np
import sys
import time
import os

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
        if len(p) >= 7:
            pops[p[1]] = p[5]

idx = []
for i, s in enumerate(sids):
    if s in pops and pops[s] == "YRI":
        idx.extend([2 * i, 2 * i + 1])
G_pop = np.ascontiguousarray(G[np.array(idx), :])
pos = np.asarray(pos)
n = G_pop.shape[0]
all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
print(f"{n} haps, {G_pop.shape[1]} sites, {len(all_pairs)} pairs", flush=True)

# Warmup
tmrca_cu.infer_blockwise(G_pop, pos, mu=1.25e-8, rho=1e-8, Ne=10000,
                          pairs=[(0, 1)], mean_only=True, auto_estimate_theta=True)

n_total = len(all_pairs)

def bench_python_chunks(label, chunk_size, streams=1):
    """Python-level chunking: multiple infer_blockwise calls."""
    t0 = time.time()
    for ci in range(0, n_total, chunk_size):
        chunk = all_pairs[ci:ci + chunk_size]
        r = tmrca_cu.infer_blockwise(G_pop, pos, mu=1.25e-8, rho=1e-8, Ne=10000,
                                      pairs=chunk, mean_only=True,
                                      auto_estimate_theta=True,
                                      max_streams=streams)
        del r
    elapsed = time.time() - t0
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    print(f"  {label:45s}: {elapsed:7.2f}s  ({n_chunks} Python calls)", flush=True)
    return elapsed

def bench_cpp_internal_chunks(label, pair_batch_size, streams=1):
    """C++ internal chunking: single infer_blockwise call, C++ loops over pair batches."""
    t0 = time.time()
    r = tmrca_cu.infer_blockwise(G_pop, pos, mu=1.25e-8, rho=1e-8, Ne=10000,
                                  pairs=all_pairs, mean_only=True,
                                  auto_estimate_theta=True,
                                  pair_batch_size=pair_batch_size,
                                  max_streams=streams,
                                  verbose=True)
    elapsed = time.time() - t0
    print(f"  {label:45s}: {elapsed:7.2f}s  (1 Python call, C++ internal batch={pair_batch_size})", flush=True)
    del r
    return elapsed

print("\n=== Baseline: Python-level chunking ===", flush=True)
results = {}
results["py_30k_1s"] = bench_python_chunks("Python 30k chunks, 1 stream", 30000, 1)

print("\n=== Iteration 1: C++ internal chunking ===", flush=True)
for pbs in [10000, 20000, 30000, 63190]:
    for streams in [1, 2]:
        key = f"cpp_{pbs}_{streams}s"
        label = f"C++ batch={pbs}, {streams} stream{'s' if streams>1 else ''}"
        try:
            results[key] = bench_cpp_internal_chunks(label, pbs, streams)
        except Exception as e:
            print(f"  {label:45s}: FAILED ({e})", flush=True)

print("\n=== Summary ===", flush=True)
best_key = min(results, key=results.get)
best_time = results[best_key]
baseline = results.get("py_30k_1s", 999)
print(f"  Baseline (Python 30k chunks): {baseline:.2f}s", flush=True)
print(f"  Best:    {best_key} = {best_time:.2f}s", flush=True)
print(f"  Improvement: {baseline/best_time:.1f}x faster", flush=True)
print(f"  vs gamma_smc (866s): {866/best_time:.0f}x speedup", flush=True)
print(f"  vs ASMC (156535s):   {156535/best_time:.0f}x speedup", flush=True)
