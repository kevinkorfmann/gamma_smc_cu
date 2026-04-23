#!/usr/bin/env python
"""Verify bit-identical output between main and perf branch builds.

Runs infer_blockwise on 10 pairs and saves the output. Run once per branch,
then compare the saved .npy files.
"""
import numpy as np, sys, os, argparse
REPO = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
sys.path.insert(0, os.path.join(REPO, "python"))
import gamma_smc_cu

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

pairs = [(0,1), (2,3), (4,5), (6,7), (8,9),
         (10,11), (12,13), (14,15), (16,17), (18,19)]

parser = argparse.ArgumentParser()
parser.add_argument("--save", required=True, help="output .npy path")
parser.add_argument("--compare", help="compare against this .npy")
args = parser.parse_args()

print(f"Running infer_blockwise on {len(pairs)} pairs...", flush=True)
r = gamma_smc_cu.infer_blockwise(G_pop, pos, mu=1.25e-8, rho=1e-8, Ne=10000,
                              pairs=pairs, mean_only=True,
                              auto_estimate_theta=True)
mean = r["mean"]
print(f"Output shape: {mean.shape}, range: [{mean.min():.2f}, {mean.max():.2f}]", flush=True)
np.save(args.save, mean)
print(f"Saved to {args.save}", flush=True)

if args.compare:
    ref = np.load(args.compare)
    if np.array_equal(mean, ref):
        print("BIT-IDENTICAL ✓", flush=True)
    else:
        diff = np.abs(mean - ref)
        print(f"NOT bit-identical. Max diff: {diff.max():.2e}, mean diff: {diff.mean():.2e}", flush=True)
        n_diff = (mean != ref).sum()
        print(f"  {n_diff} / {mean.size} values differ ({100*n_diff/mean.size:.4f}%)", flush=True)
