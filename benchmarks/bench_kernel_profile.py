"""Profile kernel time vs overhead for fwd-only cached kernel.
Separates: data upload, cudaMalloc, kernel, D2H copy."""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import msprime
import time
import tmrca_cu
from tmrca_cu import _core

MU = 1.25e-8
RHO = 1e-8
NE = 10_000

def simulate(n_hap, seq_len, seed=42):
    ts = msprime.sim_ancestry(
        samples=n_hap // 2, sequence_length=seq_len,
        recombination_rate=RHO, population_size=NE, random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=seed + 1)
    return ts

configs = [
    (50,  1_000_000),
    (100, 1_000_000),
    (200, 1_000_000),
    (200, 5_000_000),
    (400, 1_000_000),
]

FF_PATH = "/sietch_colab/kkor/gamma_smc/resources/default_flow_field.txt"

print(f"{'n_hap':>6} {'seq':>6} {'pairs':>8} {'SNPs':>8} "
      f"{'1st call':>10} {'2nd call':>10} {'3rd call':>10} "
      f"{'output MB':>10} {'sp/s (2nd)':>14}")
print("-" * 100)

for n_hap, seq_len in configs:
    n_pairs = n_hap * (n_hap - 1) // 2
    ts = simulate(n_hap, seq_len)
    G = ts.genotype_matrix().T.astype(np.uint8)
    pos = np.array([v.position for v in ts.variants()])
    S = len(pos)
    pairs = [(i, j) for i in range(n_hap) for j in range(i + 1, n_hap)]
    sp = n_pairs * S
    out_mb = n_pairs * S * 4 / 1e6

    times = []
    for rep in range(4):
        t0 = time.perf_counter()
        result = _core.gamma_smc_flow_cached_fwd(
            G, pos, pairs, float(NE), MU, RHO, FF_PATH, True, 0)
        t = time.perf_counter() - t0
        times.append(t)

    seq_str = f"{seq_len/1e6:.0f}Mb"
    sps = sp / times[1] if times[1] > 0 else 0
    print(f"{n_hap:>6} {seq_str:>6} {n_pairs:>8} {S:>8} "
          f"{times[0]:>9.3f}s {times[1]:>9.3f}s {times[2]:>9.3f}s "
          f"{out_mb:>9.1f} {sps:>13,.0f}")
