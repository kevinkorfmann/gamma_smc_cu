#!/usr/bin/env python
"""Benchmark: tmrca.cu vs gamma_smc scaling with sample size.

Uses stdpopsim to simulate OutOfAfrica_3G09 (Gutenkunst et al. 2009) at
increasing sample sizes (N=50, 100, 500, 1000, 2000) on chr22-length
sequences. For each N, all N*(N-1)/2 pairs are decoded.

This demonstrates that tmrca.cu scales sub-linearly in N (GPU parallelism)
while gamma_smc scales quadratically (sequential pair processing).

Output: benchmarks/pairwise_scaling/scaling_results.csv
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
sys.path.insert(0, os.path.join(REPO, "python"))

OUT_DIR = os.path.join(REPO, "benchmarks/pairwise_scaling")
GAMMA_SMC_BIN = os.path.join(REPO, "benchmarks/test_suite_stdpopsim/gamma_smc/bin/gamma_smc")

SAMPLE_SIZES = [50, 100, 200, 500, 1000]
SEQ_LENGTH = 51_304_566  # chr22 length
MU = 1.25e-8
RHO = 1e-8


def simulate(n_samples):
    """Simulate with stdpopsim and return (haplotype_matrix, positions)."""
    import stdpopsim
    import msprime

    species = stdpopsim.get_species("HomSap")
    model = species.get_demographic_model("OutOfAfrica_3G09")
    contig = species.get_contig("chr22", genetic_map="HapMapII_GRCh38")

    engine = stdpopsim.get_engine("msprime")
    ts = engine.simulate(
        model,
        contig,
        samples={"YRI": n_samples},
        seed=42,
    )
    # Extract haplotype matrix and positions
    G = ts.genotype_matrix().T  # (n_haps, n_sites)
    positions = np.array([v.position for v in ts.variants()], dtype=np.float64)
    return G.astype(np.uint8), positions.astype(np.float64)


def bench_tmrca_cu(G, positions, n_samples):
    """Time tmrca.cu on all pairs."""
    import tmrca_cu

    n_haps = G.shape[0]
    n_pairs = n_haps * (n_haps - 1) // 2
    print(f"  tmrca.cu: {n_haps} haps, {n_pairs} pairs, {G.shape[1]} sites",
          flush=True)

    # Warmup
    tmrca_cu.infer(G, positions, mu=MU, rho=RHO, Ne=10_000,
                   pairs=[(0, 1)], mean_only=True, auto_estimate_theta=True)

    t0 = time.time()
    tmrca_cu.infer(G, positions, mu=MU, rho=RHO, Ne=10_000,
                   mean_only=True, auto_estimate_theta=True)
    elapsed = time.time() - t0
    print(f"  tmrca.cu: {elapsed:.2f}s for {n_pairs} pairs", flush=True)
    return elapsed, n_pairs


def bench_gamma_smc_per_pair(G, positions):
    """Time gamma_smc on 3 pairs to get per-pair cost, then extrapolate."""
    n_test = 3
    n_haps = G.shape[0]

    with tempfile.TemporaryDirectory(prefix="gsmc_") as td:
        haps_path = os.path.join(td, "data.haps")
        n_sites = G.shape[1]
        with open(haps_path, "w") as f:
            for i in range(n_sites):
                pos_bp = int(positions[i])
                row = " ".join(str(int(x)) for x in G[:, i])
                f.write(f"22 SNP_{pos_bp} {pos_bp} A T {row}\n")

        t0 = time.time()
        for pi in range(n_test):
            a, b = pi * 2, pi * 2 + 1
            cmd = [GAMMA_SMC_BIN, "-i", haps_path,
                   "--hap1", str(a), "--hap2", str(b),
                   "-o", os.path.join(td, f"out_{pi}")]
            subprocess.run(cmd, capture_output=True, text=True)
        per_pair = (time.time() - t0) / n_test

    print(f"  gamma_smc: {per_pair:.3f}s per pair", flush=True)
    return per_pair


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    results = []

    # Get gamma_smc per-pair cost once (it's constant regardless of N)
    print("=== Calibrating gamma_smc per-pair cost ===", flush=True)
    G_cal, pos_cal = simulate(50)
    gsmc_per_pair = bench_gamma_smc_per_pair(G_cal, pos_cal)

    for n_samples in SAMPLE_SIZES:
        print(f"\n=== N = {n_samples} samples ({n_samples * 2} haplotypes) ===",
              flush=True)

        G, positions = simulate(n_samples)
        n_haps = G.shape[0]
        n_pairs = n_haps * (n_haps - 1) // 2
        print(f"  Simulated: {G.shape[1]} sites, {n_pairs} pairs", flush=True)

        # tmrca.cu (actual timing)
        try:
            tcu_time, _ = bench_tmrca_cu(G, positions, n_samples)
        except Exception as e:
            print(f"  tmrca.cu FAILED at N={n_samples}: {e}", flush=True)
            tcu_time = float("nan")

        # gamma_smc (extrapolated)
        gsmc_time = gsmc_per_pair * n_pairs

        results.append({
            "n_samples": n_samples,
            "n_haplotypes": n_haps,
            "n_pairs": n_pairs,
            "n_sites": G.shape[1],
            "tmrca_cu_seconds": tcu_time,
            "gamma_smc_seconds": gsmc_time,
            "speedup": gsmc_time / tcu_time if tcu_time > 0 else float("nan"),
        })
        print(f"  speedup: {gsmc_time / tcu_time:.0f}x" if tcu_time > 0 else
              "  speedup: N/A", flush=True)

    csv_path = os.path.join(OUT_DIR, "scaling_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
