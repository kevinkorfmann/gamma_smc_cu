"""
Deep benchmark: tmrca.cu vs gamma_smc scaling comparison.

Sweeps across:
  - n_haplotypes: 2, 10, 20, 50, 100, 200, 500
  - seq_lengths:  500kb, 1Mb, 5Mb, 10Mb
  - measures wall-clock, throughput (site·pairs/s), per-pair time

Saves raw timings to benchmarks/bench_gamma_deep.npz for figure generation.

Usage:
    pixi run python benchmarks/bench_gamma_smc_deep.py
"""

import subprocess
import tempfile
import time
import os
import json
import sys
import numpy as np

# ── Paths ──
GAMMA_SMC_BIN = "/sietch_colab/kkor/gamma_smc/bin/gamma_smc"

# ── Parameters ──
MU = 1.25e-8
RHO = 1e-8
NE = 10_000
K = 32

# ── Sweep grid ──
N_HAPLOTYPES = [2, 10, 20, 50, 100, 200]
SEQ_LENGTHS = [500_000, 1_000_000, 5_000_000, 10_000_000]

# For very large configs, cap gamma_smc to avoid multi-hour waits
GAMMA_MAX_WALL_S = 300  # 5 min timeout per config


def simulate(n_hap, seq_len, seed=42):
    import msprime
    ts = msprime.sim_ancestry(
        samples=n_hap // 2,
        sequence_length=seq_len,
        recombination_rate=RHO,
        population_size=NE,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=seed + 1)
    return ts


def ts_to_genotype_matrix(ts):
    G = ts.genotype_matrix().T.astype(np.uint8)
    positions = np.array([v.position for v in ts.variants()])
    return G, positions


def ts_to_vcf(ts, vcf_path):
    with open(vcf_path, "w") as f:
        ts.write_vcf(f, contig_id="chr1")
    subprocess.run(
        f"bgzip -f {vcf_path} && tabix -p vcf {vcf_path}.gz",
        shell=True, check=True, capture_output=True,
    )
    return vcf_path + ".gz"


def bench_gamma_smc(ts, tmpdir, timeout=GAMMA_MAX_WALL_S):
    """Run gamma_smc on all pairs. Returns wall-clock time or None."""
    vcf_path = ts_to_vcf(ts, os.path.join(tmpdir, "sim.vcf"))
    output_path = os.path.join(tmpdir, "gamma_out.zst")
    rho_over_mu = RHO / MU

    cmd = [
        GAMMA_SMC_BIN,
        "-i", vcf_path,
        "-o", output_path,
        "-t", str(rho_over_mu),
        "-s", "100",
    ]

    env = os.environ.copy()
    pixi_lib = "/sietch_colab/kkor/tmrca.cu/.pixi/envs/default/lib"
    env["LD_LIBRARY_PATH"] = pixi_lib + ":" + env.get("LD_LIBRARY_PATH", "")

    try:
        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env)
        t_wall = time.perf_counter() - t0
        if result.returncode != 0:
            stderr = result.stderr.decode()[:200]
            print(f"    gamma_smc error: {stderr}")
            return None
        return t_wall
    except subprocess.TimeoutExpired:
        print(f"    gamma_smc TIMEOUT ({timeout}s)")
        return None


def bench_tmrca_cu(ts, n_pairs_limit=None):
    """Run tmrca.cu batched API. Returns (wall_time, n_pairs, n_sites)."""
    from tmrca_cu import _core

    G, positions = ts_to_genotype_matrix(ts)
    n = G.shape[0]
    S = G.shape[1]

    if S < 10:
        return None, 0, S

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if n_pairs_limit is not None:
        all_pairs = all_pairs[:n_pairs_limit]

    t0 = time.perf_counter()
    _core.hmm_posterior_batched(
        G, positions, all_pairs, K, float(NE), MU, RHO, -1.0)
    t_wall = time.perf_counter() - t0

    return t_wall, len(all_pairs), S


def main():
    import msprime

    print("=" * 80)
    print("DEEP BENCHMARK: tmrca.cu vs gamma_smc")
    print("=" * 80)
    print(f"Parameters: mu={MU}, rho={RHO}, Ne={NE}, K={K}")
    print(f"Haplotypes: {N_HAPLOTYPES}")
    print(f"Seq lengths: {[f'{l/1e6:.1f}Mb' for l in SEQ_LENGTHS]}")
    print()

    # Storage: dicts of (n_hap, seq_len) -> measurement
    results = {
        "n_haplotypes": N_HAPLOTYPES,
        "seq_lengths": SEQ_LENGTHS,
        "gamma_wall": {},    # (n_hap, seq_len) -> seconds
        "cu_wall": {},       # (n_hap, seq_len) -> seconds
        "n_snps": {},        # (n_hap, seq_len) -> int
        "n_pairs": {},       # n_hap -> int
    }

    # ═══════════════════════════════════════════════════════════
    # Experiment 1: Fixed seq_len=1Mb, sweep n_haplotypes
    # ═══════════════════════════════════════════════════════════
    print("─" * 80)
    print("Experiment 1: Scaling with sample size (seq_len = 1 Mb)")
    print("─" * 80)
    hdr = (f"{'n_hap':>6s}  {'pairs':>8s}  {'SNPs':>8s}  "
           f"{'gamma_smc':>12s}  {'tmrca.cu':>12s}  "
           f"{'gamma kp/s':>12s}  {'cu kp/s':>12s}  {'Speedup':>8s}")
    print(hdr)
    print("─" * len(hdr))

    seq_len = 1_000_000
    for n_hap in N_HAPLOTYPES:
        n_pairs = n_hap * (n_hap - 1) // 2
        results["n_pairs"][n_hap] = n_pairs

        ts = simulate(n_hap=n_hap, seq_len=seq_len)
        n_snps = ts.num_mutations
        results["n_snps"][(n_hap, seq_len)] = n_snps

        print(f"{n_hap:>6d}  {n_pairs:>8d}  {n_snps:>8d}", end="  ", flush=True)

        # gamma_smc
        t_gamma = None
        with tempfile.TemporaryDirectory() as tmpdir:
            t_gamma = bench_gamma_smc(ts, tmpdir)
        results["gamma_wall"][(n_hap, seq_len)] = t_gamma

        if t_gamma is not None:
            kpairs_gamma = n_pairs * n_snps / t_gamma / 1e3
            print(f"{t_gamma:>10.2f}s", end="  ", flush=True)
        else:
            kpairs_gamma = 0
            print(f"{'TIMEOUT':>12s}", end="  ", flush=True)

        # tmrca.cu
        try:
            t_cu, np_cu, S = bench_tmrca_cu(ts)
            kpairs_cu = np_cu * S / t_cu / 1e3
            results["cu_wall"][(n_hap, seq_len)] = t_cu
            print(f"{t_cu:>10.3f}s", end="  ", flush=True)
        except Exception as e:
            t_cu = None
            kpairs_cu = 0
            results["cu_wall"][(n_hap, seq_len)] = None
            print(f"{'ERR':>12s}", end="  ", flush=True)

        # Rates (kilo-pairs·sites/s)
        if t_gamma is not None:
            print(f"{kpairs_gamma:>10.0f}", end="  ", flush=True)
        else:
            print(f"{'N/A':>12s}", end="  ", flush=True)
        if t_cu is not None:
            print(f"{kpairs_cu:>10.0f}", end="  ", flush=True)
        else:
            print(f"{'N/A':>12s}", end="  ", flush=True)

        if t_gamma is not None and t_cu is not None and t_cu > 0:
            print(f"{t_gamma/t_cu:>6.1f}x")
        else:
            print(f"{'N/A':>8s}")

    # ═══════════════════════════════════════════════════════════
    # Experiment 2: Fixed n_hap=10, sweep seq_lengths
    # ═══════════════════════════════════════════════════════════
    print()
    print("─" * 80)
    print("Experiment 2: Scaling with sequence length (n_hap = 10)")
    print("─" * 80)
    hdr2 = (f"{'seq_len':>10s}  {'pairs':>6s}  {'SNPs':>10s}  "
            f"{'gamma_smc':>12s}  {'tmrca.cu':>12s}  "
            f"{'gamma/pair':>12s}  {'cu/pair':>12s}  {'Speedup':>8s}")
    print(hdr2)
    print("─" * len(hdr2))

    n_hap = 10
    n_pairs = n_hap * (n_hap - 1) // 2

    for seq_len in SEQ_LENGTHS:
        ts = simulate(n_hap=n_hap, seq_len=seq_len)
        n_snps = ts.num_mutations
        results["n_snps"][(n_hap, seq_len)] = n_snps

        print(f"{seq_len/1e6:>8.1f}Mb  {n_pairs:>6d}  {n_snps:>10d}", end="  ", flush=True)

        # gamma_smc
        t_gamma = None
        with tempfile.TemporaryDirectory() as tmpdir:
            t_gamma = bench_gamma_smc(ts, tmpdir)
        results["gamma_wall"][(n_hap, seq_len)] = t_gamma

        if t_gamma is not None:
            per_pair_gamma = t_gamma / n_pairs
            print(f"{t_gamma:>10.2f}s", end="  ", flush=True)
        else:
            per_pair_gamma = None
            print(f"{'TIMEOUT':>12s}", end="  ", flush=True)

        # tmrca.cu
        try:
            t_cu, np_cu, S = bench_tmrca_cu(ts)
            per_pair_cu = t_cu / np_cu
            results["cu_wall"][(n_hap, seq_len)] = t_cu
            print(f"{t_cu:>10.3f}s", end="  ", flush=True)
        except Exception as e:
            t_cu = None
            per_pair_cu = None
            results["cu_wall"][(n_hap, seq_len)] = None
            print(f"{'ERR':>12s}", end="  ", flush=True)

        # Per-pair time
        if per_pair_gamma is not None:
            print(f"{per_pair_gamma*1e3:>9.1f}ms", end="  ", flush=True)
        else:
            print(f"{'N/A':>12s}", end="  ", flush=True)
        if per_pair_cu is not None:
            print(f"{per_pair_cu*1e3:>9.2f}ms", end="  ", flush=True)
        else:
            print(f"{'N/A':>12s}", end="  ", flush=True)

        if t_gamma is not None and t_cu is not None and t_cu > 0:
            print(f"{t_gamma/t_cu:>6.1f}x")
        else:
            print(f"{'N/A':>8s}")

    # ═══════════════════════════════════════════════════════════
    # Experiment 3: Full grid (haplotypes × seq_length)
    # ═══════════════════════════════════════════════════════════
    print()
    print("─" * 80)
    print("Experiment 3: Full grid — wall-clock seconds")
    print("─" * 80)

    # Header
    header = f"{'n_hap':>6s}"
    for sl in SEQ_LENGTHS:
        header += f"  {'γ ' + f'{sl/1e6:.0f}M':>10s}  {'cu ' + f'{sl/1e6:.0f}M':>10s}"
    print(header)
    print("─" * len(header))

    for n_hap in N_HAPLOTYPES:
        row = f"{n_hap:>6d}"
        for seq_len in SEQ_LENGTHS:
            # Check if already measured
            key = (n_hap, seq_len)
            if key not in results["gamma_wall"]:
                ts = simulate(n_hap=n_hap, seq_len=seq_len)
                results["n_snps"][key] = ts.num_mutations

                with tempfile.TemporaryDirectory() as tmpdir:
                    t_gamma = bench_gamma_smc(ts, tmpdir)
                results["gamma_wall"][key] = t_gamma

                try:
                    t_cu, _, _ = bench_tmrca_cu(ts)
                    results["cu_wall"][key] = t_cu
                except:
                    results["cu_wall"][key] = None

            t_g = results["gamma_wall"][key]
            t_c = results["cu_wall"][key]
            row += f"  {t_g:>9.2f}s" if t_g is not None else f"  {'TIMEOUT':>10s}"
            row += f"  {t_c:>9.3f}s" if t_c is not None else f"  {'ERR':>10s}"
        print(row)

    # ═══════════════════════════════════════════════════════════
    # Experiment 4: Speedup grid
    # ═══════════════════════════════════════════════════════════
    print()
    print("─" * 80)
    print("Experiment 4: Speedup (gamma_smc / tmrca.cu)")
    print("─" * 80)

    header = f"{'n_hap':>6s}  {'pairs':>8s}"
    for sl in SEQ_LENGTHS:
        header += f"  {f'{sl/1e6:.0f}Mb':>8s}"
    print(header)
    print("─" * len(header))

    for n_hap in N_HAPLOTYPES:
        n_pairs = n_hap * (n_hap - 1) // 2
        row = f"{n_hap:>6d}  {n_pairs:>8d}"
        for seq_len in SEQ_LENGTHS:
            key = (n_hap, seq_len)
            t_g = results["gamma_wall"].get(key)
            t_c = results["cu_wall"].get(key)
            if t_g is not None and t_c is not None and t_c > 0:
                row += f"  {t_g/t_c:>6.0f}x"
            else:
                row += f"  {'N/A':>8s}"
        print(row)

    # ═══════════════════════════════════════════════════════════
    # Save results
    # ═══════════════════════════════════════════════════════════
    out_path = "benchmarks/bench_gamma_deep.npz"

    # Convert dict keys to serializable arrays
    grid_gamma = np.full((len(N_HAPLOTYPES), len(SEQ_LENGTHS)), np.nan)
    grid_cu = np.full((len(N_HAPLOTYPES), len(SEQ_LENGTHS)), np.nan)
    grid_snps = np.full((len(N_HAPLOTYPES), len(SEQ_LENGTHS)), np.nan)

    for i, nh in enumerate(N_HAPLOTYPES):
        for j, sl in enumerate(SEQ_LENGTHS):
            key = (nh, sl)
            if results["gamma_wall"].get(key) is not None:
                grid_gamma[i, j] = results["gamma_wall"][key]
            if results["cu_wall"].get(key) is not None:
                grid_cu[i, j] = results["cu_wall"][key]
            if key in results["n_snps"]:
                grid_snps[i, j] = results["n_snps"][key]

    np.savez(out_path,
             n_haplotypes=np.array(N_HAPLOTYPES),
             seq_lengths=np.array(SEQ_LENGTHS),
             gamma_wall=grid_gamma,
             cu_wall=grid_cu,
             n_snps=grid_snps)
    print(f"\nResults saved to {out_path}")

    # ═══════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    speedups = []
    for i, nh in enumerate(N_HAPLOTYPES):
        for j, sl in enumerate(SEQ_LENGTHS):
            if not np.isnan(grid_gamma[i, j]) and not np.isnan(grid_cu[i, j]) and grid_cu[i, j] > 0:
                speedups.append(grid_gamma[i, j] / grid_cu[i, j])

    if speedups:
        print(f"Speedup range: {min(speedups):.0f}x – {max(speedups):.0f}x")
        print(f"Median speedup: {np.median(speedups):.0f}x")
        print(f"Configs measured: {len(speedups)}")

    # Per-pair cost at largest config
    for nh in reversed(N_HAPLOTYPES):
        key = (nh, SEQ_LENGTHS[-1])
        t_g = results["gamma_wall"].get(key)
        t_c = results["cu_wall"].get(key)
        if t_g is not None and t_c is not None:
            n_p = nh * (nh - 1) // 2
            print(f"\nAt n_hap={nh}, seq={SEQ_LENGTHS[-1]/1e6:.0f}Mb ({n_p} pairs):")
            print(f"  gamma_smc: {t_g:.1f}s total, {t_g/n_p*1e3:.1f}ms/pair")
            print(f"  tmrca.cu:  {t_c:.3f}s total, {t_c/n_p*1e3:.2f}ms/pair")
            print(f"  Speedup:   {t_g/t_c:.0f}x")
            break


if __name__ == "__main__":
    main()
