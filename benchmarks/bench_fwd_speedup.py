"""Benchmark: forward-only cached kernel vs Schweiger's gamma_smc.

Measures kernel time (via warmup + repeat) and end-to-end time.
Tests multiple (n_hap, seq_len) configurations.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import msprime
import time
import subprocess
import tempfile
import tmrca_cu

GAMMA_SMC_BIN = "/sietch_colab/kkor/gamma_smc/bin/gamma_smc"
MU = 1.25e-8
RHO = 1e-8
NE = 10_000
FF_PATH = "/sietch_colab/kkor/gamma_smc/resources/default_flow_field.txt"


def simulate(n_hap, seq_len, seed=42):
    ts = msprime.sim_ancestry(
        samples=n_hap // 2, sequence_length=seq_len,
        recombination_rate=RHO, population_size=NE, random_seed=seed)
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=seed + 1)
    return ts


def bench_schweiger(ts, tmpdir, timeout=300):
    vcf_path = os.path.join(tmpdir, "sim.vcf")
    with open(vcf_path, "w") as f:
        ts.write_vcf(f, contig_id="chr1")
    subprocess.run(f"bgzip -f {vcf_path} && tabix -p vcf {vcf_path}.gz",
                   shell=True, check=True, capture_output=True)
    vcf_gz = vcf_path + ".gz"
    output_path = os.path.join(tmpdir, "gamma_out.zst")

    env = os.environ.copy()
    pixi_lib = "/sietch_colab/kkor/tmrca.cu/.pixi/envs/default/lib"
    env["LD_LIBRARY_PATH"] = pixi_lib + ":" + env.get("LD_LIBRARY_PATH", "")

    cmd = [GAMMA_SMC_BIN, "-i", vcf_gz, "-o", output_path,
           "-t", str(RHO / MU), "-s", "100"]
    try:
        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env)
        t_wall = time.perf_counter() - t0
        if result.returncode != 0:
            return None
        return t_wall
    except subprocess.TimeoutExpired:
        return None


def bench_gpu_fwd(G, pos, pairs, n_warmup=1, n_repeat=3):
    """Benchmark forward-only cached kernel. Returns (first_call, median_repeat)."""
    # First call includes cache build + data upload
    t0 = time.perf_counter()
    result = tmrca_cu.gamma_smc_flow_cached_fwd(
        G, pos, pairs, Ne=NE, mu=MU, rho=RHO,
        flow_field_path=FF_PATH, mean_only=True)
    t_first = time.perf_counter() - t0

    # Warmup (cache already built)
    for _ in range(n_warmup):
        tmrca_cu.gamma_smc_flow_cached_fwd(
            G, pos, pairs, Ne=NE, mu=MU, rho=RHO,
            flow_field_path=FF_PATH, mean_only=True)

    # Timed repeats
    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        tmrca_cu.gamma_smc_flow_cached_fwd(
            G, pos, pairs, Ne=NE, mu=MU, rho=RHO,
            flow_field_path=FF_PATH, mean_only=True)
        times.append(time.perf_counter() - t0)

    return t_first, np.median(times), result


def bench_gpu_fb(G, pos, pairs, n_repeat=3):
    """Benchmark cached forward-backward for comparison."""
    # Warmup
    tmrca_cu.gamma_smc_flow_cached_fb(
        G, pos, pairs, Ne=NE, mu=MU, rho=RHO,
        flow_field_path=FF_PATH, mean_only=True)

    times = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        tmrca_cu.gamma_smc_flow_cached_fb(
            G, pos, pairs, Ne=NE, mu=MU, rho=RHO,
            flow_field_path=FF_PATH, mean_only=True)
        times.append(time.perf_counter() - t0)
    return np.median(times)


def main():
    configs = [
        (10,  1_000_000),
        (20,  1_000_000),
        (50,  1_000_000),
        (100, 1_000_000),
        (200, 1_000_000),
        (200, 5_000_000),
        (400, 1_000_000),
    ]

    print("=" * 100)
    print("FORWARD-ONLY CACHED KERNEL vs SCHWEIGER gamma_smc")
    print("=" * 100)
    print(f"{'n_hap':>6} {'seq':>6} {'pairs':>8} {'SNPs':>8} "
          f"{'Schweiger':>10} {'FwdOnly':>10} {'FB':>10} "
          f"{'Speedup':>10} {'FwdOnly sp/s':>14} {'FB sp/s':>14}")
    print("-" * 100)

    for n_hap, seq_len in configs:
        n_pairs = n_hap * (n_hap - 1) // 2
        ts = simulate(n_hap, seq_len)
        G = ts.genotype_matrix().T.astype(np.uint8)
        pos = np.array([v.position for v in ts.variants()])
        S = len(pos)
        all_pairs = [(i, j) for i in range(n_hap) for j in range(i + 1, n_hap)]
        sp = n_pairs * S

        # Schweiger
        with tempfile.TemporaryDirectory() as tmpdir:
            t_sch = bench_schweiger(ts, tmpdir)

        # GPU forward-only
        t_first, t_fwd, _ = bench_gpu_fwd(G, pos, all_pairs)

        # GPU FB
        try:
            t_fb = bench_gpu_fb(G, pos, all_pairs)
        except Exception:
            t_fb = None

        # Print
        sch_str = f"{t_sch:.2f}s" if t_sch else "TIMEOUT"
        fwd_str = f"{t_fwd:.3f}s"
        fb_str = f"{t_fb:.3f}s" if t_fb else "OOM"

        if t_sch and t_fwd > 0:
            speedup = f"{t_sch / t_fwd:.0f}x"
        else:
            speedup = "N/A"

        fwd_sps = f"{sp / t_fwd:,.0f}" if t_fwd > 0 else "N/A"
        fb_sps = f"{sp / t_fb:,.0f}" if t_fb and t_fb > 0 else "N/A"

        seq_str = f"{seq_len/1e6:.0f}Mb"
        print(f"{n_hap:>6} {seq_str:>6} {n_pairs:>8} {S:>8} "
              f"{sch_str:>10} {fwd_str:>10} {fb_str:>10} "
              f"{speedup:>10} {fwd_sps:>14} {fb_sps:>14}")

    print()
    print("Note: FwdOnly = forward-only cached (no backward pass, no fwd buffer)")
    print("      FB = forward-backward cached (full smoothing)")
    print("      Speedup = Schweiger wall-clock / FwdOnly wall-clock")


if __name__ == "__main__":
    main()
