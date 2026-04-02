"""Benchmark: FlowContext (persistent GPU) vs Schweiger gamma_smc."""

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
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/sietch_colab/kkor/tmrca.cu/.pixi/envs/default/lib:" + env.get("LD_LIBRARY_PATH", "")
    cmd = [GAMMA_SMC_BIN, "-i", vcf_path + ".gz",
           "-o", os.path.join(tmpdir, "out.zst"), "-t", str(RHO / MU), "-s", "100"]
    try:
        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env)
        t = time.perf_counter() - t0
        return t if result.returncode == 0 else None
    except subprocess.TimeoutExpired:
        return None


def main():
    configs = [
        (10,  1_000_000),
        (20,  1_000_000),
        (50,  1_000_000),
        (100, 1_000_000),
        (200, 1_000_000),
        (200, 5_000_000),
        (400, 1_000_000),
        (400, 5_000_000),
    ]

    print("=" * 120)
    print("FlowContext (persistent GPU) vs Schweiger gamma_smc")
    print("=" * 120)
    print(f"{'n_hap':>6} {'seq':>6} {'pairs':>8} {'SNPs':>8} "
          f"{'Schweiger':>10} {'ctx.fwd':>10} {'ctx.fb':>10} "
          f"{'fwd x':>8} {'fb x':>8} "
          f"{'fwd M sp/s':>12} {'fb M sp/s':>12}")
    print("-" * 120)

    for n_hap, seq_len in configs:
        n_pairs = n_hap * (n_hap - 1) // 2
        ts = simulate(n_hap, seq_len)
        G = ts.genotype_matrix().T.astype(np.uint8)
        pos = np.array([v.position for v in ts.variants()])
        S = len(pos)
        pairs = [(i, j) for i in range(n_hap) for j in range(i + 1, n_hap)]
        sp = n_pairs * S

        # Schweiger
        with tempfile.TemporaryDirectory() as tmpdir:
            t_sch = bench_schweiger(ts, tmpdir)

        # FlowContext
        ctx = tmrca_cu.FlowContext(G, pos, Ne=NE, mu=MU, rho=RHO, flow_field_path=FF_PATH)

        # Warmup
        ctx.run_fwd(pairs)

        # Forward-only timing
        fwd_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            ctx.run_fwd(pairs)
            fwd_times.append(time.perf_counter() - t0)
        t_fwd = np.median(fwd_times)

        # FB timing
        try:
            ctx.run_fb(pairs)  # warmup
            fb_times = []
            for _ in range(3):
                t0 = time.perf_counter()
                ctx.run_fb(pairs)
                fb_times.append(time.perf_counter() - t0)
            t_fb = np.median(fb_times)
        except Exception:
            t_fb = None

        # Print
        sch_s = f"{t_sch:.2f}s" if t_sch else "TIMEOUT"
        fwd_s = f"{t_fwd:.3f}s"
        fb_s = f"{t_fb:.3f}s" if t_fb else "OOM"
        fwd_x = f"{t_sch/t_fwd:.0f}x" if t_sch and t_fwd > 0 else "N/A"
        fb_x = f"{t_sch/t_fb:.0f}x" if t_sch and t_fb and t_fb > 0 else "N/A"
        fwd_sps = f"{sp/t_fwd/1e6:.0f}" if t_fwd > 0 else "N/A"
        fb_sps = f"{sp/t_fb/1e6:.0f}" if t_fb and t_fb > 0 else "N/A"

        print(f"{n_hap:>6} {seq_len/1e6:.0f}Mb{' ':>2} {n_pairs:>8} {S:>8} "
              f"{sch_s:>10} {fwd_s:>10} {fb_s:>10} "
              f"{fwd_x:>8} {fb_x:>8} "
              f"{fwd_sps:>12} {fb_sps:>12}")

        del ctx

    print()
    print("ctx.fwd = FlowContext.run_fwd() — forward-only, no fwd buffer, pinned D2H")
    print("ctx.fb  = FlowContext.run_fb()  — forward-backward smoothed, pinned D2H")


if __name__ == "__main__":
    main()
