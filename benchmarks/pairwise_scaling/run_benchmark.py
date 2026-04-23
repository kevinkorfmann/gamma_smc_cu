#!/usr/bin/env python
"""Benchmark: gamma_smc_cu vs gamma_smc vs ASMC — measured, not extrapolated.

Each method is run at increasing pair counts on chr22 YRI until a 4h
wall-time budget is approached. Only the final (largest) point is
extrapolated if it would exceed the budget.

- gamma_smc_cu: infer_blockwise(), measured at all pair counts.
- gamma_smc: VCF input (bgzip+tabix), all-pairs mode. Measured by
  varying the number of haplotypes in the VCF (2, 10, 46, 142, 356).
- ASMC: batch decode_pairs(), measured at 1, 10, 100, 1000+ pairs.
"""
from __future__ import annotations

import csv
import gzip
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO = "/vast/projects/smathi/cohort/kkor/gamma_smc_cu"
sys.path.insert(0, os.path.join(REPO, "python"))

PARSED_DIR = os.path.join(REPO, "analysis/genome_wide/cache/parsed")
SAMPLES_PATH = os.path.join(REPO, "analysis/genome_wide/data/samples.txt")
ASMC_DATA = "/vast/projects/smathi/cohort/kkor/asmc_data"
DQ_FILE = os.path.join(ASMC_DATA, "CEU_50.decodingQuantities.gz")
GAMMA_SMC_BIN = os.path.join(REPO, "benchmarks/test_suite_stdpopsim/gamma_smc/bin/gamma_smc")
FLOW_FIELD = os.path.join(REPO, "default_flow_field.txt")
OUT_DIR = os.path.join(REPO, "benchmarks/pairwise_scaling")

CHR = 22
POP = "YRI"
BUDGET_S = 4 * 3600


def load_samples():
    pops = {}
    with open(SAMPLES_PATH) as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                pops[parts[1]] = (parts[5], parts[6])
    return pops


def get_pop_haps(sample_ids, pop_map, population):
    indices = []
    for i, sid in enumerate(sample_ids):
        if sid in pop_map and pop_map[sid][0] == population:
            indices.extend([2 * i, 2 * i + 1])
    return sorted(indices)


def write_vcf(path, G_sub, positions, chr_num):
    n_haps, n_sites = G_sub.shape
    n_samples = n_haps // 2
    with open(path, "w") as f:
        f.write("##fileformat=VCFv4.1\n")
        f.write(f'##contig=<ID={chr_num}>\n')
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
        for s in range(n_samples):
            f.write(f"\tS{s}")
        f.write("\n")
        for i in range(n_sites):
            pos_bp = int(positions[i])
            gts = []
            for s in range(n_samples):
                a1 = int(G_sub[2 * s, i])
                a2 = int(G_sub[2 * s + 1, i])
                gts.append(f"{a1}|{a2}")
            f.write(f"{chr_num}\t{pos_bp}\tSNP_{i}\tA\tT\t.\tPASS\t.\tGT")
            f.write("\t" + "\t".join(gts) + "\n")


def write_asmc_input(out_root, chr_num, positions, G_sub, n_samples):
    n_haps, n_sites = G_sub.shape
    cm_positions = positions * 1e-6
    with gzip.open(out_root + ".hap.gz", "wt") as f:
        for i in range(n_sites):
            pos_bp = int(positions[i])
            haps = " ".join(str(int(x)) for x in G_sub[:, i])
            f.write(f"{chr_num}:{pos_bp}_1_2 SNP_{pos_bp}_{i} {pos_bp} 1 2 {haps}\n")
    with open(out_root + ".samples", "w") as f:
        f.write("ID_1 ID_2 missing\n0 0 0\n")
        for i in range(n_samples):
            f.write(f"{i+1}_{i+1} {i+1}_{i+1} 0\n")
    with gzip.open(out_root + ".map.gz", "wt") as f:
        for i in range(n_sites):
            pos_bp = int(positions[i])
            f.write(f"{chr_num}\tSNP_{pos_bp}_{i}\t{cm_positions[i]:.10f}\t{pos_bp}\n")


# ── gamma_smc_cu ──────────────────────────────────────────────────

def bench_gamma_smc_cu(G_pop, positions, all_pairs):
    import gamma_smc_cu
    results = []
    pair_counts = [1, 10, 100, 1000, 10000, len(all_pairs)]
    PAIR_CHUNK = 1000

    # Warmup
    gamma_smc_cu.infer_blockwise(
        G_pop, positions, mu=1.25e-8, rho=1e-8, Ne=10_000,
        pairs=[(0, 1)], mean_only=True, auto_estimate_theta=True)

    for n in pair_counts:
        pairs = all_pairs[:n]
        t0 = time.time()
        if n <= PAIR_CHUNK:
            gamma_smc_cu.infer_blockwise(
                G_pop, positions, mu=1.25e-8, rho=1e-8, Ne=10_000,
                pairs=pairs, mean_only=True, auto_estimate_theta=True)
        else:
            for ci in range(0, n, PAIR_CHUNK):
                chunk = pairs[ci:ci + PAIR_CHUNK]
                gamma_smc_cu.infer_blockwise(
                    G_pop, positions, mu=1.25e-8, rho=1e-8, Ne=10_000,
                    pairs=chunk, mean_only=True, auto_estimate_theta=True)
        elapsed = time.time() - t0
        results.append(("gamma_smc_cu", n, elapsed, "measured"))
        print(f"  gamma_smc_cu  n={n:>6}: {elapsed:.3f}s", flush=True)
    return results


# ── gamma_smc ─────────────────────────────────────────────────

def bench_gamma_smc(G_pop, positions):
    results = []
    # gamma_smc decodes ALL pairs — vary haplotype count
    # n_haps → n_pairs: 2→1, 10→45, 46→1035, 142→10011, 356→63190
    hap_counts = [2, 10, 46, 142, 356]
    t_total = 0

    with tempfile.TemporaryDirectory(prefix="gsmc_") as td:
        for nh in hap_counts:
            if nh > G_pop.shape[0]:
                break
            n_pairs = nh * (nh - 1) // 2

            # Estimate: skip if would blow budget
            if t_total > 0 and results:
                last_ppt = results[-1][2] / results[-1][1]
                est = last_ppt * n_pairs
                if t_total + est > BUDGET_S * 0.9:
                    # Extrapolate this point
                    results.append(("gamma_smc", n_pairs, est, "extrapolated"))
                    print(f"  gamma_smc {nh:>3} haps ({n_pairs:>6} pairs): "
                          f"{est:.1f}s (extrapolated, would exceed budget)", flush=True)
                    continue

            G_sub = G_pop[:nh, :]
            af = G_sub.sum(axis=0) / nh
            poly = (af > 0) & (af < 1)
            G_sub_p = G_sub[:, poly]
            pos_p = positions[poly]

            vcf_path = os.path.join(td, f"data_{nh}.vcf")
            print(f"  Writing VCF for {nh} haps ({G_sub_p.shape[1]} poly sites)...",
                  flush=True)
            write_vcf(vcf_path, G_sub_p, pos_p, CHR)
            subprocess.run(f"bgzip -f {vcf_path} && tabix -f -p vcf {vcf_path}.gz",
                           shell=True, check=True, capture_output=True)

            t0 = time.time()
            cmd = [GAMMA_SMC_BIN, "-i", vcf_path + ".gz", "-o",
                   os.path.join(td, f"out_{nh}"), "-t", "0.8",
                   "-f", FLOW_FIELD, "-h"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            elapsed = time.time() - t0
            t_total += elapsed

            if r.returncode != 0:
                print(f"  gamma_smc {nh} haps FAILED: {r.stderr[-200:]}", flush=True)
                continue

            results.append(("gamma_smc", n_pairs, elapsed, "measured"))
            print(f"  gamma_smc {nh:>3} haps ({n_pairs:>6} pairs): {elapsed:.3f}s "
                  f"({elapsed/n_pairs:.4f}s/pair)", flush=True)

    return results


# ── ASMC ──────────────────────────────────────────────────────

def bench_asmc(G_pop, positions, all_pairs):
    from asmc.asmc import ASMC
    results = []
    N_SUB = 50
    n_haps = G_pop.shape[0]
    rng = np.random.default_rng(42)

    with tempfile.TemporaryDirectory(prefix="asmc_") as td:
        pair_a, pair_b = all_pairs[0]
        others = [i for i in range(n_haps) if i not in (pair_a, pair_b)]
        picked = rng.choice(others, size=min(N_SUB * 2 - 2, len(others)),
                            replace=False).tolist()
        sub_idx = [pair_a, pair_b] + sorted(picked)
        G_sub = G_pop[np.array(sub_idx), :]
        af = G_sub.sum(axis=0) / G_sub.shape[0]
        poly = (af > 0) & (af < 1)
        G_sub = G_sub[:, poly]
        pos_sub = positions[poly]

        out_root = os.path.join(td, "data")
        write_asmc_input(out_root, CHR, pos_sub, G_sub, len(sub_idx) // 2)

        asmc = ASMC(out_root, DQ_FILE, decoding_mode="sequence")
        asmc.set_store_per_pair_posterior_mean(True)
        print("  ASMC initialized", flush=True)

        # Warmup
        asmc.decode_pairs([0], [1])
        _ = asmc.get_copy_of_results()

        # Measure at increasing pair counts
        pair_counts = [1, 10, 100, 1000, 5000]
        t_total = 0
        last_ppt = None

        for n in pair_counts:
            # Estimate and skip if over budget
            if last_ppt and t_total + last_ppt * n > BUDGET_S * 0.9:
                est = last_ppt * n
                results.append(("ASMC", n, est, "extrapolated"))
                print(f"  ASMC      n={n:>6}: {est:.1f}s (extrapolated)", flush=True)
                continue

            t0 = time.time()
            for _ in range(n):
                asmc.decode_pairs([0], [1])
                _ = asmc.get_copy_of_results()
            elapsed = time.time() - t0
            t_total += elapsed
            last_ppt = elapsed / n
            results.append(("ASMC", n, elapsed, "measured"))
            print(f"  ASMC      n={n:>6}: {elapsed:.3f}s ({last_ppt:.3f}s/pair)",
                  flush=True)

        # Extrapolate to full YRI pair count
        if last_ppt:
            for n in [10000, len(all_pairs)]:
                est = last_ppt * n
                results.append(("ASMC", n, est, "extrapolated"))
                print(f"  ASMC      n={n:>6}: {est:.1f}s (extrapolated)", flush=True)

    return results


# ── main ──────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"=== Pairwise scaling benchmark: chr{CHR} {POP} ===", flush=True)
    print(f"    Budget: {BUDGET_S}s ({BUDGET_S/3600:.0f}h)", flush=True)

    d = np.load(os.path.join(PARSED_DIR, f"chr{CHR}.npz"),
                allow_pickle=True, mmap_mode="r")
    G, positions, sample_ids = d["G"], d["positions"], d["sample_ids"]
    pop_map = load_samples()
    hap_idx = get_pop_haps(sample_ids, pop_map, POP)
    G_pop = np.ascontiguousarray(G[np.array(hap_idx), :])
    positions = np.asarray(positions)
    n = G_pop.shape[0]
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    np.random.default_rng(42).shuffle(all_pairs)
    print(f"  {n} haplotypes, {G_pop.shape[1]} sites, {len(all_pairs)} pairs",
          flush=True)

    all_results = []

    print("\n--- gamma_smc_cu (GPU, infer_blockwise) ---", flush=True)
    all_results.extend(bench_gamma_smc_cu(G_pop, positions, all_pairs))

    print("\n--- gamma_smc (CPU, VCF, all-pairs) ---", flush=True)
    try:
        all_results.extend(bench_gamma_smc(G_pop, positions))
    except Exception as e:
        print(f"  gamma_smc FAILED: {e}", flush=True)

    print("\n--- ASMC (CPU, sequential decode_pairs) ---", flush=True)
    try:
        all_results.extend(bench_asmc(G_pop, positions, all_pairs))
    except Exception as e:
        print(f"  ASMC FAILED: {e}", flush=True)

    csv_path = os.path.join(OUT_DIR, "results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "n_pairs", "seconds", "type"])
        for method, n, t, typ in all_results:
            w.writerow([method, n, f"{t:.4f}", typ])
    print(f"\nWrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
