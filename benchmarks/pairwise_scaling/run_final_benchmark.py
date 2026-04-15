#!/usr/bin/env python
"""Final clean benchmark: all three methods at consistent pair counts.

tmrca.cu uses the optimized path (single infer_blockwise call with
pair_batch_size for C++ internal chunking).
gamma_smc uses proper VCF input.
ASMC uses amortized initialization.
"""
import numpy as np, sys, os, csv, gzip, subprocess, tempfile, time

REPO = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
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
PAIR_COUNTS = [1, 10, 100, 1000, 10000, 63190]

import tmrca_cu

def load_data():
    d = np.load(os.path.join(PARSED_DIR, f"chr{CHR}.npz"), allow_pickle=True, mmap_mode="r")
    G, pos, sids = d["G"], d["positions"], d["sample_ids"]
    pops = {}
    with open(SAMPLES_PATH) as f:
        next(f)
        for line in f:
            p = line.strip().split()
            if len(p) >= 7: pops[p[1]] = p[5]
    idx = [2*i+j for i, s in enumerate(sids) if s in pops and pops[s] == POP for j in (0,1)]
    G_pop = np.ascontiguousarray(G[np.array(idx), :])
    pos = np.asarray(pos)
    n = G_pop.shape[0]
    all_pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
    np.random.default_rng(42).shuffle(all_pairs)
    return G_pop, pos, all_pairs

print(f"=== Final benchmark: chr{CHR} {POP} ===", flush=True)
G_pop, pos, all_pairs = load_data()
print(f"{G_pop.shape[0]} haps, {G_pop.shape[1]} sites, {len(all_pairs)} pairs", flush=True)

results = []

# Warmup
tmrca_cu.infer_blockwise(G_pop, pos, mu=1.25e-8, rho=1e-8, Ne=10000,
                          pairs=[(0,1)], mean_only=True, auto_estimate_theta=True)

# ── tmrca.cu (optimized: single call, C++ internal batching) ──
print("\n--- tmrca.cu (GPU, optimized) ---", flush=True)
for np_ in PAIR_COUNTS:
    pairs = all_pairs[:np_]
    t0 = time.time()
    r = tmrca_cu.infer_blockwise(G_pop, pos, mu=1.25e-8, rho=1e-8, Ne=10000,
                                  pairs=pairs, mean_only=True,
                                  auto_estimate_theta=True,
                                  pair_batch_size=min(np_, 30000))
    elapsed = time.time() - t0
    del r
    results.append(("tmrca.cu", np_, elapsed, "measured"))
    print(f"  n={np_:>6}: {elapsed:.3f}s", flush=True)

# ── gamma_smc (VCF, measured at small N, extrapolated) ──
print("\n--- gamma_smc (CPU, VCF) ---", flush=True)

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
                a1 = int(G_sub[2*s, i])
                a2 = int(G_sub[2*s+1, i])
                gts.append(f"{a1}|{a2}")
            f.write(f"{chr_num}\t{pos_bp}\tSNP_{i}\tA\tT\t.\tPASS\t.\tGT")
            f.write("\t" + "\t".join(gts) + "\n")

hap_counts = [2, 10, 46, 142, 356]
with tempfile.TemporaryDirectory(prefix="gsmc_") as td:
    per_pair_times = []
    for nh in hap_counts:
        if nh > G_pop.shape[0]: break
        np_ = nh * (nh - 1) // 2
        G_sub = G_pop[:nh, :]
        af = G_sub.sum(axis=0) / nh
        poly = (af > 0) & (af < 1)
        G_sub_p = G_sub[:, poly]
        pos_p = pos[poly]
        vcf_path = os.path.join(td, f"data_{nh}.vcf")
        write_vcf(vcf_path, G_sub_p, pos_p, CHR)
        subprocess.run(f"bgzip -f {vcf_path} && tabix -f -p vcf {vcf_path}.gz",
                       shell=True, check=True, capture_output=True)
        t0 = time.time()
        r = subprocess.run([GAMMA_SMC_BIN, "-i", vcf_path + ".gz", "-o",
                           os.path.join(td, f"out_{nh}"), "-t", "0.8",
                           "-f", FLOW_FIELD, "-h"], capture_output=True, text=True)
        elapsed = time.time() - t0
        if r.returncode != 0: continue
        per_pair_times.append(elapsed / np_)
        results.append(("gamma_smc", np_, elapsed, "measured"))
        print(f"  {nh:>3} haps ({np_:>6} pairs): {elapsed:.3f}s", flush=True)
    if per_pair_times:
        avg_ppt = np.median(per_pair_times)
        for np_ in PAIR_COUNTS:
            if not any(r[1] == np_ for r in results if r[0] == "gamma_smc"):
                ext = avg_ppt * np_
                results.append(("gamma_smc", np_, ext, "extrapolated"))
                print(f"  n={np_:>6}: {ext:.1f}s (extrapolated)", flush=True)

# ── ASMC (amortized init) ──
print("\n--- ASMC (CPU, amortized) ---", flush=True)
try:
    from asmc.asmc import ASMC
    N_SUB = 50
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory(prefix="asmc_") as td:
        pair_a, pair_b = all_pairs[0]
        others = [i for i in range(G_pop.shape[0]) if i not in (pair_a, pair_b)]
        picked = rng.choice(others, size=min(N_SUB*2-2, len(others)), replace=False).tolist()
        sub_idx = [pair_a, pair_b] + sorted(picked)
        G_sub = G_pop[np.array(sub_idx), :]
        af = G_sub.sum(axis=0) / G_sub.shape[0]
        poly = (af > 0) & (af < 1)
        G_sub = G_sub[:, poly]
        pos_sub = pos[poly]
        out_root = os.path.join(td, "data")
        # Write ASMC input
        cm = pos_sub * 1e-6
        with gzip.open(out_root + ".hap.gz", "wt") as f:
            for i in range(len(pos_sub)):
                haps = " ".join(str(int(x)) for x in G_sub[:, i])
                f.write(f"{CHR}:{int(pos_sub[i])}_1_2 SNP_{int(pos_sub[i])}_{i} {int(pos_sub[i])} 1 2 {haps}\n")
        with open(out_root + ".samples", "w") as f:
            f.write("ID_1 ID_2 missing\n0 0 0\n")
            for i in range(len(sub_idx)//2):
                f.write(f"{i+1}_{i+1} {i+1}_{i+1} 0\n")
        with gzip.open(out_root + ".map.gz", "wt") as f:
            for i in range(len(pos_sub)):
                f.write(f"{CHR}\tSNP_{int(pos_sub[i])}_{i}\t{cm[i]:.10f}\t{int(pos_sub[i])}\n")
        asmc = ASMC(out_root, DQ_FILE, decoding_mode="sequence")
        asmc.set_store_per_pair_posterior_mean(True)
        asmc.decode_pairs([0], [1]); asmc.get_copy_of_results()  # warmup
        for np_ in [1, 10, 100, 1000]:
            t0 = time.time()
            for _ in range(np_):
                asmc.decode_pairs([0], [1])
                asmc.get_copy_of_results()
            elapsed = time.time() - t0
            results.append(("ASMC", np_, elapsed, "measured"))
            print(f"  n={np_:>6}: {elapsed:.3f}s ({elapsed/np_:.3f}s/pair)", flush=True)
            if elapsed > 3600: break
        ppt = elapsed / np_
        for np2 in PAIR_COUNTS:
            if np2 > 1000 and not any(r[1]==np2 for r in results if r[0]=="ASMC"):
                ext = ppt * np2
                results.append(("ASMC", np2, ext, "extrapolated"))
                print(f"  n={np2:>6}: {ext:.1f}s (extrapolated)", flush=True)
except Exception as e:
    print(f"  ASMC FAILED: {e}", flush=True)

# Write CSV
csv_path = os.path.join(OUT_DIR, "results.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "n_pairs", "seconds", "type"])
    for method, n, t, typ in sorted(results, key=lambda x: (x[0], x[1])):
        w.writerow([method, n, f"{t:.4f}", typ])
print(f"\nWrote {csv_path}", flush=True)
