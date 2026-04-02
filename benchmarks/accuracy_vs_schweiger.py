"""
Accuracy comparison: tmrca.cu methods vs Schweiger gamma_smc vs truth (msprime).
Correct binary parsing of gamma_smc output.
"""
import subprocess, tempfile, os, msprime, json, time
import numpy as np
from scipy.stats import pearsonr
import sys

sys.path.insert(0, "/sietch_colab/kkor/tmrca.cu/python")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tmrca_cu import _core

GAMMA_SMC = "/sietch_colab/kkor/gamma_smc/bin/gamma_smc"
FF_PATH = "/sietch_colab/kkor/gamma_smc/resources/default_flow_field.txt"
MU, RHO, NE = 1.25e-8, 1e-8, 10_000


def simulate(n_hap, seq_len, seed=42):
    ts = msprime.sim_ancestry(samples=n_hap // 2, sequence_length=seq_len,
        recombination_rate=RHO, population_size=NE, random_seed=seed)
    return msprime.sim_mutations(ts, rate=MU, random_seed=seed + 1)


def true_tmrca_at_sites(ts, i, j, spos):
    t = np.empty(len(spos))
    tit = ts.trees(); tree = next(tit)
    for idx, p in enumerate(spos):
        while p >= tree.interval.right: tree = next(tit)
        t[idx] = tree.tmrca(i, j)
    return t


def run_schweiger(ts, n_hap):
    """Run Schweiger gamma_smc, return dict of pair -> mean_tmrca array."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vcf_path = os.path.join(tmpdir, "sim.vcf")
        with open(vcf_path, "w") as f:
            ts.write_vcf(f, contig_id="chr1")
        subprocess.run(f"bgzip -f {vcf_path} && tabix -p vcf {vcf_path}.gz",
                       shell=True, check=True, capture_output=True)

        out_path = os.path.join(tmpdir, "out")
        cmd = [GAMMA_SMC, "-i", vcf_path + ".gz", "-o", out_path,
               "-m", str(4*NE*MU), "-r", str(4*NE*RHO), "-f", FF_PATH, "-h"]

        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        wall = time.perf_counter() - t0

        if result.returncode != 0:
            return None, None, wall

        # Decompress
        dec = os.path.join(tmpdir, "out.bin")
        subprocess.run(["zstd", "-d", out_path, "-o", dec],
                       capture_output=True, check=True)

        with open(dec, "rb") as f:
            raw = f.read()
        with open(out_path + ".meta", "r") as f:
            meta = json.load(f)

        n_pairs = meta["num_pairs"]
        n_sites = meta["sequence_length"]
        cs = meta["chunk_size"]
        nc = (n_pairs + cs - 1) // cs
        positions = np.array(meta["output_positions"])

        arr = np.frombuffer(raw, dtype=np.float32).reshape(nc, 2, n_sites, cs)
        means = arr[:, 0, :, :].transpose(1, 0, 2).reshape(n_sites, nc * cs)[:, :n_pairs]

        # Build pair map: haplotype indices in order
        pair_map = []
        for i in range(n_hap):
            for j in range(i + 1, n_hap):
                pair_map.append((i, j))

        return {pair_map[p]: means[:, p] for p in range(n_pairs)}, positions, wall


def r_log(truth, est):
    """Pearson r on log scale."""
    lx = np.log(np.maximum(truth, 1.0))
    ly = np.log(np.maximum(est, 1.0))
    return pearsonr(lx, ly)[0]


def rmse_log(truth, est):
    lx = np.log(np.maximum(truth, 1.0))
    ly = np.log(np.maximum(est, 1.0))
    return np.sqrt(np.mean((lx - ly) ** 2))


# ================================================================
print("=" * 80)
print("Accuracy: tmrca.cu vs Schweiger gamma_smc vs truth (msprime)")
print("=" * 80)

configs = [
    (20, 1_000_000, 42),
    (20, 1_000_000, 99),
    (50, 1_000_000, 44),
    (50, 1_000_000, 55),
]

summary = {m: {"r": [], "rmse": []} for m in
           ["our_gsmc_fwd", "our_flow_fwd", "our_flow_fb", "schweiger_fb"]}

for n_hap, seq_len, seed in configs:
    print(f"\n--- n={n_hap}, seq={seq_len/1e6:.0f}Mb, seed={seed} ---")
    ts = simulate(n_hap, seq_len, seed)
    G = ts.genotype_matrix().T.astype(np.uint8)
    pos = np.array([v.position for v in ts.variants()])
    n, S = G.shape

    # Select test pairs
    rng = np.random.RandomState(seed)
    n_test = min(20, n * (n - 1) // 2)
    pset = set()
    while len(pset) < n_test:
        a, b = sorted(rng.choice(n, 2, replace=False))
        pset.add((int(a), int(b)))
    test_pairs = sorted(pset)

    # Truths at our positions
    truths = {p: true_tmrca_at_sites(ts, p[0], p[1], pos) for p in test_pairs}

    # Our methods
    gsmc = _core.gamma_smc_forward(G, pos, test_pairs, float(NE), MU, RHO, 1, True)["mean"]
    flow = _core.gamma_smc_flow_cached_fwd(G, pos, test_pairs, float(NE), MU, RHO, FF_PATH, True, 0)["mean"]
    try:
        fb = _core.gamma_smc_flow_cached_fb(G, pos, test_pairs, float(NE), MU, RHO, FF_PATH, True, 0)["mean"]
    except:
        fb = None

    # Schweiger
    schw_results, schw_pos, schw_wall = run_schweiger(ts, n)
    print(f"  Schweiger wall time: {schw_wall:.1f}s")

    # Find Schweiger scale factor (empirically, truth = factor * schweiger_raw)
    # Use all available pairs to estimate
    if schw_results:
        ratios = []
        for p in test_pairs:
            if p in schw_results:
                truth_schw = true_tmrca_at_sites(ts, p[0], p[1], schw_pos)
                raw = schw_results[p]
                med_ratio = np.median(truth_schw / raw)
                ratios.append(med_ratio)
        scale = np.median(ratios) if ratios else 1.0
        print(f"  Schweiger scale factor: {scale:.0f} (median truth/raw)")
    else:
        scale = 1.0

    # Compute metrics
    results = {}
    for method, tag in [("our_gsmc_fwd", "gsmc"), ("our_flow_fwd", "flow"),
                         ("our_flow_fb", "fb"), ("schweiger_fb", "schw")]:
        rs, rmses = [], []
        for pidx, p in enumerate(test_pairs):
            truth = truths[p]
            if tag == "gsmc":
                est = gsmc[:, pidx]
            elif tag == "flow":
                est = flow[:, pidx]
            elif tag == "fb":
                if fb is None: continue
                est = fb[:, pidx]
            elif tag == "schw":
                if schw_results is None or p not in schw_results: continue
                raw = schw_results[p]
                est = np.interp(pos, schw_pos, raw * scale)

            r = r_log(truth, est)
            rm = rmse_log(truth, est)
            rs.append(r)
            rmses.append(rm)

        results[method] = (rs, rmses)
        if rs:
            summary[method]["r"].extend(rs)
            summary[method]["rmse"].extend(rmses)

    # Table
    print(f"  {'Method':25s} {'n':>4} {'r_log':>8} {'RMSE_log':>10}")
    print(f"  {'-'*50}")
    for method in ["our_gsmc_fwd", "our_flow_fwd", "our_flow_fb", "schweiger_fb"]:
        rs, rmses = results[method]
        if rs:
            print(f"  {method:25s} {len(rs):>4} {np.mean(rs):>8.4f} {np.mean(rmses):>10.4f}")
        else:
            print(f"  {method:25s} {'N/A':>4}")

# ================================================================
print("\n" + "=" * 80)
print("OVERALL SUMMARY")
print("=" * 80)
print(f"{'Method':25s} {'n_pairs':>8} {'mean_r_log':>11} {'mean_RMSE':>10}")
print("-" * 57)
for method in ["our_gsmc_fwd", "our_flow_fwd", "our_flow_fb", "schweiger_fb"]:
    rs = summary[method]["r"]
    rmses = summary[method]["rmse"]
    if rs:
        print(f"{method:25s} {len(rs):>8} {np.mean(rs):>11.4f} {np.mean(rmses):>10.4f}")
    else:
        print(f"{method:25s} {'N/A':>8}")

print("\nNotes:")
print("  our_gsmc_fwd:  Gamma-SMC forward-only (moment-matched, fastest)")
print("  our_flow_fwd:  Flow field forward-only (Schweiger method, GPU)")
print("  our_flow_fb:   Flow field forward-backward (most accurate)")
print("  schweiger_fb:  Original Schweiger gamma_smc (CPU reference)")
