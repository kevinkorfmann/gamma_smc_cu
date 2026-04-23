"""
Two questions in one experiment:

  Q1  Does using gamma_smc's pi_hat estimator (per-individual heterozygosity
      averaged across diploids) close the remaining gap on AnoGam / DroMel /
      AraTha / CanFam?

  Q2  If pi_hat is the same on both sides, are the algorithms still equivalent?
      (i.e. is there ANYTHING left after we match the parameter source exactly?)

For each problem config we run tmrca.cu cached_fb three ways:
  (a) tmrca.cu's pairwise-pi auto-theta (current default)
  (b) gamma_smc's per-individual-het pi_hat (extracted by running the binary)
  (c) the same per-individual-het pi_hat computed in numpy (sanity check)

and compare each to gamma_smc auto_mt.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import msprime
import stdpopsim
from scipy.stats import pearsonr
from bench_paths import resolve_flow_field_path, resolve_gamma_smc_bin
from bench_inputs import materialize_binary_snp_vcf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "python"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from gamma_smc_cu import _core  # noqa: E402

FF = resolve_flow_field_path(HERE)
GSMC = resolve_gamma_smc_bin(HERE)

CONFIGS = [
    ("AnoGam", "GabonAg1000G_1A17", "GAS"),
]
SEQ_LEN = 5_000_000
N_HAP = 20
SEED = 42
NE = 10_000


def true_t(ts, i, j, positions):
    t = np.empty(len(positions))
    tit = ts.trees(); tree = next(tit)
    for idx, p in enumerate(positions):
        while p >= tree.interval.right:
            tree = next(tit)
        t[idx] = tree.tmrca(i, j)
    return t


def true_t_all_pairs(ts, all_pairs, positions):
    """Ground-truth TMRCA for ALL pairs in one tree traversal."""
    n_pairs = len(all_pairs)
    n_pos = len(positions)
    result = np.empty((n_pos, n_pairs))
    pos_idx = 0
    for tree in ts.trees():
        right = tree.interval.right
        start_idx = pos_idx
        while pos_idx < n_pos and positions[pos_idx] < right:
            pos_idx += 1
        if pos_idx > start_idx:
            for pidx, (i, j) in enumerate(all_pairs):
                result[start_idx:pos_idx, pidx] = tree.tmrca(i, j)
        if pos_idx >= n_pos:
            break
    return result


def r_log_safe(x, y):
    lx = np.log(np.maximum(x, 1e-10))
    ly = np.log(np.maximum(y, 1e-10))
    mask = np.isfinite(lx) & np.isfinite(ly)
    if mask.sum() < 10: return np.nan
    lx, ly = lx[mask], ly[mask]
    if np.std(lx) == 0 or np.std(ly) == 0: return np.nan
    return float(pearsonr(lx, ly)[0])


def schweiger_pair_order(nh):
    nd = nh // 2; pairs = []
    for dA in range(nd):
        for dB in range(dA, nd):
            if dA == dB: pairs.append((2 * dA, 2 * dA + 1))
            else:
                for ha in range(2):
                    for hb in range(2):
                        pairs.append((2 * dA + ha, 2 * dB + hb))
    return pairs


def pi_pairwise(G, positions):
    """Mean pairwise heterozygosity per bp (Tajima's pi)."""
    n, S = G.shape
    k = G.sum(axis=0, dtype=np.int64)
    total_diff = int((k * (np.int64(n) - k)).sum())
    n_pair = n * (n - 1) // 2
    seq_len = float(positions[-1] - positions[0] + 1)
    return total_diff / (float(n_pair) * seq_len)


def pi_individual_het(G, positions):
    """Average per-individual heterozygosity per bp.

    Matches gamma_smc's calculate_heterozygosity(): for each diploid,
    count het sites (haplotype 0 != haplotype 1), divide by sequence
    length, average across individuals.
    """
    n, S = G.shape
    nd = n // 2
    seq_len = float(positions[-1] - positions[0] + 1)
    rates = []
    for i in range(nd):
        h0 = G[2 * i]
        h1 = G[2 * i + 1]
        n_hets = int(np.sum(h0 != h1))
        rates.append(n_hets / seq_len)
    return float(np.mean(rates))


def run_gsmc_auto(vcf_path, nh, mu, rho, ne):
    with tempfile.TemporaryDirectory() as td:
        # Copy the filtered VCF into a temp dir for bgzip/tabix
        import shutil
        vp = os.path.join(td, "s.vcf")
        shutil.copy(vcf_path, vp)
        subprocess.run(f"bgzip -f {vp} && tabix -p vcf {vp}.gz",
                       shell=True, check=True, capture_output=True)
        out = os.path.join(td, "out")
        result = subprocess.run(
            [GSMC, "-i", vp + ".gz", "-o", out,
             "-t", str(rho / max(mu, 1e-30)), "-f", FF, "-h"],
            capture_output=True, text=True, check=True, timeout=600,
        )
        m = re.search(r"Scaled mutation rate:\s*([0-9.eE+-]+)", result.stdout)
        reported = float(m.group(1)) if m else None

        dec = os.path.join(td, "out.bin")
        subprocess.run(["zstd", "-d", out, "-o", dec], capture_output=True)
        with open(dec, "rb") as f: raw = f.read()
        with open(out + ".meta") as f: meta = json.load(f)
        n_pairs = meta["num_pairs"]; n_sites_meta = meta["sequence_length"]
        cs = meta["chunk_size"]; nc = (n_pairs + cs - 1) // cs
        positions = np.array(meta["output_positions"])
        arr = np.frombuffer(raw, dtype=np.float32).reshape(nc, 2, n_sites_meta, cs)
        pair_layout = schweiger_pair_order(nh)
        results = {}
        for pidx, p in enumerate(pair_layout[:n_pairs]):
            alpha = arr[pidx // cs, 0, :, pidx % cs]
            beta = arr[pidx // cs, 1, :, pidx % cs]
            mean_gen = (alpha / np.maximum(beta, 1e-10)) * 2 * ne
            results[tuple(sorted(p))] = mean_gen
        return results, positions, reported


def run_tmrca(G, pos, pairs, kernel_mu, kernel_rho):
    """tmrca.cu cached_fb with the given (mu, rho) passed to FlowContext."""
    return _core.gamma_smc_flow_cached_fb(
        G, pos, pairs, float(NE), kernel_mu, kernel_rho, FF, True, 0
    )["mean"]


def median_r(estimates_array, pairs, ts, pos, truth_all=None):
    if truth_all is None:
        truth_all = true_t_all_pairs(ts, pairs, pos)
    rs = []
    for pidx in range(len(pairs)):
        rs.append(r_log_safe(truth_all[:, pidx], estimates_array[:, pidx]))
    a = np.asarray(rs, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a))


def benchmark(species_id, model_id, pop):
    print(f"\n{'='*78}")
    print(f"{species_id}  {model_id}  pop={pop}")
    print('='*78)

    species = stdpopsim.get_species(species_id)
    model = species.get_demographic_model(model_id)
    contig = species.get_contig(length=SEQ_LEN)
    mu = contig.mutation_rate
    try:
        rho = float(contig.recombination_map.mean_rate)
    except Exception:
        rho = 1e-8

    samples = {pop: N_HAP // 2}
    ts_raw = stdpopsim.get_engine("msprime").simulate(
        model, contig, samples, seed=SEED
    )
    ts = msprime.sim_mutations(ts_raw, rate=mu, random_seed=SEED + 1)

    # Use the same VCF normalization as the benchmark (run_one.py)
    _tmpdir = tempfile.mkdtemp()
    vcf_path = os.path.join(_tmpdir, "s.vcf")
    prepared = materialize_binary_snp_vcf(ts, vcf_path)
    G = prepared.G
    pos = prepared.pos
    n, S = G.shape
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    print(f"  n={n}, S={S} (kept={prepared.n_kept_records}/{prepared.n_total_records} "
          f"drop_non_binary={prepared.n_dropped_non_binary})")
    print(f"  mu={mu:.2e}, rho={rho:.2e}, rho/mu={rho/mu:.4f}")

    pi_pair = pi_pairwise(G, pos)
    pi_ind = pi_individual_het(G, pos)
    print(f"  pi (pairwise, current default) = {pi_pair:.4e}")
    print(f"  pi (per-individual het, gsmc style) = {pi_ind:.4e}")
    print(f"  ratio (ind / pair) = {pi_ind / pi_pair:.4f}")

    # gsmc auto_mt to get the actual pi_hat it reports
    gsmc_results, gsmc_pos, gsmc_pi = run_gsmc_auto(vcf_path, n, mu, rho, NE)
    print(f"  pi (gamma_smc reported)         = {gsmc_pi:.4e}")
    print(f"  ratio (gsmc / pair)             = {gsmc_pi / pi_pair:.4f}")
    print(f"  ratio (gsmc / ind)              = {gsmc_pi / pi_ind:.4f}")

    # Precompute ground truth once for all pairs
    print("  ground truth: extracting ...", flush=True)
    truth_all = true_t_all_pairs(ts, pairs, pos)
    print(f"  ground truth: done", flush=True)

    # gamma_smc r_log
    rs_g = []
    for pidx, pair in enumerate(pairs):
        truth = truth_all[:, pidx]
        key = tuple(sorted(pair))
        if key in gsmc_results:
            est = np.interp(pos, gsmc_pos, gsmc_results[key])
            rs_g.append(r_log_safe(truth, est))
    r_gsmc = float(np.nanmedian(rs_g))

    # tmrca.cu with three different scaled_mu values:
    #   (a) pairwise pi (current default)
    #   (b) per-individual het computed in numpy
    #   (c) gsmc's reported pi (from the binary stdout)
    ratio = rho / max(mu, 1e-30)
    rows = []
    for label, pi_target in [
        ("pairwise (current)", pi_pair),
        ("ind-het (gsmc-style numpy)", pi_ind),
        ("gsmc reported", gsmc_pi),
    ]:
        kernel_mu = pi_target / (4.0 * NE)
        kernel_rho = pi_target * ratio / (2.0 * NE)
        out = run_tmrca(G, pos, pairs, kernel_mu, kernel_rho)
        med = median_r(out, pairs, ts, pos, truth_all=truth_all)
        rows.append((label, pi_target, med))
        print(f"  tmrca.cu [{label:<28s}] r_log = {med:.4f}  (pi={pi_target:.3e})")

    print(f"  gamma_smc auto_mt              r_log = {r_gsmc:.4f}")
    print(f"  best tmrca minus gsmc           = "
          f"{max(r for _, _, r in rows) - r_gsmc:+.4f}")

    return {
        "species": species_id, "model": model_id,
        "pi_pair": pi_pair, "pi_ind": pi_ind, "pi_gsmc": gsmc_pi,
        "r_gsmc": r_gsmc,
        "r_pair": rows[0][2],
        "r_ind": rows[1][2],
        "r_gsmcpi": rows[2][2],
    }


def main():
    rows = []
    for cfg in CONFIGS:
        rows.append(benchmark(*cfg))

    print("\n" + "="*98)
    print("SUMMARY")
    print("="*98)
    print(f"{'species':<8} {'model':<28}  "
          f"{'pair':>7} {'ind':>7} {'gsmcpi':>7} {'gsmc':>7}  "
          f"{'pair-g':>8} {'ind-g':>8} {'gpi-g':>8}")
    for r in rows:
        print(f"{r['species']:<8} {r['model'][:28]:<28}  "
              f"{r['r_pair']:>7.4f} {r['r_ind']:>7.4f} "
              f"{r['r_gsmcpi']:>7.4f} {r['r_gsmc']:>7.4f}  "
              f"{r['r_pair']-r['r_gsmc']:>+8.4f} "
              f"{r['r_ind']-r['r_gsmc']:>+8.4f} "
              f"{r['r_gsmcpi']-r['r_gsmc']:>+8.4f}")

    out = os.path.join(HERE, "diagnose_pi_estimator.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
