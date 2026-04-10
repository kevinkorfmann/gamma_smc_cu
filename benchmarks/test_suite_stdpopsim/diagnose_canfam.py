"""
Diagnose why `tmrca.cu`'s cached forward-backward is worse than the original
`gamma_smc` binary on species with small mu/rho (CanFam, BosTau).

Runs on the SAME simulated data for each config:
  1. `tmrca_cu._core.gamma_smc_flow_cached_fb` -- the cached FB used in the paper
  2. `tmrca_cu._core.gamma_smc_flow_fb`        -- the iterative (non-cached) FB
  3. the gamma_smc binary

Computes Pearson r_log vs msprime truth for each method, printed per pair.
Also writes a per-site dump for the first pair so we can eyeball where the
cached path diverges.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import msprime
import stdpopsim
from scipy.stats import pearsonr
from bench_inputs import materialize_binary_snp_vcf
from bench_paths import resolve_flow_field_path, resolve_gamma_smc_bin

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "python"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from tmrca_cu import _core  # noqa: E402

FF = resolve_flow_field_path(HERE)
GSMC = resolve_gamma_smc_bin(HERE)

CONFIGS = [
    ("HomSap", "OutOfAfrica_3G09", "YRI"),        # control (tmrca.cu wins)
    ("CanFam", "EarlyWolfAdmixture_6F14", "BSJ"), # tmrca.cu loses badly
    ("BosTau", "HolsteinFriesian_1M13", "Holstein_Friesian"),  # tmrca.cu loses
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


def r_log(x, y):
    lx = np.log(np.maximum(x, 1e-10))
    ly = np.log(np.maximum(y, 1e-10))
    return float(pearsonr(lx, ly)[0])


def rmse_log(x, y):
    lx = np.log(np.maximum(x, 1e-10))
    ly = np.log(np.maximum(y, 1e-10))
    return float(np.sqrt(np.mean((lx - ly) ** 2)))


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


def run_gsmc(vcf_path, nh, mu, rho, ne):
    td = os.path.dirname(vcf_path)
    subprocess.run(f"bgzip -f {vcf_path} && tabix -f -p vcf {vcf_path}.gz",
                   shell=True, check=True, capture_output=True)
    out = os.path.join(td, "out")
    subprocess.run(
        [GSMC, "-i", vcf_path + ".gz", "-o", out,
         "-m", str(4 * ne * mu), "-r", str(4 * ne * rho),
         "-f", FF, "-h"],
        check=True, capture_output=True, timeout=600,
    )
    dec = os.path.join(td, "out.bin")
    subprocess.run(["zstd", "-d", out, "-o", dec], capture_output=True)
    with open(dec, "rb") as f:
        raw = f.read()
    with open(out + ".meta") as f:
        meta = json.load(f)
    n_pairs = meta["num_pairs"]; n_sites = meta["sequence_length"]
    cs = meta["chunk_size"]; nc = (n_pairs + cs - 1) // cs
    positions = np.array(meta["output_positions"])
    arr = np.frombuffer(raw, dtype=np.float32).reshape(nc, 2, n_sites, cs)
    pair_layout = schweiger_pair_order(nh)
    results = {}
    for pidx, p in enumerate(pair_layout[:n_pairs]):
        alpha = arr[pidx // cs, 0, :, pidx % cs]
        beta = arr[pidx // cs, 1, :, pidx % cs]
        mean_gen = (alpha / np.maximum(beta, 1e-10)) * 2 * ne
        results[tuple(sorted(p))] = mean_gen
    return results, positions


def diagnose(species_id, model_id, pop):
    print(f"\n{'='*70}")
    print(f"{species_id}  {model_id}  pop={pop}")
    print('='*70)

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

    with tempfile.TemporaryDirectory() as td:
        vcf_path = os.path.join(td, "s.vcf")
        prepared = materialize_binary_snp_vcf(ts, vcf_path)
        G = prepared.G
        pos = prepared.pos
        n, S = G.shape
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        print(f"  n={n}, S={S}, pairs={len(pairs)}")
        print(f"  mu={mu:.2e}, rho={rho:.2e}")
        print(f"  4*Ne*mu={4*NE*mu:.2e}, 2*Ne*rho={2*NE*rho:.2e}")
        print(
            f"  normalized-vcf: kept={prepared.n_kept_records}/{prepared.n_total_records} "
            f"drop_non_snp={prepared.n_dropped_non_snp} "
            f"drop_non_binary={prepared.n_dropped_non_binary} "
            f"drop_missing={prepared.n_dropped_missing}"
        )

        # tmrca.cu cached FB
        t0 = time.perf_counter()
        cached = _core.gamma_smc_flow_cached_fb(
            G, pos, pairs, float(NE), mu, rho, FF, True, 0
        )["mean"]
        t_cached = time.perf_counter() - t0
        print(f"  cached_fb: {t_cached:.2f}s")

        # tmrca.cu iterative FB
        t0 = time.perf_counter()
        iterative = _core.gamma_smc_flow_fb(
            G, pos, pairs, float(NE), mu, rho, FF, True
        )["mean"]
        t_iter = time.perf_counter() - t0
        print(f"  iterative_fb: {t_iter:.2f}s")

        # gamma_smc binary
        t0 = time.perf_counter()
        gsmc_results, gsmc_pos = run_gsmc(vcf_path, n, mu, rho, NE)
        t_gsmc = time.perf_counter() - t0
        print(f"  gamma_smc: {t_gsmc:.2f}s")

        rows = []
        for pidx, pair in enumerate(pairs):
            truth = true_t(ts, pair[0], pair[1], pos)
            r_cached = r_log(truth, cached[:, pidx])
            r_iter = r_log(truth, iterative[:, pidx])
            key = tuple(sorted(pair))
            if key in gsmc_results:
                est_g = np.interp(pos, gsmc_pos, gsmc_results[key])
                r_gsmc = r_log(truth, est_g)
            else:
                r_gsmc = np.nan
            rows.append((r_cached, r_iter, r_gsmc))

    arr = np.array(rows)
    print(f"\n  per-pair r_log summary (n={len(rows)} pairs):")
    print(f"    cached_fb    median={np.median(arr[:,0]):.4f}  "
          f"q25={np.quantile(arr[:,0],.25):.4f}  q75={np.quantile(arr[:,0],.75):.4f}")
    print(f"    iterative_fb median={np.median(arr[:,1]):.4f}  "
          f"q25={np.quantile(arr[:,1],.25):.4f}  q75={np.quantile(arr[:,1],.75):.4f}")
    print(f"    gamma_smc    median={np.nanmedian(arr[:,2]):.4f}  "
          f"q25={np.nanquantile(arr[:,2],.25):.4f}  q75={np.nanquantile(arr[:,2],.75):.4f}")

    print(f"\n  cached vs iterative delta: "
          f"median={np.median(arr[:,0]-arr[:,1]):+.4f}")
    print(f"  iterative vs gamma_smc delta: "
          f"median={np.nanmedian(arr[:,1]-arr[:,2]):+.4f}")

    # Raw estimate magnitudes for first pair (ballpark sanity check)
    truth0 = true_t(ts, 0, 1, pos)
    print(f"\n  pair (0,1) TMRCA stats (generations, log-mean):")
    print(f"    truth:        {np.exp(np.mean(np.log(np.maximum(truth0,1)))):.1f}")
    print(f"    cached_fb:    {np.exp(np.mean(np.log(np.maximum(cached[:,0],1)))):.1f}")
    print(f"    iterative_fb: {np.exp(np.mean(np.log(np.maximum(iterative[:,0],1)))):.1f}")
    if (0, 1) in gsmc_results:
        gsmc0 = np.interp(pos, gsmc_pos, gsmc_results[(0, 1)])
        print(f"    gamma_smc:    {np.exp(np.mean(np.log(np.maximum(gsmc0,1)))):.1f}")

    return {
        "species": species_id, "model": model_id, "pop": pop,
        "n_sites": int(S), "n_pairs": int(len(pairs)),
        "mu": mu, "rho": rho,
        "r_cached_median": float(np.median(arr[:,0])),
        "r_iterative_median": float(np.median(arr[:,1])),
        "r_gsmc_median": float(np.nanmedian(arr[:,2])),
        "t_cached": round(t_cached, 2),
        "t_iterative": round(t_iter, 2),
        "t_gsmc": round(t_gsmc, 2),
    }


def main():
    all_results = []
    for spec in CONFIGS:
        all_results.append(diagnose(*spec))

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'species':<8} {'model':<30} {'cached':>8} {'iter':>8} {'gsmc':>8}  "
          f"{'cached-gsmc':>12} {'iter-gsmc':>10}")
    for r in all_results:
        print(f"{r['species']:<8} {r['model'][:30]:<30} "
              f"{r['r_cached_median']:>8.4f} "
              f"{r['r_iterative_median']:>8.4f} "
              f"{r['r_gsmc_median']:>8.4f}  "
              f"{r['r_cached_median']-r['r_gsmc_median']:>+12.4f} "
              f"{r['r_iterative_median']-r['r_gsmc_median']:>+10.4f}")

    out = os.path.join(HERE, "diagnose_canfam.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
