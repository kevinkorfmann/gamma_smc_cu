"""
Smoking-gun experiment for the factor-of-2 suspicion on scaled_rho.

Internally, tmrca.cu's FlowContext sets scaled_rho = 2*Ne*rho while the gamma_smc
binary expects 4*Ne*rho as its -r argument. The two tools share the same flow
field file, so if gamma_smc's convention is correct then tmrca.cu advances the
flow at half the expected rate per base pair.

This script recomputes the benchmark on one HomSap config (should already be
good) and the two bad species (CanFam, BosTau) using:

  (a) rho passed as-is (what the paper used)
  (b) rho doubled at the Python level (same effect as patching scaled_rho to
      4*Ne*rho inside bindings.cpp)

If hypothesis is correct, (b) should rescue CanFam/BosTau. If HomSap stays the
same or improves under (b), the fix is safe to land. If HomSap regresses, the
convention story is more subtle and the fix needs to be conditional.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import msprime
import stdpopsim
from scipy.stats import pearsonr
from bench_paths import resolve_flow_field_path, resolve_gamma_smc_bin

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "python"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from gamma_smc_cu import _core  # noqa: E402

FF = resolve_flow_field_path(HERE)
GSMC = resolve_gamma_smc_bin(HERE)

CONFIGS = [
    ("HomSap", "OutOfAfrica_3G09", "YRI"),
    ("HomSap", "Africa_1T12", "AFR"),
    ("CanFam", "EarlyWolfAdmixture_6F14", "BSJ"),
    ("BosTau", "HolsteinFriesian_1M13", "Holstein_Friesian"),
    ("AraTha", "SouthMiddleAtlas_1D17", "SouthMiddleAtlas"),
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


def r_log_safe(x, y):
    lx = np.log(np.maximum(x, 1e-10))
    ly = np.log(np.maximum(y, 1e-10))
    mask = np.isfinite(lx) & np.isfinite(ly)
    if mask.sum() < 10:
        return np.nan
    lx, ly = lx[mask], ly[mask]
    if np.std(lx) == 0 or np.std(ly) == 0:
        return np.nan
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


def run_gsmc(ts, nh, mu, rho, ne):
    with tempfile.TemporaryDirectory() as td:
        vp = os.path.join(td, "s.vcf")
        with open(vp, "w") as f:
            ts.write_vcf(f, contig_id="chr1", allow_position_zero=True)
        subprocess.run(f"bgzip -f {vp} && tabix -p vcf {vp}.gz",
                       shell=True, check=True, capture_output=True)
        out = os.path.join(td, "out")
        subprocess.run(
            [GSMC, "-i", vp + ".gz", "-o", out,
             "-m", str(4 * ne * mu), "-r", str(4 * ne * rho),
             "-f", FF, "-h"],
            check=True, capture_output=True, timeout=600,
        )
        dec = os.path.join(td, "out.bin")
        subprocess.run(["zstd", "-d", out, "-o", dec], capture_output=True)
        with open(dec, "rb") as f: raw = f.read()
        with open(out + ".meta") as f: meta = json.load(f)
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


def benchmark(species_id, model_id, pop):
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

    G = ts.genotype_matrix().T.astype(np.uint8)
    pos = np.array([v.position for v in ts.variants()], dtype=np.float64)
    n, S = G.shape
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    print(f"  n={n}, S={S}, mu={mu:.2e}, rho={rho:.2e}")
    print(f"  4*Ne*mu={4*NE*mu:.2e}")
    print(f"  tmrca.cu internal 2*Ne*rho={2*NE*rho:.2e}  "
          f"gamma_smc -r=4*Ne*rho={4*NE*rho:.2e}")

    # Current tmrca.cu path (rho as-is)
    out_a = _core.gamma_smc_flow_cached_fb(
        G, pos, pairs, float(NE), mu, rho, FF, True, 0
    )["mean"]

    # Doubled rho -- same effect as patching bindings.cpp to use 4*Ne*rho
    out_b = _core.gamma_smc_flow_cached_fb(
        G, pos, pairs, float(NE), mu, 2 * rho, FF, True, 0
    )["mean"]

    # gamma_smc binary
    gsmc_results, gsmc_pos = run_gsmc(ts, n, mu, rho, NE)

    rows = []
    for pidx, pair in enumerate(pairs):
        truth = true_t(ts, pair[0], pair[1], pos)
        r_a = r_log_safe(truth, out_a[:, pidx])
        r_b = r_log_safe(truth, out_b[:, pidx])
        key = tuple(sorted(pair))
        if key in gsmc_results:
            est_g = np.interp(pos, gsmc_pos, gsmc_results[key])
            r_gsmc = r_log_safe(truth, est_g)
        else:
            r_gsmc = np.nan
        rows.append((r_a, r_b, r_gsmc))

    arr = np.array(rows)
    med = np.nanmedian(arr, axis=0)
    q25 = np.nanquantile(arr, 0.25, axis=0)
    q75 = np.nanquantile(arr, 0.75, axis=0)
    print(f"\n  median r_log (n={np.sum(np.isfinite(arr[:,0]))} / "
          f"{np.sum(np.isfinite(arr[:,1]))} / "
          f"{np.sum(np.isfinite(arr[:,2]))} pairs):")
    print(f"    tmrca.cu as-is       : {med[0]:.4f}  "
          f"[{q25[0]:.4f}, {q75[0]:.4f}]")
    print(f"    tmrca.cu with 2*rho  : {med[1]:.4f}  "
          f"[{q25[1]:.4f}, {q75[1]:.4f}]")
    print(f"    gamma_smc binary     : {med[2]:.4f}  "
          f"[{q25[2]:.4f}, {q75[2]:.4f}]")
    print(f"\n  delta (2*rho - as-is): {med[1]-med[0]:+.4f}")
    print(f"  delta (2*rho - gsmc) : {med[1]-med[2]:+.4f}")

    # Quick TMRCA magnitude check for pair (0,1)
    truth0 = true_t(ts, 0, 1, pos)
    def gm(a): return np.exp(np.nanmean(np.log(np.maximum(a, 1))))
    print(f"\n  pair (0,1) geometric mean TMRCA:")
    print(f"    truth        : {gm(truth0):>10.1f}")
    print(f"    as-is        : {gm(out_a[:, 0]):>10.1f}")
    print(f"    2*rho        : {gm(out_b[:, 0]):>10.1f}")
    if (0, 1) in gsmc_results:
        g = np.interp(pos, gsmc_pos, gsmc_results[(0, 1)])
        print(f"    gamma_smc    : {gm(g):>10.1f}")

    return {
        "species": species_id, "model": model_id, "pop": pop,
        "mu": mu, "rho": rho,
        "r_asis_median": float(med[0]),
        "r_2rho_median": float(med[1]),
        "r_gsmc_median": float(med[2]),
    }


def main():
    out = []
    for c in CONFIGS:
        out.append(benchmark(*c))

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'species':<8} {'model':<34} "
          f"{'as-is':>8} {'2*rho':>8} {'gsmc':>8}  "
          f"{'2rho-asis':>10}  {'2rho-gsmc':>10}")
    for r in out:
        print(f"{r['species']:<8} {r['model'][:34]:<34} "
              f"{r['r_asis_median']:>8.4f} "
              f"{r['r_2rho_median']:>8.4f} "
              f"{r['r_gsmc_median']:>8.4f}  "
              f"{r['r_2rho_median']-r['r_asis_median']:>+10.4f}  "
              f"{r['r_2rho_median']-r['r_gsmc_median']:>+10.4f}")

    with open(os.path.join(HERE, "diagnose_rho_scaling.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
