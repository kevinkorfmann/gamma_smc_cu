"""
Does gamma_smc's advantage on CanFam/BosTau come from its heterozygosity-based
auto-estimation of theta?

Three modes for gamma_smc, same data each time:

  explicit : -m 4*Ne*mu  -r 4*Ne*rho              (what our test suite does)
  auto_m   : no -m       -r 4*Ne*rho              (let gsmc learn theta, keep rho)
  auto_mt  : no -m       -t rho/mu                (fully data-driven; rho derived)

Reports per-pair median r_log (log-TMRCA vs msprime truth) for each mode plus
the gsmc-reported scaled mutation rate in each mode, so we can see how far the
data-estimated theta is from the passed-in theta.
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "python"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from tmrca_cu import _core  # noqa: E402

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


def run_gsmc_mode(ts, nh, mu, rho, ne, mode):
    """Returns (per_pair_means_dict, positions, reported_scaled_mu_or_None)."""
    with tempfile.TemporaryDirectory() as td:
        vp = os.path.join(td, "s.vcf")
        with open(vp, "w") as f:
            ts.write_vcf(f, contig_id="chr1", allow_position_zero=True)
        subprocess.run(f"bgzip -f {vp} && tabix -p vcf {vp}.gz",
                       shell=True, check=True, capture_output=True)
        out = os.path.join(td, "out")

        cmd = [GSMC, "-i", vp + ".gz", "-o", out, "-f", FF, "-h"]
        if mode == "explicit":
            cmd += ["-m", str(4 * ne * mu), "-r", str(4 * ne * rho)]
        elif mode == "auto_m":
            cmd += ["-r", str(4 * ne * rho)]
        elif mode == "auto_mt":
            cmd += ["-t", str(rho / mu)]
        else:
            raise ValueError(mode)

        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=600, check=True)
        # Parse reported scaled mutation rate from stdout
        m = re.search(r"Scaled mutation rate:\s*([0-9.eE+-]+)", result.stdout)
        reported_scaled_mu = float(m.group(1)) if m else None

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
            # gsmc output is in units where the point estimate of T is alpha/beta
            # and we rescale to generations by * 2*Ne -- same for all three modes.
            mean_gen = (alpha / np.maximum(beta, 1e-10)) * 2 * ne
            results[tuple(sorted(p))] = mean_gen
        return results, positions, reported_scaled_mu


def benchmark(species_id, model_id, pop):
    print(f"\n{'='*75}")
    print(f"{species_id}  {model_id}  pop={pop}")
    print('='*75)

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

    # Observed pi (segregating sites / (n_hap * seq_len)) for sanity check --
    # not exactly Watterson theta but a quick ballpark for "data-implied" theta.
    obs_het = float(np.mean(G.sum(0) != 0))   # fraction of sites where any hap differs
    watterson = S / SEQ_LEN  # very rough: segregating density
    print(f"  n={n}, S={S}, mu={mu:.2e}, rho={rho:.2e}")
    print(f"  4*Ne*mu (passed)    = {4*NE*mu:.3e}")
    print(f"  S/L (rough density) = {watterson:.3e}")

    # tmrca.cu baseline (what we currently report)
    tmrca_out = _core.gamma_smc_flow_cached_fb(
        G, pos, pairs, float(NE), mu, rho, FF, True, 0
    )["mean"]

    rows_tmrca = []
    for pidx, pair in enumerate(pairs):
        truth = true_t(ts, pair[0], pair[1], pos)
        rows_tmrca.append(r_log_safe(truth, tmrca_out[:, pidx]))
    r_tmrca = np.nanmedian(rows_tmrca)
    print(f"\n  tmrca.cu (our current benchmark) median r_log: {r_tmrca:.4f}")

    results = {"tmrca_cu": r_tmrca}

    for mode in ("explicit", "auto_m", "auto_mt"):
        try:
            gsmc_results, gsmc_pos, reported = run_gsmc_mode(
                ts, n, mu, rho, NE, mode
            )
        except Exception as e:
            print(f"  gsmc [{mode:8s}]: FAILED ({e})")
            results[f"gsmc_{mode}"] = np.nan
            results[f"gsmc_{mode}_theta"] = None
            continue

        rs = []
        for pidx, pair in enumerate(pairs):
            truth = true_t(ts, pair[0], pair[1], pos)
            key = tuple(sorted(pair))
            if key not in gsmc_results:
                continue
            est = np.interp(pos, gsmc_pos, gsmc_results[key])
            rs.append(r_log_safe(truth, est))
        med = float(np.nanmedian(rs))
        print(f"  gsmc [{mode:8s}]: median r_log={med:.4f}  "
              f"(reported scaled_mu={reported})")
        results[f"gsmc_{mode}"] = med
        results[f"gsmc_{mode}_theta"] = reported

    return {
        "species": species_id, "model": model_id, "pop": pop,
        "mu": mu, "rho": rho, "n_sites": int(S),
        "theta_passed": 4 * NE * mu,
        **results,
    }


def main():
    all_rows = []
    for cfg in CONFIGS:
        all_rows.append(benchmark(*cfg))

    print("\n" + "="*95)
    print("SUMMARY")
    print("="*95)
    print(f"{'species':<8} {'model':<30}  "
          f"{'tmrca':>7}  {'g:expl':>7} {'g:autoM':>7} {'g:autoMT':>8}  "
          f"{'theta_passed':>12} {'theta_auto':>12}")
    for r in all_rows:
        theta_auto = r.get("gsmc_auto_m_theta")
        theta_auto_s = f"{theta_auto:.3e}" if theta_auto else "---"
        print(f"{r['species']:<8} {r['model'][:30]:<30}  "
              f"{r['tmrca_cu']:>7.4f}  "
              f"{r['gsmc_explicit']:>7.4f} "
              f"{r['gsmc_auto_m']:>7.4f} "
              f"{r['gsmc_auto_mt']:>8.4f}  "
              f"{r['theta_passed']:>12.3e} "
              f"{theta_auto_s:>12}")

    out = os.path.join(HERE, "diagnose_auto_estimate.json")
    with open(out, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
