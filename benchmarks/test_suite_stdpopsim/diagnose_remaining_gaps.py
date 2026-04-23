"""
Investigate the three configs where gamma_smc still beats tmrca.cu after
both methods adopt data-driven scaled rates:

  AraTha  African2Epoch_1H18    Δ = -0.132   (biggest)
  DroMel  African3Epoch_1S16    Δ = -0.049
  AnoGam  GabonAg1000G_1A17     Δ = -0.028

Hypothesis pool:
  H1  cache vs iterative     -- the cached transition table loses precision
                                relative to the iterative kernel on dense /
                                high-theta sites.
  H2  flow-field clipping    -- on extreme configs the (mean, cv) state walks
                                outside the calibrated grid; tmrca.cu's
                                fmin/fmax clamp differs from gamma_smc's edge
                                behavior.
  H3  step granularity       -- tmrca.cu uses int(pos - prev_pos + 0.5) for
                                gap_steps; with ~25 bp average gaps fractional
                                rounding compounds.

This script tests H1 cleanly: run BOTH tmrca.cu kernels (cached and iterative)
on the same data with the same auto-estimated scaled rates, then compare
per-pair r_log against the msprime ground truth. Same gamma_smc binary call as
the test suite. If iterative beats cached, the cache is the issue and the fix
is to route the test suite through `gamma_smc_flow_fb` instead of
`gamma_smc_flow_cached_fb`. If both lose to gamma_smc by the same margin, the
issue is something else (clipping or precision) and we need a deeper look.
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
from bench_inputs import materialize_binary_snp_vcf
from bench_paths import resolve_flow_field_path, resolve_gamma_smc_bin

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "python"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from gamma_smc_cu import _core  # noqa: E402
from gamma_smc_cu.infer import _estimate_scaled_params  # noqa: E402

FF = resolve_flow_field_path(HERE)
GSMC = resolve_gamma_smc_bin(HERE)

# (species, model, pop) tuples that still lose to gamma_smc after auto-theta.
CONFIGS = [
    ("HomSap", "OutOfAfrica_3G09", "YRI"),               # control (matches)
    ("AnoGam", "GabonAg1000G_1A17", "GAS"),
    ("DroMel", "African3Epoch_1S16", "AFR"),
    ("AraTha", "African2Epoch_1H18", "SouthMiddleAtlas"),
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


def run_gsmc_auto(vcf_path, nh, mu, rho, ne):
    """gamma_smc auto_mt: omit -m, pass -t (rho/mu)."""
    td = os.path.dirname(vcf_path)
    subprocess.run(f"bgzip -f {vcf_path} && tabix -f -p vcf {vcf_path}.gz",
                   shell=True, check=True, capture_output=True)
    out = os.path.join(td, "out")
    result = subprocess.run(
        [GSMC, "-i", vcf_path + ".gz", "-o", out,
         "-t", str(rho / max(mu, 1e-30)), "-f", FF, "-h"],
        capture_output=True, text=True, check=True, timeout=600,
    )
    m = re.search(r"Scaled mutation rate:\s*([0-9.eE+-]+)", result.stdout)
    reported = float(m.group(1)) if m else None

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
    return results, positions, reported


def run_config(species_id, model_id, pop):
    print(f"\n{'='*72}")
    print(f"{species_id}  {model_id}  pop={pop}")
    print('='*72)

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
        print(f"  n={n}, S={S}, mu={mu:.2e}, rho={rho:.2e}")
        print(
            f"  normalized-vcf: kept={prepared.n_kept_records}/{prepared.n_total_records} "
            f"drop_non_snp={prepared.n_dropped_non_snp} "
            f"drop_non_binary={prepared.n_dropped_non_binary} "
            f"drop_missing={prepared.n_dropped_missing}"
        )

        # auto-theta both ways
        kernel_mu, kernel_rho = _estimate_scaled_params(G, pos, mu, rho, NE)
        pi_hat = 4 * NE * kernel_mu
        print(f"  pi_hat (tmrca.cu auto-theta)     = {pi_hat:.4e}")

        # tmrca.cu cached FB with auto-theta
        cached = _core.gamma_smc_flow_cached_fb(
            G, pos, pairs, float(NE), kernel_mu, kernel_rho, FF, True, 0
        )["mean"]

        # tmrca.cu iterative FB (no cache) with auto-theta
        try:
            iterative = _core.gamma_smc_flow_fb(
                G, pos, pairs, float(NE), kernel_mu, kernel_rho, FF, True
            )["mean"]
        except Exception as e:
            print(f"  iterative_fb FAILED: {e}")
            iterative = None

        # gamma_smc auto_mt
        gsmc_results, gsmc_pos, reported_mu = run_gsmc_auto(vcf_path, n, mu, rho, NE)
        print(f"  pi_hat (gamma_smc auto_mt)        = {reported_mu:.4e}")
        print(f"  ratio (gsmc/tmrca)               = "
              f"{(reported_mu/pi_hat) if pi_hat>0 else float('nan'):.4f}")

        # per-pair r_log against truth
        r_cached = []
        r_iterative = []
        r_gsmc = []
        for pidx, pair in enumerate(pairs):
            truth = true_t(ts, pair[0], pair[1], pos)
            r_cached.append(r_log_safe(truth, cached[:, pidx]))
            if iterative is not None:
                r_iterative.append(r_log_safe(truth, iterative[:, pidx]))
            key = tuple(sorted(pair))
            if key in gsmc_results:
                est = np.interp(pos, gsmc_pos, gsmc_results[key])
                r_gsmc.append(r_log_safe(truth, est))

    def stats(v):
        a = np.asarray(v, dtype=float)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return None, None, None
        return float(np.median(a)), float(np.quantile(a, 0.25)), float(np.quantile(a, 0.75))

    med_c, q25_c, q75_c = stats(r_cached)
    med_i, q25_i, q75_i = stats(r_iterative) if iterative is not None else (None, None, None)
    med_g, q25_g, q75_g = stats(r_gsmc)

    print(f"\n  median r_log:")
    print(f"    cached_fb     : {med_c:.4f}  [{q25_c:.4f}, {q75_c:.4f}]")
    if med_i is not None:
        print(f"    iterative_fb  : {med_i:.4f}  [{q25_i:.4f}, {q75_i:.4f}]")
    print(f"    gamma_smc auto: {med_g:.4f}  [{q25_g:.4f}, {q75_g:.4f}]")

    if med_i is not None:
        print(f"\n  delta (cached - iterative)   = {med_c - med_i:+.4f}")
        print(f"  delta (cached - gsmc_auto)   = {med_c - med_g:+.4f}")
        print(f"  delta (iterative - gsmc_auto)= {med_i - med_g:+.4f}")
    else:
        print(f"\n  delta (cached - gsmc_auto)   = {med_c - med_g:+.4f}")

    return {
        "species": species_id, "model": model_id, "pop": pop,
        "n_sites": int(S), "pi_hat_tmrca": pi_hat, "pi_hat_gsmc": reported_mu,
        "r_cached": med_c, "r_iterative": med_i, "r_gsmc_auto": med_g,
    }


def main():
    rows = []
    for cfg in CONFIGS:
        rows.append(run_config(*cfg))

    print("\n" + "="*86)
    print("SUMMARY")
    print("="*86)
    print(f"{'species':<8} {'model':<28}  "
          f"{'cached':>8} {'iter':>8} {'gsmc':>8}  "
          f"{'c-g':>8}  {'i-g':>8}")
    for r in rows:
        c = r['r_cached']
        i = r['r_iterative']
        g = r['r_gsmc_auto']
        i_str = f"{i:.4f}" if i is not None else "----"
        ig_str = f"{i-g:+.4f}" if i is not None else "----"
        print(f"{r['species']:<8} {r['model'][:28]:<28}  "
              f"{c:>8.4f} {i_str:>8} {g:>8.4f}  "
              f"{c-g:>+8.4f}  {ig_str:>8}")

    out = os.path.join(HERE, "diagnose_remaining_gaps.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
