"""
Run a single stdpopsim config: simulate, run tmrca.cu and gamma_smc
(Schweiger and Durbin, 2023), compute accuracy and wall time, write JSON.

Invoked by slurm_array.sh as:
    python run_one.py --config-idx ${SLURM_ARRAY_TASK_ID}
"""
import argparse
import gc
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np
import msprime
import stdpopsim
from scipy.stats import pearsonr
from bench_inputs import materialize_binary_snp_vcf
from bench_paths import resolve_flow_field_path, resolve_gamma_smc_bin

# --- betty paths ---------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PY_MOD = os.path.join(REPO, "python")
FF = resolve_flow_field_path(HERE)
GSMC = resolve_gamma_smc_bin(HERE)
CONFIGS_JSON = os.path.join(HERE, "configs.json")
RESULTS_DIR = os.path.join(HERE, "results")

sys.path.insert(0, PY_MOD)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Imported after sys.path edit so the local tmrca.cu build is found.
from tmrca_cu import _core  # noqa: E402
from tmrca_cu.infer import _estimate_scaled_params  # noqa: E402


# --- helpers (copied from benchmarks/bench_demographics.py) --------------
def true_t(ts, i, j, positions):
    """Ground-truth TMRCA at each site position from tree sequence."""
    t = np.empty(len(positions))
    tit = ts.trees()
    tree = next(tit)
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
    """Pair layout used by the gamma_smc binary output."""
    nd = nh // 2
    pairs = []
    for dA in range(nd):
        for dB in range(dA, nd):
            if dA == dB:
                pairs.append((2 * dA, 2 * dA + 1))
            else:
                for ha in range(2):
                    for hb in range(2):
                        pairs.append((2 * dA + ha, 2 * dB + hb))
    return pairs


def run_schweiger(vcf_path, nh, mu_d, rho_d, ne_d):
    """Run gamma_smc binary in auto-estimation mode; return
    ``(dict[pair -> mean_array], positions, wall_total, wall_compute)``.

    Invoked WITHOUT ``-m`` so gamma_smc reads the scaled mutation rate
    from observed heterozygosity (``calculate_heterozygosity()``,
    Schweiger and Durbin 2023), and with ``-t (rho/mu)`` so the scaled
    recombination rate is derived from the same data-driven base via
    the user-supplied ratio. This is the same parameterization that
    ``tmrca_cu.infer(auto_estimate_theta=True)`` uses, so the two
    methods compete on equal footing.

    wall_compute = just the gamma_smc subprocess itself (no VCF/bgzip/zstd).
    wall_total   = everything including I/O wrapping.
    """
    t_total0 = time.perf_counter()
    td = os.path.dirname(vcf_path)
    subprocess.run(
        f"bgzip -f {vcf_path} && tabix -f -p vcf {vcf_path}.gz",
        shell=True, check=True, capture_output=True,
    )
    out = os.path.join(td, "out")

    t_c0 = time.perf_counter()
    result = subprocess.run(
        [GSMC, "-i", vcf_path + ".gz", "-o", out,
         "-t", str(rho_d / max(mu_d, 1e-30)),
         "-f", FF, "-h"],
        capture_output=True, text=True, timeout=600,
    )
    wall_compute = time.perf_counter() - t_c0

    if result.returncode != 0:
        raise RuntimeError(
            f"gamma_smc failed: rc={result.returncode}\n"
            f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
        )

    dec = os.path.join(td, "out.bin")
    subprocess.run(["zstd", "-d", out, "-o", dec], capture_output=True)
    if not os.path.exists(dec):
        if os.path.exists(out):
            shutil.copy(out, dec)
        else:
            raise RuntimeError("gamma_smc output not found after decompress")

    with open(dec, "rb") as f:
        raw = f.read()
    meta_path = out + ".meta"
    if not os.path.exists(meta_path):
        metas = glob.glob(os.path.join(td, "*.meta"))
        meta_path = metas[0] if metas else meta_path
    with open(meta_path) as f:
        meta = json.load(f)

    n_pairs = meta["num_pairs"]
    n_sites = meta["sequence_length"]
    cs = meta["chunk_size"]
    nc = (n_pairs + cs - 1) // cs
    positions = np.array(meta["output_positions"])

    arr = np.frombuffer(raw, dtype=np.float32).reshape(nc, 2, n_sites, cs)
    pair_layout = schweiger_pair_order(nh)
    results = {}
    for pidx, p in enumerate(pair_layout[:n_pairs]):
        alpha = arr[pidx // cs, 0, :, pidx % cs]
        beta = arr[pidx // cs, 1, :, pidx % cs]
        mean_gen = (alpha / np.maximum(beta, 1e-10)) * 2 * ne_d
        results[tuple(sorted(p))] = mean_gen

    wall_total = time.perf_counter() - t_total0
    return results, positions, wall_total, wall_compute


# --- main ----------------------------------------------------------------
def simulate_config(cfg):
    species = stdpopsim.get_species(cfg["species"])
    model = species.get_demographic_model(cfg["model_id"])
    contig = species.get_contig(length=cfg["seq_len"])
    samples = {cfg["pop"]: cfg["n_hap"] // 2}
    engine = stdpopsim.get_engine("msprime")
    ts_raw = engine.simulate(model, contig, samples, seed=cfg["seed"])
    ts = msprime.sim_mutations(ts_raw, rate=cfg["mu"], random_seed=cfg["seed"] + 1)
    return ts


def run(cfg):
    # Tool health check (compute node may not have bgzip/tabix/zstd on PATH).
    for tool in ("bgzip", "tabix", "zstd"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} not found on PATH")
    if not os.path.exists(GSMC):
        raise RuntimeError(f"gamma_smc binary missing at {GSMC}")
    if not os.path.exists(FF):
        raise RuntimeError(f"flow field missing at {FF}")

    print(f"[{cfg['species']}/{cfg['model_id']}] simulating ...", flush=True)
    t0 = time.perf_counter()
    ts = simulate_config(cfg)
    t_sim = time.perf_counter() - t0
    print(f"  sim: {t_sim:.1f}s", flush=True)

    with tempfile.TemporaryDirectory() as td:
        vcf_path = os.path.join(td, "s.vcf")
        prepared = materialize_binary_snp_vcf(ts, vcf_path)
        G = prepared.G
        pos = prepared.pos
        print(
            f"  normalized-vcf: kept={prepared.n_kept_records}/{prepared.n_total_records} "
            f"drop_non_snp={prepared.n_dropped_non_snp} "
            f"drop_non_binary={prepared.n_dropped_non_binary} "
            f"drop_missing={prepared.n_dropped_missing}",
            flush=True,
        )
        n, S = G.shape
        if S < 50:
            raise RuntimeError(f"too few segregating sites after VCF normalization: S={S}")
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        n_pairs = len(all_pairs)
        print(f"  n={n}, S={S}, pairs={n_pairs}", flush=True)

        ne = 10_000  # constant Ne assumption (the point of a misspec benchmark)
        mu, rho = cfg["mu"], cfg["rho"]

        # ------ tmrca.cu timing (warmup + 3 reps, take min) ------------------
        # Replace (mu, rho) with the data-driven auto-estimate that matches
        # gamma_smc's auto_mt mode -- the scaled mutation rate becomes the
        # observed pairwise heterozygosity and the scaled recombination rate
        # becomes pi_hat * (rho/mu). See tmrca_cu.infer._estimate_scaled_params.
        # Ne is still used to invert the kernel's internal per-bp scaling.
        kernel_mu, kernel_rho = _estimate_scaled_params(G, pos, mu, rho, ne)
        print(
            f"  auto-theta: pi_hat={4*ne*kernel_mu:.3e} "
            f"(vs passed 4*Ne*mu={4*ne*mu:.3e})", flush=True,
        )
        print("  tmrca.cu: warmup ...", flush=True)
        ctx = _core.FlowContext(G, pos, float(ne), kernel_mu, kernel_rho, FF, 0)
        _ = ctx.run_fb_summary([(0, 1)])  # small warmup to page in kernels
        del ctx

        tmrca_times = []
        tmrca_mean = None
        for rep in range(3):
            t0 = time.perf_counter()
            out = _core.gamma_smc_flow_cached_fb(
                G, pos, all_pairs, float(ne), kernel_mu, kernel_rho, FF, True, 0
            )["mean"]  # shape (S, n_pairs)
            dt = time.perf_counter() - t0
            tmrca_times.append(dt)
            if rep == 0:
                tmrca_mean = out
        t_tmrca_cu_compute = float(min(tmrca_times))
        t_tmrca_cu_total = t_tmrca_cu_compute  # no I/O wrapping
        print(f"  tmrca.cu: {t_tmrca_cu_compute:.3f}s (min of {len(tmrca_times)})", flush=True)

        # ------ gamma_smc ----------------------------------------------------
        print("  gamma_smc: running ...", flush=True)
        gsmc_results, gsmc_pos, t_gsmc_total, t_gsmc_compute = run_schweiger(
            vcf_path, n, mu, rho, ne
        )
        print(
            f"  gamma_smc: total={t_gsmc_total:.2f}s compute={t_gsmc_compute:.2f}s",
            flush=True,
        )

        # ------ accuracy (per pair) --------------------------------------
        r_tmrca, r_gsmc = [], []
        rmse_tmrca, rmse_gsmc = [], []
        for pidx, pair in enumerate(all_pairs):
            truth = true_t(ts, pair[0], pair[1], pos)
            est_t = tmrca_mean[:, pidx]
            r_tmrca.append(r_log(truth, est_t))
            rmse_tmrca.append(rmse_log(truth, est_t))

            key = tuple(sorted(pair))
            if key in gsmc_results:
                raw = gsmc_results[key]
                est_g = np.interp(pos, gsmc_pos, raw)
                r_gsmc.append(r_log(truth, est_g))
                rmse_gsmc.append(rmse_log(truth, est_g))

    def _qstats(v):
        a = np.asarray(v, dtype=float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return {"median": None, "q25": None, "q75": None, "n": 0}
        return {
            "median": float(np.median(a)),
            "q25": float(np.quantile(a, 0.25)),
            "q75": float(np.quantile(a, 0.75)),
            "n": int(a.size),
        }

    r_t = _qstats(r_tmrca)
    r_g = _qstats(r_gsmc)
    e_t = _qstats(rmse_tmrca)
    e_g = _qstats(rmse_gsmc)

    speedup_total = (t_gsmc_total / t_tmrca_cu_total) if t_tmrca_cu_total > 0 else None
    speedup_compute = (t_gsmc_compute / t_tmrca_cu_compute) if t_tmrca_cu_compute > 0 else None

    result = {
        "config_idx": cfg["config_idx"],
        "species": cfg["species"],
        "model_id": cfg["model_id"],
        "pop": cfg["pop"],
        "n_hap": n,
        "seq_len": cfg["seq_len"],
        "n_sites": int(S),
        "n_pairs": int(n_pairs),
        "mu": mu,
        "rho": rho,
        "rho_source": cfg.get("rho_source", "model"),
        "t_sim": round(t_sim, 3),
        "t_tmrca_cu_total": round(t_tmrca_cu_total, 4),
        "t_tmrca_cu_compute": round(t_tmrca_cu_compute, 4),
        "t_gsmc_total": round(t_gsmc_total, 4),
        "t_gsmc_compute": round(t_gsmc_compute, 4),
        "speedup_total": round(speedup_total, 2) if speedup_total else None,
        "speedup_compute": round(speedup_compute, 2) if speedup_compute else None,
        "r_tmrca_cu_median": r_t["median"],
        "r_tmrca_cu_q25": r_t["q25"],
        "r_tmrca_cu_q75": r_t["q75"],
        "r_gsmc_median": r_g["median"],
        "r_gsmc_q25": r_g["q25"],
        "r_gsmc_q75": r_g["q75"],
        "rmse_tmrca_cu_median": e_t["median"],
        "rmse_gsmc_median": e_g["median"],
        "n_pairs_evaluated_tmrca": r_t["n"],
        "n_pairs_evaluated_gsmc": r_g["n"],
        "status": "ok",
    }

    del G, ts, tmrca_mean, gsmc_results
    gc.collect()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-idx", type=int, required=True)
    args = ap.parse_args()

    if not os.path.exists(CONFIGS_JSON):
        sys.exit(f"configs.json not found at {CONFIGS_JSON}; run configs.py first")
    with open(CONFIGS_JSON) as f:
        configs = json.load(f)

    matches = [c for c in configs if c["config_idx"] == args.config_idx]
    if not matches:
        # Fall back: treat --config-idx as a positional index into the list.
        if 0 <= args.config_idx < len(configs):
            cfg = configs[args.config_idx]
        else:
            sys.exit(f"config idx {args.config_idx} not found in configs.json")
    else:
        cfg = matches[0]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"config_{args.config_idx:03d}.json")
    fail_path = os.path.join(RESULTS_DIR, f"config_{args.config_idx:03d}.FAILED")

    # Clean stale markers
    for p in (out_path, fail_path):
        if os.path.exists(p):
            os.remove(p)

    try:
        result = run(cfg)
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        with open(fail_path, "w") as f:
            f.write(json.dumps({
                "config_idx": args.config_idx,
                "species": cfg.get("species"),
                "model_id": cfg.get("model_id"),
                "error": tb,
            }, indent=2))
        sys.exit(1)

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out_path}")
    print(
        f"  r_tmrca_cu={result['r_tmrca_cu_median']:.3f} "
        f"r_gsmc={result['r_gsmc_median']:.3f} "
        f"speedup_total={result['speedup_total']}x"
    )


if __name__ == "__main__":
    main()
