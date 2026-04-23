"""
Direct tmrca.cu-vs-gamma_smc posterior diff on one stdpopsim config.

This is the parity harness we actually need once truth-correlation gets close:
run both implementations on the same normalized VCF, align one pair exactly,
and compare the posterior mean / alpha / beta arrays site-by-site.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

import msprime
import numpy as np
import stdpopsim
from scipy.stats import pearsonr

from bench_inputs import materialize_binary_snp_vcf
from bench_paths import resolve_flow_field_path, resolve_gamma_smc_bin
from configs import expand_configs

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PY_MOD = os.path.join(REPO, "python")
sys.path.insert(0, PY_MOD)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from gamma_smc_cu import _core  # noqa: E402
from gamma_smc_cu.infer import _estimate_scaled_params  # noqa: E402

FF = resolve_flow_field_path(HERE)
GSMC = resolve_gamma_smc_bin(HERE)
NE = 10_000


def _log_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lx = np.log(np.maximum(x, 1e-10))
    ly = np.log(np.maximum(y, 1e-10))
    mask = np.isfinite(lx) & np.isfinite(ly)
    if mask.sum() < 10:
        return float("nan")
    return float(pearsonr(lx[mask], ly[mask])[0])


def _summarize_rel_err(a, b):
    rel = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)) / np.maximum(
        np.asarray(b, dtype=float), 1e-10
    )
    return {
        "median_rel_err": float(np.median(rel)),
        "p95_rel_err": float(np.quantile(rel, 0.95)),
        "max_rel_err": float(np.max(rel)),
    }


def _run_gsmc(vcf_path, mu, rho, pair, mode):
    td = os.path.dirname(vcf_path)
    subprocess.run(
        f"bgzip -f {vcf_path} && tabix -f -p vcf {vcf_path}.gz",
        shell=True,
        check=True,
        capture_output=True,
    )
    out = os.path.join(td, "out")
    cmd = [GSMC, "-i", vcf_path + ".gz", "-o", out, "-f", FF, "-h"]
    if mode == "auto":
        cmd += ["-t", str(rho / max(mu, 1e-30))]
    elif mode == "explicit":
        cmd += ["-m", str(4.0 * NE * mu), "-r", str(4.0 * NE * rho)]
    else:
        raise ValueError(mode)
    result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)
    m = re.search(r"Scaled mutation rate:\s*([0-9.eE+-]+)", result.stdout)
    reported_scaled_mu = float(m.group(1)) if m else None

    dec = os.path.join(td, "out.bin")
    subprocess.run(["zstd", "-d", out, "-o", dec], capture_output=True)
    with open(dec, "rb") as f:
        raw = f.read()
    with open(out + ".meta") as f:
        meta = json.load(f)

    pairs = meta["pairs"]
    pair_list = [list(pair), tuple(pair)]
    target = None
    for candidate in pair_list:
        if candidate in pairs:
            target = pairs.index(candidate)
            break
    if target is None:
        raise RuntimeError(f"pair {pair} not found in gamma_smc output")

    cs = meta["chunk_size"]
    arr = np.frombuffer(raw, dtype=np.float32).reshape(
        (meta["num_pairs"] + cs - 1) // cs, 2, meta["sequence_length"], cs
    )
    chunk = target // cs
    off = target % cs
    alpha = arr[chunk, 0, :, off].astype(np.float32)
    beta = arr[chunk, 1, :, off].astype(np.float32)
    positions = np.asarray(meta["output_positions"], dtype=np.float64)
    mean = (alpha / np.maximum(beta, 1e-10)) * 2.0 * NE
    return {
        "positions": positions,
        "alpha": alpha,
        "beta": beta,
        "mean": mean.astype(np.float32),
        "reported_scaled_mu": reported_scaled_mu,
    }


def _run_tmrca(G, pos, mu, rho, pair, mode):
    if mode == "auto":
        kernel_mu, kernel_rho = _estimate_scaled_params(G, pos, mu, rho, NE)
    elif mode == "explicit":
        kernel_mu, kernel_rho = float(mu), float(rho)
    else:
        raise ValueError(mode)
    ctx = _core.FlowContext(G, pos, float(NE), kernel_mu, kernel_rho, FF, 0)
    out = ctx.run_fb([pair], mean_only=True, return_posterior=True)
    return {
        "mean": out["mean"][:, 0].astype(np.float32),
        "alpha": out["posterior_alpha"][:, 0].astype(np.float32),
        "beta": out["posterior_beta"][:, 0].astype(np.float32),
        "scaled_mu": float(4.0 * NE * kernel_mu),
        "scaled_rho": float(4.0 * NE * kernel_rho),
    }


def _top_sites(pos, tm_mean, gm_mean, tm_alpha, gm_alpha, tm_beta, gm_beta, k):
    rel = np.abs(tm_mean - gm_mean) / np.maximum(gm_mean, 1e-10)
    order = np.argsort(rel)[-k:][::-1]
    prev = np.concatenate(([-1.0], pos[:-1]))
    rows = []
    for idx in order:
        rows.append(
            {
                "site_index": int(idx),
                "position": float(pos[idx]),
                "gap_from_prev": float(pos[idx] - prev[idx]),
                "mean_gamma_smc_cu": float(tm_mean[idx]),
                "mean_gamma_smc": float(gm_mean[idx]),
                "mean_rel_err": float(rel[idx]),
                "alpha_gamma_smc_cu": float(tm_alpha[idx]),
                "alpha_gamma_smc": float(gm_alpha[idx]),
                "beta_gamma_smc_cu": float(tm_beta[idx]),
                "beta_gamma_smc": float(gm_beta[idx]),
            }
        )
    return rows


def simulate_config(cfg):
    species = stdpopsim.get_species(cfg["species"])
    model = species.get_demographic_model(cfg["model_id"])
    contig = species.get_contig(length=cfg["seq_len"])
    samples = {cfg["pop"]: cfg["n_hap"] // 2}
    engine = stdpopsim.get_engine("msprime")
    ts_raw = engine.simulate(model, contig, samples, seed=cfg["seed"])
    return msprime.sim_mutations(ts_raw, rate=cfg["mu"], random_seed=cfg["seed"] + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-idx", type=int, required=True)
    parser.add_argument("--pair-i", type=int, default=0)
    parser.add_argument("--pair-j", type=int, default=1)
    parser.add_argument("--mode", choices=("auto", "explicit"), default="auto")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    configs = expand_configs()
    cfg = next(c for c in configs if c["config_idx"] == args.config_idx)
    pair = (args.pair_i, args.pair_j)

    print(
        f"[{cfg['species']}/{cfg['model_id']}] pair={pair} mode={args.mode}",
        flush=True,
    )
    ts = simulate_config(cfg)

    with tempfile.TemporaryDirectory() as td:
        vcf_path = os.path.join(td, "s.vcf")
        prepared = materialize_binary_snp_vcf(ts, vcf_path)
        G = prepared.G
        pos = prepared.pos
        tm = _run_tmrca(G, pos, cfg["mu"], cfg["rho"], pair, args.mode)
        gm = _run_gsmc(vcf_path, cfg["mu"], cfg["rho"], pair, args.mode)

        if not np.array_equal(pos, gm["positions"]):
            raise RuntimeError("position mismatch between tmrca.cu input and gamma_smc output")

        summary = {
            "config_idx": cfg["config_idx"],
            "species": cfg["species"],
            "model_id": cfg["model_id"],
            "pair": list(pair),
            "mode": args.mode,
            "n_sites": int(len(pos)),
            "tmrca_scaled_mu": tm["scaled_mu"],
            "tmrca_scaled_rho": tm["scaled_rho"],
            "gamma_smc_scaled_mu": gm["reported_scaled_mu"],
            "mean_log_corr": _log_corr(tm["mean"], gm["mean"]),
            "alpha_log_corr": _log_corr(tm["alpha"], gm["alpha"]),
            "beta_log_corr": _log_corr(tm["beta"], gm["beta"]),
            "mean_rel": _summarize_rel_err(tm["mean"], gm["mean"]),
            "alpha_rel": _summarize_rel_err(tm["alpha"], gm["alpha"]),
            "beta_rel": _summarize_rel_err(tm["beta"], gm["beta"]),
            "top_sites": _top_sites(
                pos,
                tm["mean"],
                gm["mean"],
                tm["alpha"],
                gm["alpha"],
                tm["beta"],
                gm["beta"],
                args.top_k,
            ),
        }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
