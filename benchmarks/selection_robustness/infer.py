"""Run one method in a separate process and preserve raw plus aligned estimates."""
import argparse
import gzip
import json
import time
from pathlib import Path

import msprime
import numpy as np

from common import digest, interval_means, seed_for, snp_cells, write_json


def asmc_input(out, G, positions, rmap):
    root = out/"input"
    pos1 = positions.astype(np.int64)+1
    if not np.all(positions == pos1-1):
        raise ValueError("ASMC input requires integral physical positions")
    # Genetic distances integrate the native zero-based map, in centimorgans.
    cm = np.interp(positions, rmap.position,
                   np.r_[0, np.cumsum(np.diff(rmap.position)*rmap.rate)])*100
    with gzip.open(str(root)+".hap.gz", "wt") as f:
        for j, pos in enumerate(pos1):
            f.write(f"1 snp{j} {pos} A G " + " ".join(map(str, G[:,j]))+"\n")
    with gzip.open(str(root)+".map.gz", "wt") as f:
        for j, pos in enumerate(pos1):
            f.write(f"1\tsnp{j}\t{cm[j]:.12f}\t{pos}\n")
    with open(str(root)+".samples", "w") as f:
        f.write("ID_1 ID_2 missing\n0 0 0\n")
        for j in range(G.shape[0]//2):
            f.write(f"s{j} s{j} 0\n")
    return str(root)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--method", choices=["gamma_fixed", "gamma_auto", "asmc_uniform", "asmc_map", "cxt"], required=True)
    p.add_argument("--cxt-draws", type=int, default=3)
    p.add_argument("--attempt", type=int, default=1)
    args = p.parse_args()
    if args.attempt < 1:
        raise ValueError("Attempt must be positive")
    out = args.run/(args.method if args.attempt == 1 else f"{args.method}_attempt{args.attempt}")
    out.mkdir(exist_ok=False)
    x = np.load(args.run/"input.npz")
    cfg = json.loads((args.run/"config.json").read_text())
    qc = json.loads((args.run/"qc.json").read_text())
    if digest(args.run/"input.npz") != qc["input_sha256"]:
        raise ValueError("Input hash changed")
    G, pos, pairs, edges = (x[k] for k in ["G", "positions", "pairs", "edges"])
    start = time.perf_counter()
    info = dict(method=args.method, input_sha256=qc["input_sha256"],
                script_sha256=digest(__file__), pairs=pairs.tolist())
    if args.method.startswith("gamma"):
        import gamma_smc_cu
        result = gamma_smc_cu.infer(G, pos, pairs=pairs.tolist(), mu=cfg["mu"],
                  rho=float(x["mean_r"]), Ne=cfg["Ne"], mean_only=True,
                  auto_estimate_theta=args.method=="gamma_auto")
        raw = result["mean"]
        if not np.array_equal(result["positions"], pos):
            raise ValueError("Gamma-SMC-CU unexpectedly changed shared SNP set")
        left, right = snp_cells(pos)
        aligned, coverage = interval_means(left, right, raw, edges)
        info["map_input"] = "scalar region mean recombination rate"
    elif args.method.startswith("asmc"):
        from asmc.asmc import ASMC
        rmap = msprime.RateMap(position=x["map_position"], rate=x["map_rate"])
        if args.method == "asmc_uniform":
            rmap = msprime.RateMap.uniform(edges[-1], rmap.mean_rate)
        inp = asmc_input(out, G, pos, rmap)
        dq = args.root/"tools"/f"asmc_constant{cfg['Ne']}"/"prior.decodingQuantities.gz"
        provenance = json.loads((dq.parent/"provenance.json").read_text())
        if (digest(dq) != provenance["dq_sha256"] or provenance["diploid_Ne"] != cfg["Ne"]
                or provenance["mutation_rate"] != cfg["mu"]):
            raise ValueError("ASMC prior provenance mismatch")
        method = ASMC(inp, str(dq), decoding_mode="sequence")
        method.set_store_per_pair_posterior_mean(True)
        method.decode_pairs(pairs[:,0].tolist(), pairs[:,1].tolist())
        raw = np.asarray(method.get_copy_of_results().per_pair_posterior_means).T
        if raw.shape != (len(pos), len(pairs)):
            raise ValueError(f"Unexpected ASMC posterior shape {raw.shape}")
        left, right = snp_cells(pos)
        aligned, coverage = interval_means(left, right, raw, edges)
        info.update(dq_sha256=digest(dq), map_input=args.method)
    else:
        import torch
        import cxt
        torch.set_num_threads(4)
        if edges[-1] != 1_000_000 or len(G) != 50:
            raise ValueError("Initial cxt adapter validated only for 50 haps and one 1-Mb block")
        checkpoint = args.root/"tools/cxt_broad.ckpt"
        expected = json.loads((args.root/"tools/cxt_provenance.json").read_text())["checkpoint_sha256"]
        if digest(checkpoint) != expected:
            raise ValueError("Checkpoint changed")
        model = cxt.load_model("broad", device="cuda:0", force_local=str(checkpoint))
        Y, index = cxt.translate((G.astype(np.int32), pos.astype(np.float32)), model,
                         blocks=[(0, 1_000_000)], pivot_pairs=pairs.tolist(),
                         n_reps=args.cxt_draws, base_seed=seed_for("cxt",cfg["seed"]),
                         devices=["cuda:0"], B=2, B_per_device=2, build_workers=1,
                         progress=False, data_type="gm", mutation_rate=None)
        # Pinned implementation returns (n_reps, n_items, 500), unlike some
        # historical README examples. Verify index_map and explicit dimensions.
        if args.cxt_draws == 1:
            Y = np.asarray(Y)[None,:,:]
        if Y.shape != (args.cxt_draws, len(pairs), 500):
            raise ValueError(f"Unexpected cxt shape {Y.shape}")
        index = np.asarray(index)
        if not np.array_equal(index, np.column_stack([np.zeros(len(pairs),int), np.arange(len(pairs))])):
            raise ValueError(f"Unexpected cxt row mapping {index}")
        raw = np.asarray(Y)
        aligned = np.exp(raw).mean(axis=0).T
        coverage = np.ones(500)
        torch.cuda.synchronize()
        info.update(checkpoint_sha256=expected, draws=args.cxt_draws,
                    raw_axes="draw,pair,window", mutation_correction=False,
                    map_input="none", uncertainty_calibration_claim=False)
    if aligned.shape != x["truth"].shape or np.any(aligned[coverage > .999] <= 0):
        raise ValueError("Invalid prediction shape/scale")
    if not np.all(np.isfinite(aligned[coverage > .999])):
        raise ValueError("Nonfinite prediction on supported grid")
    np.savez_compressed(out/"predictions.npz", raw=raw, prediction=aligned, coverage=coverage)
    # Smoke scoring excludes 100 kb on each side and incomplete projections.
    mask = (edges[:-1] >= 100_000) & (edges[1:] <= edges[-1]-100_000) & (coverage > .999)
    if not np.any(mask):
        raise ValueError("No supported interior scoring windows")
    err = np.log(aligned[mask])-np.log(x["truth"][mask])
    info.update(total_seconds=time.perf_counter()-start,
                scored_windows=int(mask.sum()), mean_log_error=float(err.mean()),
                rmse_log=float(np.sqrt(np.mean(err**2))),
                report_type="technical smoke only, no biological or comparative claim")
    write_json(out/"metrics.json",info)
    print(json.dumps(info,indent=2))


if __name__ == "__main__":
    main()
