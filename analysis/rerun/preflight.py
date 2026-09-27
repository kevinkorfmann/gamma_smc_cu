"""Empirical full-pass versus blockwise diagnostic, before production jobs."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "python"), str(ROOT / "analysis/genome_wide")]
from analysis.rerun.cache import load_chromosome
from rerun_support import calibrate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--samples", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--chr", type=int, default=22)
    p.add_argument("--pop", default="ASW")
    p.add_argument("--core-sites", type=int, default=65536)
    p.add_argument("--flank-sites", type=int, default=8192)
    a = p.parse_args()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    import gamma_smc_cu
    d = load_chromosome(a.cache_dir, a.chr)
    membership = {}
    for line in Path(a.samples).read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 7:
            membership[parts[1]] = parts[5]
    indices = [j for i, sid in enumerate(d["sample_ids"]) if membership.get(str(sid)) == a.pop for j in (2*i, 2*i+1)]
    G = np.ascontiguousarray(d["G"][indices])
    positions = d["positions"]
    calibration = calibrate(G, positions, 1.25e-8, 1e-8)
    all_pairs = np.array([(i,j) for i in range(len(G)) for j in range(i+1,len(G))], dtype=np.int32)
    pairs = all_pairs[np.sort(np.random.default_rng(20260925).choice(len(all_pairs), 20, replace=False))]
    kwargs = dict(mu=1.25e-8, rho=1e-8, Ne=calibration["calibrated_Ne"],
                  auto_estimate_theta=False, mean_only=True, pairs=pairs)
    start = time.time()
    full = gamma_smc_cu.infer(G, positions, **kwargs)
    full_seconds = time.time()-start
    start = time.time()
    block = gamma_smc_cu.infer_blockwise(G, positions, core_block_sites=a.core_sites,
                                       flank_sites=a.flank_sites, **kwargs)
    block_seconds = time.time()-start
    np.testing.assert_array_equal(full["positions"], block["positions"])
    for data in (full["mean"], block["mean"]):
        if not np.all(np.isfinite(data)) or not np.all(data > 0):
            raise ValueError("Invalid posterior means")
    if len(full["positions"]) <= a.core_sites:
        raise ValueError("Preflight must exercise multiple core blocks")
    diff = np.abs(np.log(block["mean"].astype(float)) - np.log(full["mean"].astype(float)))
    quantiles = dict(zip(("median", "p95", "p99", "max"), map(float, np.quantile(diff,[.5,.95,.99,1]))))
    genes = pd.read_csv(Path(a.cache_dir)/"genes"/f"chr{a.chr}_genes.tsv", sep="\t")
    rows = []
    for gene in genes.itertuples():
        keep = (full["positions"] >= gene.start) & (full["positions"] <= gene.end)
        if np.count_nonzero(keep) >= 2:
            f = float(np.exp(np.log(full["mean"][keep].astype(float)).mean()))
            b = float(np.exp(np.log(block["mean"][keep].astype(float)).mean()))
            rows.append(dict(gene_name=gene.gene_name, n_sites=int(keep.sum()), full=f, blockwise=b, abs_log_error=abs(np.log(b/f))))
    table = pd.DataFrame(rows)
    table.to_csv(out/"gene_comparison.csv", index=False)
    # Preserve a deterministic small cross-host check without exporting dense chromosome arrays.
    sample_sites = np.unique(np.linspace(0, len(full["positions"])-1, 2000, dtype=int))
    np.savez_compressed(out/"cross_host.npz", positions=full["positions"][sample_sites], pairs=pairs,
                        full=full["mean"][sample_sites], blockwise=block["mean"][sample_sites])
    passed = quantiles["p99"] <= .01 and quantiles["max"] <= .1 and float(table.abs_log_error.max()) <= .01
    result = dict(passed=passed, settings=vars(a), calibration=calibration,
                  diagnostic_limits={"site_p99_abs_log":.01,"site_max_abs_log":.1,"gene_max_abs_log":.01},
                  site_abs_log_error=quantiles, gene_max_abs_log_error=float(table.abs_log_error.max()),
                  retained_sites=len(full["positions"]), samples=len(indices)//2, pairs=pairs.tolist(),
                  full_seconds=full_seconds, blockwise_seconds=block_seconds,
                  hostname=platform.node(), gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  full_metadata=full.get("metadata"), blockwise_metadata=block.get("metadata"))
    (out/"validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
