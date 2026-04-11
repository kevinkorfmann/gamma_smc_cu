#!/usr/bin/env python
"""Run selscan iHS and nSL at one (chr, pop) task.

For each task we:
  1. Load the parsed chromosome NPZ (mmap).
  2. Subset rows to the target population's haplotypes.
  3. Write a .hap file (space-separated 0/1 per hap per row) and a
     PLINK-style .map file (chr snp_id cm_pos bp_pos) using 1 cM/Mb.
  4. Call the selscan binary twice: --ihs and --nsl.
  5. Leave the per-task output at
     analysis/orthogonal_v41/selscan/{chr}_{pop}/
     with selscan's canonical filenames.

The per-gene aggregation is done by a separate postprocess step.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

import numpy as np

REPO = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
DATA_DIR = os.path.join(REPO, "analysis/genome_wide")
PARSED_DIR = os.path.join(DATA_DIR, "cache/parsed")
SAMPLES_PATH = os.path.join(DATA_DIR, "data/samples.txt")
OUT_DIR = os.path.join(REPO, "analysis/orthogonal_v41/selscan")
SELSCAN_BIN = "/vast/projects/smathi/cohort/kkor/tools/selscan/bin/linux/selscan"


def load_samples():
    pops = {}
    with open(SAMPLES_PATH) as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                pops[parts[1]] = (parts[5], parts[6])
    return pops


def load_chr_npz(chr_num):
    path = os.path.join(PARSED_DIR, f"chr{chr_num}.npz")
    d = np.load(path, allow_pickle=True, mmap_mode="r")
    return d["G"], d["positions"], d["sample_ids"]


def get_pop_haps(sample_ids, pop_map, population):
    indices = []
    for i, sid in enumerate(sample_ids):
        if sid in pop_map and pop_map[sid][0] == population:
            indices.extend([2 * i, 2 * i + 1])
    return sorted(indices)


def write_hap_map(out_root: str, chr_num: int, positions, G_pop):
    """Write .hap and .map files in selscan format.

    .hap: one row per HAPLOTYPE, space-separated 0/1 per site
    .map: one row per site, 'chr snp_id cm_pos bp_pos'
    Uses 1 cM/Mb for the genetic map.
    """
    hap_path = out_root + ".hap"
    map_path = out_root + ".map"
    n_haps, n_sites = G_pop.shape

    with open(hap_path, "w") as f:
        for h in range(n_haps):
            row = " ".join(str(int(x)) for x in G_pop[h, :])
            f.write(row + "\n")

    with open(map_path, "w") as f:
        for i in range(n_sites):
            pos_bp = int(positions[i])
            cm = pos_bp * 1e-6
            snp_id = f"chr{chr_num}_{pos_bp}"
            f.write(f"{chr_num}\t{snp_id}\t{cm:.10f}\t{pos_bp}\n")

    return hap_path, map_path


def run_selscan(stat: str, hap_path: str, map_path: str, out_prefix: str,
                threads: int = 4):
    flag = {"ihs": "--ihs", "nsl": "--nsl"}[stat]
    cmd = [
        SELSCAN_BIN, flag,
        "--hap", hap_path,
        "--map", map_path,
        "--out", out_prefix,
        "--threads", str(threads),
    ]
    print("  " + " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("  stderr:", r.stderr[-800:], flush=True)
        raise SystemExit(f"selscan {stat} failed with code {r.returncode}")
    print(f"  {stat} OK ({time.time()-t0:.1f}s)", flush=True)
    return r.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chr", type=int, required=True)
    parser.add_argument("--pop", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    task_dir = os.path.join(OUT_DIR, f"chr{args.chr}_{args.pop}")
    os.makedirs(task_dir, exist_ok=True)
    print(f"=== selscan chr{args.chr} {args.pop} -> {task_dir} ===", flush=True)

    G, positions, sample_ids = load_chr_npz(args.chr)
    pop_map = load_samples()

    hap_idx = get_pop_haps(sample_ids, pop_map, args.pop)
    print(f"{len(hap_idx)} haplotypes", flush=True)
    if len(hap_idx) < 4:
        print("  too few, skipping", flush=True)
        return

    G_pop = np.ascontiguousarray(G[np.array(hap_idx), :])
    # Drop monomorphic sites (selscan requires polymorphic)
    af = G_pop.sum(axis=0) / G_pop.shape[0]
    poly = (af > 0) & (af < 1)
    G_pop = G_pop[:, poly]
    positions_poly = np.asarray(positions)[poly]
    print(f"{G_pop.shape[1]} polymorphic sites", flush=True)

    data_root = os.path.join(task_dir, "data")
    hap_path, map_path = write_hap_map(data_root, args.chr, positions_poly, G_pop)

    run_selscan("ihs", hap_path, map_path, os.path.join(task_dir, "ihs"),
                threads=args.threads)
    run_selscan("nsl", hap_path, map_path, os.path.join(task_dir, "nsl"),
                threads=args.threads)

    # Remove the large .hap/.map to keep disk tidy
    os.remove(hap_path)
    os.remove(map_path)
    print("=== done ===", flush=True)


if __name__ == "__main__":
    main()
