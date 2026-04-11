#!/usr/bin/env python
"""Run cxt + tmrca.cu pairwise mode at one (gene, population) window.

For one task in the SLURM array we:
  1. Load the parsed chromosome NPZ.
  2. Subset to the focal population's haplotypes.
  3. Cut a +/-500 kb window around the gene midpoint.
  4. Pick a fixed set of pivot pairs (capped at PAIR_CAP).
  5. Run cxt regional inference -> log-TMRCA per pair per window
  6. Run tmrca.cu pairwise mode (the existing tmrca_cu.infer) -> per-site
     posterior mean TMRCA per pair
  7. Save both results to one NPZ at
     analysis/orthogonal_v41/three_method/{gene}_{pop}_{group}.npz

ASMC will be added in a follow-up sub-task once the file-format
conversion (NPZ -> ASMC binary input + decoding-quantities generation)
is in place; until then the figure has 2 of 3 method columns populated.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

# IMPORTANT: import torch/cxt FIRST so the newer libcudart 12.8 (shipped by
# nvidia-cuda-runtime-cu12 in the pip cache) is loaded before tmrca.cu pulls
# in the older pixi libcudart 12.2. The dynamic linker only loads each
# soname once per process, so the first one wins. tmrca.cu (built against
# 12.2) is forward-compatible with the 12.8 runtime.
try:
    import torch  # noqa: F401
    import cxt    # noqa: F401
except Exception:
    pass

import numpy as np

REPO = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
sys.path.insert(0, os.path.join(REPO, "python"))

DATA_DIR = os.path.join(REPO, "analysis/genome_wide")
PARSED_DIR = os.path.join(DATA_DIR, "cache/parsed")
SAMPLES_PATH = os.path.join(DATA_DIR, "data/samples.txt")
OUT_DIR = os.path.join(REPO, "analysis/orthogonal_v41/three_method")

WINDOW_BP = 500_000   # +/- around gene midpoint
PAIR_CAP = 20         # cap number of pairs per (gene, pop) for the figure


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
    # Use mmap_mode so we don't materialize the full ~30 GB G matrix.
    # The pop slice happens in main() and is the only data fully read.
    d = np.load(path, allow_pickle=True, mmap_mode="r")
    return d["G"], d["positions"], d["sample_ids"]


def get_pop_haps(sample_ids, pop_map, population):
    indices = []
    for i, sid in enumerate(sample_ids):
        if sid in pop_map and pop_map[sid][0] == population:
            indices.extend([2 * i, 2 * i + 1])
    return sorted(indices)


def gene_midpoint(gene_chr, gene_name):
    import pandas as pd
    path = os.path.join(DATA_DIR, "cache/genes", f"chr{gene_chr}_genes.tsv")
    df = pd.read_csv(path, sep="\t")
    row = df[df["gene_name"] == gene_name]
    if row.empty:
        raise SystemExit(f"gene {gene_name} not found in chr{gene_chr}_genes.tsv")
    r = row.iloc[0]
    return int((r["start"] + r["end"]) / 2), int(r["start"]), int(r["end"])


def run_tmrcacu(G_pop_full, positions_full, site_mask, pairs):
    """Run tmrca.cu on the FULL chromosome so auto_estimate_theta sees
    chromosome-wide heterozygosity (not the gene-window's depleted pi),
    then slice the output back to the gene window for plotting.

    Passing only the gene window with auto_estimate_theta=True gave
    systematically inflated TMRCAs (~150k vs ASMC/cxt ~2k for the same
    pairs) because sweep regions are heterozygosity-depleted and the
    window-local pi biases theta down -> biases TMRCAs up.
    """
    import tmrca_cu
    t0 = time.time()
    result = tmrca_cu.infer(
        G_pop_full, positions_full,
        mu=1.25e-8, rho=1e-8, Ne=10_000,
        pairs=pairs,
        mean_only=True,
        auto_estimate_theta=True,
    )
    out_positions = result["positions"]
    mean = result["mean"]
    # Slice the output to the gene window
    out_mask = np.isin(out_positions, positions_full[site_mask])
    return {
        "method": "tmrca_cu_pairwise",
        "mean": mean[out_mask, :],
        "positions": out_positions[out_mask],
        "elapsed": time.time() - t0,
    }


def run_cxt(G_pop, positions, pairs, win_lo, win_hi):
    """cxt regional inference (transformer-based).

    Runs on CPU. cxt and torch are imported at the top of this module
    BEFORE tmrca.cu, so the newer libcudart 12.8 from the pip-shipped
    nvidia-cuda-runtime-cu12 wheel wins the dynamic-linker race and
    libc10_cuda.so resolves cleanly.

    The cxt 'broad' model expects exactly num_samples=50 haplotypes per
    inference call. We subsample G_pop to 50 haps (keeping the pivot
    pairs at indices 0 and 1) and remap pairs accordingly.
    """
    try:
        import cxt
    except Exception as exc:
        return {"method": "cxt", "error": f"import: {exc}"}
    t0 = time.time()
    try:
        model = cxt.load_model(model_type="broad", device="cpu")
        N_SAMPLES = 50  # cxt 'broad' preset hardcoded num_samples

        n_haps = G_pop.shape[0]
        rng = np.random.default_rng(123)
        results_per_pair = []

        for (a, b) in pairs:
            # Build a 50-haplotype subset that includes the focal pair at
            # positions 0 and 1, plus 48 other random haplotypes from G_pop.
            others = [i for i in range(n_haps) if i not in (a, b)]
            if len(others) < N_SAMPLES - 2:
                # Not enough haps; pad with the focal pair (won't happen for
                # 1KG populations which all have >=74 haps)
                pick = others
            else:
                pick = rng.choice(others, size=N_SAMPLES - 2, replace=False).tolist()
            sub_idx = [a, b] + sorted(pick)
            G_sub = G_pop[np.array(sub_idx), :]

            Y, index_map = cxt.translate(
                (G_sub.astype(np.int32), positions.astype(np.float32)),
                model,
                blocks=[(int(win_lo), int(win_hi))],
                pivot_pairs=[(0, 1)],
                n_reps=3,
                devices=["cpu"],
                progress=False,
                data_type="gm",
            )
            # Y shape: (n_items=1, [n_reps=3,] n_windows)
            results_per_pair.append(np.asarray(Y))

        return {
            "method": "cxt",
            "log_tmrca": np.stack(results_per_pair, axis=0),  # (n_pairs, ...)
            "elapsed": time.time() - t0,
        }
    except Exception as exc:
        return {
            "method": "cxt",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            "elapsed": time.time() - t0,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", required=True)
    parser.add_argument("--chr", type=int, required=True)
    parser.add_argument("--pop", required=True)
    parser.add_argument("--group", required=True,
                        choices=["novel","positive","neutral","control"])
    args = parser.parse_args()

    print(f"=== {args.gene} chr{args.chr} {args.pop} ({args.group}) ===")
    os.makedirs(OUT_DIR, exist_ok=True)

    G, positions, sample_ids = load_chr_npz(args.chr)
    pop_map = load_samples()
    print(f"Loaded chr{args.chr}: G {G.shape}, positions {positions.shape}", flush=True)

    midpoint, gstart, gend = gene_midpoint(args.chr, args.gene)
    win_lo = midpoint - WINDOW_BP
    win_hi = midpoint + WINDOW_BP
    site_mask = (positions >= win_lo) & (positions <= win_hi)
    pos_win = positions[site_mask]
    print(f"Window: chr{args.chr}:{win_lo}-{win_hi}, sites: {site_mask.sum()}", flush=True)

    hap_idx = get_pop_haps(sample_ids, pop_map, args.pop)
    print(f"Population {args.pop}: {len(hap_idx)} haplotypes", flush=True)
    if len(hap_idx) < 4:
        print(f"  too few haplotypes, skipping", flush=True)
        return

    G_pop_full = np.ascontiguousarray(G[np.array(hap_idx), :])
    G_pop_win = np.ascontiguousarray(G_pop_full[:, site_mask])
    print(f"G_pop_full: {G_pop_full.shape} (used by tmrca.cu)", flush=True)
    print(f"G_pop_win: {G_pop_win.shape} (used by cxt+ASMC)", flush=True)

    n_pop = len(hap_idx)
    all_pairs = [(i, j) for i in range(n_pop) for j in range(i + 1, n_pop)]
    rng = np.random.default_rng(42)
    if len(all_pairs) > PAIR_CAP:
        idx = rng.choice(len(all_pairs), PAIR_CAP, replace=False)
        idx.sort()
        pairs = [all_pairs[i] for i in idx]
    else:
        pairs = all_pairs
    print(f"Using {len(pairs)} pivot pairs", flush=True)

    results = {}
    print("Running tmrca.cu pairwise (full chromosome for stable auto-theta)...", flush=True)
    results["tmrca_cu"] = run_tmrcacu(G_pop_full, positions, site_mask, pairs)

    print("Running cxt...", flush=True)
    results["cxt"] = run_cxt(G_pop_win, pos_win, pairs, win_lo, win_hi)

    out_path = os.path.join(OUT_DIR, f"{args.gene}_{args.pop}_{args.group}.npz")
    save_dict = {
        "gene": args.gene,
        "chr": args.chr,
        "pop": args.pop,
        "group": args.group,
        "midpoint": midpoint,
        "gstart": gstart,
        "gend": gend,
        "win_lo": win_lo,
        "win_hi": win_hi,
        "positions": pos_win,
        "pairs": np.array(pairs),
    }
    for name, r in results.items():
        if r.get("mean") is not None:
            save_dict[f"{name}_mean"] = r["mean"]
            save_dict[f"{name}_positions"] = r.get("positions", pos_win)
        if r.get("log_tmrca") is not None:
            save_dict[f"{name}_log_tmrca"] = r["log_tmrca"]
        if r.get("index_map") is not None:
            save_dict[f"{name}_index_map"] = r["index_map"]
        if r.get("error") is not None:
            save_dict[f"{name}_error"] = r["error"]
        if r.get("elapsed") is not None:
            save_dict[f"{name}_elapsed"] = r["elapsed"]

    np.savez_compressed(out_path, **save_dict)
    print(f"Wrote {out_path}", flush=True)

    for name, r in results.items():
        if r.get("error"):
            print(f"  {name}: ERROR -- {r['error'][:200]}", flush=True)
        else:
            elapsed = r.get("elapsed", 0)
            arr = r.get("mean") if r.get("mean") is not None else r.get("log_tmrca")
            shape = arr.shape if arr is not None else None
            print(f"  {name}: OK ({elapsed:.1f}s, shape={shape})", flush=True)


if __name__ == "__main__":
    main()
