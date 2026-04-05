#!/usr/bin/env python
"""
Run cxt (transformer-based TMRCA) on a specific genomic region for a focal population.
Produces per-site log-TMRCA averaged over a small set of within-population pairs.

Usage:
    python run_cxt_region.py \
        --region SH2B3_ALDH2 --chr 12 --start 110000000 --end 113000000 \
        --pop TSI --n-pairs 100 --outdir results/
"""

import argparse
import numpy as np
import os
import sys

# ---------------------------------------------------------------------------
# Paths on betty
# ---------------------------------------------------------------------------
BETTY_BASE = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide"
PARSED_DIR = os.path.join(BETTY_BASE, "cache/parsed")        # chr{N}.npz
SAMPLES_FILE = os.path.join(BETTY_BASE, "data/samples.txt")   # 1KG sample metadata


def load_population_indices(samples_file, pop):
    """Return haplotype indices for a given population.

    Handles the 1KG sample metadata format:
      FamilyID SampleID FatherID MotherID Sex Population Superpopulation
    Space- or tab-delimited.
    """
    indices = []
    with open(samples_file) as f:
        header = f.readline().strip().split()
        pop_col = None
        for j, col in enumerate(header):
            if col.lower() in ("pop", "population"):
                pop_col = j
                break
        if pop_col is None:
            pop_col = 5  # fallback: 'Population' is typically column 5
            print(f"Warning: no 'pop'/'Population' column in header {header}, using col {pop_col}")
        for i, line in enumerate(f):
            fields = line.strip().split()
            if len(fields) > pop_col and fields[pop_col] == pop:
                indices.extend([2 * i, 2 * i + 1])
    if not indices:
        raise ValueError(f"No samples found for population '{pop}' in {samples_file}")
    return np.array(indices)


def main():
    parser = argparse.ArgumentParser(description="Run cxt on a genomic region")
    parser.add_argument("--region", required=True, help="Region name for output files")
    parser.add_argument("--chr", type=int, required=True, help="Chromosome number")
    parser.add_argument("--start", type=int, required=True, help="Start position (bp)")
    parser.add_argument("--end", type=int, required=True, help="End position (bp)")
    parser.add_argument("--pop", required=True, help="Focal population code (e.g. TSI)")
    parser.add_argument("--n-pairs", type=int, default=100, help="Number of random pairs")
    parser.add_argument("--n-reps", type=int, default=15, help="Stochastic replicates")
    parser.add_argument("--model-type", default="broad", help="cxt model variant")
    parser.add_argument("--device", default="cuda:0", help="GPU device")
    parser.add_argument("--outdir", default="results", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mutation-rate", type=float, default=1.29e-8)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # --- Load parsed data ---
    npz_path = os.path.join(PARSED_DIR, f"chr{args.chr}.npz")
    print(f"Loading {npz_path} ...")
    data = np.load(npz_path)
    G_full = data["G"]           # (n_haplotypes, n_sites)
    pos_full = data["positions"]  # (n_sites,) in bp
    print(f"  Full chromosome: {G_full.shape[0]} haplotypes, {G_full.shape[1]} sites")

    # --- Subset to region ---
    mask = (pos_full >= args.start) & (pos_full <= args.end)
    pos = pos_full[mask]
    G_region = G_full[:, mask]
    del G_full, pos_full  # free memory
    print(f"Region chr{args.chr}:{args.start}-{args.end}: {mask.sum()} sites")

    # --- Subset to focal population ---
    hap_idx = load_population_indices(SAMPLES_FILE, args.pop)
    G_pop_full = G_region[hap_idx]
    del G_region
    n_haps_full = len(hap_idx)
    print(f"Population {args.pop}: {n_haps_full} haplotypes ({n_haps_full // 2} individuals)")

    # --- Subsample to 50 haplotypes (model expects num_samples=50) ---
    rng = np.random.default_rng(args.seed)
    N_MODEL = 50
    if n_haps_full > N_MODEL:
        sub_idx = rng.choice(n_haps_full, size=N_MODEL, replace=False)
        sub_idx.sort()
        G_pop = G_pop_full[sub_idx]
        n_haps = N_MODEL
        print(f"Subsampled to {N_MODEL} haplotypes for model compatibility")
    else:
        G_pop = G_pop_full
        n_haps = n_haps_full
    del G_pop_full

    # --- Pick random pairs ---
    n_total_pairs = n_haps * (n_haps - 1) // 2
    n_pick = min(args.n_pairs, n_total_pairs)

    # Sample pair indices without enumerating all pairs
    pair_set = set()
    while len(pair_set) < n_pick:
        i = rng.integers(0, n_haps)
        j = rng.integers(0, n_haps)
        if i != j:
            pair_set.add((min(i, j), max(i, j)))
    pivot_pairs = list(pair_set)
    print(f"Selected {n_pick} pairs from {n_total_pairs} total")

    # --- Run cxt ---
    import cxt

    print(f"Loading cxt model '{args.model_type}' on {args.device} ...")
    model = cxt.load_model(args.model_type, device=args.device)

    # --- Tile region into uniform 1 Mb blocks ---
    # All blocks must be the same size for cxt (np.stack requirement).
    # Extend the last block to full size; output will be trimmed when plotting.
    BLOCK_SIZE = 1_000_000
    blocks = []
    bp = int(args.start)
    while bp < int(args.end):
        blocks.append((bp, bp + BLOCK_SIZE))
        bp += BLOCK_SIZE
    print(f"Tiled into {len(blocks)} blocks of {BLOCK_SIZE/1e6:.0f} Mb: {blocks}")

    print("Running cxt.translate ...")
    log_tmrca_raw, index_map = cxt.translate(
        (G_pop, pos.astype(float)),
        model,
        blocks=blocks,
        pivot_pairs=pivot_pairs,
        devices=[args.device],
        B=128,
        n_reps=args.n_reps,
        mutation_rate=args.mutation_rate,
    )

    # log_tmrca_raw shape: (n_items, n_reps, n_windows) where n_items = n_blocks * n_pairs
    # index_map shape: (n_items, 2) -> [block_idx, pivot_idx]
    print(f"Raw output shape: {log_tmrca_raw.shape}, index_map: {index_map.shape}")

    # --- Save raw + summaries ---
    out_prefix = os.path.join(args.outdir, f"cxt_{args.region}_{args.pop}")
    np.savez(
        f"{out_prefix}.npz",
        log_tmrca_raw=log_tmrca_raw,
        index_map=index_map,
        blocks=np.array(blocks),
        region=args.region,
        pop=args.pop,
        chr=args.chr,
        start=args.start,
        end=args.end,
        n_pairs=n_pick,
        n_reps=args.n_reps,
        pivot_pairs=np.array(pivot_pairs),
    )
    print(f"Saved: {out_prefix}.npz")
    print(f"  raw shape: {log_tmrca_raw.shape} (items x reps x windows)")
    tmrca_mean = np.exp(log_tmrca_raw.mean(axis=1).mean(axis=0))
    print(f"  mean TMRCA range: {tmrca_mean.min():.0f} - {tmrca_mean.max():.0f}")
    print("Done.")


if __name__ == "__main__":
    main()
