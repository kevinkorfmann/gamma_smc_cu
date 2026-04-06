#!/usr/bin/env python
"""
Run tmrca_cu inference for GRK2 fire plot (Schweiger Fig 5B style).
200k pairwise coalescence times across the GRK2 region for SAS + YRI.

Computes the 2D histogram (position × log10-years) in chunks to avoid
storing the full (n_sites, n_pairs) matrix, which can exceed 80 GB.

Usage (on betty, via slurm):
    python run_fire_grk2.py --pop SAS --n-pairs 200000 --outdir results/fire/
    python run_fire_grk2.py --pop YRI --n-pairs 200000 --outdir results/fire/
"""

import argparse
import numpy as np
import os
import sys
import time

# ---------------------------------------------------------------------------
# Paths on betty
# ---------------------------------------------------------------------------
BETTY_BASE = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide"
PARSED_DIR = os.path.join(BETTY_BASE, "cache/parsed")
SAMPLES_FILE = os.path.join(BETTY_BASE, "data/samples.txt")
FLOW_FIELD = "/vast/projects/smathi/cohort/kkor/tmrca.cu/default_flow_field.txt"

# GRK2 region
CHR = 11
REGION_START = 66_000_000
REGION_END = 68_000_000

GENERATION_TIME = 29  # years per generation

# 2D histogram bins
N_POS_BINS = 400       # position bins across region
N_TIME_BINS = 300      # log10(years) bins
TIME_RANGE = (3.0, 6.5)  # log10(years): 1,000 - 3,162,278 years

# Chunk size: how many pairs to infer at once
CHUNK_SIZE = 10_000

SUPERPOP_POPS = {
    "SAS": ["BEB", "GIH", "ITU", "PJL", "STU"],
    "AFR": ["YRI", "LWK", "GWD", "MSL", "ESN", "ACB", "ASW"],
    "EUR": ["CEU", "TSI", "FIN", "GBR", "IBS"],
    "EAS": ["CHB", "JPT", "CHS", "CDX", "KHV"],
}


def load_population_indices(samples_file, pop):
    """Return haplotype indices for a population or superpopulation."""
    pops_to_load = SUPERPOP_POPS.get(pop, [pop])
    indices = []
    with open(samples_file) as f:
        header = f.readline().strip().split()
        pop_col = None
        for j, col in enumerate(header):
            if col.lower() in ("pop", "population"):
                pop_col = j
                break
        if pop_col is None:
            pop_col = 5
        for i, line in enumerate(f):
            fields = line.strip().split()
            if len(fields) > pop_col and fields[pop_col] in pops_to_load:
                indices.extend([2 * i, 2 * i + 1])
    if not indices:
        raise ValueError(f"No samples found for '{pop}' in {samples_file}")
    return np.array(indices)


def make_pairs(n_haps, n_pick, rng):
    """Generate pair indices."""
    n_total = n_haps * (n_haps - 1) // 2
    n_pick = min(n_pick, n_total)
    if n_pick == n_total:
        pairs = []
        for i in range(n_haps):
            for j in range(i + 1, n_haps):
                pairs.append((i, j))
        return np.array(pairs), n_pick
    else:
        pair_set = set()
        while len(pair_set) < n_pick:
            batch = rng.integers(0, n_haps, size=(n_pick * 2, 2))
            for row in batch:
                if row[0] != row[1]:
                    pair_set.add((min(row[0], row[1]), max(row[0], row[1])))
                if len(pair_set) >= n_pick:
                    break
        return np.array(list(pair_set)[:n_pick]), n_pick


def main():
    parser = argparse.ArgumentParser(description="tmrca_cu fire plot inference")
    parser.add_argument("--pop", required=True,
                        help="Population (BEB, YRI) or superpopulation (SAS, AFR)")
    parser.add_argument("--n-pairs", type=int, default=0,
                        help="Number of pairs (0 = all pairs)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--outdir", default="results/fire")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # --- Load chromosome data ---
    npz_path = os.path.join(PARSED_DIR, f"chr{CHR}.npz")
    print(f"Loading {npz_path} ...")
    data = np.load(npz_path)
    G_full = data["G"]
    pos_full = data["positions"]
    print(f"  Full chr{CHR}: {G_full.shape[0]} haplotypes, {G_full.shape[1]} sites")

    # --- Subset to GRK2 region ---
    mask = (pos_full >= REGION_START) & (pos_full <= REGION_END)
    pos = pos_full[mask]
    G_region = G_full[:, mask]
    del G_full, pos_full
    n_sites = mask.sum()
    print(f"  Region {REGION_START/1e6:.0f}-{REGION_END/1e6:.0f} Mb: {n_sites} sites")

    # --- Get population haplotypes ---
    hap_idx = load_population_indices(SAMPLES_FILE, args.pop)
    G_pop = G_region[hap_idx]
    del G_region
    n_haps = len(hap_idx)
    n_total_pairs = n_haps * (n_haps - 1) // 2
    print(f"  {args.pop}: {n_haps} haplotypes ({n_haps // 2} individuals)")
    print(f"  Total possible pairs: {n_total_pairs:,}")

    # --- Sample pairs (0 = all) ---
    n_request = n_total_pairs if args.n_pairs <= 0 else args.n_pairs
    pairs, n_pick = make_pairs(n_haps, n_request, rng)
    print(f"  Using {n_pick:,} pairs")

    # --- Set up 2D histogram ---
    pos_mb = pos / 1e6
    pos_edges = np.linspace(pos_mb.min(), pos_mb.max(), N_POS_BINS + 1)
    time_edges = np.linspace(TIME_RANGE[0], TIME_RANGE[1], N_TIME_BINS + 1)
    H = np.zeros((N_POS_BINS, N_TIME_BINS), dtype=np.int64)

    # Pre-compute position bin for each site
    site_pos_bin = np.digitize(pos_mb, pos_edges) - 1
    site_pos_bin = np.clip(site_pos_bin, 0, N_POS_BINS - 1)

    # --- Run tmrca_cu in chunks, accumulate histogram ---
    sys.path.insert(0, "/vast/projects/smathi/cohort/kkor/tmrca.cu/python")
    import tmrca_cu

    n_chunks = (n_pick + args.chunk_size - 1) // args.chunk_size
    print(f"\nRunning tmrca_cu.infer() in {n_chunks} chunks of {args.chunk_size} ...")
    t0 = time.time()

    for c in range(n_chunks):
        c_start = c * args.chunk_size
        c_end = min(c_start + args.chunk_size, n_pick)
        chunk_pairs = pairs[c_start:c_end]

        result = tmrca_cu.infer(
            G_pop, positions=pos,
            mu=1.29e-8, rho=1e-8, Ne=10_000,
            pairs=chunk_pairs.tolist(),
            flow_field_path=FLOW_FIELD,
            mean_only=True,
        )

        # result["mean"]: (n_sites, n_chunk_pairs) in generations
        tmrca_gen = result["mean"]  # (n_sites, n_chunk_pairs)

        # Convert to log10(years), clamp to valid range
        tmrca_years = tmrca_gen * GENERATION_TIME
        log_years = np.log10(np.clip(tmrca_years, 10**TIME_RANGE[0], 10**TIME_RANGE[1]))

        # Accumulate into 2D histogram
        time_bin = np.digitize(log_years, time_edges) - 1
        time_bin = np.clip(time_bin, 0, N_TIME_BINS - 1)

        # Vectorized accumulation
        for s in range(n_sites):
            pb = site_pos_bin[s]
            np.add.at(H[pb], time_bin[s], 1)

        elapsed_chunk = time.time() - t0
        pairs_done = c_end
        rate = pairs_done / elapsed_chunk if elapsed_chunk > 0 else 0
        eta = (n_pick - pairs_done) / rate if rate > 0 else 0
        print(f"  Chunk {c+1}/{n_chunks}: {pairs_done:,}/{n_pick:,} pairs "
              f"({elapsed_chunk:.1f}s, {rate:.0f} pairs/s, ETA {eta:.0f}s)")

        del tmrca_gen, tmrca_years, log_years, time_bin

    elapsed = time.time() - t0
    print(f"\nTotal: {elapsed:.1f}s ({n_pick / elapsed:.0f} pairs/sec)")

    # --- Save histogram + metadata ---
    out_path = os.path.join(args.outdir, f"fire_GRK2_{args.pop}.npz")
    np.savez(
        out_path,
        histogram=H,               # (N_POS_BINS, N_TIME_BINS)
        pos_edges=pos_edges,        # (N_POS_BINS + 1,) in Mb
        time_edges=time_edges,      # (N_TIME_BINS + 1,) in log10(years)
        pop=args.pop,
        chr=CHR,
        start=REGION_START,
        end=REGION_END,
        n_pairs=n_pick,
        n_haplotypes=n_haps,
        elapsed_sec=elapsed,
    )
    print(f"Saved: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    print(f"  Histogram shape: {H.shape}")
    print(f"  Non-zero bins: {(H > 0).sum():,} / {H.size:,}")
    print(f"  Max bin count: {H.max():,}")


if __name__ == "__main__":
    main()
