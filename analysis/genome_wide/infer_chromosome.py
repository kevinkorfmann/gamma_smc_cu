#!/usr/bin/env python
"""Per-chromosome, per-population TMRCA inference (full-chromosome mode).

Runs gamma_smc_cu.infer_blockwise() over the full chromosome in pair chunks,
accumulating per-gene summary statistics in both linear and log space,
plus a histogram of per-pair values for offline post-hoc quantile
computation.

Output per (chromosome, population):
    results/chr{N}/{POP}.csv   -- human-readable per-gene summary
                                   (primary stat: geometric mean TMRCA)
    results/chr{N}/{POP}.npz   -- raw accumulators for offline re-aggregation

The NPZ contains, per gene:
    count        : number of pairs contributing
    lin_sum      : sum of per-pair linear TMRCA (arith mean = lin_sum / count)
    log_sum      : sum of per-pair log TMRCA   (geom mean  = exp(log_sum/count))
    log_sq_sum   : sum of (per-pair log TMRCA)^2  (-> log variance)
    min_lin      : minimum per-pair linear TMRCA
    min_log      : minimum per-pair log TMRCA (== log(min_lin))
    histogram    : (n_genes, n_bins) counts of per-pair log-TMRCA
    bin_edges    : (n_bins+1,) natural-log edges spanning ln(10)..ln(1e6)

With these saved, any order statistic / quantile / threshold-fraction can
be recomputed offline without rerunning inference.

Usage:
    python infer_chromosome.py --chr 21
    python infer_chromosome.py --chr 1 --populations YRI CEU CHB
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np

BASE = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
DATA = os.path.join(BASE, "analysis/genome_wide")
CACHE_DIR = os.path.join(DATA, "cache")
RESULTS_DIR = os.path.join(DATA, "results")

MU = 1.25e-8
RHO = 1e-8
NE = 10_000

# Pair chunk size: controls peak RAM for the (n_sites, n_chunk_pairs) output.
PAIR_CHUNK = 1000

# Histogram: 50 natural-log bins from ln(10) to ln(1e6) generations.
# Width per bin ≈ 0.23 nats ≈ factor 1.26 in linear space.
HIST_NBINS = 50
HIST_LOG_LO = np.log(10.0)
HIST_LOG_HI = np.log(1_000_000.0)
HIST_EDGES = np.linspace(HIST_LOG_LO, HIST_LOG_HI, HIST_NBINS + 1)

# Floor for log-safety (avoids log(0) on any numerical zero).
TMRCA_FLOOR = 1.0

ALL_POPULATIONS = [
    "ACB", "ASW", "BEB", "CDX", "CEU", "CHB", "CHS", "CLM",
    "ESN", "FIN", "GBR", "GIH", "GWD", "IBS", "ITU", "JPT",
    "KHV", "LWK", "MSL", "MXL", "PEL", "PJL", "PUR", "STU",
    "TSI", "YRI",
]


def load_samples(samples_path):
    pops = {}
    with open(samples_path) as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                pops[parts[1]] = (parts[5], parts[6])
    return pops


def load_genes(chr_num):
    path = os.path.join(CACHE_DIR, "genes", f"chr{chr_num}_genes.tsv")
    genes = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            genes.append((parts[0], parts[1], int(parts[2]), int(parts[3])))
    return genes


def get_population_haplotype_indices(sample_ids, pop_map, population):
    indices = []
    for i, sid in enumerate(sample_ids):
        if sid in pop_map and pop_map[sid][0] == population:
            indices.extend([2 * i, 2 * i + 1])
    return sorted(indices)


def make_pairs(n_haps):
    return [(i, j) for i in range(n_haps) for j in range(i + 1, n_haps)]


def compute_gene_site_indices(positions, genes):
    """For each gene, return the array of site indices falling inside it.

    Returned as a list of np.int64 arrays so the chunk loop can do a
    single fancy-index per gene per chunk, avoiding recomputing masks.
    """
    result = []
    for _, _, gstart, gend in genes:
        mask = (positions >= gstart) & (positions <= gend)
        result.append(np.where(mask)[0].astype(np.int64))
    return result


def run_chromosome(chr_num, populations):
    import gamma_smc_cu

    print(f"=== Chromosome {chr_num} ===", flush=True)
    t0 = time.time()

    npz_path = os.path.join(CACHE_DIR, "parsed", f"chr{chr_num}.npz")
    print(f"Loading {npz_path}...", flush=True)
    data = np.load(npz_path, allow_pickle=True)
    G = data["G"]
    positions = data["positions"]
    sample_ids = data["sample_ids"]
    print(f"  G: {G.shape}, positions: {positions.shape}, samples: {len(sample_ids)}",
          flush=True)

    pop_map = load_samples(os.path.join(DATA, "data", "samples.txt"))
    genes = load_genes(chr_num)
    n_genes = len(genes)
    print(f"  {n_genes} genes", flush=True)

    out_dir = os.path.join(RESULTS_DIR, f"chr{chr_num}")
    os.makedirs(out_dir, exist_ok=True)

    for pop in populations:
        pop_t0 = time.time()
        hap_idx = get_population_haplotype_indices(sample_ids, pop_map, pop)
        n_pop = len(hap_idx)
        if n_pop < 4:
            print(f"  {pop}: skipped (only {n_pop} haplotypes)", flush=True)
            continue

        G_pop = np.ascontiguousarray(G[np.array(hap_idx), :])

        all_pairs = make_pairs(n_pop)
        n_pairs_total = len(all_pairs)
        n_chunks = (n_pairs_total + PAIR_CHUNK - 1) // PAIR_CHUNK
        print(f"  {pop}: {n_pop} haplotypes, {n_pairs_total} pairs, "
              f"{n_chunks} chunks of {PAIR_CHUNK}", flush=True)

        # Per-gene accumulators
        count       = np.zeros(n_genes, dtype=np.int64)
        lin_sum     = np.zeros(n_genes, dtype=np.float64)
        log_sum     = np.zeros(n_genes, dtype=np.float64)
        log_sq_sum  = np.zeros(n_genes, dtype=np.float64)
        min_lin     = np.full(n_genes, np.inf, dtype=np.float64)
        min_log     = np.full(n_genes, np.inf, dtype=np.float64)
        histogram   = np.zeros((n_genes, HIST_NBINS), dtype=np.int64)
        n_sites_per_gene = np.zeros(n_genes, dtype=np.int32)

        # Will be filled on first chunk
        gene_site_idx = None

        for ci in range(n_chunks):
            chunk_start = ci * PAIR_CHUNK
            chunk_end = min(chunk_start + PAIR_CHUNK, n_pairs_total)
            chunk_pairs = all_pairs[chunk_start:chunk_end]
            n_chunk_pairs = len(chunk_pairs)

            result = gamma_smc_cu.infer_blockwise(
                G_pop,
                positions,
                mu=MU,
                rho=RHO,
                Ne=NE,
                pairs=chunk_pairs,
                mean_only=True,
                auto_estimate_theta=True,
            )

            mean = result["mean"]          # (n_filtered_sites, n_chunk_pairs)
            out_positions = result["positions"]

            # Guard against numerical zero or negative values in mean
            mean_safe = np.maximum(mean, TMRCA_FLOOR)
            log_mean = np.log(mean_safe)   # (n_filtered_sites, n_chunk_pairs)

            if gene_site_idx is None:
                gene_site_idx = compute_gene_site_indices(out_positions, genes)
                for gi, idxs in enumerate(gene_site_idx):
                    n_sites_per_gene[gi] = len(idxs)

            for gi, idxs in enumerate(gene_site_idx):
                n_gene_sites = idxs.size
                if n_gene_sites < 2:
                    continue

                # Per-pair values for this gene (one number per pair):
                #   linear: arithmetic mean TMRCA across sites
                #   log:    arithmetic mean of log(TMRCA) across sites
                gene_lin = mean_safe[idxs, :]       # (n_gene_sites, n_chunk_pairs)
                gene_log = log_mean[idxs, :]
                per_pair_lin = gene_lin.mean(axis=0)
                per_pair_log = gene_log.mean(axis=0)

                count[gi]      += n_chunk_pairs
                lin_sum[gi]    += per_pair_lin.sum()
                log_sum[gi]    += per_pair_log.sum()
                log_sq_sum[gi] += (per_pair_log * per_pair_log).sum()

                chunk_min_lin = per_pair_lin.min()
                chunk_min_log = per_pair_log.min()
                if chunk_min_lin < min_lin[gi]:
                    min_lin[gi] = chunk_min_lin
                if chunk_min_log < min_log[gi]:
                    min_log[gi] = chunk_min_log

                # Histogram of per-pair log TMRCA
                bins = np.digitize(per_pair_log, HIST_EDGES) - 1
                np.clip(bins, 0, HIST_NBINS - 1, out=bins)
                np.add.at(histogram[gi], bins, 1)

            del result, mean, mean_safe, log_mean

            if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
                elapsed = time.time() - pop_t0
                print(f"    chunk {ci+1}/{n_chunks} done ({elapsed:.1f}s)", flush=True)

        # Sanitize: genes with zero contributions get NaN in the CSV
        with np.errstate(divide="ignore", invalid="ignore"):
            geom_mean = np.where(count > 0, np.exp(log_sum / count), np.nan)
            arith_mean = np.where(count > 0, lin_sum / count, np.nan)

        # Write CSV (primary stat = geometric mean)
        csv_path = os.path.join(out_dir, f"{pop}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["gene_id", "gene_name", "start", "end",
                 "geom_mean_tmrca", "arith_mean_tmrca",
                 "min_tmrca", "n_pairs", "n_sites"]
            )
            for gi, (gene_id, gene_name, gstart, gend) in enumerate(genes):
                gm = f"{geom_mean[gi]:.2f}" if np.isfinite(geom_mean[gi]) else ""
                am = f"{arith_mean[gi]:.2f}" if np.isfinite(arith_mean[gi]) else ""
                mn = f"{min_lin[gi]:.2f}" if np.isfinite(min_lin[gi]) else ""
                writer.writerow(
                    [gene_id, gene_name, gstart, gend, gm, am, mn,
                     int(count[gi]), int(n_sites_per_gene[gi])]
                )

        # Write NPZ with all raw accumulators
        npz_out = os.path.join(out_dir, f"{pop}.npz")
        gene_ids = np.array([g[0] for g in genes])
        gene_names = np.array([g[1] for g in genes])
        gene_starts = np.array([g[2] for g in genes], dtype=np.int64)
        gene_ends = np.array([g[3] for g in genes], dtype=np.int64)
        np.savez_compressed(
            npz_out,
            gene_id=gene_ids,
            gene_name=gene_names,
            start=gene_starts,
            end=gene_ends,
            count=count,
            lin_sum=lin_sum,
            log_sum=log_sum,
            log_sq_sum=log_sq_sum,
            min_lin=min_lin,
            min_log=min_log,
            histogram=histogram,
            bin_edges=HIST_EDGES,
            n_sites_per_gene=n_sites_per_gene,
            n_haplotypes=np.int64(n_pop),
            n_pairs_total=np.int64(n_pairs_total),
        )

        pop_dt = time.time() - pop_t0
        print(f"    {pop} done in {pop_dt:.1f}s -> {csv_path}", flush=True)

    dt = time.time() - t0
    print(f"=== chr{chr_num} complete in {dt:.1f}s ===", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Per-chromosome TMRCA inference")
    parser.add_argument("--chr", type=int, required=True, help="Chromosome number (1-22)")
    parser.add_argument(
        "--populations",
        nargs="*",
        default=None,
        help="Populations to run (default: all 26)",
    )
    args = parser.parse_args()

    pops = args.populations if args.populations else ALL_POPULATIONS
    run_chromosome(args.chr, pops)


if __name__ == "__main__":
    main()
