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
    min_log      : minimum across pairs of mean log TMRCA across sites
    histogram    : (n_genes, n_bins) counts of per-pair log-TMRCA
    bin_edges    : (n_bins+1,) natural-log edges spanning ln(10)..ln(1e6)

Histograms give approximate quantiles within finite bins, with underflow and
overflow retained separately. Exact counts below 1000 generations are saved for
per-pair geometric and arithmetic means across sites; arbitrary exact quantiles
or thresholds cannot be recovered from the binned summaries.

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
from pathlib import Path
from lead_summaries import load_leads, lead_arrays
from rerun_support import add_arguments, check_arguments, fingerprint, calibrate, run_identity, completed_or_reserve, finish, load_input, validate_posterior_means, per_pair_moments, pair_distribution_counts

CACHE_DIR = None
RESULTS_DIR = None

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

    Sorted coordinates permit two binary searches per interval, avoiding a
    chromosome-sized boolean scan for every gene and lead window.
    """
    result = []
    for _, _, gstart, gend in genes:
        if gstart > gend:
            raise ValueError('Interval start is greater than end')
        left = np.searchsorted(positions, gstart, side='left')
        right = np.searchsorted(positions, gend, side='right')
        result.append(np.arange(left, right, dtype=np.int64))
    return result


def run_chromosome(chr_num, populations, args):
    import gamma_smc_cu

    print(f"=== Chromosome {chr_num} ===", flush=True)
    t0 = time.time()

    npz_path = os.path.join(CACHE_DIR, "parsed", f"chr{chr_num}.npz")
    print(f"Loading {npz_path}...", flush=True)
    data, cache_identity = load_input(CACHE_DIR, chr_num)
    G = data["G"]
    positions = data["positions"]
    sample_ids = data["sample_ids"]
    print(f"  G: {G.shape}, positions: {positions.shape}, samples: {len(sample_ids)}",
          flush=True)

    pop_map = load_samples(args.samples)
    genes = load_genes(chr_num)
    n_genes = len(genes)
    lead_file = getattr(args, "lead_variants", None)
    lead_half_bp = getattr(args, "lead_half_bp", 25000)
    leads = load_leads(lead_file, chr_num)
    regions = genes + [(row["rsid"], row["rsid"], max(1,row["center_pos"]-lead_half_bp),
                       row["center_pos"]+lead_half_bp) for row in leads]
    n_regions = len(regions)
    print(f"  {len(leads)} lead windows share the full-chromosome posterior", flush=True)
    print(f"  {n_genes} genes", flush=True)

    out_dir = os.path.join(RESULTS_DIR, f"chr{chr_num}")
    os.makedirs(out_dir, exist_ok=True)

    inputs = {'cache': cache_identity, 'samples': fingerprint(args.samples),
              'genes': fingerprint(Path(CACHE_DIR)/'genes'/f'chr{chr_num}_genes.tsv')}
    if lead_file:
        inputs["lead_variants"] = fingerprint(lead_file)
    for pop in populations:
        identity = run_identity(args, inputs, {'chromosome': chr_num, 'population': pop})
        if completed_or_reserve(out_dir, pop, identity, args.resume):
            print(f"  {pop}: matching complete outputs, skipped", flush=True)
            continue
        pop_t0 = time.time()
        hap_idx = get_population_haplotype_indices(sample_ids, pop_map, pop)
        n_pop = len(hap_idx)
        if n_pop < 4:
            print(f"  {pop}: skipped (only {n_pop} haplotypes)", flush=True)
            continue

        G_pop = np.ascontiguousarray(G[np.array(hap_idx), :])
        calibration = calibrate(G_pop, positions, args.mu, args.rho, args.sequence_length)
        decoder_metadata = None

        all_pairs = make_pairs(n_pop)
        n_pairs_total = len(all_pairs)
        n_chunks = (n_pairs_total + PAIR_CHUNK - 1) // PAIR_CHUNK
        print(f"  {pop}: {n_pop} haplotypes, {n_pairs_total} pairs, "
              f"{n_chunks} chunks of {PAIR_CHUNK}", flush=True)

        # Per-gene accumulators
        count       = np.zeros(n_regions, dtype=np.int64)
        lin_sum     = np.zeros(n_regions, dtype=np.float64)
        log_sum     = np.zeros(n_regions, dtype=np.float64)
        log_sq_sum  = np.zeros(n_regions, dtype=np.float64)
        min_lin     = np.full(n_regions, np.inf, dtype=np.float64)
        min_log     = np.full(n_regions, np.inf, dtype=np.float64)
        histogram   = np.zeros((n_regions, HIST_NBINS), dtype=np.int64)
        histogram_underflow = np.zeros(n_regions, dtype=np.int64)
        histogram_overflow = np.zeros(n_regions, dtype=np.int64)
        n_geom_lt_1000 = np.zeros(n_regions, dtype=np.int64)
        n_arith_lt_1000 = np.zeros(n_regions, dtype=np.int64)
        n_sites_per_gene = np.zeros(n_regions, dtype=np.int32)

        # Will be filled on first chunk
        gene_site_idx = None
        retained_positions = None

        region_context = None
        region_rows = {}
        if getattr(args, 'gpu_region_moments', False):
            if args.flank_sites != 0 or args.core_block_sites < G_pop.shape[1]:
                raise ValueError('--gpu-region-moments requires full-chromosome core sites and zero flanks')
            region_context = gamma_smc_cu.RegionMomentContext(
                G_pop, positions, mu=args.mu, rho=args.rho,
                Ne=calibration['calibrated_Ne'], physical_mu=args.mu,
                auto_estimate_theta=False)
            retained_positions = region_context.positions.copy()
            gene_site_idx = compute_gene_site_indices(retained_positions, regions)
            region_bounds = []
            for gi, idxs in enumerate(gene_site_idx):
                n_sites_per_gene[gi] = len(idxs)
                if len(idxs) >= 2:
                    region_rows[gi] = len(region_bounds)
                    region_bounds.append((int(idxs[0]), int(idxs[-1])+1))

        for ci in range(n_chunks):
            chunk_start = ci * PAIR_CHUNK
            chunk_end = min(chunk_start + PAIR_CHUNK, n_pairs_total)
            chunk_pairs = all_pairs[chunk_start:chunk_end]
            n_chunk_pairs = len(chunk_pairs)

            if region_context is not None:
                result = region_context.run(chunk_pairs, region_bounds,
                    tile_sites=args.checkpoint_sites)
                mean = None
            else:
                result = gamma_smc_cu.infer_blockwise(
                    G_pop,
                    positions,
                    mu=args.mu,
                    rho=args.rho,
                    Ne=calibration["calibrated_Ne"],
                    physical_mu=args.mu,
                    core_block_sites=args.core_block_sites,
                    flank_sites=args.flank_sites,
                    pairs=chunk_pairs,
                    mean_only=True,
                    auto_estimate_theta=False,
                )

                mean = result['mean']
            decoder_metadata = result.get('metadata', {})
            out_positions = result["positions"]

            if mean is not None:
                validate_posterior_means(mean, len(out_positions), n_chunk_pairs)
            else:
                values = result['region_mean']
                logs = result['region_mean_log']
                if (values.shape != (len(region_bounds), n_chunk_pairs)
                        or logs.shape != values.shape or not np.isfinite(values).all()
                        or np.any(values <= 0) or not np.isfinite(logs).all()):
                    raise ValueError('Invalid GPU region moments')
            np.testing.assert_array_equal(result["pairs"], chunk_pairs)
            if decoder_metadata.get("time_units") != "generations":
                raise ValueError("Decoder must return physically calibrated generation units")

            if gene_site_idx is None:
                retained_positions = np.asarray(out_positions).copy()
                gene_site_idx = compute_gene_site_indices(out_positions, regions)
                for gi, idxs in enumerate(gene_site_idx):
                    n_sites_per_gene[gi] = len(idxs)
            else:
                np.testing.assert_array_equal(out_positions, retained_positions)

            for gi, idxs in enumerate(gene_site_idx):
                n_gene_sites = idxs.size
                if n_gene_sites < 2:
                    continue

                # Per-pair values for this gene (one number per pair):
                #   linear: arithmetic mean TMRCA across sites
                #   log:    arithmetic mean of log(TMRCA) across sites
                if region_context is None:
                    per_pair_lin, per_pair_log = per_pair_moments(mean, idxs)
                else:
                    row = region_rows[gi]
                    per_pair_lin = result['region_mean'][row]
                    per_pair_log = result['region_mean_log'][row]

                count[gi]      += n_chunk_pairs
                lin_sum[gi]    += per_pair_lin.sum(dtype=np.float64)
                log_sum[gi]    += per_pair_log.sum(dtype=np.float64)
                log_sq_sum[gi] += (per_pair_log * per_pair_log).sum(dtype=np.float64)

                chunk_min_lin = per_pair_lin.min()
                chunk_min_log = per_pair_log.min()
                if chunk_min_lin < min_lin[gi]:
                    min_lin[gi] = chunk_min_lin
                if chunk_min_log < min_log[gi]:
                    min_log[gi] = chunk_min_log

                distribution = pair_distribution_counts(per_pair_lin, per_pair_log, HIST_EDGES)
                histogram[gi] += distribution["histogram"]
                histogram_underflow[gi] += distribution["histogram_underflow"]
                histogram_overflow[gi] += distribution["histogram_overflow"]
                n_geom_lt_1000[gi] += distribution["n_geom_lt_1000"]
                n_arith_lt_1000[gi] += distribution["n_arith_lt_1000"]

            del result, mean

            if (ci + 1) % 5 == 0 or ci == n_chunks - 1:
                elapsed = time.time() - pop_t0
                print(f"    chunk {ci+1}/{n_chunks} done ({elapsed:.1f}s)", flush=True)

        np.testing.assert_array_equal(histogram.sum(axis=1) + histogram_underflow + histogram_overflow, count)
        # Genes with fewer than two retained markers have no contributions.
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
                 "min_tmrca", "n_pairs", "n_sites",
                 "frac_pairs_geom_lt_1000", "frac_pairs_arith_lt_1000"]
            )
            for gi, (gene_id, gene_name, gstart, gend) in enumerate(genes):
                gm = f"{geom_mean[gi]:.17g}" if np.isfinite(geom_mean[gi]) else ""
                am = f"{arith_mean[gi]:.17g}" if np.isfinite(arith_mean[gi]) else ""
                mn = f"{min_lin[gi]:.17g}" if np.isfinite(min_lin[gi]) else ""
                writer.writerow(
                    [gene_id, gene_name, gstart, gend, gm, am, mn,
                     int(count[gi]), int(n_sites_per_gene[gi]),
                     f"{n_geom_lt_1000[gi]/count[gi]:.17g}" if count[gi] else "",
                     f"{n_arith_lt_1000[gi]/count[gi]:.17g}" if count[gi] else ""]
                )

        # Write NPZ with all raw accumulators
        npz_out = os.path.join(out_dir, f"{pop}.npz")
        gene_ids = np.array([g[0] for g in genes])
        gene_names = np.array([g[1] for g in genes])
        gene_starts = np.array([g[2] for g in genes], dtype=np.int64)
        gene_ends = np.array([g[3] for g in genes], dtype=np.int64)
        retained_leads = lead_arrays(leads, lead_half_bp, {
            "count":count, "lin_sum":lin_sum, "log_sum":log_sum, "log_sq_sum":log_sq_sum,
            "min_lin":min_lin, "min_log":min_log, "histogram":histogram,
            "histogram_underflow":histogram_underflow, "histogram_overflow":histogram_overflow,
            "n_geom_lt_1000":n_geom_lt_1000, "n_arith_lt_1000":n_arith_lt_1000,
            "n_sites":n_sites_per_gene}, n_genes, bool(lead_file))
        np.savez_compressed(
            npz_out,
            gene_id=gene_ids,
            gene_name=gene_names,
            start=gene_starts,
            end=gene_ends,
            count=count[:n_genes],
            lin_sum=lin_sum[:n_genes],
            log_sum=log_sum[:n_genes],
            log_sq_sum=log_sq_sum[:n_genes],
            min_lin=min_lin[:n_genes],
            min_log=min_log[:n_genes],
            histogram=histogram[:n_genes], histogram_underflow=histogram_underflow[:n_genes],
            histogram_overflow=histogram_overflow[:n_genes],
            n_geom_lt_1000=n_geom_lt_1000[:n_genes], n_arith_lt_1000=n_arith_lt_1000[:n_genes],
            threshold_generations=np.float64(1000),
            histogram_tail_policy=np.array("separate underflow/overflow; no clipping"),
            bin_edges=HIST_EDGES,
            n_sites_per_gene=n_sites_per_gene[:n_genes],
            n_haplotypes=np.int64(n_pop),
            n_pairs_total=np.int64(n_pairs_total),
            **retained_leads,
        )

        finish(out_dir, pop, identity, calibration, decoder_metadata)
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
    add_arguments(parser)
    parser.add_argument("--lead-variants", help="GRCh38 one-based lead TSV; summarize windows from the same full-chromosome posterior")
    parser.add_argument("--lead-half-bp", type=int, default=25000)
    parser.add_argument('--gpu-region-moments', action='store_true',
                        help='Compute full-context per-pair gene/window moments on GPU; avoid dense posterior transfers')
    parser.add_argument('--checkpoint-sites', type=int, default=4096)
    args = parser.parse_args()
    if args.checkpoint_sites < 1:
        raise ValueError('Checkpoint site count must be positive')
    check_arguments(args)
    if args.lead_half_bp <= 0:
        raise ValueError("Lead window half width must be positive")
    global CACHE_DIR, RESULTS_DIR, PAIR_CHUNK
    CACHE_DIR, RESULTS_DIR, PAIR_CHUNK = args.cache_dir, args.output_dir, args.pair_chunk

    pops = args.populations if args.populations else ALL_POPULATIONS
    run_chromosome(args.chr, pops, args)


if __name__ == "__main__":
    main()
