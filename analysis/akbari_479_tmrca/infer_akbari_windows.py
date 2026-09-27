#!/usr/bin/env python
"""Per-chromosome, per-population TMRCA at Akbari 2026 lead variants.

Instead of decoding the full chromosome, iterate over lead variants and pass
gamma_smc_cu.infer_blockwise() a G-slice [lead - SLICE_HALF_BP, lead + SLICE_HALF_BP]
per variant. The blockwise API's ``flank_sites`` padding handles HMM boundaries,
local boundary effects are reduced by additional context. Agreement with a
full-chromosome decoder must be checked empirically, not assumed.

Output per (chr, pop):
    results/chr{N}/{POP}.csv   -- one row per lead variant
    results/chr{N}/{POP}.npz   -- raw accumulators (per-lead geom/arith/min + histogram)

Usage:
    python infer_akbari_windows.py --chr 11 --populations GBR
    python infer_akbari_windows.py --chr 2 --populations GBR CEU GIH
"""

from __future__ import annotations

import argparse
import csv
import os
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "genome_wide"))
from rerun_support import add_arguments, check_arguments, fingerprint, calibrate, run_identity, completed_or_reserve, finish, load_input, validate_posterior_means, per_pair_moments, pair_distribution_counts
import numpy as np
import pandas as pd

CACHE_DIR = None
LEADS_TSV = None
RESULTS_DIR = None

MU = 1.25e-8
RHO = 1e-8
NE = 10_000

# HMM context around each lead variant. Boundary sensitivity must be checked
# against a full chromosome; a finite flank does not guarantee identical output.
SLICE_HALF_BP = 500_000     # +/- 500 kb sliced from G
AGG_HALF_BP   = 25_000      # +/- 25 kb aggregated into the reported TMRCA

# Per-variant pair-chunking; keeps the (n_sites, n_chunk_pairs) output in-RAM.
PAIR_CHUNK = 2000

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


def load_akbari_windows(chr_num):
    df = pd.read_csv(LEADS_TSV, sep="\t")
    df = df[df.CHROM.astype(str) == str(chr_num)].reset_index(drop=True)
    rows = []
    for _, r in df.iterrows():
        c = int(r.POS)
        rows.append({
            "rsid": r.RSID if pd.notna(r.RSID) else f"chr{chr_num}_{c}",
            "chrom": str(r.CHROM),
            "center_pos": c,
            "akbari_X": float(r.X),
            "akbari_S": float(r.S),
            "akbari_posterior": float(r.POSTERIOR),
        })
    return rows


def get_population_haplotype_indices(sample_ids, pop_map, population):
    indices = []
    for i, sid in enumerate(sample_ids):
        if sid in pop_map and pop_map[sid][0] == population:
            indices.extend([2 * i, 2 * i + 1])
    return sorted(indices)


def make_pairs(n_haps):
    return [(i, j) for i in range(n_haps) for j in range(i + 1, n_haps)]



def run_chromosome(chr_num, populations, args):
    import gamma_smc_cu

    print(f"=== Chromosome {chr_num} ===", flush=True)
    t_chr = time.time()

    npz_path = os.path.join(CACHE_DIR, "parsed", f"chr{chr_num}.npz")
    print(f"Loading {npz_path}...", flush=True)
    data, cache_identity = load_input(CACHE_DIR, chr_num)
    G_full = data["G"]
    positions = data["positions"].astype(np.int64)
    sample_ids = data["sample_ids"]
    print(f"  G: {G_full.shape}, positions: {positions.shape}, samples: {len(sample_ids)}",
          flush=True)

    pop_map = load_samples(args.samples)
    variants = load_akbari_windows(chr_num)
    n_v = len(variants)
    print(f"  {n_v} Akbari lead variants on chr{chr_num}", flush=True)
    if n_v == 0:
        print("  no variants, exiting early.")
        return

    out_dir = os.path.join(RESULTS_DIR, f"chr{chr_num}")
    os.makedirs(out_dir, exist_ok=True)

    inputs = {'cache': cache_identity, 'samples': fingerprint(args.samples), 'leads': fingerprint(LEADS_TSV)}
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

        G_pop = np.ascontiguousarray(G_full[np.array(hap_idx), :])
        all_pairs = make_pairs(n_pop)
        n_pairs_total = len(all_pairs)

        calibration = calibrate(G_pop, positions, args.mu, args.rho, args.sequence_length)
        decoder_metadata = None
        print(f"  {pop}: {n_pop} haplotypes, {n_pairs_total} pairs, "
              f"chromosome theta={calibration['theta']:.8g}, Ne={calibration['calibrated_Ne']:.8g}", flush=True)

        count       = np.zeros(n_v, dtype=np.int64)
        lin_sum     = np.zeros(n_v, dtype=np.float64)
        log_sum     = np.zeros(n_v, dtype=np.float64)
        log_sq_sum  = np.zeros(n_v, dtype=np.float64)
        min_lin     = np.full(n_v, np.inf, dtype=np.float64)
        min_log     = np.full(n_v, np.inf, dtype=np.float64)
        histogram   = np.zeros((n_v, HIST_NBINS), dtype=np.int64)
        histogram_underflow = np.zeros(n_v, dtype=np.int64)
        histogram_overflow = np.zeros(n_v, dtype=np.int64)
        n_geom_lt_1000 = np.zeros(n_v, dtype=np.int64)
        n_arith_lt_1000 = np.zeros(n_v, dtype=np.int64)
        n_sites_agg = np.zeros(n_v, dtype=np.int32)

        for vi, v in enumerate(variants):
            c = v["center_pos"]
            slice_lo = c - SLICE_HALF_BP
            slice_hi = c + SLICE_HALF_BP
            slice_mask = (positions >= slice_lo) & (positions <= slice_hi)
            site_idx = np.where(slice_mask)[0]
            n_slice = site_idx.size
            if n_slice < 200:
                print(f"    {v['rsid']} chr{chr_num}:{c}: only {n_slice} sites, skipping",
                      flush=True)
                continue

            G_slice = np.ascontiguousarray(G_pop[:, site_idx])
            pos_slice = positions[site_idx]

            # Which decoded positions fall in the +/- AGG_HALF_BP aggregation
            # window? Computed post-filter below (infer may drop monomorphic).
            n_chunks = (n_pairs_total + PAIR_CHUNK - 1) // PAIR_CHUNK
            retained_positions = None

            for ci in range(n_chunks):
                chunk_pairs = all_pairs[ci * PAIR_CHUNK:(ci + 1) * PAIR_CHUNK]
                n_chunk_pairs = len(chunk_pairs)

                result = gamma_smc_cu.infer_blockwise(
                    G_slice, pos_slice,
                    mu=args.mu, rho=args.rho, Ne=calibration["calibrated_Ne"],
                    physical_mu=args.mu,
                    core_block_sites=args.core_block_sites, flank_sites=args.flank_sites,
                    pairs=chunk_pairs,
                    mean_only=True,
                    auto_estimate_theta=False,
                )
                decoder_metadata = result.get("metadata", {})
                mean = result["mean"]
                out_positions = result["positions"]
                validate_posterior_means(mean, len(out_positions), n_chunk_pairs)
                np.testing.assert_array_equal(result["pairs"], chunk_pairs)
                if decoder_metadata.get("time_units") != "generations":
                    raise ValueError("Decoder must return physically calibrated generation units")
                if retained_positions is None:
                    retained_positions = np.asarray(out_positions).copy()
                else:
                    np.testing.assert_array_equal(out_positions, retained_positions)

                agg_mask = (out_positions >= c - AGG_HALF_BP) & \
                           (out_positions <= c + AGG_HALF_BP)
                n_agg_sites = int(agg_mask.sum())
                if n_agg_sites < 2:
                    del result, mean
                    continue
                if ci == 0:
                    n_sites_agg[vi] = n_agg_sites

                per_pair_lin, per_pair_log = per_pair_moments(mean, agg_mask)

                count[vi]      += n_chunk_pairs
                lin_sum[vi]    += per_pair_lin.sum(dtype=np.float64)
                log_sum[vi]    += per_pair_log.sum(dtype=np.float64)
                log_sq_sum[vi] += (per_pair_log * per_pair_log).sum(dtype=np.float64)

                chunk_min_lin = per_pair_lin.min()
                chunk_min_log = per_pair_log.min()
                if chunk_min_lin < min_lin[vi]:
                    min_lin[vi] = chunk_min_lin
                if chunk_min_log < min_log[vi]:
                    min_log[vi] = chunk_min_log

                distribution = pair_distribution_counts(per_pair_lin, per_pair_log, HIST_EDGES)
                histogram[vi] += distribution["histogram"]
                histogram_underflow[vi] += distribution["histogram_underflow"]
                histogram_overflow[vi] += distribution["histogram_overflow"]
                n_geom_lt_1000[vi] += distribution["n_geom_lt_1000"]
                n_arith_lt_1000[vi] += distribution["n_arith_lt_1000"]

                del result, mean

            if (vi + 1) % 5 == 0 or vi == n_v - 1:
                el = time.time() - pop_t0
                print(f"    {pop}: {vi+1}/{n_v} leads done ({el:.1f}s)", flush=True)

        np.testing.assert_array_equal(histogram.sum(axis=1) + histogram_underflow + histogram_overflow, count)
        with np.errstate(divide="ignore", invalid="ignore"):
            geom_mean = np.where(count > 0, np.exp(log_sum / count), np.nan)
            arith_mean = np.where(count > 0, lin_sum / count, np.nan)

        csv_path = os.path.join(out_dir, f"{pop}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["rsid", "chrom", "center_pos", "window_half_bp",
                 "akbari_X", "akbari_S", "akbari_posterior",
                 "geom_mean_tmrca", "arith_mean_tmrca", "min_tmrca",
                 "n_pairs", "n_sites", "frac_pairs_geom_lt_1000",
                 "frac_pairs_arith_lt_1000"]
            )
            for vi, v in enumerate(variants):
                gm = f"{geom_mean[vi]:.17g}" if np.isfinite(geom_mean[vi]) else ""
                am = f"{arith_mean[vi]:.17g}" if np.isfinite(arith_mean[vi]) else ""
                mn = f"{min_lin[vi]:.17g}" if np.isfinite(min_lin[vi]) else ""
                writer.writerow(
                    [v["rsid"], v["chrom"], v["center_pos"], AGG_HALF_BP,
                     v["akbari_X"], v["akbari_S"], v["akbari_posterior"],
                     gm, am, mn, int(count[vi]), int(n_sites_agg[vi]),
                     f"{n_geom_lt_1000[vi]/count[vi]:.17g}" if count[vi] else "",
                     f"{n_arith_lt_1000[vi]/count[vi]:.17g}" if count[vi] else ""]
                )

        np.savez_compressed(
            os.path.join(out_dir, f"{pop}.npz"),
            rsid=np.array([v["rsid"] for v in variants]),
            chrom=np.array([v["chrom"] for v in variants]),
            center_pos=np.array([v["center_pos"] for v in variants], dtype=np.int64),
            akbari_X=np.array([v["akbari_X"] for v in variants], dtype=np.float64),
            akbari_S=np.array([v["akbari_S"] for v in variants], dtype=np.float64),
            akbari_posterior=np.array([v["akbari_posterior"] for v in variants], dtype=np.float64),
            count=count, lin_sum=lin_sum, log_sum=log_sum, log_sq_sum=log_sq_sum,
            min_lin=min_lin, min_log=min_log, histogram=histogram, bin_edges=HIST_EDGES,
            histogram_underflow=histogram_underflow, histogram_overflow=histogram_overflow,
            n_geom_lt_1000=n_geom_lt_1000, n_arith_lt_1000=n_arith_lt_1000,
            threshold_generations=np.float64(1000),
            histogram_tail_policy=np.array("separate underflow/overflow; no clipping"),
            n_sites_per_window=n_sites_agg,
            n_haplotypes=np.int64(n_pop),
            n_pairs_total=np.int64(n_pairs_total),
            slice_half_bp=np.int64(SLICE_HALF_BP),
            agg_half_bp=np.int64(AGG_HALF_BP),
        )
        finish(out_dir, pop, identity, calibration, decoder_metadata)
        pop_dt = time.time() - pop_t0
        print(f"    {pop} done in {pop_dt:.1f}s ({pop_dt/n_v:.17g}s/variant) -> {csv_path}",
              flush=True)

    print(f"=== chr{chr_num} complete in {time.time() - t_chr:.1f}s ===", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chr", type=int, required=True)
    p.add_argument("--populations", nargs="*", default=None)
    add_arguments(p)
    p.add_argument('--leads', required=True, help='GRCh38 lead-variant TSV')
    p.add_argument('--slice-half-bp', type=int, default=500000)
    p.add_argument('--aggregate-half-bp', type=int, default=25000)
    args = p.parse_args()
    check_arguments(args)
    if args.aggregate_half_bp <= 0 or args.slice_half_bp <= args.aggregate_half_bp:
        raise ValueError('Slice context must be wider than the positive aggregation window')
    global CACHE_DIR, RESULTS_DIR, LEADS_TSV, PAIR_CHUNK, SLICE_HALF_BP, AGG_HALF_BP
    CACHE_DIR, RESULTS_DIR, LEADS_TSV = args.cache_dir, args.output_dir, args.leads
    PAIR_CHUNK, SLICE_HALF_BP, AGG_HALF_BP = args.pair_chunk, args.slice_half_bp, args.aggregate_half_bp
    pops = args.populations if args.populations else ALL_POPULATIONS
    run_chromosome(args.chr, pops, args)


if __name__ == "__main__":
    main()
