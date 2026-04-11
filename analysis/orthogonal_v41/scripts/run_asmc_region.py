#!/usr/bin/env python
"""Run ASMC at one (gene, population) window and append to the existing
three_method NPZ alongside the cxt + tmrca.cu results.

ASMC needs three input files in a per-task directory:
  <root>.hap.gz  : haplotype matrix in PLINK .hap format
                   columns: chr snp_id pos_cm pos_bp ref alt h0 h1 ... hN
  <root>.samples : two header lines + sample lines (FID IID 0)
  <root>.map     : PLINK genetic map (chr snp_id pos_cm pos_bp)

ASMC then loads these via DecodingParams + Data and exposes the per-pair
posterior mean TMRCA via decode_pairs + get_copy_of_results.
"""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np

REPO = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
sys.path.insert(0, os.path.join(REPO, "python"))

DATA_DIR = os.path.join(REPO, "analysis/genome_wide")
PARSED_DIR = os.path.join(DATA_DIR, "cache/parsed")
SAMPLES_PATH = os.path.join(DATA_DIR, "data/samples.txt")
THREE_METHOD_DIR = os.path.join(REPO, "analysis/orthogonal_v41/three_method")
ASMC_DATA = "/vast/projects/smathi/cohort/kkor/asmc_data"
DQ_FILE = os.path.join(ASMC_DATA, "CEU_50.decodingQuantities.gz")

WINDOW_BP = 500_000
PAIR_CAP = 20
N_SAMPLES_ASMC = 50  # match cxt's num_samples and the dq file's sample count


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
    # Use mmap_mode so we don't pull the full ~25 GB G matrix into RAM
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


def write_asmc_input(out_root: str, chr_num: int, positions, G_sub, n_samples):
    """Write <out_root>.hap.gz, <out_root>.samples, <out_root>.map.gz for ASMC.

    G_sub: (n_haps, n_sites) uint8 with haplotypes 0/1.
    positions: 1D float64 in bp.
    n_samples: number of DIPLOID samples = n_haps / 2.

    Format reference (from PalamaraLab/ASMC_data examples/asmc/exampleFile.n100.*):
        .hap.gz   chr_id snp_id pos_bp ref alt h0 h1 ... h(2N-1)
        .samples  three header lines (ID_1 ID_2 missing / 0 0 0 / per-sample lines)
        .map.gz   chr_int snp_id pos_cm pos_bp  (TAB separated)
    """
    n_haps, n_sites = G_sub.shape
    assert n_haps == 2 * n_samples, f"{n_haps} != 2 * {n_samples}"

    # 1 cM/Mb (approximation; cM only affects transition rates and is fine for
    # the per-gene-window comparison figure)
    cm_positions = positions * 1e-6

    # .hap.gz
    hap_path = out_root + ".hap.gz"
    with gzip.open(hap_path, "wt") as f:
        for i in range(n_sites):
            pos_bp = int(positions[i])
            chrom_id = f"{chr_num}:{pos_bp}_1_2"
            snp_id = f"SNP_{pos_bp}_{i}"
            a1, a2 = "1", "2"   # alleles encoded as 1/2 like the example
            haps = " ".join(str(int(x)) for x in G_sub[:, i])
            f.write(f"{chrom_id} {snp_id} {pos_bp} {a1} {a2} {haps}\n")

    # .samples
    samples_path = out_root + ".samples"
    with open(samples_path, "w") as f:
        f.write("ID_1 ID_2 missing\n")
        f.write("0 0 0\n")
        for i in range(n_samples):
            f.write(f"{i+1}_{i+1} {i+1}_{i+1} 0\n")

    # .map.gz
    map_path = out_root + ".map.gz"
    with gzip.open(map_path, "wt") as f:
        for i in range(n_sites):
            pos_bp = int(positions[i])
            snp_id = f"SNP_{pos_bp}_{i}"
            f.write(f"{chr_num}\t{snp_id}\t{cm_positions[i]:.10f}\t{pos_bp}\n")

    return hap_path, samples_path, map_path


def run_asmc_for_pair(out_root: str, hap_a: int, hap_b: int):
    """Run ASMC on the prebuilt input directory and decode one pair."""
    from asmc.asmc import ASMC
    asmc = ASMC(out_root, DQ_FILE, decoding_mode="sequence")
    asmc.set_store_per_pair_posterior_mean(True)
    asmc.decode_pairs([hap_a], [hap_b])
    res = asmc.get_copy_of_results()
    # res.per_pair_posterior_means is a list/array of per-site posterior means;
    # for one pair it's [array_of_length_n_sites]
    pp = np.asarray(res.per_pair_posterior_means)
    if pp.ndim == 2:
        return pp[0]
    return pp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", required=True)
    parser.add_argument("--chr", type=int, required=True)
    parser.add_argument("--pop", required=True)
    parser.add_argument("--group", required=True,
                        choices=["novel","positive","neutral","control"])
    args = parser.parse_args()

    print(f"=== ASMC: {args.gene} chr{args.chr} {args.pop} ({args.group}) ===", flush=True)

    if not os.path.exists(DQ_FILE):
        print(f"ERROR: missing decoding quantities file: {DQ_FILE}", flush=True)
        sys.exit(1)

    npz_path = os.path.join(THREE_METHOD_DIR, f"{args.gene}_{args.pop}_{args.group}.npz")
    if not os.path.exists(npz_path):
        print(f"ERROR: missing three_method NPZ: {npz_path}", flush=True)
        sys.exit(1)

    G, positions, sample_ids = load_chr_npz(args.chr)
    pop_map = load_samples()

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

    G_pop_win = np.ascontiguousarray(G[np.array(hap_idx), :][:, site_mask])
    n_pop_haps = G_pop_win.shape[0]

    # Pivot pairs (same RNG seed as run_three_method.py to use the same set)
    all_pairs = [(i, j) for i in range(n_pop_haps) for j in range(i + 1, n_pop_haps)]
    rng = np.random.default_rng(42)
    if len(all_pairs) > PAIR_CAP:
        idx = rng.choice(len(all_pairs), PAIR_CAP, replace=False)
        idx.sort()
        pairs = [all_pairs[i] for i in idx]
    else:
        pairs = all_pairs
    print(f"Decoding {len(pairs)} pairs via ASMC", flush=True)

    # We follow the same subsampling strategy as cxt: 50 haps per call,
    # with the focal pair at indices 0 and 1.
    rng2 = np.random.default_rng(123)
    asmc_per_pair = []
    asmc_positions = pos_win.copy()
    asmc_errors = []

    t0 = time.time()
    for (a, b) in pairs:
        others = [i for i in range(n_pop_haps) if i not in (a, b)]
        if len(others) < N_SAMPLES_ASMC * 2 - 2:
            picked_haps = others
        else:
            picked_haps = rng2.choice(others, size=N_SAMPLES_ASMC * 2 - 2,
                                       replace=False).tolist()
        sub_idx = [a, b] + sorted(picked_haps)
        G_sub = G_pop_win[np.array(sub_idx), :]

        with tempfile.TemporaryDirectory(prefix="asmc_") as td:
            out_root = os.path.join(td, "data")
            try:
                write_asmc_input(out_root, args.chr, pos_win, G_sub,
                                 n_samples=len(sub_idx) // 2)
                # ASMC haplotype indices: pair (0, 1) since we placed the focal
                # haps at positions 0 and 1.
                pp = run_asmc_for_pair(out_root, 0, 1)
                asmc_per_pair.append(pp)
            except Exception as exc:
                asmc_errors.append(f"pair {a},{b}: {type(exc).__name__}: {exc}")
                # Append NaN-filled array of expected shape
                asmc_per_pair.append(np.full(len(pos_win), np.nan))

    elapsed = time.time() - t0
    print(f"ASMC done in {elapsed:.1f}s", flush=True)
    if asmc_errors:
        print(f"  {len(asmc_errors)} pair errors:", flush=True)
        for e in asmc_errors[:3]:
            print(f"    {e}", flush=True)

    # Stack to (n_pairs, n_sites)
    if asmc_per_pair:
        # The per-pair arrays may have different lengths if ASMC trimmed; pad to common
        max_len = max(p.shape[0] for p in asmc_per_pair)
        stacked = np.full((len(asmc_per_pair), max_len), np.nan, dtype=np.float32)
        for j, p in enumerate(asmc_per_pair):
            stacked[j, :len(p)] = p
    else:
        stacked = np.zeros((0, 0), dtype=np.float32)

    # Append to existing NPZ
    existing = dict(np.load(npz_path, allow_pickle=True))
    existing["asmc_mean"] = stacked
    existing["asmc_positions"] = asmc_positions
    existing["asmc_elapsed"] = float(elapsed)
    if asmc_errors:
        existing["asmc_partial_errors"] = "\n".join(asmc_errors)

    np.savez_compressed(npz_path, **existing)
    print(f"Updated {npz_path}", flush=True)
    print(f"  asmc_mean shape: {stacked.shape}", flush=True)


if __name__ == "__main__":
    main()
