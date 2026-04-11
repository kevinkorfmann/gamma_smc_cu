#!/usr/bin/env python
"""Re-extract variant-level evidence at the v4.1 candidate genes.

For each candidate (gene, chr, focal_pop):
  1. Load chr NPZ.
  2. Subset to a +/-500 kb window around the gene midpoint.
  3. For each polymorphic variant in the window, compute:
       focal_af              allele frequency in the focal population
       super_af[s]           allele frequency in each superpopulation (5 of them)
       fst_focal_vs_others   Hudson FST between focal pop and the union of all
                             non-focal-superpop haplotypes
       direction             "depleted" if focal_af < global_af, "enriched"
                             if focal_af > global_af (relative to non-focal)
  4. Summarize per gene:
       n_variants, n_depleted, n_enriched,
       depleted_to_enriched_ratio,
       max_fst, mean_top10_fst,
       most_diff_variant_pos, most_diff_variant_focal_af,
       most_diff_variant_other_af
  5. Write to analysis/orthogonal_v41/variant_evidence/{gene}_{pop}.json

Run as a per-gene SLURM array (15 tasks).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
DATA_DIR = os.path.join(REPO, "analysis/genome_wide")
PARSED_DIR = os.path.join(DATA_DIR, "cache/parsed")
GENES_DIR = os.path.join(DATA_DIR, "cache/genes")
SAMPLES_PATH = os.path.join(DATA_DIR, "data/samples.txt")
OUT_DIR = os.path.join(REPO, "analysis/orthogonal_v41/variant_evidence")

WINDOW_BP = 500_000

# 1KG superpopulation lookup (sample_id -> super_pop)
SUPERPOPS = ["AFR", "AMR", "EAS", "EUR", "SAS"]


def load_samples():
    pops = {}
    with open(SAMPLES_PATH) as f:
        next(f)
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                pops[parts[1]] = (parts[5], parts[6])  # (pop, super_pop)
    return pops


def load_chr_npz(chr_num):
    path = os.path.join(PARSED_DIR, f"chr{chr_num}.npz")
    d = np.load(path, allow_pickle=True, mmap_mode="r")
    return d["G"], d["positions"], d["sample_ids"]


def get_indices_by_pop(sample_ids, pop_map, key, kind):
    """key='GIH' kind='pop' or key='SAS' kind='super'."""
    indices = []
    for i, sid in enumerate(sample_ids):
        if sid in pop_map:
            pop, super_pop = pop_map[sid]
            if (kind == "pop" and pop == key) or (kind == "super" and super_pop == key):
                indices.extend([2 * i, 2 * i + 1])
    return sorted(indices)


def hudson_fst(p1: np.ndarray, p2: np.ndarray, n1: int, n2: int) -> np.ndarray:
    """Per-site Hudson FST between two populations.

    Numerator:   (p1-p2)^2 - p1*(1-p1)/(n1-1) - p2*(1-p2)/(n2-1)
    Denominator: p1*(1-p2) + p2*(1-p1)
    """
    num = (p1 - p2) ** 2
    if n1 > 1:
        num = num - p1 * (1 - p1) / (n1 - 1)
    if n2 > 1:
        num = num - p2 * (1 - p2) / (n2 - 1)
    den = p1 * (1 - p2) + p2 * (1 - p1)
    fst = np.where(den > 0, num / den, np.nan)
    return fst


def gene_midpoint(chr_num, gene_name):
    df = pd.read_csv(os.path.join(GENES_DIR, f"chr{chr_num}_genes.tsv"), sep="\t")
    row = df[df["gene_name"] == gene_name]
    if row.empty:
        raise SystemExit(f"gene {gene_name} not found")
    r = row.iloc[0]
    return int((r["start"] + r["end"]) / 2), int(r["start"]), int(r["end"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", required=True)
    parser.add_argument("--chr", type=int, required=True)
    parser.add_argument("--pop", required=True, help="focal population (e.g. GIH)")
    parser.add_argument("--super", required=True,
                        help="focal superpopulation of the focal pop (e.g. SAS)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"=== variant evidence: {args.gene} chr{args.chr} {args.pop}/{getattr(args, 'super')} ===",
          flush=True)

    G, positions, sample_ids = load_chr_npz(args.chr)
    pop_map = load_samples()

    midpoint, gstart, gend = gene_midpoint(args.chr, args.gene)
    win_lo = midpoint - WINDOW_BP
    win_hi = midpoint + WINDOW_BP
    site_mask = (positions >= win_lo) & (positions <= win_hi)
    pos_win = np.asarray(positions)[site_mask]
    print(f"  window chr{args.chr}:{win_lo}-{win_hi}, sites: {site_mask.sum()}",
          flush=True)

    # Slice G to window once (this is what materializes the mmap)
    G_win = np.ascontiguousarray(G[:, site_mask])

    # Build index lists
    focal_pop_idx = get_indices_by_pop(sample_ids, pop_map, args.pop, "pop")
    focal_super = getattr(args, "super")
    focal_super_idx = get_indices_by_pop(sample_ids, pop_map, focal_super, "super")
    other_super_idx = sorted(set(range(2 * len(sample_ids))) - set(focal_super_idx))
    print(f"  focal pop {args.pop}: {len(focal_pop_idx)} haps; "
          f"focal super {focal_super}: {len(focal_super_idx)} haps; "
          f"other supers: {len(other_super_idx)} haps", flush=True)

    G_focal = G_win[np.array(focal_pop_idx), :]
    G_other = G_win[np.array(other_super_idx), :]

    n_focal = G_focal.shape[0]
    n_other = G_other.shape[0]
    p_focal = G_focal.sum(axis=0) / n_focal
    p_other = G_other.sum(axis=0) / n_other

    # Drop monomorphic in BOTH groups
    keep = ~((p_focal == 0) & (p_other == 0)) & ~((p_focal == 1) & (p_other == 1))
    p_focal = p_focal[keep]
    p_other = p_other[keep]
    pos_win = pos_win[keep]

    fst = hudson_fst(p_focal, p_other, n_focal, n_other)
    diff = p_focal - p_other  # >0 enriched in focal, <0 depleted

    # Summary
    n_variants = int(len(pos_win))
    n_enriched = int((diff > 0).sum())
    n_depleted = int((diff < 0).sum())
    ratio = (n_depleted / n_enriched) if n_enriched > 0 else float("inf")

    fst_finite = fst[np.isfinite(fst)]
    max_fst = float(fst_finite.max()) if len(fst_finite) else float("nan")
    top10_fst = float(np.sort(fst_finite)[-10:].mean()) if len(fst_finite) >= 10 else float("nan")
    most_diff_idx = int(np.nanargmax(np.abs(diff)))

    summary = {
        "gene": args.gene,
        "chr": args.chr,
        "pop": args.pop,
        "super": focal_super,
        "midpoint": midpoint,
        "gstart": gstart,
        "gend": gend,
        "win_lo": int(win_lo),
        "win_hi": int(win_hi),
        "n_variants": n_variants,
        "n_enriched": n_enriched,
        "n_depleted": n_depleted,
        "depleted_to_enriched_ratio": float(ratio),
        "max_fst": max_fst,
        "mean_top10_fst": top10_fst,
        "most_diff_variant_pos": int(pos_win[most_diff_idx]),
        "most_diff_variant_focal_af": float(p_focal[most_diff_idx]),
        "most_diff_variant_other_af": float(p_other[most_diff_idx]),
        "most_diff_variant_diff": float(diff[most_diff_idx]),
    }
    out_path = os.path.join(OUT_DIR, f"{args.gene}_{args.pop}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {out_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
