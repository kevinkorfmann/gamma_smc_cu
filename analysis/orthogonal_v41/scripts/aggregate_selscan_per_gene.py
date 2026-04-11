#!/usr/bin/env python
"""Aggregate selscan iHS and nSL output to gene-level scores per population.

For one population:
  1. Read iHS and nSL outputs from analysis/orthogonal_v41/selscan/chr*_<POP>/
  2. Frequency-bin-normalize raw iHS and nSL within the population (z-score
     within 20 DAF bins of width 0.05) — this is the standard Voight 2006
     normalization that selscan-norm performs.
  3. For each protein-coding gene on each chromosome, compute:
       n_sites              number of polymorphic sites in the gene
       max_abs_ihs_norm     max |iHS_norm| over sites in the gene
       frac_ihs_extreme     fraction of sites with |iHS_norm| > 2.0
       max_abs_nsl_norm     same for nSL
       frac_nsl_extreme     same for nSL
  4. Compute within-population rank percentiles for max_abs_ihs_norm and
     frac_ihs_extreme (lower rank = stronger signal -> higher max |iHS|).
  5. Write the per-population gene table to
       analysis/orthogonal_v41/selscan_genelevel/{POP}.csv

Gene-window definition: [gene_start - FLANK, gene_end + FLANK] with FLANK = 0.
We can rerun with FLANK = 50 kb if a flanking-region version is wanted later.

Usage:
    python aggregate_selscan_per_gene.py --pop YRI
    # or all 26 in one process:
    python aggregate_selscan_per_gene.py --all
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
GENES_DIR = os.path.join(REPO, "analysis/genome_wide/cache/genes")
SELSCAN_DIR = os.path.join(REPO, "analysis/orthogonal_v41/selscan")
OUT_DIR = os.path.join(REPO, "analysis/orthogonal_v41/selscan_genelevel")

CHRS = list(range(1, 23))

POPS = [
    "ACB", "ASW", "BEB", "CDX", "CEU", "CHB", "CHS", "CLM", "ESN", "FIN",
    "GBR", "GIH", "GWD", "IBS", "ITU", "JPT", "KHV", "LWK", "MSL", "MXL",
    "PEL", "PJL", "PUR", "STU", "TSI", "YRI",
]

DAF_BINS = np.linspace(0.0, 1.0, 21)  # 20 bins of width 0.05
EXTREME_THRESHOLD = 2.0  # |iHS_norm| or |nSL_norm| > 2 -> extreme
FLANK_BP = 0  # extend gene window by this many bp on each side


def load_selscan_table(path: str, stat: str) -> pd.DataFrame:
    """Load selscan .out table.

    Columns (no header in newer versions, header in some): chr id pos freq
    ihh1 ihh0 <stat>
    """
    if not os.path.exists(path):
        return None
    # Selscan v3 includes a header row; sniff it
    with open(path) as f:
        first = f.readline()
    has_header = first.startswith("chr") or first.startswith("id")
    df = pd.read_csv(
        path, sep="\t",
        header=0 if has_header else None,
        names=None if has_header else ["chr", "id", "pos", "freq", "ihh1", "ihh0", stat],
    )
    # Standardize column names
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if stat not in df.columns:
        # Some selscan builds put unstandardized names
        last = df.columns[-1]
        df = df.rename(columns={last: stat})
    df["abs_stat"] = df[stat].abs()
    return df[["pos", "freq", stat, "abs_stat"]]


def normalize_within_bins(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Z-normalize the raw stat within DAF bins.

    Replicates selscan-norm's behavior: bin by derived allele frequency (we use
    the freq column directly since selscan reports DAF), compute mean and std
    within each bin, then standardize.
    """
    df = df.copy()
    df["bin"] = pd.cut(df["freq"], bins=DAF_BINS, include_lowest=True, labels=False)
    norm = np.full(len(df), np.nan, dtype=np.float64)
    for b in range(len(DAF_BINS) - 1):
        mask = (df["bin"] == b) & np.isfinite(df[stat])
        if mask.sum() < 50:
            continue
        vals = df.loc[mask, stat].values
        mu = vals.mean()
        sd = vals.std(ddof=0)
        if sd == 0 or not np.isfinite(sd):
            continue
        norm[mask.values] = (vals - mu) / sd
    df[f"{stat}_norm"] = norm
    df[f"{stat}_norm_abs"] = np.abs(norm)
    return df


def aggregate_one_chr(chr_num: int, pop: str) -> pd.DataFrame:
    """Aggregate one chromosome's selscan output to per-gene scores."""
    task_dir = os.path.join(SELSCAN_DIR, f"chr{chr_num}_{pop}")
    ihs_path = os.path.join(task_dir, "ihs.ihs.out")
    nsl_path = os.path.join(task_dir, "nsl.nsl.out")

    if not (os.path.exists(ihs_path) and os.path.exists(nsl_path)):
        print(f"  chr{chr_num} {pop}: missing selscan output, skipping",
              flush=True)
        return None

    ihs = load_selscan_table(ihs_path, "ihs")
    nsl = load_selscan_table(nsl_path, "nsl")
    ihs = normalize_within_bins(ihs, "ihs")
    nsl = normalize_within_bins(nsl, "nsl")

    # Load genes for this chromosome
    genes_path = os.path.join(GENES_DIR, f"chr{chr_num}_genes.tsv")
    genes = pd.read_csv(genes_path, sep="\t")
    # Standardize column names defensively
    cols = {c.lower(): c for c in genes.columns}
    name_col = cols.get("gene_name", cols.get("gene", "gene_name"))
    start_col = cols.get("start", "start")
    end_col = cols.get("end", "end")

    rows = []
    ihs_pos = ihs["pos"].values
    nsl_pos = nsl["pos"].values
    for _, gene in genes.iterrows():
        gname = gene[name_col]
        gstart = int(gene[start_col]) - FLANK_BP
        gend = int(gene[end_col]) + FLANK_BP

        ihs_mask = (ihs_pos >= gstart) & (ihs_pos <= gend)
        nsl_mask = (nsl_pos >= gstart) & (nsl_pos <= gend)
        n_ihs = int(ihs_mask.sum())
        n_nsl = int(nsl_mask.sum())

        if n_ihs == 0 and n_nsl == 0:
            continue

        ihs_sub = ihs.loc[ihs_mask, "ihs_norm_abs"].dropna()
        nsl_sub = nsl.loc[nsl_mask, "nsl_norm_abs"].dropna()

        max_ihs = float(ihs_sub.max()) if len(ihs_sub) else np.nan
        max_nsl = float(nsl_sub.max()) if len(nsl_sub) else np.nan
        frac_ihs = float((ihs_sub > EXTREME_THRESHOLD).mean()) if len(ihs_sub) else np.nan
        frac_nsl = float((nsl_sub > EXTREME_THRESHOLD).mean()) if len(nsl_sub) else np.nan

        rows.append({
            "chr": chr_num,
            "gene_name": gname,
            "gstart": gene[start_col],
            "gend": gene[end_col],
            "n_ihs_sites": n_ihs,
            "n_nsl_sites": n_nsl,
            "max_abs_ihs_norm": max_ihs,
            "frac_ihs_extreme": frac_ihs,
            "max_abs_nsl_norm": max_nsl,
            "frac_nsl_extreme": frac_nsl,
        })

    return pd.DataFrame(rows)


def aggregate_one_pop(pop: str) -> pd.DataFrame:
    """Aggregate all 22 chromosomes for one population."""
    print(f"=== {pop} ===", flush=True)
    parts = []
    for chr_num in CHRS:
        t0 = time.time()
        df = aggregate_one_chr(chr_num, pop)
        if df is not None and len(df):
            parts.append(df)
            print(f"  chr{chr_num}: {len(df)} genes ({time.time()-t0:.1f}s)",
                  flush=True)
    if not parts:
        return None
    full = pd.concat(parts, ignore_index=True)

    # Compute within-population ranks (lower rank = stronger signal because
    # we want HIGH max |iHS| at the top -> rank from largest to smallest)
    n_genes = len(full)
    for col in ["max_abs_ihs_norm", "frac_ihs_extreme",
                "max_abs_nsl_norm", "frac_nsl_extreme"]:
        # Rank descending so the largest value gets rank 1
        ranks = full[col].rank(method="average", ascending=False, na_option="bottom")
        full[f"{col}_rank"] = ranks / n_genes

    return full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop", help="Single population to aggregate")
    parser.add_argument("--all", action="store_true",
                        help="Run for all 26 populations")
    args = parser.parse_args()

    if not args.pop and not args.all:
        parser.error("Pass --pop POP or --all")

    os.makedirs(OUT_DIR, exist_ok=True)
    pops = POPS if args.all else [args.pop]

    for pop in pops:
        full = aggregate_one_pop(pop)
        if full is None:
            print(f"  {pop}: no data, skipping", flush=True)
            continue
        out_path = os.path.join(OUT_DIR, f"{pop}.csv")
        full.to_csv(out_path, index=False)
        print(f"  wrote {out_path} ({len(full)} genes)", flush=True)


if __name__ == "__main__":
    main()
