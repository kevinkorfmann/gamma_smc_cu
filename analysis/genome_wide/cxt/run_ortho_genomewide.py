#!/usr/bin/env python
"""
Compute orthogonal selection statistics (Tajima's D, pi ratio, max |iHS|, max FST)
in ±500kb windows around ALL protein-coding genes genome-wide for EAS populations.
Then rank each gene's stats to produce percentiles.

Output: one CSV with 18,884 rows × stats + percentiles.
"""

import numpy as np
import allel
import pandas as pd
import os
import sys
import time

BETTY_BASE = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide"
PARSED_DIR = os.path.join(BETTY_BASE, "cache/parsed")
GENE_DIR = os.path.join(BETTY_BASE, "cache/genes")
SAMPLES_FILE = os.path.join(BETTY_BASE, "data/samples.txt")
OUTDIR = os.path.join(BETTY_BASE, "cxt/results/orthogonal")
os.makedirs(OUTDIR, exist_ok=True)

EAS_POPS = ["CHB", "JPT", "CHS", "CDX", "KHV"]
WINDOW = 500_000


def load_pop_indices(samples_file):
    pops = {}
    with open(samples_file) as f:
        header = f.readline().strip().split()
        pop_col = next(j for j, c in enumerate(header) if c.lower() in ("pop", "population"))
        for i, line in enumerate(f):
            fields = line.strip().split()
            pop = fields[pop_col]
            if pop not in pops:
                pops[pop] = []
            pops[pop].append(i)
    return pops


def get_hap_idx(pop_indices, pop_list):
    idx = []
    for pop in pop_list:
        for si in pop_indices[pop]:
            idx.extend([2 * si, 2 * si + 1])
    return np.array(idx)


def compute_gene_stats(pos_r, h_f, h_nf):
    """Compute all four stats for one gene window. Returns dict or None on failure."""
    ac_f = h_f.count_alleles()
    ac_nf = h_nf.count_alleles()

    # Tajima's D
    try:
        tajd = float(allel.tajima_d(ac_f, pos=pos_r))
    except:
        tajd = np.nan

    # Pi ratio
    try:
        pi_f = allel.sequence_diversity(pos_r, ac_f)
        pi_nf = allel.sequence_diversity(pos_r, ac_nf)
        pi_ratio = float(pi_f / pi_nf) if pi_nf > 0 else np.nan
    except:
        pi_ratio = np.nan

    # Max |iHS|
    try:
        is_seg = ac_f.is_segregating() & ac_f.is_biallelic()
        if is_seg.sum() > 10:
            ihs_raw = allel.ihs(h_f[is_seg], pos_r[is_seg], use_threads=False)
            ihs_std = allel.standardize_by_allele_count(
                ihs_raw, ac_f[is_seg][:, 1], diagnostics=False)
            ihs_vals = np.abs(ihs_std[0])
            ihs_vals = ihs_vals[~np.isnan(ihs_vals)]
            max_ihs = float(np.nanmax(ihs_vals)) if len(ihs_vals) else np.nan
        else:
            max_ihs = np.nan
    except:
        max_ihs = np.nan

    # Max FST
    try:
        num, den = allel.hudson_fst(ac_f, ac_nf)
        fst = num / den
        fst = fst[~np.isnan(fst)]
        max_fst = float(np.nanmax(fst)) if len(fst) else np.nan
    except:
        max_fst = np.nan

    return {"tajima_d": tajd, "pi_ratio": pi_ratio, "max_ihs": max_ihs, "max_fst": max_fst}


def main():
    # Which chromosome to process (pass as arg for parallelization)
    if len(sys.argv) > 1:
        chromosomes = [int(sys.argv[1])]
    else:
        chromosomes = list(range(1, 23))

    pop_indices = load_pop_indices(SAMPLES_FILE)
    focal_pops = EAS_POPS
    nonfocal_pops = [p for p in pop_indices if p not in focal_pops]
    focal_hap_global = get_hap_idx(pop_indices, focal_pops)
    nonfocal_hap_global = get_hap_idx(pop_indices, nonfocal_pops)

    all_results = []

    for chrn in chromosomes:
        gene_file = os.path.join(GENE_DIR, f"chr{chrn}_genes.tsv")
        if not os.path.exists(gene_file):
            continue
        genes = pd.read_csv(gene_file, sep="\t")
        genes["midpoint"] = (genes["start"] + genes["end"]) / 2

        print(f"Loading chr{chrn}...", flush=True)
        t0 = time.time()
        npz = np.load(os.path.join(PARSED_DIR, f"chr{chrn}.npz"))
        G_full = npz["G"]
        pos_full = npz["positions"]
        print(f"  Loaded: {G_full.shape}, {time.time()-t0:.1f}s", flush=True)

        n_genes = len(genes)
        print(f"  Processing {n_genes} genes...", flush=True)

        for gi, (_, gene) in enumerate(genes.iterrows()):
            if gi % 50 == 0:
                print(f"  chr{chrn}: {gi}/{n_genes} ({gi/n_genes*100:.0f}%)", flush=True)

            gs, ge = gene["start"], gene["end"]
            rs, re = max(0, gs - WINDOW), ge + WINDOW
            mask = (pos_full >= rs) & (pos_full <= re)

            if mask.sum() < 20:
                all_results.append({
                    "gene": gene["gene_name"], "chr": chrn,
                    "tajima_d": np.nan, "pi_ratio": np.nan,
                    "max_ihs": np.nan, "max_fst": np.nan,
                })
                continue

            pos_r = pos_full[mask]
            G_r = G_full[:, mask]

            h_f = allel.HaplotypeArray(G_r[focal_hap_global].T)
            h_nf = allel.HaplotypeArray(G_r[nonfocal_hap_global].T)

            stats = compute_gene_stats(pos_r, h_f, h_nf)
            stats["gene"] = gene["gene_name"]
            stats["chr"] = chrn
            all_results.append(stats)

        print(f"  chr{chrn} done: {time.time()-t0:.0f}s", flush=True)

    # Save per-chromosome results
    df = pd.DataFrame(all_results)
    if len(chromosomes) == 1:
        outf = os.path.join(OUTDIR, f"ortho_chr{chromosomes[0]}.csv")
    else:
        outf = os.path.join(OUTDIR, "ortho_genomewide.csv")
    df.to_csv(outf, index=False)
    print(f"\nSaved {len(df)} genes to {outf}")

    # If genome-wide, compute percentiles
    if len(chromosomes) > 1:
        for stat in ["tajima_d", "pi_ratio", "max_ihs", "max_fst"]:
            valid = df[stat].dropna()
            df[f"{stat}_pctl"] = df[stat].rank(pct=True, na_option="keep")
            # For tajima_d and pi_ratio, low = sweep, so invert percentile
            if stat in ["tajima_d", "pi_ratio"]:
                df[f"{stat}_pctl"] = df[f"{stat}_pctl"]  # low value = low percentile = sweep signal

        df.to_csv(outf, index=False)
        print(f"Added percentiles, re-saved to {outf}")

        # Print candidates
        candidates = ["CLEC6A", "TRAF6", "TNFRSF13C", "JCHAIN",
                       "GRK2", "BPIFA2", "CCDC92", "SLC6A15",
                       "TTBK1", "CCDC70", "CZIB", "PER1", "CUL1"]
        print(f"\n{'Gene':<14} {'TajD':>8} {'pctl':>6} {'piR':>8} {'pctl':>6} {'iHS':>8} {'pctl':>6} {'FST':>8} {'pctl':>6}")
        for _, row in df[df["gene"].isin(candidates)].iterrows():
            print(f"{row['gene']:<14} {row['tajima_d']:>8.3f} {row['tajima_d_pctl']:>6.1%} "
                  f"{row['pi_ratio']:>8.3f} {row['pi_ratio_pctl']:>6.1%} "
                  f"{row['max_ihs']:>8.2f} {row['max_ihs_pctl']:>6.1%} "
                  f"{row['max_fst']:>8.3f} {row['max_fst_pctl']:>6.1%}")


if __name__ == "__main__":
    main()
