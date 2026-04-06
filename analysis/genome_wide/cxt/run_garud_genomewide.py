#!/usr/bin/env python
"""
Compute Garud's H statistics (H1, H12, H123, H2/H1) genome-wide for EAS.
Uses 1000-SNP windows centered on each gene.

H12 detects both hard and soft sweeps via haplotype homozygosity.
H2/H1 distinguishes them: low = hard sweep, high = soft sweep.
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
WINDOW_SNP = 1000  # number of SNPs per window (standard for Garud's H)


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


def main():
    chrn = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    pop_indices = load_pop_indices(SAMPLES_FILE)
    focal_hap_idx = get_hap_idx(pop_indices, EAS_POPS)

    gene_file = os.path.join(GENE_DIR, f"chr{chrn}_genes.tsv")
    if not os.path.exists(gene_file):
        print(f"No gene file for chr{chrn}")
        return
    genes = pd.read_csv(gene_file, sep="\t")
    genes["midpoint"] = ((genes["start"] + genes["end"]) / 2).astype(int)

    print(f"Loading chr{chrn}...", flush=True)
    t0 = time.time()
    npz = np.load(os.path.join(PARSED_DIR, f"chr{chrn}.npz"))
    G_full = npz["G"]
    pos_full = npz["positions"]
    print(f"  Loaded: {G_full.shape}, {time.time()-t0:.1f}s", flush=True)

    # Subset to EAS haplotypes for the whole chromosome
    G_eas = G_full[focal_hap_idx]
    del G_full

    results = []
    n_genes = len(genes)
    print(f"  Computing Garud's H for {n_genes} genes...", flush=True)

    for gi, (_, gene) in enumerate(genes.iterrows()):
        if gi % 50 == 0:
            print(f"  chr{chrn}: {gi}/{n_genes} ({gi/n_genes*100:.0f}%)", flush=True)

        mid = gene["midpoint"]
        # Find the closest SNP to gene midpoint
        mid_idx = np.searchsorted(pos_full, mid)
        mid_idx = min(mid_idx, len(pos_full) - 1)

        # Window of WINDOW_SNP SNPs centered on gene
        half = WINDOW_SNP // 2
        start_idx = max(0, mid_idx - half)
        end_idx = min(len(pos_full), mid_idx + half)

        if end_idx - start_idx < 50:
            results.append({
                "gene": gene["gene_name"], "chr": chrn,
                "H1": np.nan, "H12": np.nan, "H123": np.nan, "H2_H1": np.nan,
            })
            continue

        # Get haplotype array for this window
        h_window = allel.HaplotypeArray(G_eas[:, start_idx:end_idx].T)

        try:
            h1, h12, h123, h2_h1 = allel.garud_h(h_window)
            results.append({
                "gene": gene["gene_name"], "chr": chrn,
                "H1": float(h1), "H12": float(h12),
                "H123": float(h123), "H2_H1": float(h2_h1),
            })
        except Exception as e:
            results.append({
                "gene": gene["gene_name"], "chr": chrn,
                "H1": np.nan, "H12": np.nan, "H123": np.nan, "H2_H1": np.nan,
            })

    df = pd.DataFrame(results)
    outf = os.path.join(OUTDIR, f"garud_chr{chrn}.csv")
    df.to_csv(outf, index=False)
    print(f"  Saved {len(df)} genes to {outf} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
