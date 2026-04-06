#!/usr/bin/env python
"""
Compute XP-EHH (cross-population extended haplotype homozygosity) genome-wide.
For each gene, compute max |XP-EHH| in the focal population vs YRI (African reference).

XP-EHH > 0: selection in focal population
XP-EHH < 0: selection in reference population

Run per chromosome: python run_xpehh_genomewide.py CHR FOCAL_SUPERPOP
Example: python run_xpehh_genomewide.py 12 EAS
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

SUPERPOP_MAP = {
    "EAS": ["CHB", "JPT", "CHS", "CDX", "KHV"],
    "SAS": ["GIH", "PJL", "BEB", "STU", "ITU"],
    "EUR": ["CEU", "TSI", "FIN", "GBR", "IBS"],
}
REF_POPS = ["YRI"]  # African reference
WINDOW = 500_000


def load_pop_indices(samples_file):
    pops = {}
    with open(samples_file) as f:
        header = f.readline().strip().split()
        pop_col = next(j for j, c in enumerate(header) if c.lower() in ("pop", "population"))
        for i, line in enumerate(f):
            fields = line.strip().split()
            pops.setdefault(fields[pop_col], []).append(i)
    return pops


def get_hap_idx(pop_indices, pop_list):
    idx = []
    for pop in pop_list:
        for si in pop_indices[pop]:
            idx.extend([2 * si, 2 * si + 1])
    return np.array(idx)


def main():
    chrn = int(sys.argv[1])
    focal_superpop = sys.argv[2] if len(sys.argv) > 2 else "EAS"
    focal_pops = SUPERPOP_MAP[focal_superpop]

    pop_indices = load_pop_indices(SAMPLES_FILE)
    focal_hap_idx = get_hap_idx(pop_indices, focal_pops)
    ref_hap_idx = get_hap_idx(pop_indices, REF_POPS)

    gene_file = os.path.join(GENE_DIR, f"chr{chrn}_genes.tsv")
    genes = pd.read_csv(gene_file, sep="\t")
    genes["midpoint"] = ((genes["start"] + genes["end"]) / 2).astype(int)

    print(f"Loading chr{chrn}...", flush=True)
    t0 = time.time()
    npz = np.load(os.path.join(PARSED_DIR, f"chr{chrn}.npz"))
    G = npz["G"]
    pos = npz["positions"]
    print(f"  {G.shape}, {time.time()-t0:.1f}s", flush=True)

    # Compute chromosome-wide XP-EHH once
    print(f"  Computing chromosome-wide XP-EHH ({focal_superpop} vs YRI)...", flush=True)
    h_focal = allel.HaplotypeArray(G[focal_hap_idx].T)
    h_ref = allel.HaplotypeArray(G[ref_hap_idx].T)
    del G

    ac_focal = h_focal.count_alleles()
    ac_ref = h_ref.count_alleles()

    # Filter to biallelic segregating in both
    is_ok = (ac_focal.is_segregating() | ac_ref.is_segregating()) & ac_focal.is_biallelic() & ac_ref.is_biallelic()
    pos_filt = pos[is_ok]
    h_focal_filt = h_focal[is_ok]
    h_ref_filt = h_ref[is_ok]

    print(f"  {is_ok.sum()} variants after filtering, computing XP-EHH...", flush=True)
    t1 = time.time()
    try:
        xpehh_raw = allel.xpehh(h_focal_filt, h_ref_filt, pos_filt, use_threads=False)
        print(f"  XP-EHH done in {time.time()-t1:.0f}s", flush=True)
    except Exception as e:
        print(f"  XP-EHH FAILED: {e}")
        return

    # For each gene, get max XP-EHH in ±WINDOW
    results = []
    n_genes = len(genes)
    for gi, (_, gene) in enumerate(genes.iterrows()):
        if gi % 100 == 0:
            print(f"  chr{chrn}: {gi}/{n_genes} ({gi/n_genes*100:.0f}%)", flush=True)

        gs, ge = gene["start"], gene["end"]
        mask = (pos_filt >= gs - WINDOW) & (pos_filt <= ge + WINDOW)
        if mask.sum() < 5:
            results.append({"gene": gene["gene_name"], "chr": chrn,
                            "max_xpehh": np.nan, "mean_xpehh": np.nan,
                            "n_extreme": 0})
            continue

        vals = xpehh_raw[mask]
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            results.append({"gene": gene["gene_name"], "chr": chrn,
                            "max_xpehh": np.nan, "mean_xpehh": np.nan,
                            "n_extreme": 0})
            continue

        # Positive XP-EHH = selection in focal pop
        max_xpehh = float(np.nanmax(vals))
        mean_xpehh = float(np.nanmean(vals))
        n_extreme = int((vals > 2).sum())

        results.append({"gene": gene["gene_name"], "chr": chrn,
                        "max_xpehh": max_xpehh, "mean_xpehh": mean_xpehh,
                        "n_extreme": n_extreme})

    df = pd.DataFrame(results)
    outf = os.path.join(OUTDIR, f"xpehh_{focal_superpop}_chr{chrn}.csv")
    df.to_csv(outf, index=False)
    print(f"  Saved {len(df)} genes to {outf} ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
