#!/usr/bin/env python
"""
Compute iHS and orthogonal stats (Tajima's D, pi ratio, max FST)
for the 3 mucosal immunity genes missing these stats: TRAF6, TNFRSF13C, JCHAIN.
CLEC6A already has them.
"""

import numpy as np
import allel
import os
import sys

BETTY_BASE = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide"
PARSED_DIR = os.path.join(BETTY_BASE, "cache/parsed")
SAMPLES_FILE = os.path.join(BETTY_BASE, "data/samples.txt")
OUTDIR = os.path.join(BETTY_BASE, "cxt/results/orthogonal")

os.makedirs(OUTDIR, exist_ok=True)

# Genes to analyze
GENES = [
    ("TRAF6", 11, 36483769, 36510272, "EAS"),
    ("TNFRSF13C", 22, 41922032, 41926806, "EAS"),
    ("JCHAIN", 4, 70655541, 70681817, "EAS"),
]

EAS_POPS = ["CHB", "JPT", "CHS", "CDX", "KHV"]
WINDOW = 500_000  # ±500 kb flanking


def load_pop_indices(samples_file):
    """Return dict: pop -> list of sample indices."""
    pops = {}
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
            pop = fields[pop_col]
            if pop not in pops:
                pops[pop] = []
            pops[pop].append(i)
    return pops


def get_haplotype_indices(pop_indices, pop_list):
    """Convert sample indices to haplotype indices for a list of populations."""
    hap_idx = []
    for pop in pop_list:
        for si in pop_indices[pop]:
            hap_idx.extend([2 * si, 2 * si + 1])
    return np.array(hap_idx)


def compute_stats(gene_name, chrn, gene_start, gene_end, focal_superpop):
    print(f"\n{'='*60}")
    print(f"Processing {gene_name} (chr{chrn}:{gene_start}-{gene_end})")

    # Load data
    npz = np.load(os.path.join(PARSED_DIR, f"chr{chrn}.npz"))
    G = npz["G"]
    pos = npz["positions"]

    # Region: gene ± WINDOW
    region_start = max(0, gene_start - WINDOW)
    region_end = gene_end + WINDOW
    mask = (pos >= region_start) & (pos <= region_end)
    pos_region = pos[mask]
    G_region = G[:, mask]
    print(f"  Region: {region_start}-{region_end}, {mask.sum()} variants")

    # Population indices
    pop_indices = load_pop_indices(SAMPLES_FILE)

    # Focal (EAS) haplotypes
    focal_hap_idx = get_haplotype_indices(pop_indices, EAS_POPS)
    focal_haps = G_region[focal_hap_idx]

    # Non-focal haplotypes (all non-EAS)
    non_eas_pops = [p for p in pop_indices if p not in EAS_POPS]
    non_focal_hap_idx = get_haplotype_indices(pop_indices, non_eas_pops)
    non_focal_haps = G_region[non_focal_hap_idx]

    # Convert to allel format: (n_variants, n_haplotypes)
    h_focal = allel.HaplotypeArray(focal_haps.T)
    h_non_focal = allel.HaplotypeArray(non_focal_haps.T)

    # --- iHS ---
    print("  Computing iHS...")
    ac_focal = h_focal.count_alleles()
    # Filter: biallelic, not fixed
    is_seg = ac_focal.is_segregating() & ac_focal.is_biallelic()
    pos_seg = pos_region[is_seg]
    h_focal_seg = h_focal[is_seg]

    try:
        ihs_raw = allel.ihs(h_focal_seg, pos_seg, use_threads=False)
        ihs_std = allel.standardize_by_allele_count(
            ihs_raw, ac_focal[is_seg][:, 1], diagnostics=False
        )
        ihs_vals = np.abs(ihs_std[0])
        ihs_vals = ihs_vals[~np.isnan(ihs_vals)]
        max_ihs = np.nanmax(ihs_vals) if len(ihs_vals) > 0 else 0
        mean_ihs = np.nanmean(ihs_vals) if len(ihs_vals) > 0 else 0
        n_extreme = int((ihs_vals > 2).sum())
        print(f"  iHS: max|iHS|={max_ihs:.3f}, mean={mean_ihs:.3f}, n_extreme(>2)={n_extreme}")
    except Exception as e:
        print(f"  iHS failed: {e}")
        max_ihs, mean_ihs, n_extreme = 0, 0, 0

    # --- Tajima's D ---
    print("  Computing Tajima's D...")
    ac_focal_all = h_focal.count_alleles()
    try:
        tajd = allel.tajima_d(ac_focal_all, pos=pos_region)
        print(f"  Tajima's D = {tajd:.3f}")
    except Exception as e:
        print(f"  Tajima's D failed: {e}")
        tajd = np.nan

    # --- Pi (nucleotide diversity) ---
    print("  Computing pi ratio...")
    try:
        pi_focal = allel.sequence_diversity(pos_region, ac_focal_all)
        ac_nf = h_non_focal.count_alleles()
        pi_nonfocal = allel.sequence_diversity(pos_region, ac_nf)
        pi_ratio = pi_focal / pi_nonfocal if pi_nonfocal > 0 else np.nan
        print(f"  pi_focal={pi_focal:.6f}, pi_nonfocal={pi_nonfocal:.6f}, ratio={pi_ratio:.3f}")
    except Exception as e:
        print(f"  Pi failed: {e}")
        pi_ratio = np.nan

    # --- Max FST ---
    print("  Computing max FST...")
    try:
        ac_f = h_focal.count_alleles()
        ac_nf = h_non_focal.count_alleles()
        fst_per_snp = allel.hudson_fst(ac_f, ac_nf)
        # hudson_fst returns (num, den)
        num, den = fst_per_snp
        fst_vals = num / den
        fst_vals = fst_vals[~np.isnan(fst_vals)]
        max_fst = np.nanmax(fst_vals) if len(fst_vals) > 0 else 0
        print(f"  Max FST = {max_fst:.3f}")
    except Exception as e:
        print(f"  FST failed: {e}")
        max_fst = np.nan

    # Save
    result = {
        "gene": gene_name, "chr": chrn,
        "max_abs_ihs": max_ihs, "mean_abs_ihs": mean_ihs, "n_extreme": n_extreme,
        "tajima_d": tajd, "pi_ratio": pi_ratio, "max_fst": max_fst,
    }
    outf = os.path.join(OUTDIR, f"{gene_name}_orthogonal.npz")
    np.savez(outf, **{k: np.array(v) for k, v in result.items()})
    print(f"  Saved: {outf}")
    return result


results = []
for gene_name, chrn, start, end, superpop in GENES:
    r = compute_stats(gene_name, chrn, start, end, superpop)
    results.append(r)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Gene':<15} {'max|iHS|':>10} {'Tajima D':>10} {'pi ratio':>10} {'max FST':>10}")
for r in results:
    print(f"{r['gene']:<15} {r['max_abs_ihs']:>10.2f} {r['tajima_d']:>10.3f} {r['pi_ratio']:>10.3f} {r['max_fst']:>10.3f}")
