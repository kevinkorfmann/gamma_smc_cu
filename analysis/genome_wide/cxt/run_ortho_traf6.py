#!/usr/bin/env python
"""
Compute variant depletion/enrichment and max FST for TRAF6 (chr11:36,488,025-36,512,297).
Focal: EAS. Non-focal: all other superpopulations.
"""

import numpy as np
import allel
import os

BETTY_BASE = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide"
PARSED_DIR = os.path.join(BETTY_BASE, "cache/parsed")
SAMPLES_FILE = os.path.join(BETTY_BASE, "data/samples.txt")

# TRAF6 coordinates (GRCh38)
CHR = 11
GENE_START = 36_488_025
GENE_END = 36_512_297

EAS_POPS = ["CHB", "JPT", "CHS", "CDX", "KHV"]
SAS_POPS = ["BEB", "GIH", "ITU", "PJL", "STU"]
EUR_POPS = ["CEU", "TSI", "FIN", "GBR", "IBS"]
AFR_POPS = ["YRI", "LWK", "GWD", "MSL", "ESN", "ACB", "ASW"]
AMR_POPS = ["MXL", "PUR", "CLM", "PEL"]


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
        for si in pop_indices.get(pop, []):
            idx.extend([2 * si, 2 * si + 1])
    return np.array(idx)


def main():
    # Load data
    print(f"Loading chr{CHR}...")
    data = np.load(os.path.join(PARSED_DIR, f"chr{CHR}.npz"))
    G = data["G"]
    pos = data["positions"]
    print(f"  {G.shape[0]} haplotypes, {G.shape[1]} sites")

    # Subset to gene body
    mask = (pos >= GENE_START) & (pos <= GENE_END)
    G_gene = G[:, mask]
    pos_gene = pos[mask]
    n_sites = mask.sum()
    print(f"  TRAF6 gene body: {n_sites} sites")

    # Load population indices
    pop_idx = load_pop_indices(SAMPLES_FILE)

    eas_idx = get_hap_idx(pop_idx, EAS_POPS)
    sas_idx = get_hap_idx(pop_idx, SAS_POPS)
    eur_idx = get_hap_idx(pop_idx, EUR_POPS)
    afr_idx = get_hap_idx(pop_idx, AFR_POPS)
    amr_idx = get_hap_idx(pop_idx, AMR_POPS)

    nonfocal_idx = np.concatenate([sas_idx, eur_idx, afr_idx, amr_idx])

    # Per-population allele frequencies
    print("\nPer-superpopulation allele frequencies at each site:")
    eas_af = G_gene[eas_idx].mean(axis=0)
    sas_af = G_gene[sas_idx].mean(axis=0)
    eur_af = G_gene[eur_idx].mean(axis=0)
    afr_af = G_gene[afr_idx].mean(axis=0)
    amr_af = G_gene[amr_idx].mean(axis=0)

    # --- Variant depletion/enrichment ---
    # Depleted: AF < 10% in EAS but > 30% in at least one non-focal superpop
    depleted = (eas_af < 0.10) & (
        (sas_af > 0.30) | (eur_af > 0.30) | (afr_af > 0.30) | (amr_af > 0.30)
    )
    # Enriched: AF > 50% in EAS and < 30% in all non-focal superpops
    enriched = (eas_af > 0.50) & (sas_af < 0.30) & (eur_af < 0.30) & (afr_af < 0.30) & (amr_af < 0.30)

    n_depleted = depleted.sum()
    n_enriched = enriched.sum()
    print(f"\nVariant depletion/enrichment:")
    print(f"  Depleted: {n_depleted}")
    print(f"  Enriched: {n_enriched}")
    print(f"  Ratio: {n_depleted}:{n_enriched}")

    # --- Max FST (EAS vs non-focal) ---
    # Hudson's FST per site
    n_eas = len(eas_idx)
    n_nf = len(nonfocal_idx)

    fst_values = []
    for s in range(n_sites):
        p1 = G_gene[eas_idx, s].mean()
        p2 = G_gene[nonfocal_idx, s].mean()
        # Hudson's estimator
        num = (p1 - p2)**2 - (p1*(1-p1))/(n_eas-1) - (p2*(1-p2))/(n_nf-1)
        den = p1*(1-p2) + p2*(1-p1)
        if den > 0:
            fst_values.append(max(0, num/den))
        else:
            fst_values.append(0)

    fst_values = np.array(fst_values)
    max_fst = fst_values.max()
    max_fst_idx = fst_values.argmax()
    max_fst_pos = pos_gene[max_fst_idx]

    print(f"\nMax FST:")
    print(f"  FST = {max_fst:.3f} at position {max_fst_pos}")
    print(f"  EAS AF = {eas_af[max_fst_idx]:.3f}")
    print(f"  SAS AF = {sas_af[max_fst_idx]:.3f}")
    print(f"  EUR AF = {eur_af[max_fst_idx]:.3f}")
    print(f"  AFR AF = {afr_af[max_fst_idx]:.3f}")
    print(f"  AMR AF = {amr_af[max_fst_idx]:.3f}")

    # Print all depleted variants
    if n_depleted > 0:
        print(f"\nDepleted variant positions:")
        for i in np.where(depleted)[0]:
            print(f"  {pos_gene[i]}: EAS={eas_af[i]:.3f} SAS={sas_af[i]:.3f} "
                  f"EUR={eur_af[i]:.3f} AFR={afr_af[i]:.3f}")

    if n_enriched > 0:
        print(f"\nEnriched variant positions:")
        for i in np.where(enriched)[0]:
            print(f"  {pos_gene[i]}: EAS={eas_af[i]:.3f} SAS={sas_af[i]:.3f} "
                  f"EUR={eur_af[i]:.3f} AFR={afr_af[i]:.3f}")


if __name__ == "__main__":
    main()
