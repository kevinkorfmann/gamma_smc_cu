#!/usr/bin/env python
"""
HGDP independent validation of novel sweep findings.

Downloads HGDP phased VCFs (Bergstrom et al. 2020) for targeted regions,
computes H12 and allele frequency patterns at candidate loci, and compares
with 1KG results.

Usage: python hgdp_validation.py (run on betty via slurm)
"""

import numpy as np
import allel
import os
import subprocess
import gzip
import sys

OUTDIR = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/hgdp"
HGDP_BASE = "https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/hgdp_wgs.20190516.full"
WINDOW_SNPS = 1000  # SNPs for H12 window

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(os.path.join(OUTDIR, "vcf"), exist_ok=True)

# Candidate loci: (name, chr, start, end, focal_superpop, focal_desc)
CANDIDATES = [
    ("GRK2",      11, 66500000, 68000000, "CENTRAL_SOUTH_ASIA", "Cardiovascular"),
    ("CLEC6A",    12,  7500000, 10000000, "EAST_ASIA",          "Mucosal immunity"),
    ("TRAF6",     11, 35500000, 37500000, "EAST_ASIA",          "Mucosal immunity"),
    ("TNFRSF13C", 22, 41000000, 42500000, "EAST_ASIA",          "Mucosal immunity"),
    ("JCHAIN",     4, 69500000, 71500000, "EAST_ASIA",          "Mucosal immunity"),
    ("BPIFA2",    20, 32500000, 34000000, "CENTRAL_SOUTH_ASIA", "Salivary antimicrobial"),
    ("CCDC92",    12, 123000000,125000000,"EAST_ASIA",          "Metabolic"),
    ("SLC6A15",   12, 84000000, 86000000, "EAST_ASIA",          "Brain transporter"),
]

# HGDP population-to-region mapping (Bergstrom 2020)
REGION_POPS = {
    "EAST_ASIA": ["Han", "Japanese", "Tujia", "Yi", "Miao", "She", "Naxi",
                  "Tu", "Mongola", "Daur", "Hezhen", "Oroqen", "Xibo",
                  "Cambodian", "Dai", "Lahu", "Yakut"],
    "CENTRAL_SOUTH_ASIA": ["Balochi", "Brahui", "Burusho", "Hazara", "Kalash",
                           "Makrani", "Pathan", "Sindhi", "Uygur"],
    "EUROPE": ["Adygei", "Basque", "French", "Bergamo", "Sardinian",
               "Tuscan", "Orcadian", "Russian"],
    "AFRICA": ["BantuKenya", "BantuSouthAfrica", "BiakaPygmy", "MbutiPygmy",
               "Mandenka", "Yoruba", "San"],
}


def download_vcf(chrn):
    """Download HGDP phased VCF for a chromosome if not already present."""
    vcf_url = f"{HGDP_BASE}.chr{chrn}.vcf.gz"
    tbi_url = f"{vcf_url}.tbi"
    vcf_local = os.path.join(OUTDIR, "vcf", f"hgdp_chr{chrn}.vcf.gz")
    tbi_local = f"{vcf_local}.tbi"

    if os.path.exists(vcf_local) and os.path.getsize(vcf_local) > 1_000_000:
        print(f"  chr{chrn} VCF already downloaded ({os.path.getsize(vcf_local)/1e9:.1f} GB)")
        return vcf_local

    print(f"  Downloading chr{chrn} VCF...")
    subprocess.run(["wget", "--no-check-certificate", "-q", "-O", vcf_local, vcf_url], check=True)
    print(f"  Downloading chr{chrn} index...")
    subprocess.run(["wget", "--no-check-certificate", "-q", "-O", tbi_local, tbi_url], check=True)
    print(f"  Downloaded ({os.path.getsize(vcf_local)/1e9:.1f} GB)")
    return vcf_local


def load_hgdp_samples(vcf_path):
    """Load sample IDs and their population labels from HGDP VCF."""
    # HGDP sample IDs encode population: HGDP00001, etc.
    # We need the metadata file
    meta_url = "https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/metadata/hgdp_wgs.20190516.metadata.txt"
    readme_url = "https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/metadata/README.hgdp_wgs.20190516.metadata.txt"
    meta_local = os.path.join(OUTDIR, "hgdp_metadata.txt")

    if not os.path.exists(meta_local):
        print("  Downloading HGDP metadata...")
        subprocess.run(["wget", "-q", "-O", meta_local, meta_url], check=True)

    # Parse metadata
    sample_pop = {}
    sample_region = {}
    with open(meta_local) as f:
        header = f.readline().strip().split("\t")
        # Find columns
        sample_col = 0
        pop_col = None
        region_col = None
        for i, h in enumerate(header):
            if h.lower() in ("population", "pop"):
                pop_col = i
            if h.lower() in ("region", "continent"):
                region_col = i
        if pop_col is None:
            # Try common formats
            pop_col = 6 if len(header) > 6 else 1
        if region_col is None:
            region_col = 7 if len(header) > 7 else 2
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) > max(pop_col, region_col):
                sid = fields[sample_col]
                sample_pop[sid] = fields[pop_col]
                sample_region[sid] = fields[region_col]

    # Get sample list from VCF header
    callset = allel.read_vcf(vcf_path, fields=[], numbers={"GT": 2},
                              region=None, tabix="tabix")
    # Fallback: read header manually
    import io
    if vcf_path.endswith(".gz"):
        fh = gzip.open(vcf_path, "rt")
    else:
        fh = open(vcf_path, "r")
    for line in fh:
        if line.startswith("#CHROM"):
            samples = line.strip().split("\t")[9:]
            break
    fh.close()

    return samples, sample_pop, sample_region


def compute_h12_at_gene(vcf_path, chrn, gene_start, gene_end, hap_idx, n_snps=1000):
    """Compute Garud's H12 in a window centered on a gene."""
    mid = (gene_start + gene_end) // 2

    # Read a broad region and find SNPs
    region = f"chr{chrn}" if chrn != "X" else "chrX"
    try:
        callset = allel.read_vcf(
            vcf_path,
            region=f"{region}:{max(1, mid - 2_000_000)}-{mid + 2_000_000}",
            fields=["variants/POS", "calldata/GT"],
            numbers={"GT": 2},
        )
    except Exception:
        # Try without chr prefix
        region = str(chrn)
        callset = allel.read_vcf(
            vcf_path,
            region=f"{region}:{max(1, mid - 2_000_000)}-{mid + 2_000_000}",
            fields=["variants/POS", "calldata/GT"],
            numbers={"GT": 2},
        )

    if callset is None:
        return None, None, None

    pos = callset["variants/POS"]
    gt = allel.GenotypeArray(callset["calldata/GT"])

    # Subset to focal population haplotypes
    gt_sub = gt.take(hap_idx, axis=1)
    haps = gt_sub.to_haplotypes()

    # Find window of n_snps centered on gene midpoint
    center_idx = np.searchsorted(pos, mid)
    start_idx = max(0, center_idx - n_snps // 2)
    end_idx = min(len(pos), start_idx + n_snps)
    if end_idx - start_idx < 100:
        return None, None, None

    hap_window = haps[start_idx:end_idx]

    # Compute haplotype frequencies
    hap_strings = ["".join(str(x) for x in hap_window[:, i]) for i in range(hap_window.shape[1])]
    from collections import Counter
    counts = Counter(hap_strings)
    freqs = np.array(sorted(counts.values(), reverse=True)) / len(hap_strings)

    h1 = np.sum(freqs**2)
    h12 = h1 + 2 * freqs[0] * freqs[1] if len(freqs) > 1 else h1
    h2 = h1 - freqs[0]**2
    h2h1 = h2 / h1 if h1 > 0 else 0

    return h12, h2h1, len(hap_strings)


def compute_af_at_gene(vcf_path, chrn, gene_start, gene_end, pop_hap_indices):
    """Compute per-population allele frequencies at a gene."""
    region_prefix = f"chr{chrn}"
    try:
        callset = allel.read_vcf(
            vcf_path,
            region=f"{region_prefix}:{gene_start}-{gene_end}",
            fields=["variants/POS", "calldata/GT"],
            numbers={"GT": 2},
        )
    except Exception:
        region_prefix = str(chrn)
        callset = allel.read_vcf(
            vcf_path,
            region=f"{region_prefix}:{gene_start}-{gene_end}",
            fields=["variants/POS", "calldata/GT"],
            numbers={"GT": 2},
        )

    if callset is None:
        return None, None

    gt = allel.GenotypeArray(callset["calldata/GT"])
    pos = callset["variants/POS"]

    # Per-population AF
    afs = {}
    for pop_name, hidx in pop_hap_indices.items():
        if len(hidx) > 0:
            haps = gt.take(hidx, axis=1).to_haplotypes()
            afs[pop_name] = haps.mean(axis=1)

    return pos, afs


# Gene coordinates (GRCh38) for the specific genes
GENE_COORDS = {
    "GRK2":      (11, 67242000, 67264000),
    "CLEC6A":    (12, 8295819,  8314573),
    "TRAF6":     (11, 36488025, 36512297),
    "TNFRSF13C": (22, 41901811, 41912652),
    "JCHAIN":    (4,  70574239, 70591222),
    "BPIFA2":    (20, 33148000, 33195000),
    "CCDC92":    (12, 123930000,123960000),
    "SLC6A15":   (12, 84860000, 84920000),
}


def main():
    # Determine which chromosomes we need
    chrs_needed = sorted(set(c[1] for c in CANDIDATES))
    print(f"Chromosomes needed: {chrs_needed}")

    # Download VCFs
    print("\n=== Downloading HGDP VCFs ===")
    vcf_paths = {}
    for chrn in chrs_needed:
        vcf_paths[chrn] = download_vcf(chrn)

    # Load sample metadata
    print("\n=== Loading HGDP sample metadata ===")
    first_vcf = vcf_paths[chrs_needed[0]]
    samples, sample_pop, sample_region = load_hgdp_samples(first_vcf)
    print(f"  {len(samples)} samples in VCF")
    print(f"  {len(sample_pop)} samples in metadata")

    # Build population -> sample index mapping
    pop_sample_idx = {}
    region_sample_idx = {}
    for i, s in enumerate(samples):
        pop = sample_pop.get(s, "unknown")
        reg = sample_region.get(s, "unknown")
        pop_sample_idx.setdefault(pop, []).append(i)
        region_sample_idx.setdefault(reg, []).append(i)

    print(f"  Regions: {list(region_sample_idx.keys())}")
    for r in sorted(region_sample_idx.keys()):
        print(f"    {r}: {len(region_sample_idx[r])} samples")

    # Build haplotype indices per region (diploid -> 2 haplotypes)
    def to_hap_idx(sample_indices):
        idx = []
        for si in sample_indices:
            idx.extend([2 * si, 2 * si + 1])
        return np.array(idx)

    region_hap_idx = {r: to_hap_idx(idxs) for r, idxs in region_sample_idx.items()}

    # === Run validation for each candidate ===
    print("\n=== Validation Results ===\n")
    results = []

    for gene_name, chrn, reg_start, reg_end, focal_region, desc in CANDIDATES:
        print(f"--- {gene_name} (chr{chrn}, {desc}) ---")
        print(f"  Focal region: {focal_region}")

        vcf_path = vcf_paths[chrn]
        gene_chr, gene_start, gene_end = GENE_COORDS[gene_name]

        # Find matching region name in HGDP metadata
        # Try exact match first, then fuzzy
        focal_key = None
        for r in region_hap_idx:
            if focal_region.lower().replace("_", " ") in r.lower().replace("_", " "):
                focal_key = r
                break
            if focal_region.lower() in r.lower():
                focal_key = r
                break
        if focal_key is None:
            # Try REGION_POPS mapping
            for r in region_hap_idx:
                if r in REGION_POPS.get(focal_region, []):
                    focal_key = r
                    break
        if focal_key is None:
            print(f"  WARNING: Could not find region '{focal_region}' in HGDP metadata")
            print(f"  Available: {list(region_hap_idx.keys())}")
            continue

        focal_haps = region_hap_idx[focal_key]
        print(f"  Matched region: {focal_key} ({len(focal_haps)} haplotypes)")

        # H12
        h12, h2h1, n_haps = compute_h12_at_gene(vcf_path, chrn, gene_start, gene_end, focal_haps)
        if h12 is not None:
            print(f"  H12 = {h12:.4f}, H2/H1 = {h2h1:.4f} (from {n_haps} haplotypes)")
        else:
            print(f"  H12: could not compute")

        # Allele frequencies
        pos, afs = compute_af_at_gene(vcf_path, chrn, gene_start, gene_end, region_hap_idx)
        if pos is not None and len(pos) > 0:
            n_sites = len(pos)
            focal_af = afs.get(focal_key, np.zeros(n_sites))

            # Find African control
            afr_key = None
            for r in region_hap_idx:
                if "afri" in r.lower():
                    afr_key = r
                    break
            afr_af = afs.get(afr_key, np.zeros(n_sites)) if afr_key else np.zeros(n_sites)

            # Compute depletion/enrichment
            other_afs = {r: af for r, af in afs.items() if r != focal_key}
            max_other = np.max([af for af in other_afs.values()], axis=0) if other_afs else np.zeros(n_sites)

            depleted = ((focal_af < 0.10) & (max_other > 0.30)).sum()
            enriched = ((focal_af > 0.50) & np.all([af < 0.30 for af in other_afs.values()], axis=0)).sum()

            # Max FST (focal vs rest)
            nonfocal_samples = []
            for r, hidx in region_hap_idx.items():
                if r != focal_key:
                    nonfocal_samples.extend(hidx.tolist())
            nonfocal_haps = np.array(nonfocal_samples)

            # Simple FST computation
            gt_data = allel.read_vcf(
                vcf_path,
                region=f"chr{chrn}:{gene_start}-{gene_end}" if "chr" in str(chrn) else f"{chrn}:{gene_start}-{gene_end}",
                fields=["calldata/GT"],
                numbers={"GT": 2},
            )
            if gt_data is not None:
                gt_arr = allel.GenotypeArray(gt_data["calldata/GT"])
                focal_si = [i for i in range(len(samples)) if i in set(region_sample_idx.get(focal_key, []))]
                nonfocal_si = [i for i in range(len(samples)) if i not in set(region_sample_idx.get(focal_key, []))]
                ac1 = gt_arr.take(focal_si, axis=1).count_alleles()
                ac2 = gt_arr.take(nonfocal_si, axis=1).count_alleles()
                num, den = allel.hudson_fst(ac1, ac2)
                per_site_fst = np.where(den > 0, num / den, 0)
                per_site_fst = np.clip(per_site_fst, 0, 1)
                max_fst = per_site_fst.max()
            else:
                max_fst = float("nan")

            print(f"  {n_sites} sites in gene body")
            print(f"  Depleted:Enriched = {depleted}:{enriched}")
            print(f"  Max FST = {max_fst:.3f}")
        else:
            print(f"  Could not load allele frequencies")
            depleted = enriched = 0
            max_fst = float("nan")

        results.append({
            "gene": gene_name,
            "chr": chrn,
            "desc": desc,
            "focal_region": focal_key,
            "h12": h12,
            "h2h1": h2h1,
            "n_haps": n_haps,
            "depleted": depleted,
            "enriched": enriched,
            "max_fst": max_fst,
        })
        print()

    # Save results
    import csv
    outf = os.path.join(OUTDIR, "hgdp_validation_results.csv")
    with open(outf, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to: {outf}")


if __name__ == "__main__":
    main()
