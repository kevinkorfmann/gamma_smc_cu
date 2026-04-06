#!/usr/bin/env python
"""
HGDP validation for a single gene. Run one per chromosome in parallel.

Usage: python hgdp_validate_single.py --gene GRK2
"""

import numpy as np
import os

# Ensure pixi bin is in PATH so allel can find tabix
PIXI_BIN = "/vast/projects/smathi/cohort/kkor/tmrca.cu/.pixi/envs/default/bin"
os.environ["PATH"] = PIXI_BIN + ":" + os.environ.get("PATH", "")

import allel
import subprocess
import gzip
import argparse
from collections import Counter

OUTDIR = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/hgdp"
HGDP_BASE = "https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/hgdp_wgs.20190516.full"
META_URL = "https://ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/metadata/hgdp_wgs.20190516.metadata.txt"
WINDOW_SNPS = 1000

GENE_INFO = {
    # gene: (chr, gene_start, gene_end, region_start, region_end, focal_region)
    "GRK2":      (11, 67242000,  67264000,  66500000,  68000000, "CENTRAL_SOUTH_ASIA"),
    "CLEC6A":    (12, 8295819,   8314573,   7500000,   10000000, "EAST_ASIA"),
    "TRAF6":     (11, 36488025,  36512297,  35500000,  37500000, "EAST_ASIA"),
    "TNFRSF13C": (22, 41901811,  41912652,  41000000,  42500000, "EAST_ASIA"),
    "JCHAIN":    (4,  70574239,  70591222,  69500000,  71500000, "EAST_ASIA"),
    "BPIFA2":    (20, 33148000,  33195000,  32500000,  34000000, "CENTRAL_SOUTH_ASIA"),
    "CCDC92":    (12, 123930000, 123960000, 123000000, 125000000, "EAST_ASIA"),
    "SLC6A15":   (12, 84860000,  84920000,  84000000,  86000000, "EAST_ASIA"),
}

REGION_MAP = {
    "EAST_ASIA": "EAST_ASIA",
    "CENTRAL_SOUTH_ASIA": "CENTRAL_SOUTH_ASIA",
    "EUROPE": "EUROPE",
    "AFRICA": "AFRICA",
}


def download_vcf(chrn):
    os.makedirs(os.path.join(OUTDIR, "vcf"), exist_ok=True)
    vcf_local = os.path.join(OUTDIR, "vcf", f"hgdp_chr{chrn}.vcf.gz")
    tbi_local = f"{vcf_local}.tbi"
    if os.path.exists(vcf_local) and os.path.getsize(vcf_local) > 1_000_000:
        print(f"  chr{chrn} already downloaded ({os.path.getsize(vcf_local)/1e9:.1f} GB)")
        return vcf_local
    vcf_url = f"{HGDP_BASE}.chr{chrn}.vcf.gz"
    tbi_url = f"{vcf_url}.tbi"
    print(f"  Downloading chr{chrn} VCF...")
    subprocess.run(["wget", "--no-check-certificate", "-q", "-O", vcf_local, vcf_url], check=True)
    print(f"  Downloading chr{chrn} index...")
    subprocess.run(["wget", "--no-check-certificate", "-q", "-O", tbi_local, tbi_url], check=True)
    print(f"  Done ({os.path.getsize(vcf_local)/1e9:.1f} GB)")
    return vcf_local


def load_metadata():
    meta_local = os.path.join(OUTDIR, "hgdp_metadata.txt")
    if not os.path.exists(meta_local):
        os.makedirs(OUTDIR, exist_ok=True)
        subprocess.run(["wget", "--no-check-certificate", "-q", "-O", meta_local, META_URL], check=True)
    sample_pop = {}
    sample_region = {}
    with open(meta_local) as f:
        header = f.readline().strip().split("\t")
        # Find columns by name
        pop_col = header.index("population") if "population" in header else 5
        region_col = header.index("region") if "region" in header else 8
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) > max(pop_col, region_col):
                sid = fields[0]
                sample_pop[sid] = fields[pop_col]
                sample_region[sid] = fields[region_col]
    return sample_pop, sample_region


def get_vcf_samples(vcf_path):
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#CHROM"):
                return line.strip().split("\t")[9:]
    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene", required=True, choices=list(GENE_INFO.keys()))
    args = parser.parse_args()

    gene = args.gene
    chrn, gene_start, gene_end, reg_start, reg_end, focal_region = GENE_INFO[gene]
    focal_label = REGION_MAP.get(focal_region, focal_region)

    print(f"=== HGDP Validation: {gene} (chr{chrn}:{gene_start}-{gene_end}) ===")
    print(f"  Focal region: {focal_label}")

    # Download
    vcf_path = download_vcf(chrn)

    # Metadata
    sample_pop, sample_region = load_metadata()
    samples = get_vcf_samples(vcf_path)
    print(f"  {len(samples)} samples in VCF, {len(sample_pop)} in metadata")

    # Build region -> sample indices
    region_si = {}
    for i, s in enumerate(samples):
        r = sample_region.get(s, "unknown")
        region_si.setdefault(r, []).append(i)

    print(f"  Regions: {', '.join(f'{r}({len(v)})' for r, v in sorted(region_si.items()))}")

    # Find focal region
    focal_key = None
    for r in region_si:
        if focal_label.lower() in r.lower():
            focal_key = r
            break
    if focal_key is None:
        print(f"  ERROR: Could not match region '{focal_label}'")
        print(f"  Available: {list(region_si.keys())}")
        return
    focal_si = region_si[focal_key]
    print(f"  Focal match: '{focal_key}' ({len(focal_si)} samples)")

    # Determine chromosome prefix in VCF
    # Try reading a small region to figure out if chr prefix is used
    chr_prefix = ""
    for prefix in [f"chr{chrn}", str(chrn)]:
        try:
            test = allel.read_vcf(vcf_path, region=f"{prefix}:{gene_start}-{gene_end}",
                                  fields=["variants/POS"], numbers={"GT": 2})
            if test is not None and len(test["variants/POS"]) > 0:
                chr_prefix = prefix
                break
        except Exception:
            continue
    if not chr_prefix:
        print(f"  ERROR: Could not read region from VCF")
        return
    print(f"  Using chromosome prefix: '{chr_prefix}'")

    # === H12 ===
    # Use a single focal population (not entire superpopulation) to match 1KG methodology
    # Pick the largest population within the focal region
    focal_pops_in_region = {}
    for i in focal_si:
        pop = sample_pop.get(samples[i], "unknown")
        focal_pops_in_region.setdefault(pop, []).append(i)
    best_pop = max(focal_pops_in_region, key=lambda p: len(focal_pops_in_region[p]))
    best_pop_si = focal_pops_in_region[best_pop]
    print(f"\n  Computing H12 in {best_pop} ({len(best_pop_si)} samples, {len(best_pop_si)*2} haplotypes)...")

    mid = (gene_start + gene_end) // 2
    h12, h2h1 = None, None
    try:
        callset = allel.read_vcf(
            vcf_path,
            region=f"{chr_prefix}:{max(1, mid - 2_000_000)}-{mid + 2_000_000}",
            fields=["variants/POS", "variants/REF", "variants/ALT", "calldata/GT"],
            numbers={"ALT": 1, "GT": 2},
            types={"calldata/GT": "i1"},
        )
        if callset is not None:
            gt_all = allel.GenotypeArray(callset["calldata/GT"])
            pos = callset["variants/POS"]
            ref = callset["variants/REF"]
            alt = callset["variants/ALT"]

            # Filter: biallelic SNPs only (single-char REF and ALT)
            is_snp = np.array([len(r) == 1 and len(a) == 1 and a != ""
                               for r, a in zip(ref, alt)])
            # Also filter sites with any missing (-1) in focal pop
            gt_focal = gt_all.take(best_pop_si, axis=1)
            gt_focal_arr = np.array(gt_focal)
            no_missing = ~np.any(gt_focal_arr < 0, axis=(1, 2))
            keep = is_snp & no_missing

            gt_filt = gt_focal.compress(keep, axis=0)
            pos_filt = pos[keep]
            print(f"  {keep.sum()} biallelic SNPs in ±2Mb window (from {len(pos)} total variants)")

            haps = gt_filt.to_haplotypes()

            center_idx = np.searchsorted(pos_filt, mid)
            s = max(0, center_idx - WINDOW_SNPS // 2)
            e = min(len(pos_filt), s + WINDOW_SNPS)

            hap_window = haps[s:e]
            n_haps = hap_window.shape[1]
            print(f"  Window: {pos_filt[s]}-{pos_filt[e-1]} ({e-s} SNPs, {(pos_filt[e-1]-pos_filt[s])/1e6:.2f} Mb)")

            # Convert to haplotype strings and count
            hap_strs = ["".join(str(x) for x in hap_window[:, i]) for i in range(n_haps)]
            counts = Counter(hap_strs)
            freqs = np.array(sorted(counts.values(), reverse=True)) / n_haps
            n_unique = len(counts)

            h1 = np.sum(freqs**2)
            h12 = h1 + 2 * freqs[0] * freqs[1] if len(freqs) > 1 else h1
            h2 = h1 - freqs[0]**2
            h2h1 = h2 / h1 if h1 > 0 else 0

            print(f"  H12 = {h12:.4f}, H2/H1 = {h2h1:.4f}")
            print(f"  {n_unique} unique haplotypes from {n_haps}, top freq = {freqs[0]:.4f}")
            del gt_all, gt_focal, gt_filt, haps, callset
        else:
            print(f"  Could not load H12 region")
    except Exception as ex:
        import traceback
        print(f"  H12 error: {ex}")
        traceback.print_exc()

    # === Allele frequencies + depletion/enrichment + FST ===
    print(f"\n  Computing variant stats at gene body...")
    try:
        callset = allel.read_vcf(
            vcf_path,
            region=f"{chr_prefix}:{gene_start}-{gene_end}",
            fields=["variants/POS", "calldata/GT"],
            numbers={"GT": 2},
        )
        if callset is not None and len(callset["variants/POS"]) > 0:
            gt = allel.GenotypeArray(callset["calldata/GT"])
            pos = callset["variants/POS"]
            n_sites = len(pos)

            # Per-region allele frequencies
            afs = {}
            for r, si_list in region_si.items():
                if len(si_list) > 0:
                    h = gt.take(si_list, axis=1).to_haplotypes()
                    afs[r] = h.mean(axis=1)

            focal_af = afs[focal_key]
            other_afs = {r: af for r, af in afs.items() if r != focal_key}

            # Depletion/enrichment
            max_other = np.max(list(other_afs.values()), axis=0)
            all_other_below_30 = np.all([af < 0.30 for af in other_afs.values()], axis=0)

            depleted = int(((focal_af < 0.10) & (max_other > 0.30)).sum())
            enriched = int(((focal_af > 0.50) & all_other_below_30).sum())

            # FST (focal vs all others)
            nonfocal_si = [i for r, si_list in region_si.items() if r != focal_key for i in si_list]
            ac1 = gt.take(focal_si, axis=1).count_alleles()
            ac2 = gt.take(nonfocal_si, axis=1).count_alleles()
            num, den = allel.hudson_fst(ac1, ac2)
            per_site_fst = np.where(den > 0, num / den, 0)
            per_site_fst = np.clip(per_site_fst, 0, 1)
            max_fst = float(per_site_fst.max())

            print(f"  {n_sites} sites in gene body")
            print(f"  Depleted:Enriched = {depleted}:{enriched}")
            print(f"  Max FST = {max_fst:.3f}")

            # Print per-region AFs at max FST site
            max_fst_idx = per_site_fst.argmax()
            print(f"  Max FST variant (pos {pos[max_fst_idx]}):")
            for r in sorted(afs.keys()):
                print(f"    {r}: AF = {afs[r][max_fst_idx]:.3f}")
        else:
            print(f"  No variants in gene body")
            depleted = enriched = 0
            max_fst = float("nan")
    except Exception as ex:
        print(f"  Variant stats error: {ex}")
        depleted = enriched = 0
        max_fst = float("nan")

    # Save result
    out = os.path.join(OUTDIR, f"validation_{gene}.txt")
    with open(out, "w") as f:
        f.write(f"gene\t{gene}\n")
        f.write(f"chr\t{chrn}\n")
        f.write(f"focal_region\t{focal_key}\n")
        f.write(f"h12\t{h12}\n")
        f.write(f"h2h1\t{h2h1}\n")
        f.write(f"depleted\t{depleted}\n")
        f.write(f"enriched\t{enriched}\n")
        f.write(f"max_fst\t{max_fst}\n")
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
