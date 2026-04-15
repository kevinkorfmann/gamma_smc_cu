#!/usr/bin/env python3
"""Why doesn't 1000G EAS show a TMRCA trough at chr11q13 despite Voight 2006's
HapMap CHB+JPT iHS=3.55 at the neighbour genes?

Three candidate explanations:
 (a) Voight's iHS hit was a HapMap-era artefact (low SNP density → spurious
     long haplotype): check whether the high-frequency "sweep allele" exists
     in EAS at all.
 (b) Same sweep is in EAS but has decayed more than in SAS/EUR: expect EAS
     sweep-allele frequency between AFR and SAS, haplotype similarity to
     SAS/EUR preserved.
 (c) Distinct sweep events at an overlapping haplotype: expect EAS sweep
     haplotypes to differ substantially from SAS sweep haplotypes.

Test: use the ±25 kb window around chr11:67,407,126 already cached from
verify/22_grk2_haplotype_sharing.py. Report:
 - per-population sweep-allele frequency in EAS (CDX/CHB/CHS/JPT/KHV)
 - mean pairwise Hamming distance rate: EAS-sweep × EAS-sweep,
   EAS-sweep × SAS-sweep, EAS-sweep × AFR-sweep, EAS-sweep × EAS-non-sweep
 - ratio SAS-EAS / SAS-SAS (= 1 under shared haplotype; >> 1 under distinct)
"""
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path("/Users/kevinkorfmann/Projects/tmrca.cu")
SAMPLES_FILE = REPO / "analysis/genome_wide/data/samples.txt"
CACHE_VCF = Path("/tmp/grk2_hapshare/grk2_win_25000.vcf")
VCF_URL = ("http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
           "1000G_2504_high_coverage/working/"
           "20220422_3202_phased_SNV_INDEL_SV/"
           "1kGP_high_coverage_Illumina.chr11.filtered.SNV_INDEL_SV_phased_panel.vcf.gz")
FOCAL_POS = 67_407_126
WINDOW_BP = 25_000
MAX_N = 200
SEED = 42

if not CACHE_VCF.exists():
    CACHE_VCF.parent.mkdir(parents=True, exist_ok=True)
    region = f"chr11:{FOCAL_POS - WINDOW_BP}-{FOCAL_POS + WINDOW_BP}"
    print(f"Downloading {region} ...")
    with open(CACHE_VCF, "w") as fh:
        subprocess.check_call(["tabix", "-h", VCF_URL, region], stdout=fh)

# Parse VCF.
sample_ids, variants = [], []
with open(CACHE_VCF) as f:
    for line in f:
        if line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            sample_ids = line.strip().split("\t")[9:]; continue
        flds = line.rstrip().split("\t")
        pos = int(flds[1]); ref, alt = flds[3], flds[4]
        if len(ref) != 1 or len(alt) != 1: continue
        gts = flds[9:]
        h = np.zeros(2 * len(gts), dtype=np.int8)
        for i, gt in enumerate(gts):
            a, b = gt.split("|"); h[2*i] = int(a); h[2*i+1] = int(b)
        variants.append((pos, ref, alt, h))
pos_arr = np.array([v[0] for v in variants])
H = np.stack([v[3] for v in variants], axis=1)
print(f"Window: {len(variants)} SNVs × {H.shape[0]} haps")

sample_map = pd.read_csv(SAMPLES_FILE, sep=r"\s+").set_index("SampleID")
hap_super = np.repeat([sample_map.loc[s, "Superpopulation"] if s in sample_map.index else "?" for s in sample_ids], 2)
hap_pop   = np.repeat([sample_map.loc[s, "Population"]      if s in sample_map.index else "?" for s in sample_ids], 2)

focal_idx = int(np.where(pos_arr == FOCAL_POS)[0][0])
sweep_mask = (H[:, focal_idx] == 0)  # sweep allele = REF=G (per previous analysis)

print("\n=== Sweep-allele (REF=G @ chr11:67,407,126) carrier frequency ===")
print(f"{'group':<12} {'sweep':>6}/{'total':>5}  {'freq':>6}")
for sp in ["AFR", "EUR", "SAS", "EAS", "AMR"]:
    m = (hap_super == sp)
    sw = int((sweep_mask & m).sum())
    tot = int(m.sum())
    print(f"{sp:<12} {sw:>6}/{tot:>5}  {100*sw/tot:>5.1f}%")

print("\n=== EAS per-population sweep-allele frequency ===")
for pop in ["CDX", "CHB", "CHS", "JPT", "KHV"]:
    m = (hap_pop == pop)
    sw = int((sweep_mask & m).sum())
    tot = int(m.sum())
    print(f"  {pop}: {sw:>4}/{tot:<4} ({100*sw/tot:.1f}%)")

# Haplotype sharing. Exclude focal site.
non_focal = np.ones(H.shape[1], dtype=bool); non_focal[focal_idx] = False
H_nf = H[:, non_focal]

def hamming_rate(A, B, same_pool=False):
    if A.shape[0] == 0 or B.shape[0] == 0:
        return float("nan")
    a1 = A.sum(axis=0); a0 = A.shape[0] - a1
    b1 = B.sum(axis=0); b0 = B.shape[0] - b1
    total_disagree = int((a1 * b0 + a0 * b1).sum())
    n_pairs = A.shape[0] * (A.shape[0] - 1) if same_pool else A.shape[0] * B.shape[0]
    if n_pairs == 0: return float("nan")
    return total_disagree / (n_pairs * A.shape[1])

def subsample(Hx, rng, n=MAX_N):
    if Hx.shape[0] <= n: return Hx
    return Hx[rng.choice(Hx.shape[0], n, replace=False)]

rng = np.random.default_rng(SEED)
pools = {
    "SAS_sweep":    H_nf[(hap_super == "SAS") & sweep_mask],
    "EUR_sweep":    H_nf[(hap_super == "EUR") & sweep_mask],
    "EAS_sweep":    H_nf[(hap_super == "EAS") & sweep_mask],
    "AFR_sweep":    H_nf[(hap_super == "AFR") & sweep_mask],
    "EAS_nonsweep": H_nf[(hap_super == "EAS") & ~sweep_mask],
}
print(f"\nPool sizes: {{k: v.shape[0] for k, v in pools.items()}}")
for k, v in pools.items():
    print(f"  {k:<14} n={v.shape[0]}")

sub = {k: subsample(v, rng) for k, v in pools.items()}
d_sas_sas = hamming_rate(sub["SAS_sweep"], sub["SAS_sweep"], same_pool=True)
d_eas_eas = hamming_rate(sub["EAS_sweep"], sub["EAS_sweep"], same_pool=True)
d_sas_eas = hamming_rate(sub["SAS_sweep"], sub["EAS_sweep"])
d_eas_afr = hamming_rate(sub["EAS_sweep"], sub["AFR_sweep"])
d_eas_eur = hamming_rate(sub["EAS_sweep"], sub["EUR_sweep"])
d_eas_noneas = hamming_rate(sub["EAS_sweep"], sub["EAS_nonsweep"])

print(f"\n=== Mean per-site Hamming distance rate ===")
for label, v in [
    ("within SAS-sweep",    d_sas_sas),
    ("within EAS-sweep",    d_eas_eas),
    ("SAS-sweep × EAS-sweep", d_sas_eas),
    ("EAS-sweep × EUR-sweep", d_eas_eur),
    ("EAS-sweep × AFR-sweep", d_eas_afr),
    ("EAS-sweep × EAS-nonsweep (local control)", d_eas_noneas),
]:
    print(f"  {label:<42} {v:.4f}")

print(f"\n=== Key ratios ===")
print(f"  SAS × EAS / within-SAS       = {d_sas_eas / d_sas_sas:.2f}  "
      f"(~1 = same haplotype family; >>1 = distinct sweeps)")
print(f"  EAS-nonsweep / within-EAS    = {d_eas_noneas / d_eas_eas:.2f}  "
      f"(>>1 = EAS sweep haplotypes cluster tightly — sweep is real in EAS)")
print(f"  EAS × AFR / within-EAS       = {d_eas_afr / d_eas_eas:.2f}")

# Save a JSON with everything.
import json
out = {
    "focal_chr11_67407126_sweep_allele_freq": {
        sp: float((sweep_mask & (hap_super == sp)).sum() / max((hap_super == sp).sum(), 1))
        for sp in ["AFR", "EUR", "SAS", "EAS", "AMR"]
    },
    "eas_per_population_sweep_freq": {
        pop: float((sweep_mask & (hap_pop == pop)).sum() / max((hap_pop == pop).sum(), 1))
        for pop in ["CDX", "CHB", "CHS", "JPT", "KHV"]
    },
    "hamming_distance_rates": {
        "within_SAS_sweep":     d_sas_sas,
        "within_EAS_sweep":     d_eas_eas,
        "SAS_sweep_x_EAS_sweep": d_sas_eas,
        "EAS_sweep_x_EUR_sweep": d_eas_eur,
        "EAS_sweep_x_AFR_sweep": d_eas_afr,
        "EAS_sweep_x_EAS_nonsweep": d_eas_noneas,
    },
    "ratios": {
        "SAS_x_EAS_over_within_SAS":       d_sas_eas / d_sas_sas,
        "EAS_nonsweep_over_within_EAS":    d_eas_noneas / d_eas_eas,
        "EAS_x_AFR_over_within_EAS":       d_eas_afr / d_eas_eas,
    },
    "pool_sizes": {k: int(v.shape[0]) for k, v in pools.items()},
    "n_snvs_in_window": int(H.shape[1]),
    "focal_pos": FOCAL_POS,
    "window_bp": WINDOW_BP,
}
out_path = Path(__file__).parent / "grk2_eas_haplotype_sharing.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nwrote: {out_path}")
