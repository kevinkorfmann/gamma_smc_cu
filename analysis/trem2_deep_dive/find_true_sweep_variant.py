#!/usr/bin/env python3
"""Find the true sweep variant inside the 41.15 Mb H12 peak.

Strategy: among common variants (MAF>=5%) in the window 41.10-41.20 Mb,
rank by:
  - IBS vs non-AFR dAF (|AF_IBS - AF_non-AFR_mean|) — elevated in IBS
  - In-IBS frequency being intermediate/high (sweep allele should be frequent)
  - Highest r^2 to the top H12 haplotype identity vector

Also compute dAF for CEU, FIN, GBR, TSI, JPT to identify whether any variant
is specifically elevated in IBS vs other non-AFR.
"""
from __future__ import annotations

import gzip
import sys

import numpy as np

VCF = "/Users/kevinkorfmann/Projects/tmrca.cu/private/manuscript/v5/figures/data/trem2_pm500kb.vcf.gz"

# Load panel
PANEL = "/tmp/1kgp_panel.txt"
pop_of = {}
with open(PANEL) as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] != "sample":
            pop_of[parts[0]] = parts[1]

# IBS samples from v5 figure script
IBS_IDS = set(
    "HG01500 HG01501 HG01503 HG01504 HG01506 HG01507 HG01509 HG01510 HG01512 HG01513 HG01515 HG01516 HG01518 HG01519 HG01521 HG01522 HG01524 HG01525 HG01527 HG01528 HG01530 HG01531 HG01536 HG01537 HG01602 HG01603 HG01605 HG01606 HG01608 HG01609 HG01610 HG01612 HG01613 HG01615 HG01616 HG01618 HG01619 HG01620 HG01623 HG01624 HG01625 HG01626 HG01628 HG01630 HG01631 HG01632 HG01668 HG01669 HG01670 HG01672 HG01673 HG01675 HG01676 HG01678 HG01679 HG01680 HG01682 HG01684 HG01685 HG01686 HG01694 HG01697 HG01699 HG01700 HG01702 HG01704 HG01705 HG01707 HG01708 HG01709 HG01710 HG01746 HG01747 HG01756 HG01757 HG01761 HG01762 HG01765 HG01766 HG01767 HG01768 HG01770 HG01771 HG01773 HG01775 HG01776 HG01777 HG01779 HG01781 HG01783 HG01784 HG01786 HG01787 HG01789 HG01790 HG01791 HG02220 HG02221 HG02223 HG02224 HG02231 HG02232 HG02235 HG02236 HG02238 HG02239".split()
)

POPS = {"IBS": IBS_IDS}
for p in ["CEU","FIN","GBR","TSI","JPT","CHB","CHS","CDX","KHV","BEB","GIH","STU","ITU","PJL","YRI","LWK"]:
    POPS[p] = {s for s, pp in pop_of.items() if pp == p}

SUPER = {
    "EUR": ["CEU","FIN","GBR","IBS","TSI"],
    "EAS": ["CDX","CHB","CHS","JPT","KHV"],
    "SAS": ["BEB","GIH","ITU","PJL","STU"],
    "AFR": ["YRI","LWK"],
}

LO, HI = 41_100_000, 41_200_000  # around the H12 peak


def af_by_pop(vcf_path, lo, hi):
    hdr_samples = None
    results = []
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                hdr_samples = line.rstrip("\n").split("\t")[9:]
                pop_cols = {p: [i for i, s in enumerate(hdr_samples) if s in ids]
                            for p, ids in POPS.items()}
                continue
            parts = line.rstrip("\n").split("\t")
            pos = int(parts[1])
            if pos < lo: continue
            if pos > hi: break
            ref, alt = parts[3], parts[4]
            if len(ref) != 1 or len(alt) != 1 or "," in alt:
                continue
            rsid = parts[2]
            gts = parts[9:]
            afs = {}
            for p, cols in pop_cols.items():
                a, n = 0, 0
                for ci in cols:
                    g = gts[ci]
                    if g[0] == "." or g[2] == ".": continue
                    a += (g[0] != "0") + (g[2] != "0")
                    n += 2
                afs[p] = a / n if n > 0 else None
            results.append((pos, rsid, alt, afs))
    return results


def main():
    rows = af_by_pop(VCF, LO, HI)
    print(f"[info] {len(rows)} biallelic SNPs in chr6:{LO}-{HI}", file=sys.stderr)

    scored = []
    for pos, rsid, alt, afs in rows:
        ibs = afs.get("IBS")
        if ibs is None: continue
        eur_other = [afs[p] for p in ["CEU","FIN","GBR","TSI"] if afs.get(p) is not None]
        eas = [afs[p] for p in SUPER["EAS"] if afs.get(p) is not None]
        afr = [afs[p] for p in SUPER["AFR"] if afs.get(p) is not None]
        sas = [afs[p] for p in SUPER["SAS"] if afs.get(p) is not None]
        if not eur_other or not afr: continue
        mean_eur_other = np.mean(eur_other)
        mean_afr = np.mean(afr)
        mean_eas = np.mean(eas) if eas else np.nan
        mean_sas = np.mean(sas) if sas else np.nan
        # MAF filter
        maf_ibs = min(ibs, 1 - ibs)
        if maf_ibs < 0.05: continue
        # delta IBS vs non-IBS EUR (IBS-specific?)
        dAF_ibs_eur = ibs - mean_eur_other
        # delta IBS vs AFR (OoA-shared?)
        dAF_ibs_afr = ibs - mean_afr
        scored.append({
            "pos": pos, "rsid": rsid, "alt": alt,
            "AF_IBS": ibs, "AF_EUR_other": mean_eur_other,
            "AF_EAS": mean_eas, "AF_SAS": mean_sas, "AF_AFR": mean_afr,
            "dAF_IBS_EURother": dAF_ibs_eur,
            "dAF_IBS_AFR": dAF_ibs_afr,
        })

    print(f"[info] {len(scored)} common (MAF>=5%) variants in IBS", file=sys.stderr)

    print("\n=== Top 15 variants by |dAF(IBS vs other-EUR)| (IBS-specific candidates) ===")
    print(f"{'pos':>10} {'rsid':>14} {'ALT':>4} {'IBS':>6} {'EUR*':>6} {'EAS':>6} {'SAS':>6} {'AFR':>6} {'dEUR':>7} {'dAFR':>7}")
    for r in sorted(scored, key=lambda x: abs(x["dAF_IBS_EURother"]), reverse=True)[:15]:
        print(f"{r['pos']:>10} {r['rsid'][:14]:>14} {r['alt']:>4} "
              f"{r['AF_IBS']:6.3f} {r['AF_EUR_other']:6.3f} "
              f"{r['AF_EAS']:6.3f} {r['AF_SAS']:6.3f} {r['AF_AFR']:6.3f} "
              f"{r['dAF_IBS_EURother']:+7.3f} {r['dAF_IBS_AFR']:+7.3f}")

    print("\n=== Top 15 variants by |dAF(IBS vs AFR)| (OoA sweep candidates) ===")
    print(f"{'pos':>10} {'rsid':>14} {'ALT':>4} {'IBS':>6} {'EUR*':>6} {'EAS':>6} {'SAS':>6} {'AFR':>6} {'dEUR':>7} {'dAFR':>7}")
    for r in sorted(scored, key=lambda x: abs(x["dAF_IBS_AFR"]), reverse=True)[:15]:
        print(f"{r['pos']:>10} {r['rsid'][:14]:>14} {r['alt']:>4} "
              f"{r['AF_IBS']:6.3f} {r['AF_EUR_other']:6.3f} "
              f"{r['AF_EAS']:6.3f} {r['AF_SAS']:6.3f} {r['AF_AFR']:6.3f} "
              f"{r['dAF_IBS_EURother']:+7.3f} {r['dAF_IBS_AFR']:+7.3f}")


if __name__ == "__main__":
    main()
