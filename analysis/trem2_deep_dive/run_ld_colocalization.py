#!/usr/bin/env python3
"""Compute LD (r^2, D') in IBS between our sweep peak and FOXP4 GWAS/eQTL lead.

Inputs (local, already present):
  - private/manuscript/v5/figures/data/trem2_pm500kb.vcf.gz (1KG NYGC chr6)
  - Hard-coded IBS sample IDs from the existing v5 fig script

Focal variants:
  - Sweep peak (ours):          chr6:41,485,209 (most-differentiated in TREM2_IBS.json)
  - FOXP4 sweep-region peak:    chr6:41,470,132 (focal in trem2_haplotype_sharing.json)
  - Akbari FOXP4-intron subthr: chr6:41,522,016 (rs11760063, POSTERIOR=0.75)
  - FOXP4 long-COVID GWAS lead: chr6:41,515,652 (rs9367106)
  - FOXP4 severe-COVID GWAS:    chr6:41,515,239 (rs2496644 — closest we can find)
"""
from __future__ import annotations

import gzip
import os
import sys

import numpy as np

VCF = "/Users/kevinkorfmann/Projects/tmrca.cu/private/manuscript/v5/figures/data/trem2_pm500kb.vcf.gz"

IBS = set(
    "HG01500 HG01501 HG01503 HG01504 HG01506 HG01507 HG01509 HG01510 HG01512 HG01513 "
    "HG01515 HG01516 HG01518 HG01519 HG01521 HG01522 HG01524 HG01525 HG01527 HG01528 "
    "HG01530 HG01531 HG01536 HG01537 HG01602 HG01603 HG01605 HG01606 HG01608 HG01609 "
    "HG01610 HG01612 HG01613 HG01615 HG01616 HG01618 HG01619 HG01620 HG01623 HG01624 "
    "HG01625 HG01626 HG01628 HG01630 HG01631 HG01632 HG01668 HG01669 HG01670 HG01672 "
    "HG01673 HG01675 HG01676 HG01678 HG01679 HG01680 HG01682 HG01684 HG01685 HG01686 "
    "HG01694 HG01697 HG01699 HG01700 HG01702 HG01704 HG01705 HG01707 HG01708 HG01709 "
    "HG01710 HG01746 HG01747 HG01756 HG01757 HG01761 HG01762 HG01765 HG01766 HG01767 "
    "HG01768 HG01770 HG01771 HG01773 HG01775 HG01776 HG01777 HG01779 HG01781 HG01783 "
    "HG01784 HG01786 HG01787 HG01789 HG01790 HG01791 HG02220 HG02221 HG02223 HG02224 "
    "HG02231 HG02232 HG02235 HG02236 HG02238 HG02239".split()
)

TARGETS = {
    "sweep_peak_41485209":     41_485_209,   # our most-differentiated
    "focal_41470132":          41_470_132,   # haplotype-sharing focal
    "rs11760063_41522016":     41_522_016,   # Akbari subthreshold, FOXP4 intron
    "rs9367106_41515652":      41_515_652,   # long-COVID GWAS lead
    "rs2496644_approx":        41_515_239,   # severe-COVID lead (approximate)
}


def extract_haps(vcf_path: str, positions: dict, sample_set: set):
    """Return dict name -> np.array (2*n_sample,) of 0/1 haplotype states, plus sample list."""
    wanted_pos = set(positions.values())
    found = {}  # pos -> np.array
    ibs_cols = None
    sample_names = None
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                header = line.rstrip("\n").split("\t")
                samples = header[9:]
                ibs_cols = [i for i, s in enumerate(samples) if s in sample_set]
                sample_names = [samples[i] for i in ibs_cols]
                print(f"[info] {len(ibs_cols)} IBS samples matched from {len(samples)} total", file=sys.stderr)
                continue
            fields = line.rstrip("\n").split("\t")
            pos = int(fields[1])
            if pos not in wanted_pos:
                continue
            genos = fields[9:]
            hap = []
            for i in ibs_cols:
                gt = genos[i].split(":")[0]
                if "|" in gt:
                    a, b = gt.split("|")
                elif "/" in gt:
                    a, b = gt.split("/")
                else:
                    a = b = "."
                hap.append(0 if a == "0" else (1 if a == "1" else -1))
                hap.append(0 if b == "0" else (1 if b == "1" else -1))
            found[pos] = np.array(hap, dtype=np.int8)
            if len(found) == len(wanted_pos):
                break
    return found, sample_names


def r2_and_d(h1: np.ndarray, h2: np.ndarray):
    mask = (h1 >= 0) & (h2 >= 0)
    a, b = h1[mask], h2[mask]
    n = len(a)
    pA = a.mean()
    pB = b.mean()
    pAB = ((a == 1) & (b == 1)).mean()
    D = pAB - pA * pB
    denom = pA * (1 - pA) * pB * (1 - pB)
    r2 = (D ** 2) / denom if denom > 0 else float("nan")
    # D'
    if D >= 0:
        Dmax = min(pA * (1 - pB), (1 - pA) * pB)
    else:
        Dmax = min(pA * pB, (1 - pA) * (1 - pB))
    Dprime = D / Dmax if Dmax > 0 else float("nan")
    return {"r2": r2, "D": D, "Dprime": Dprime, "pA": pA, "pB": pB, "pAB": pAB, "n_haps": n}


def main():
    haps, samps = extract_haps(VCF, TARGETS, IBS)
    print(f"[info] Variants found at positions: {sorted(haps.keys())}", file=sys.stderr)
    for name, pos in TARGETS.items():
        if pos not in haps:
            print(f"[warn] {name} (chr6:{pos}) NOT in VCF", file=sys.stderr)

    print("\n=== LD in IBS between all target pairs (r^2, D') ===\n")
    keys = [(name, pos) for name, pos in TARGETS.items() if pos in haps]
    print(f"{'var1':>25} {'var2':>25} {'r2':>8} {'Dprime':>8} {'pA':>6} {'pB':>6} {'n':>5}")
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            n1, p1 = keys[i]; n2, p2 = keys[j]
            s = r2_and_d(haps[p1], haps[p2])
            print(f"{n1:>25} {n2:>25} {s['r2']:8.3f} {s['Dprime']:8.3f} {s['pA']:6.3f} {s['pB']:6.3f} {s['n_haps']:5d}")


if __name__ == "__main__":
    main()
