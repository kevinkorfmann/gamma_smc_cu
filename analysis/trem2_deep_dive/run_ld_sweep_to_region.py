#!/usr/bin/env python3
"""Scan r^2 in IBS between the sweep peak/focal and every common (MAF>=5%) variant
in the FOXP4 gene body (±20 kb).

Goal: is the sweep actually colocalized with ANY FOXP4 regulatory variant?
"""
from __future__ import annotations

import gzip
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

SWEEP_PEAK = 41_485_209    # most-differentiated ALT AF=0.176 in IBS
FOCAL = 41_470_132         # haplotype-sharing focal ALT AF=0.818 in IBS
# FOXP4 gene body ±20kb
FOXP4_LO = 41_546_363 - 20_000
FOXP4_HI = 41_602_384 + 20_000
# TREM cluster gene body (for comparison)
TREM_LO = 41_140_000
TREM_HI = 41_360_000


def r2(h1: np.ndarray, h2: np.ndarray) -> float:
    mask = (h1 >= 0) & (h2 >= 0)
    a, b = h1[mask], h2[mask]
    if len(a) == 0:
        return float("nan")
    pA, pB = a.mean(), b.mean()
    denom = pA * (1 - pA) * pB * (1 - pB)
    if denom == 0:
        return float("nan")
    pAB = ((a == 1) & (b == 1)).mean()
    D = pAB - pA * pB
    return (D ** 2) / denom


def stream(vcf_path, lo, hi, sample_set):
    with gzip.open(vcf_path, "rt") as f:
        ibs_cols = None
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                header = line.rstrip("\n").split("\t")
                samples = header[9:]
                ibs_cols = [i for i, s in enumerate(samples) if s in sample_set]
                yield ("HEAD", ibs_cols)
                continue
            fields = line.rstrip("\n").split("\t")
            pos = int(fields[1])
            if pos < lo:
                continue
            if pos > hi:
                break
            rsid = fields[2]
            alt = fields[4]
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
            yield (pos, rsid, alt, np.array(hap, dtype=np.int8))


def fetch(vcf_path, pos, sample_set):
    for rec in stream(vcf_path, pos - 1, pos + 1, sample_set):
        if rec[0] == "HEAD":
            continue
        p = rec[0]
        if p == pos:
            return rec[3]
    return None


def scan_region(name, lo, hi):
    print(f"\n=== {name} (chr6:{lo}-{hi}) ===", file=sys.stderr)
    sweep_hap = fetch(VCF, SWEEP_PEAK, IBS)
    focal_hap = fetch(VCF, FOCAL, IBS)
    if sweep_hap is None or focal_hap is None:
        print("ERROR: cannot find anchor variants", file=sys.stderr)
        return
    # Scan
    results = []
    for rec in stream(VCF, lo, hi, IBS):
        if rec[0] == "HEAD":
            continue
        pos, rsid, alt, hap = rec
        mask = hap >= 0
        if mask.sum() == 0:
            continue
        maf = min(hap[mask].mean(), 1 - hap[mask].mean())
        if maf < 0.05:
            continue
        r2_sweep = r2(hap, sweep_hap)
        r2_focal = r2(hap, focal_hap)
        results.append((pos, rsid, alt, maf, r2_sweep, r2_focal))
    results.sort(key=lambda x: max(x[4] if not np.isnan(x[4]) else 0,
                                    x[5] if not np.isnan(x[5]) else 0),
                 reverse=True)
    print(f"{'pos':>10} {'rsid':>14} {'ALT':>5} {'MAF':>6} {'r2_sweep':>9} {'r2_focal':>9}")
    for pos, rsid, alt, maf, rs, rf in results[:20]:
        rs_s = f"{rs:9.3f}" if not np.isnan(rs) else "      nan"
        rf_s = f"{rf:9.3f}" if not np.isnan(rf) else "      nan"
        print(f"{pos:>10} {rsid[:14]:>14} {alt[:5]:>5} {maf:6.3f} {rs_s} {rf_s}")
    n_high_sweep = sum(1 for r in results if not np.isnan(r[4]) and r[4] >= 0.5)
    n_high_focal = sum(1 for r in results if not np.isnan(r[5]) and r[5] >= 0.5)
    print(f"[{name}] {len(results)} common variants; r^2>=0.5 with sweep={n_high_sweep}, focal={n_high_focal}")


if __name__ == "__main__":
    scan_region("FOXP4", FOXP4_LO, FOXP4_HI)
    scan_region("TREM_cluster", TREM_LO, TREM_HI)
