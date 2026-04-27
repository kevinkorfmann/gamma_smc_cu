#!/usr/bin/env python3
"""Compare TREM cluster vs FOXP4 per-population ranks in our genome-wide scan,
and (b) run H12 in the swept region for IBS, CEU, TSI, GBR, FIN, JPT, YRI.

Purpose:
- If FOXP4 also ranks low in IBS (independent of the TREM cluster peak), the
  signal extends across the whole 400 kb block -> cluster-level sweep.
- If only TREM cluster ranks low, FOXP4 is unrelated bystander.
- H12 profile across EUR pops distinguishes IBS-specific vs shared EUR signal.
"""
from __future__ import annotations

import gzip
import sys
from collections import Counter

import numpy as np
import pandas as pd

RANKS = "/Users/kevinkorfmann/Projects/tmrca.cu/analysis/genome_wide/results/genome_wide_ranks.csv"
VCF = "/Users/kevinkorfmann/Projects/tmrca.cu/private/manuscript/v5/figures/data/trem2_pm500kb.vcf.gz"
PANEL_SAMPLES = {
    # 1KG NYGC high-coverage IDs from the v5 trem2 dive script for IBS,
    # plus we'll build others from allele-frequency panel info parsed from VCF.
    "IBS": "HG01500 HG01501 HG01503 HG01504 HG01506 HG01507 HG01509 HG01510 HG01512 HG01513 HG01515 HG01516 HG01518 HG01519 HG01521 HG01522 HG01524 HG01525 HG01527 HG01528 HG01530 HG01531 HG01536 HG01537 HG01602 HG01603 HG01605 HG01606 HG01608 HG01609 HG01610 HG01612 HG01613 HG01615 HG01616 HG01618 HG01619 HG01620 HG01623 HG01624 HG01625 HG01626 HG01628 HG01630 HG01631 HG01632 HG01668 HG01669 HG01670 HG01672 HG01673 HG01675 HG01676 HG01678 HG01679 HG01680 HG01682 HG01684 HG01685 HG01686 HG01694 HG01697 HG01699 HG01700 HG01702 HG01704 HG01705 HG01707 HG01708 HG01709 HG01710 HG01746 HG01747 HG01756 HG01757 HG01761 HG01762 HG01765 HG01766 HG01767 HG01768 HG01770 HG01771 HG01773 HG01775 HG01776 HG01777 HG01779 HG01781 HG01783 HG01784 HG01786 HG01787 HG01789 HG01790 HG01791 HG02220 HG02221 HG02223 HG02224 HG02231 HG02232 HG02235 HG02236 HG02238 HG02239".split(),
}

# Pull panel from NYGC 1KG panel file distributed with 1KG project
# (Prefix the pixi env default if needed; we'll use a URL fetch fallback)
PANEL_LOCAL = "/tmp/1kgp_panel.txt"

GENES_6P21 = [
    ("TREML1", 41149337, 41154347),
    ("TREM2", 41158506, 41163186),
    ("TREML2", 41189749, 41201149),
    ("TREML4", 41228339, 41238882),
    ("TREM1", 41267926, 41286682),
    ("NCR2", 41335608, 41350889),
    ("FOXP4", 41546381, 41602384),
]


def load_panel():
    # Minimal fallback: use VCF header + sample IDs in IBS hard-coded above.
    # Also parse the 1kGP panel file if present (list of pops per sample).
    import os
    pop_of = {}
    with open(PANEL_LOCAL) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] == "sample":
                continue
            if len(parts) >= 2 and len(parts[1]) == 3:
                pop_of[parts[0]] = parts[1]
    return pop_of


def print_ranks():
    df = pd.read_csv(RANKS)
    df = df[df.gene_name.isin([g for g, _, _ in GENES_6P21])]
    rank_cols = [c for c in df.columns if c.endswith("_rank")]
    print("\n=== Genome-wide rank (%) across populations for 6p21.1 genes ===\n")
    print("(lower = rarer TMRCA at this gene, suggesting selection)\n")
    # print per pop
    eur = ["CEU", "FIN", "GBR", "IBS", "TSI"]
    eas = ["CDX", "CHB", "CHS", "JPT", "KHV"]
    afr = ["YRI", "LWK"]
    sas = ["GIH", "BEB"]
    hdr = ["gene"] + eur + eas + afr + sas
    print(" ".join(f"{h:>7}" for h in hdr))
    for _, r in df.iterrows():
        row = [r["gene_name"]]
        for p in eur + eas + afr + sas:
            v = r.get(f"{p}_rank")
            row.append(f"{v*100:6.2f}" if pd.notna(v) else "     -")
        print(" ".join(f"{c:>7}" for c in row))


def load_haps(vcf_path, lo, hi, samples_keep):
    positions, haps_rows, sample_cols, kept_samples = [], [], None, []
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                header = line.rstrip("\n").split("\t")
                samples = header[9:]
                sample_cols = [i for i, s in enumerate(samples) if s in samples_keep]
                kept_samples = [samples[i] for i in sample_cols]
                continue
            parts = line.rstrip("\n").split("\t")
            pos = int(parts[1])
            if pos < lo:
                continue
            if pos > hi:
                break
            ref, alt = parts[3], parts[4]
            if len(ref) != 1 or len(alt) != 1 or "," in alt:
                continue
            gts = parts[9:]
            row = np.zeros(len(sample_cols) * 2, dtype=np.uint8)
            bad = False
            for i, ci in enumerate(sample_cols):
                g = gts[ci]
                if g[0] == "." or g[2] == ".":
                    bad = True; break
                row[2 * i] = 0 if g[0] == "0" else 1
                row[2 * i + 1] = 0 if g[2] == "0" else 1
            if bad:
                continue
            positions.append(pos)
            haps_rows.append(row)
    positions = np.array(positions)
    if not haps_rows:
        return positions, np.zeros((0,0), dtype=np.uint8)
    haps = np.array(haps_rows, dtype=np.uint8).T
    return positions, haps


def h12_track(G, positions, win_snps=400, step_snps=50):
    n_haps, n_snps = G.shape
    if n_snps < win_snps:
        return np.array([]), np.array([])
    mids, vals = [], []
    for s in range(0, n_snps - win_snps + 1, step_snps):
        e = s + win_snps
        window = np.ascontiguousarray(G[:, s:e])
        tuples = [row.tobytes() for row in window]
        counts = Counter(tuples)
        freqs = np.array(sorted(counts.values(), reverse=True)) / n_haps
        h12 = (freqs[0] + freqs[1]) ** 2 + np.sum(freqs[2:] ** 2) if len(freqs) >= 2 else 1.0
        mids.append((positions[s] + positions[e - 1]) / 2)
        vals.append(h12)
    return np.array(mids), np.array(vals)


def print_h12(pop_of):
    pops_eur = ["CEU", "FIN", "GBR", "IBS", "TSI"]
    pops_other = ["JPT", "CHB", "YRI"]
    pops = pops_eur + pops_other
    print("\n=== H12 at chr6:41.0-41.7 Mb per population ===\n")
    print(f"{'pop':>6} {'n_hap':>6} {'max_H12':>8} {'H12_at_peak':>12} {'peak_pos_Mb':>13}")
    LO, HI = 41_000_000, 41_700_000
    focal = 41_470_132
    for p in pops:
        if p == "IBS":
            samples = set(PANEL_SAMPLES["IBS"])
        else:
            samples = {s for s, pp in pop_of.items() if pp == p}
        if len(samples) < 30:
            print(f"{p:>6} {len(samples):>6}  (too few samples in VCF panel)")
            continue
        positions, G = load_haps(VCF, LO, HI, samples)
        if G.size == 0:
            print(f"{p:>6} {len(samples):>6}  (no variants)")
            continue
        mids, vals = h12_track(G, positions, win_snps=400, step_snps=50)
        if len(vals) == 0:
            continue
        imax = int(np.argmax(vals))
        # H12 at focal
        ifocal = int(np.argmin(np.abs(mids - focal)))
        h12_peak = vals[imax]
        h12_atfocal = vals[ifocal]
        peak_mb = mids[imax] / 1e6
        print(f"{p:>6} {G.shape[0]:>6} {h12_peak:>8.3f} {h12_atfocal:>12.3f} {peak_mb:>13.3f}")


def main():
    print_ranks()
    pop_of = load_panel()
    if not pop_of:
        print("[warn] no panel file found; running H12 only for IBS", file=sys.stderr)
        pop_of = {s: "IBS" for s in PANEL_SAMPLES["IBS"]}
    print_h12(pop_of)


if __name__ == "__main__":
    main()
