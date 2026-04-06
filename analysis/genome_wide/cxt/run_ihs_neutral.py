#!/usr/bin/env python
"""Compute orthogonal stats for 5 neutral control genes (EAS mean rank ~50%)."""

import numpy as np
import allel
import os

BETTY_BASE = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide"
PARSED_DIR = os.path.join(BETTY_BASE, "cache/parsed")
SAMPLES_FILE = os.path.join(BETTY_BASE, "data/samples.txt")
OUTDIR = os.path.join(BETTY_BASE, "cxt/results/orthogonal")
os.makedirs(OUTDIR, exist_ok=True)

GENES = [
    ("TTBK1", 6, 43243481, 43288258, "EAS"),
    ("CCDC70", 13, 51861969, 51866232, "EAS"),
    ("CZIB", 1, 53214099, 53220634, "EAS"),
    ("PER1", 17, 8140472, 8156506, "EAS"),
    ("CUL1", 7, 148697914, 148801110, "EAS"),
]

EAS_POPS = ["CHB", "JPT", "CHS", "CDX", "KHV"]
WINDOW = 500_000


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
        for si in pop_indices[pop]:
            idx.extend([2 * si, 2 * si + 1])
    return np.array(idx)


def compute(gene_name, chrn, gs, ge, superpop):
    print(f"\n{'='*50}\n{gene_name} chr{chrn}:{gs}-{ge} (NEUTRAL CONTROL)")

    npz = np.load(os.path.join(PARSED_DIR, f"chr{chrn}.npz"))
    G, pos = npz["G"], npz["positions"]
    rs, re = max(0, gs - WINDOW), ge + WINDOW
    mask = (pos >= rs) & (pos <= re)
    pos_r, G_r = pos[mask], G[:, mask]
    print(f"  {mask.sum()} variants")

    pop_idx = load_pop_indices(SAMPLES_FILE)
    focal_pops = EAS_POPS
    nonfocal_pops = [p for p in pop_idx if p not in focal_pops]

    h_f = allel.HaplotypeArray(G_r[get_hap_idx(pop_idx, focal_pops)].T)
    h_nf = allel.HaplotypeArray(G_r[get_hap_idx(pop_idx, nonfocal_pops)].T)
    ac_f = h_f.count_alleles()

    # iHS
    is_seg = ac_f.is_segregating() & ac_f.is_biallelic()
    try:
        ihs_raw = allel.ihs(h_f[is_seg], pos_r[is_seg], use_threads=False)
        ihs_std = allel.standardize_by_allele_count(ihs_raw, ac_f[is_seg][:, 1], diagnostics=False)
        ihs_vals = np.abs(ihs_std[0])
        ihs_vals = ihs_vals[~np.isnan(ihs_vals)]
        max_ihs = float(np.nanmax(ihs_vals)) if len(ihs_vals) else 0
    except:
        max_ihs = 0

    # Tajima's D
    try:
        tajd = float(allel.tajima_d(ac_f, pos=pos_r))
    except:
        tajd = float('nan')

    # Pi ratio
    try:
        ac_nf = h_nf.count_alleles()
        pi_f = allel.sequence_diversity(pos_r, ac_f)
        pi_nf = allel.sequence_diversity(pos_r, ac_nf)
        pi_ratio = float(pi_f / pi_nf) if pi_nf > 0 else float('nan')
    except:
        pi_ratio = float('nan')

    # Max FST
    try:
        ac_nf = h_nf.count_alleles()
        num, den = allel.hudson_fst(ac_f, ac_nf)
        fst = num / den
        max_fst = float(np.nanmax(fst[~np.isnan(fst)])) if np.any(~np.isnan(fst)) else 0
    except:
        max_fst = float('nan')

    print(f"  max|iHS|={max_ihs:.2f}, Tajima D={tajd:.3f}, pi ratio={pi_ratio:.3f}, max FST={max_fst:.3f}")
    np.savez(os.path.join(OUTDIR, f"{gene_name}_neutral_orthogonal.npz"),
             gene=gene_name, chr=chrn, max_abs_ihs=max_ihs,
             tajima_d=tajd, pi_ratio=pi_ratio, max_fst=max_fst)
    return {"gene": gene_name, "max_ihs": max_ihs, "tajd": tajd, "pi_ratio": pi_ratio, "max_fst": max_fst}


results = [compute(*g) for g in GENES]
print(f"\n{'='*50}\nNEUTRAL CONTROLS SUMMARY\n{'='*50}")
print(f"{'Gene':<12} {'EAS rank':>10} {'max|iHS|':>10} {'Tajima D':>10} {'pi ratio':>10} {'max FST':>10}")
for r, g in zip(results, GENES):
    print(f"{r['gene']:<12} {'~50%':>10} {r['max_ihs']:>10.2f} {r['tajd']:>10.3f} {r['pi_ratio']:>10.3f} {r['max_fst']:>10.3f}")
