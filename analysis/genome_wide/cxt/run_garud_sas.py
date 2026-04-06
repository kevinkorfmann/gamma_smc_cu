#!/usr/bin/env python
"""Compute Garud's H12 for SAS populations on chr11 (GRK2) and chr20 (BPIFA2)."""

import numpy as np
import allel
import pandas as pd
import os

BETTY_BASE = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide"
PARSED_DIR = os.path.join(BETTY_BASE, "cache/parsed")
GENE_DIR = os.path.join(BETTY_BASE, "cache/genes")
SAMPLES_FILE = os.path.join(BETTY_BASE, "data/samples.txt")
OUTDIR = os.path.join(BETTY_BASE, "cxt/results/orthogonal")

SAS_POPS = ["GIH", "PJL", "BEB", "STU", "ITU"]
WINDOW_SNP = 1000


def load_pop_indices(samples_file):
    pops = {}
    with open(samples_file) as f:
        header = f.readline().strip().split()
        pop_col = next(j for j, c in enumerate(header) if c.lower() in ("pop", "population"))
        for i, line in enumerate(f):
            fields = line.strip().split()
            pops.setdefault(fields[pop_col], []).append(i)
    return pops


def get_hap_idx(pop_indices, pop_list):
    idx = []
    for pop in pop_list:
        for si in pop_indices[pop]:
            idx.extend([2 * si, 2 * si + 1])
    return np.array(idx)


pop_indices = load_pop_indices(SAMPLES_FILE)
sas_hap_idx = get_hap_idx(pop_indices, SAS_POPS)

for chrn in [11, 20]:
    print(f"\nLoading chr{chrn}...")
    npz = np.load(os.path.join(PARSED_DIR, f"chr{chrn}.npz"))
    G_sas = npz["G"][sas_hap_idx]
    pos = npz["positions"]

    genes = pd.read_csv(os.path.join(GENE_DIR, f"chr{chrn}_genes.tsv"), sep="\t")
    genes["midpoint"] = ((genes["start"] + genes["end"]) / 2).astype(int)

    results = []
    for _, gene in genes.iterrows():
        mid_idx = np.searchsorted(pos, gene["midpoint"])
        mid_idx = min(mid_idx, len(pos) - 1)
        half = WINDOW_SNP // 2
        si, ei = max(0, mid_idx - half), min(len(pos), mid_idx + half)
        if ei - si < 50:
            results.append({"gene": gene["gene_name"], "chr": chrn,
                            "H1": np.nan, "H12": np.nan, "H123": np.nan, "H2_H1": np.nan})
            continue
        h = allel.HaplotypeArray(G_sas[:, si:ei].T)
        try:
            h1, h12, h123, h2_h1 = allel.garud_h(h)
            results.append({"gene": gene["gene_name"], "chr": chrn,
                            "H1": float(h1), "H12": float(h12),
                            "H123": float(h123), "H2_H1": float(h2_h1)})
        except:
            results.append({"gene": gene["gene_name"], "chr": chrn,
                            "H1": np.nan, "H12": np.nan, "H123": np.nan, "H2_H1": np.nan})

    df = pd.DataFrame(results)
    outf = os.path.join(OUTDIR, f"garud_sas_chr{chrn}.csv")
    df.to_csv(outf, index=False)

    # Report GRK2 / BPIFA2
    df["H12_pctl"] = df["H12"].rank(pct=True, na_option="keep")
    for target in ["GRK2", "BPIFA2"]:
        row = df[df["gene"] == target]
        if len(row):
            r = row.iloc[0]
            print(f"  {target}: H12={r['H12']:.4f}, percentile={r['H12_pctl']:.1%}, H2/H1={r['H2_H1']:.4f}")
