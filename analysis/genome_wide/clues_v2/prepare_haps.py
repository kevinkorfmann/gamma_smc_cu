#!/usr/bin/env python
"""
Prepare population-specific haplotype files for Relate from 1KG VCFs.
Outputs: {POP}_chr{N}.haps.gz, .sample.gz, .dist.gz
"""

import numpy as np
import gzip
import os
import argparse

BETTY_BASE = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide"
PARSED_DIR = os.path.join(BETTY_BASE, "cache/parsed")
SAMPLES_FILE = os.path.join(BETTY_BASE, "data/samples.txt")
RECOMB_DIR = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide/clues/1kg_trees/data"


def load_pop_indices(samples_file, pop):
    indices = []
    sample_ids = []
    with open(samples_file) as f:
        header = f.readline().strip().split()
        pop_col = next(j for j, c in enumerate(header) if c.lower() in ("pop", "population"))
        id_col = next(j for j, c in enumerate(header) if c.lower() in ("sampleid", "sample"))
        for i, line in enumerate(f):
            fields = line.strip().split()
            if fields[pop_col] == pop:
                indices.append(i)
                sample_ids.append(fields[id_col])
    return indices, sample_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop", required=True)
    parser.add_argument("--chr", type=int, required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--suffix", default="", help="Output file suffix")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load parsed data
    print(f"Loading chr{args.chr}...")
    npz = np.load(os.path.join(PARSED_DIR, f"chr{args.chr}.npz"))
    G = npz["G"]  # (n_haplotypes, n_sites)
    pos = npz["positions"]

    # Get population indices
    sample_indices, sample_ids = load_pop_indices(SAMPLES_FILE, args.pop)
    hap_indices = []
    for si in sample_indices:
        hap_indices.extend([2 * si, 2 * si + 1])
    hap_indices = np.array(hap_indices)

    # Subset to region if specified
    if args.start and args.end:
        mask = (pos >= args.start) & (pos <= args.end)
        pos = pos[mask]
        G = G[:, mask]
        print(f"  Region {args.start}-{args.end}: {mask.sum()} sites")

    G_pop = G[hap_indices]
    del G
    n_haps = len(hap_indices)
    n_sites = len(pos)
    print(f"  {args.pop}: {n_haps} haplotypes, {n_sites} sites")

    # Remove duplicate positions (keep first)
    _, unique_idx = np.unique(pos, return_index=True)
    if len(unique_idx) < n_sites:
        print(f"  Removing {n_sites - len(unique_idx)} duplicate positions")
        pos = pos[unique_idx]
        G_pop = G_pop[:, unique_idx]
        n_sites = len(pos)

    # Write .haps.gz (Relate format: 5 columns + haplotypes)
    # Format: chr rsid pos ref alt hap1 hap2 ...
    sfx = args.suffix
    haps_file = os.path.join(args.outdir, f"{args.pop}_chr{args.chr}{sfx}.haps.gz")
    print(f"  Writing {haps_file}...")
    with gzip.open(haps_file, "wt") as f:
        for i in range(n_sites):
            haps = " ".join(str(int(G_pop[j, i])) for j in range(n_haps))
            f.write(f"{args.chr} snp_{i} {int(pos[i])} A T {haps}\n")

    # Write .sample.gz
    sample_file = os.path.join(args.outdir, f"{args.pop}_chr{args.chr}{sfx}.sample.gz")
    print(f"  Writing {sample_file}...")
    with gzip.open(sample_file, "wt") as f:
        f.write("ID_1\tID_2\tmissing\n")
        f.write("0\t0\t0\n")
        for sid in sample_ids:
            f.write(f"{sid}\t{sid}\t0\n")

    # Write .poplabels
    poplabels_file = os.path.join(args.outdir, f"{args.pop}_chr{args.chr}{sfx}.poplabels")
    with open(poplabels_file, "w") as f:
        f.write("sample population group sex\n")
        for sid in sample_ids:
            f.write(f"{sid} {args.pop} {args.pop} NA\n")

    # Write .dist.gz (genetic distances in cM, from physical positions)
    # Simple approximation: 1 cM per Mb
    dist_file = os.path.join(args.outdir, f"{args.pop}_chr{args.chr}{sfx}.dist.gz")
    print(f"  Writing {dist_file}...")
    with gzip.open(dist_file, "wt") as f:
        dists = np.diff(pos) * 1e-6  # approximate cM
        dists = np.insert(dists, 0, 0)
        f.write(" ".join(f"{d:.6f}" for d in dists) + "\n")

    print("  Done.")


if __name__ == "__main__":
    main()
