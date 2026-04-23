"""Compute per-gene H12 and H2/H1 in a ±500 kb window around each gene midpoint
for one (chr, pop). Uses 400-SNP sliding windows with 50-SNP step, Garud's H12.

Output: analysis/orthogonal_v41/h12_500kb/{chr}_{pop}.csv with columns
    gene_name, gstart, gend, midpoint, n_seg, max_h12_500kb, max_h2h1_500kb, n_windows
"""
from __future__ import annotations
import argparse, os, sys, time
from collections import Counter
import numpy as np
import pandas as pd

REPO = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
PARSED = os.path.join(REPO, "analysis/genome_wide/cache/parsed")
GENES_DIR = os.path.join(REPO, "analysis/genome_wide/cache/genes")
SAMPLES = os.path.join(REPO, "analysis/genome_wide/data/samples.txt")
OUT_DIR = os.path.join(REPO, "analysis/orthogonal_v41/h12_500kb_v2")

WIN_SNPS = 400
STEP = 50
HALF_BP = 500_000

def pop_hap_idx(sample_ids, pop):
    df = pd.read_csv(SAMPLES, sep=" ")
    pop_samples = set(df[df["Population"] == pop]["SampleID"])
    sidx = [i for i, s in enumerate(sample_ids) if s in pop_samples]
    return np.array([x for i in sidx for x in (2*i, 2*i+1)])

def h12_h2h1_track(G):
    """Return per-window (h12, h2h1) arrays for segmented pop G (n_haps, n_snps)."""
    n_h, n_s = G.shape
    if n_s < WIN_SNPS:
        return np.array([]), np.array([])
    h12s, h2h1s = [], []
    for s in range(0, n_s - WIN_SNPS + 1, STEP):
        w = np.ascontiguousarray(G[:, s:s + WIN_SNPS])
        counts = Counter(r.tobytes() for r in w)
        f = np.array(sorted(counts.values(), reverse=True)) / n_h
        if len(f) < 2:
            h12 = 1.0; h1 = 1.0; h2h1 = 1.0
        else:
            h12 = (f[0] + f[1])**2 + float(np.sum(f[2:]**2))
            h1 = float(np.sum(f**2))
            h2 = h12 - f[0]**2 - f[1]**2 + (f[0] + f[1])**2 - f[0]**2  # no, re-derive
            # H2 = sum_{i>=2} p_i^2 ; H1 = sum p_i^2 ; h2/h1
            h2 = float(np.sum(f[1:]**2))
            h2h1 = h2 / h1 if h1 > 0 else 0.0
        h12s.append(h12); h2h1s.append(h2h1)
    return np.array(h12s), np.array(h2h1s)

def main(chr_num, pop):
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"chr{chr_num}_{pop}.csv")
    if os.path.exists(out_path):
        print(f"already exists, skipping: {out_path}", file=sys.stderr)
        return
    t0 = time.time()
    d = np.load(os.path.join(PARSED, f"chr{chr_num}.npz"), allow_pickle=True, mmap_mode="r")
    G = d["G"]; positions = d["positions"]; sample_ids = d["sample_ids"]
    hap = pop_hap_idx(sample_ids, pop)
    # Keep ALL VCF-polymorphic sites (do NOT drop monomorphic-in-pop). This matches
    # the standard Garud 2015 convention and multistat_demo/verify-39 (H12_max=0.65 for CLEC6A).
    G_pop_seg = np.asarray(G[hap, :])
    pos_seg = positions
    print(f"chr{chr_num} {pop}: n_hap={len(hap)} all_sites={len(pos_seg)} (load {time.time()-t0:.1f}s)", file=sys.stderr)

    genes = pd.read_csv(os.path.join(GENES_DIR, f"chr{chr_num}_genes.tsv"), sep="\t")
    rows = []
    for _, g in genes.iterrows():
        mid = (g["start"] + g["end"]) // 2
        lo_bp = mid - HALF_BP
        hi_bp = mid + HALF_BP
        a = np.searchsorted(pos_seg, lo_bp)
        b = np.searchsorted(pos_seg, hi_bp)
        G_win = G_pop_seg[:, a:b]
        h12_arr, h2h1_arr = h12_h2h1_track(G_win)
        if len(h12_arr) == 0:
            mh12 = np.nan; mh2h1 = np.nan; nw = 0
        else:
            argmax = int(np.argmax(h12_arr))
            mh12 = float(h12_arr[argmax])
            mh2h1 = float(h2h1_arr[argmax])  # H2/H1 at the peak-H12 window (Garud 2015)
            nw = len(h12_arr)
        rows.append({
            "gene_name": g["gene_name"], "gstart": int(g["start"]), "gend": int(g["end"]),
            "midpoint": int(mid), "n_seg": int(b - a),
            "max_h12_500kb": mh12, "max_h2h1_500kb": mh2h1, "n_windows": nw,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"wrote {out_path}  n_genes={len(rows)}  total {time.time()-t0:.1f}s", file=sys.stderr)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chr", type=int, required=True)
    ap.add_argument("--pop", type=str, required=True)
    args = ap.parse_args()
    main(args.chr, args.pop)
