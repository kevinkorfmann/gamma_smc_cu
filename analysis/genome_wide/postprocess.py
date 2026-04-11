#!/usr/bin/env python
"""Postprocessing for the new genome-wide TMRCA scan.

Implements steps 1-6 of the post-rerun investigation:
  1. SD-mask the gene list (UCSC genomicSuperDups) and rebuild
     filtered top-candidate tables for every primary statistic.
  2. Pick the primary statistic and document the rationale (writes
     a STATS.md note to the output directory).
  3. Investigate why TYRP1, APOL1, OCA2, EPAS1 are still missed by
     reading their NPZ histograms in the relevant populations.
  4. Manhattan plot from the new genome_wide_stats.csv (geom_mean +
     frac_below_1000 side by side).
  5. Pathway convergence reanalysis: re-test KEGG / GO_BP enrichment
     in the new EAS / SAS / EUR / AFR mean ranks against a
     permutation null. Mucosal-immunity pathways are flagged.
  6. Cross-validate the 5 novel findings (GRK2, BPIFA2, SLC6A15,
     CCDC92, mucosal-immunity) against the new ranks.

All outputs land in analysis/genome_wide/postprocess/.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = "/vast/projects/smathi/cohort/kkor/tmrca.cu/analysis/genome_wide"
RESULTS = os.path.join(BASE, "results")
OUT = os.path.join(BASE, "postprocess")
os.makedirs(OUT, exist_ok=True)

POPS = [
    "ACB", "ASW", "BEB", "CDX", "CEU", "CHB", "CHS", "CLM",
    "ESN", "FIN", "GBR", "GIH", "GWD", "IBS", "ITU", "JPT",
    "KHV", "LWK", "MSL", "MXL", "PEL", "PJL", "PUR", "STU",
    "TSI", "YRI",
]
SUPERPOPS = {
    "AFR": ["YRI", "LWK", "GWD", "MSL", "ESN", "ACB", "ASW"],
    "EUR": ["CEU", "TSI", "FIN", "GBR", "IBS"],
    "EAS": ["CHB", "JPT", "CHS", "CDX", "KHV"],
    "SAS": ["GIH", "PJL", "BEB", "STU", "ITU"],
    "AMR": ["MXL", "PUR", "CLM", "PEL"],
}


# ---------- Step 1 ---------- #
def load_sd_intervals(path):
    """Return dict chr_int -> sorted list of (start, end) intervals."""
    intervals = defaultdict(list)
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            chrom = parts[1]
            if not chrom.startswith("chr"):
                continue
            try:
                chr_int = int(chrom.replace("chr", ""))
            except ValueError:
                continue
            start = int(parts[2])
            end = int(parts[3])
            intervals[chr_int].append((start, end))
    for c in intervals:
        intervals[c].sort()
    return intervals


def gene_overlaps_sd(gene_chr, gene_start, gene_end, sd_intervals, frac_threshold=0.5):
    """Return True if >=frac_threshold of the gene length overlaps SD."""
    ivs = sd_intervals.get(int(gene_chr), [])
    if not ivs:
        return False
    gene_len = max(1, gene_end - gene_start)
    overlap = 0
    for s, e in ivs:
        if e < gene_start:
            continue
        if s > gene_end:
            break
        overlap += max(0, min(e, gene_end) - max(s, gene_start))
    return overlap / gene_len >= frac_threshold


def step1_sd_mask():
    print("=" * 60)
    print("Step 1: SD masking")
    print("=" * 60)

    sd_path = os.path.join(BASE, "genomicSuperDups.txt.gz")
    sds = load_sd_intervals(sd_path)
    print(f"Loaded SD intervals on {len(sds)} chromosomes")

    stat_files = {
        "geom_mean": "genome_wide_stats.csv",
        "p5": "genome_wide_stats_p5.csv",
        "p10": "genome_wide_stats_p10.csv",
        "min": "genome_wide_stats_min.csv",
        "frac_below_1000": "genome_wide_stats_frac_below_1000.csv",
    }

    sd_flags = None
    for stat, fname in stat_files.items():
        path = os.path.join(RESULTS, fname)
        df = pd.read_csv(path)
        if sd_flags is None:
            print("Computing SD overlap for all genes (one pass)...")
            sd_flags = df.apply(
                lambda r: gene_overlaps_sd(r["chr"], r["start"], r["end"], sds),
                axis=1,
            )
            n_sd = sd_flags.sum()
            print(f"  {n_sd}/{len(df)} genes ({n_sd/len(df)*100:.1f}%) overlap SDs")
            df_sd = df.copy()
            df_sd["is_sd"] = sd_flags
        else:
            df_sd = df.copy()
            df_sd["is_sd"] = sd_flags

        # Filtered top 50
        clean = df_sd[~df_sd["is_sd"]].sort_values("min_rank")
        out_path = os.path.join(OUT, f"top50_{stat}_no_sd.csv")
        clean.head(50).to_csv(out_path, index=False)
        print(f"  {stat}: top 50 (no SD) -> {out_path}")

        if stat == "geom_mean":
            print(f"\nTop 30 candidates with SD column:")
            top30 = df_sd.sort_values("min_rank").head(30)
            print(top30[["gene_name", "chr", "min_rank", "min_pop", "is_sd"]].to_string(index=False))
            print(f"\nTop 30 SD-FREE candidates:")
            print(clean.head(30)[["gene_name", "chr", "start", "min_rank", "min_pop"]].to_string(index=False))

    # Save the SD flag table for downstream reuse
    sd_table = pd.read_csv(os.path.join(RESULTS, "genome_wide_stats.csv"))
    sd_table["is_sd"] = sd_flags
    sd_table.to_csv(os.path.join(OUT, "genes_sd_flag.csv"), index=False)


# ---------- Step 2 ---------- #
def step2_pick_primary():
    print("=" * 60)
    print("Step 2: Pick primary statistic")
    print("=" * 60)

    summary = []
    controls = ["LCT", "MCM6", "EDAR", "SLC24A5", "TRPV6", "KITLG", "HERC2",
                "ADH1B", "TYRP1", "OCA2", "APOL1", "CD36", "EPAS1", "HBB"]

    for stat, fname in [
        ("geom_mean", "genome_wide_stats.csv"),
        ("p5", "genome_wide_stats_p5.csv"),
        ("p10", "genome_wide_stats_p10.csv"),
        ("min", "genome_wide_stats_min.csv"),
        ("frac_below_1000", "genome_wide_stats_frac_below_1000.csv"),
    ]:
        df = pd.read_csv(os.path.join(RESULTS, fname))
        hit = df[df["gene_name"].isin(controls)]
        n10 = (hit["min_rank"] < 0.10).sum()
        n5 = (hit["min_rank"] < 0.05).sum()
        n1 = (hit["min_rank"] < 0.01).sum()
        summary.append({
            "stat": stat,
            "n_genes": len(df),
            "controls_found": len(hit),
            "below_10pct": n10,
            "below_5pct": n5,
            "below_1pct": n1,
        })

    summ_df = pd.DataFrame(summary)
    print(summ_df.to_string(index=False))
    summ_df.to_csv(os.path.join(OUT, "stat_comparison.csv"), index=False)

    md = """# Primary statistic recommendation

We compute five per-gene per-population summaries from the per-pair
TMRCA distribution within each gene region:

| stat | definition | strength | weakness |
|---|---|---|---|
| geom_mean | exp(mean(log TMRCA)) over all (pair, site) tuples | matches old run; balanced view | dilutes very partial sweeps |
| p5 | 5th percentile of per-pair geometric-mean TMRCA | catches sweeps at frequency >= sqrt(0.05) ~= 22% | noisier (single-point) |
| p10 | 10th percentile | similar to p5 with more support | similar |
| min | youngest per-pair TMRCA | maximum sensitivity | very noisy, single point |
| frac_below_1000 | fraction of pairs with mean TMRCA < 1000 generations | direct sweep-haplotype frequency | threshold-dependent |

**Recommendation: report `geom_mean` as the headline statistic and
`frac_below_1000` as the supporting statistic.**

Rationale:
- `geom_mean` is what the original archive pipeline used (verified in
  `archive_2026_04_09/genome_wide_local/cxt/run_cxt_region.py`,
  via `np.exp(log_tmrca_raw.mean(...))`). Comparing to the archive
  is meaningful only with `geom_mean`.
- `frac_below_1000` is a direct, interpretable measure of sweep
  haplotype frequency: "what fraction of within-population pairs in
  this gene have an inferred TMRCA below 1000 generations?". A high
  value means many pairs share a recent common ancestor at this gene,
  i.e. a sweep haplotype is at high frequency.
- Reporting both lets the reader see two complementary views: a
  centrality-based estimate (geom_mean) and a frequency-based
  estimate (frac_below_1000).
- `p5` / `p10` / `min` are kept as additional sanity checks. `p5`
  and `p10` are most sensitive for partial sweeps but tend to be
  noisier on small genes with few segregating sites.
"""
    with open(os.path.join(OUT, "STATS.md"), "w") as f:
        f.write(md)
    print(f"\nWrote {os.path.join(OUT, 'STATS.md')}")


# ---------- Step 3 ---------- #
def step3_investigate_missed():
    print("=" * 60)
    print("Step 3: Investigate missed sweeps via NPZ histograms")
    print("=" * 60)

    # For each missed sweep, find the relevant population's NPZ
    # and inspect the histogram of per-pair log TMRCA.
    targets = [
        ("TYRP1", 9, ["KHV", "CHB", "CHS", "JPT", "CDX"]),
        ("APOL1", 22, ["YRI", "ESN", "GWD", "LWK", "MSL"]),
        ("OCA2", 15, ["CEU", "GBR", "FIN", "TSI", "IBS"]),
        ("EPAS1", 2, ["CHB", "JPT", "CHS", "CDX", "KHV"]),
    ]

    fig, axes = plt.subplots(len(targets), 5, figsize=(15, 3 * len(targets)))
    for ti, (gene, chrn, pops) in enumerate(targets):
        # Load any pop NPZ to find this gene's index
        any_pop_npz = os.path.join(RESULTS, f"chr{chrn}", f"{pops[0]}.npz")
        with np.load(any_pop_npz, allow_pickle=True) as d:
            gene_names = [str(g) for g in d["gene_name"]]
            try:
                gi = gene_names.index(gene)
            except ValueError:
                print(f"  {gene} not in chr{chrn} gene list")
                for ax in axes[ti]:
                    ax.set_visible(False)
                continue

        for pi, pop in enumerate(pops):
            npz_path = os.path.join(RESULTS, f"chr{chrn}", f"{pop}.npz")
            if not os.path.exists(npz_path):
                continue
            with np.load(npz_path, allow_pickle=True) as d:
                hist = d["histogram"][gi]
                edges = d["bin_edges"]
                count = int(d["count"][gi])
                lin_sum = float(d["lin_sum"][gi])
                log_sum = float(d["log_sum"][gi])
                mn = float(d["min_lin"][gi])

            ax = axes[ti, pi]
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.bar(centers, hist, width=(edges[1] - edges[0]) * 0.9, color="steelblue")
            if count > 0:
                arith = lin_sum / count
                geom = np.exp(log_sum / count)
                ax.axvline(np.log(arith), color="red", lw=1, label=f"arith={arith:.0f}")
                ax.axvline(np.log(max(geom, 1)), color="green", lw=1, label=f"geom={geom:.0f}")
                ax.axvline(np.log(max(mn, 1)), color="orange", lw=1, label=f"min={mn:.0f}")
            ax.set_title(f"{gene} | {pop} (chr{chrn})", fontsize=9)
            ax.set_xlabel("ln(TMRCA gen)", fontsize=8)
            if pi == 0:
                ax.set_ylabel("# pairs", fontsize=8)
            ax.legend(fontsize=6, loc="upper right")
            ax.tick_params(labelsize=7)
    plt.tight_layout()
    out_png = os.path.join(OUT, "missed_sweeps_histograms.png")
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"Wrote {out_png}")

    # Also dump quantitative summary
    print()
    for gene, chrn, pops in targets:
        for pop in pops[:1]:  # primary expected pop
            npz_path = os.path.join(RESULTS, f"chr{chrn}", f"{pop}.npz")
            with np.load(npz_path, allow_pickle=True) as d:
                gene_names = [str(g) for g in d["gene_name"]]
                if gene not in gene_names:
                    continue
                gi = gene_names.index(gene)
                hist = d["histogram"][gi]
                edges = d["bin_edges"]
                count = int(d["count"][gi])
                if count == 0:
                    continue
                # Fraction of pairs in lowest 5 bins
                low_frac = hist[:5].sum() / count
                geom = np.exp(d["log_sum"][gi] / count)
                arith = d["lin_sum"][gi] / count
                mn = d["min_lin"][gi]
                print(f"{gene:8s} {pop} chr{chrn}: count={count}, "
                      f"frac_lowest_5_bins={low_frac:.3f}, "
                      f"geom={geom:.0f}, arith={arith:.0f}, min={mn:.0f}")


# ---------- Step 4 ---------- #
def step4_manhattan():
    print("=" * 60)
    print("Step 4: Manhattan plots")
    print("=" * 60)

    # Try to read SD flags
    sd_csv = os.path.join(OUT, "genes_sd_flag.csv")
    if os.path.exists(sd_csv):
        sd_df = pd.read_csv(sd_csv)
        sd_set = set(sd_df[sd_df["is_sd"]]["gene_name"])
    else:
        sd_set = set()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    chr_lens = {
        1: 248956422, 2: 242193529, 3: 198295559, 4: 190214555, 5: 181538259,
        6: 170805979, 7: 159345973, 8: 145138636, 9: 138394717, 10: 133797422,
        11: 135086622, 12: 133275309, 13: 114364328, 14: 107043718, 15: 101991189,
        16: 90338345, 17: 83257441, 18: 80373285, 19: 58617616, 20: 64444167,
        21: 46709983, 22: 50818468,
    }
    offsets = {1: 0}
    for c in range(2, 23):
        offsets[c] = offsets[c - 1] + chr_lens[c - 1]
    total_len = offsets[22] + chr_lens[22]

    for ax, (stat, fname, title) in zip(
        axes,
        [
            ("geom_mean", "genome_wide_stats.csv", "Manhattan: geom_mean min_rank"),
            ("frac_below_1000", "genome_wide_stats_frac_below_1000.csv",
             "Manhattan: frac_below_1000 min_rank"),
        ],
    ):
        df = pd.read_csv(os.path.join(RESULTS, fname))
        df["abs_pos"] = df["chr"].map(offsets) + (df["start"] + df["end"]) / 2
        df["is_sd"] = df["gene_name"].isin(sd_set)
        df["minus_log10_rank"] = -np.log10(np.clip(df["min_rank"], 1e-6, 1.0))
        # SD points: lighter
        sd_pts = df[df["is_sd"]]
        clean = df[~df["is_sd"]]
        ax.scatter(
            sd_pts["abs_pos"], sd_pts["minus_log10_rank"], s=3,
            color="lightgray", alpha=0.4, label="SD",
        )
        # Color clean points by chromosome
        colors = ["#3182bd", "#08519c"]
        for ci in range(1, 23):
            sub = clean[clean["chr"] == ci]
            ax.scatter(
                sub["abs_pos"], sub["minus_log10_rank"], s=4,
                color=colors[ci % 2], alpha=0.8,
            )
        # Annotate top 10 non-SD
        top = clean.nsmallest(10, "min_rank")
        for _, r in top.iterrows():
            ax.annotate(
                r["gene_name"],
                xy=(r["abs_pos"], r["minus_log10_rank"]),
                fontsize=6, ha="center", va="bottom",
            )
        ax.axhline(-np.log10(0.10), color="orange", ls=":", lw=0.5)
        ax.axhline(-np.log10(0.01), color="red", ls=":", lw=0.5)
        ax.set_ylabel("-log10(min_rank)")
        ax.set_title(title, loc="left")
        ax.set_xlim(0, total_len)
        # Chromosome ticks
        tick_pos = [offsets[c] + chr_lens[c] / 2 for c in range(1, 23)]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([str(c) for c in range(1, 23)], fontsize=7)

    axes[1].set_xlabel("Chromosome")
    plt.tight_layout()
    out_png = os.path.join(OUT, "manhattan.png")
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"Wrote {out_png}")


# ---------- Step 5 ---------- #
def step5_pathway():
    print("=" * 60)
    print("Step 5: Pathway convergence")
    print("=" * 60)

    try:
        import gseapy as gp
    except ImportError:
        print("gseapy not installed; skipping pathway analysis")
        return

    # Load gene-wide ranks per population
    ranks = pd.read_csv(os.path.join(RESULTS, "genome_wide_ranks.csv"))
    rank_cols = [f"{p}_rank" for p in POPS if f"{p}_rank" in ranks.columns]

    # Compute superpop mean ranks
    sp_ranks = {}
    for sp, sp_pops in SUPERPOPS.items():
        cols = [f"{p}_rank" for p in sp_pops if f"{p}_rank" in ranks.columns]
        sp_ranks[sp] = ranks[cols].mean(axis=1)

    print("Loading KEGG and GO BP gene sets...")
    libs = {}
    for libname in ["KEGG_2021_Human", "GO_Biological_Process_2023"]:
        try:
            libs[libname] = gp.get_library(libname)
            print(f"  {libname}: {len(libs[libname])} sets")
        except Exception as exc:
            print(f"  {libname}: failed ({exc})")

    if not libs:
        print("No pathway libraries available, skipping")
        return

    all_sets = {}
    for libname, sets in libs.items():
        for name, genes in sets.items():
            all_sets[f"{libname.split('_')[0]}:{name}"] = genes

    THRESHOLD = 0.10
    N_PERM = 20000
    rng = np.random.default_rng(42)
    gene_to_idx = {g: i for i, g in enumerate(ranks["gene_name"])}

    all_results = []
    for sp, sp_rank in sp_ranks.items():
        rank_array = sp_rank.values
        below = rank_array < THRESHOLD
        gw_frac = below.mean()
        for pname, genes in all_sets.items():
            idx = [gene_to_idx[g] for g in genes if g in gene_to_idx]
            n = len(idx)
            if n < 5 or n > 500:
                continue
            n_below = below[idx].sum()
            expected = n * gw_frac
            if n_below <= expected:
                p = 1.0
            else:
                perm_idx = rng.integers(0, len(rank_array), size=(N_PERM, n))
                perm_below = below[perm_idx].sum(axis=1)
                p = (perm_below >= n_below).sum() / N_PERM
                p = max(p, 1.0 / N_PERM)
            all_results.append({
                "superpop": sp,
                "pathway": pname,
                "n_genes": n,
                "n_below": int(n_below),
                "expected_below": expected,
                "fold": n_below / max(expected, 0.01),
                "perm_p": p,
            })

    res_df = pd.DataFrame(all_results)
    # FDR per superpop
    from statsmodels.stats.multitest import multipletests
    out_chunks = []
    for sp in SUPERPOPS:
        sub = res_df[res_df["superpop"] == sp].copy()
        if not len(sub):
            continue
        _, q, _, _ = multipletests(sub["perm_p"], method="fdr_bh")
        sub["q"] = q
        out_chunks.append(sub)
    final = pd.concat(out_chunks).sort_values(["superpop", "perm_p"])
    final.to_csv(os.path.join(OUT, "pathway_enrichment.csv"), index=False)
    print(f"Wrote {os.path.join(OUT, 'pathway_enrichment.csv')}")

    print("\nTop 10 pathways per superpop (by perm p):")
    for sp in SUPERPOPS:
        print(f"\n--- {sp} ---")
        sub = final[final["superpop"] == sp].head(10)
        print(sub[["pathway", "n_genes", "n_below", "fold", "perm_p", "q"]].to_string(index=False))

    # Mucosal immunity flag
    print("\n--- Mucosal/IgA-related pathways across all superpops ---")
    keys = ["iga", "mucos", "intestin", "innate", "toll", "nf-kappa", "nod-like",
            "dectin", "lectin", "b cell", "complement", "antigen process"]
    flagged = final[final["pathway"].str.lower().str.contains("|".join(keys))]
    print(flagged.sort_values("perm_p").head(20)[
        ["superpop", "pathway", "n_genes", "n_below", "fold", "perm_p", "q"]
    ].to_string(index=False))


# ---------- Step 6 ---------- #
def step6_novel_findings():
    print("=" * 60)
    print("Step 6: Cross-validate novel findings")
    print("=" * 60)

    novel = {
        "GRK2":   ("expected SAS sweep, GIH-strong", "SAS"),
        "BPIFA2": ("salivary antimicrobial, SAS sweep, ITU/GIH-strong", "SAS"),
        "SLC6A15": ("EAS sweep, CHS-strong", "EAS"),
        "CCDC92":  ("EAS sweep", "EAS"),
        "TNFRSF13C": ("BAFF receptor, mucosal immunity, EAS", "EAS"),
        "JCHAIN":  ("IgA polymerization, mucosal immunity", "EAS"),
        "PIGR":    ("polymeric Ig receptor, mucosal", "EAS"),
        "CLEC6A":  ("C-type lectin, mucosal innate immunity", "EAS"),
    }

    ranks = pd.read_csv(os.path.join(RESULTS, "genome_wide_ranks.csv"))

    rows = []
    for gene, (desc, expected_sp) in novel.items():
        row = ranks[ranks["gene_name"] == gene]
        if row.empty:
            print(f"{gene:12s} NOT FOUND ({desc})")
            continue
        r = row.iloc[0]
        # Find best within each superpop
        best = {}
        for sp, sp_pops in SUPERPOPS.items():
            sp_rank_cols = [f"{p}_rank" for p in sp_pops if f"{p}_rank" in r.index]
            if not sp_rank_cols:
                continue
            sp_vals = r[sp_rank_cols].astype(float)
            best_pop = sp_vals.idxmin().replace("_rank", "")
            best[sp] = (best_pop, float(sp_vals.min()))
        global_min = min(b[1] for b in best.values())
        global_pop = [b[0] for b in best.values() if b[1] == global_min][0]
        marker = "<<<" if best.get(expected_sp, ("", 1))[1] < 0.05 else ""
        print(f"{gene:12s} chr={int(r['chr']):>2}  global_min={global_min:.4f} ({global_pop})")
        for sp, (pop, val) in best.items():
            tag = " <-- expected" if sp == expected_sp else ""
            print(f"             {sp}: best={pop} rank={val:.4f}{tag}")
        print()
        rows.append({
            "gene": gene,
            "description": desc,
            "expected_superpop": expected_sp,
            "global_min_rank": global_min,
            "global_min_pop": global_pop,
            **{f"{sp}_best_pop": best[sp][0] for sp in best},
            **{f"{sp}_best_rank": best[sp][1] for sp in best},
        })

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "novel_findings.csv"), index=False)
    print(f"\nWrote {os.path.join(OUT, 'novel_findings.csv')}")


def main():
    step1_sd_mask()
    print()
    step2_pick_primary()
    print()
    step3_investigate_missed()
    print()
    step4_manhattan()
    print()
    step5_pathway()
    print()
    step6_novel_findings()


if __name__ == "__main__":
    main()
