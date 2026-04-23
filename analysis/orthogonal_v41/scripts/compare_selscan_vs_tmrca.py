#!/usr/bin/env python
"""Compare selscan iHS/nSL gene-level ranks against gamma_smc_cu TMRCA ranks for
the v4.1 candidate set (5 novel + 5 positive controls + 5 neutral controls).

For each (gene, focal_pop) pair we report:
  - gamma_smc_cu rank percentile (geom_mean primary stat from genome_wide_ranks.csv)
  - selscan max |iHS_norm| rank percentile in the focal pop
  - selscan frac_ihs_extreme rank percentile
  - selscan max |nSL_norm| rank percentile
  - selscan frac_nsl_extreme rank percentile

Output:
  analysis/orthogonal_v41/selscan_genelevel/v41_candidate_comparison.csv
  analysis/orthogonal_v41/figures/selscan_vs_tmrca.{png,pdf}

Run after aggregate_selscan_per_gene.py --all has produced one CSV per pop.
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
TMRCA_RANKS = os.path.join(
    REPO, "analysis/genome_wide/results/genome_wide_ranks.csv"
)
SELSCAN_GENE_DIR = os.path.join(REPO, "analysis/orthogonal_v41/selscan_genelevel")
OUT_CSV = os.path.join(SELSCAN_GENE_DIR, "v41_candidate_comparison.csv")
OUT_FIG_DIR = os.path.join(REPO, "analysis/orthogonal_v41/figures")

# 5 novel + 5 positive controls + 5 neutral controls (matches gene_list.py
# and the three_method NPZ task list)
CANDIDATES = [
    # gene, chr, focal_pop, group, color
    ("GRK2",     11, "GIH", "novel",    "purple"),
    ("BPIFA2",   20, "GIH", "novel",    "purple"),
    ("SLC6A15",  12, "CHS", "novel",    "purple"),
    ("CCDC92",   12, "CDX", "novel",    "purple"),
    ("CLEC6A",   12, "CDX", "novel",    "purple"),
    ("SLC24A5",  15, "GBR", "positive", "blue"),
    ("LCT",       2, "CEU", "positive", "blue"),
    ("EDAR",      2, "CHB", "positive", "blue"),
    ("ABCC11",   16, "CHB", "positive", "blue"),
    ("KITLG",    12, "MXL", "positive", "blue"),
    ("AQP3",      9, "GWD", "neutral",  "gray"),
    ("MYO1F",    19, "CEU", "neutral",  "gray"),
    ("NUP35",     2, "CDX", "neutral",  "gray"),
    ("DSC1",     18, "ITU", "neutral",  "gray"),
    ("SEC13",     3, "PEL", "neutral",  "gray"),
]


def main():
    os.makedirs(OUT_FIG_DIR, exist_ok=True)

    # Load gamma_smc_cu ranks
    tmrca = pd.read_csv(TMRCA_RANKS)
    tmrca_lookup = {(row["gene_name"], row["chr"]): row for _, row in tmrca.iterrows()}

    # Load all selscan per-pop tables once, keyed by pop
    pop_tables = {}
    for fname in os.listdir(SELSCAN_GENE_DIR):
        if not fname.endswith(".csv") or fname.startswith("v41_"):
            continue
        pop = fname.replace(".csv", "")
        pop_tables[pop] = pd.read_csv(os.path.join(SELSCAN_GENE_DIR, fname))

    rows = []
    for gene, chr_num, pop, group, color in CANDIDATES:
        # gamma_smc_cu rank
        key = (gene, chr_num)
        t = tmrca_lookup.get(key)
        if t is None:
            print(f"  WARN: {gene} chr{chr_num} not in gamma_smc_cu ranks", flush=True)
            tmrca_rank = np.nan
        else:
            tmrca_rank = float(t.get(f"{pop}_rank", np.nan))

        # selscan ranks
        if pop not in pop_tables:
            print(f"  WARN: no selscan output for {pop}", flush=True)
            sel_row = {}
        else:
            df = pop_tables[pop]
            match = df[(df["gene_name"] == gene) & (df["chr"] == chr_num)]
            if match.empty:
                print(f"  WARN: {gene} chr{chr_num} not in selscan {pop}",
                      flush=True)
                sel_row = {}
            else:
                sel_row = match.iloc[0].to_dict()

        rows.append({
            "gene": gene,
            "chr": chr_num,
            "pop": pop,
            "group": group,
            "tmrca_rank": tmrca_rank,
            "n_ihs_sites": sel_row.get("n_ihs_sites", np.nan),
            "max_abs_ihs_norm": sel_row.get("max_abs_ihs_norm", np.nan),
            "max_abs_ihs_norm_rank": sel_row.get("max_abs_ihs_norm_rank", np.nan),
            "frac_ihs_extreme": sel_row.get("frac_ihs_extreme", np.nan),
            "frac_ihs_extreme_rank": sel_row.get("frac_ihs_extreme_rank", np.nan),
            "max_abs_nsl_norm": sel_row.get("max_abs_nsl_norm", np.nan),
            "max_abs_nsl_norm_rank": sel_row.get("max_abs_nsl_norm_rank", np.nan),
            "frac_nsl_extreme": sel_row.get("frac_nsl_extreme", np.nan),
            "frac_nsl_extreme_rank": sel_row.get("frac_nsl_extreme_rank", np.nan),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}", flush=True)
    print(out.to_string(), flush=True)

    # Figure: scatter tmrca rank vs selscan max |iHS_norm| rank, colored by group
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    color_map = {"novel": "purple", "positive": "#1f77b4", "neutral": "#888888"}

    for stat_col, ax, title in [
        ("max_abs_ihs_norm_rank", axes[0], "iHS"),
        ("max_abs_nsl_norm_rank", axes[1], "nSL"),
    ]:
        for group, sub in out.groupby("group"):
            ax.scatter(
                sub["tmrca_rank"], sub[stat_col],
                color=color_map[group], s=70, edgecolor="k", lw=0.6,
                label=group, alpha=0.85,
            )
            for _, r in sub.iterrows():
                if not np.isnan(r["tmrca_rank"]) and not np.isnan(r[stat_col]):
                    ax.annotate(
                        f"{r['gene']}",
                        (r["tmrca_rank"], r[stat_col]),
                        fontsize=7, xytext=(4, 3), textcoords="offset points",
                    )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(0.05, color="gray", lw=0.6, ls=":")
        ax.axvline(0.05, color="gray", lw=0.6, ls=":")
        ax.set_xlabel("gamma_smc_cu rank (geom mean)")
        ax.set_ylabel(f"selscan {title} max |Z| rank")
        ax.set_title(title, loc="left", fontweight="normal")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    plt.tight_layout()
    out_png = os.path.join(OUT_FIG_DIR, "selscan_vs_tmrca.png")
    out_pdf = os.path.join(OUT_FIG_DIR, "selscan_vs_tmrca.pdf")
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
