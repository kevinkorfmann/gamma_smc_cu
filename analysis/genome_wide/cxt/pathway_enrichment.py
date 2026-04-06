#!/usr/bin/env python
"""
Genome-wide pathway enrichment test for EAS TMRCA ranks.

For every KEGG pathway, test whether its genes have lower EAS mean ranks
than expected by chance (permutation test). Apply FDR correction across
all pathways. This is an unbiased scan — no pathway is privileged.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
import gseapy as gp
import os
import json

np.random.seed(42)

# --- Load genome-wide EAS ranks ---
GW_STATS = "/Users/kevinkorfmann/Projects/tmrca.cu/docs_local/genome_wide_results/genome_wide_stats.csv"
gw = pd.read_csv(GW_STATS, index_col=0)

# Compute EAS mean rank from per-chromosome gene_ranks files
GW_RESULTS = "/Users/kevinkorfmann/Projects/tmrca.cu/docs_local/genome_wide_results"
EAS_POPS = ["CHB", "JPT", "CHS", "CDX", "KHV"]

# Load all per-chromosome ranks and compute EAS mean per gene
all_ranks = {}
for chrn in range(1, 23):
    ranks_file = os.path.join(GW_RESULTS, f"chr{chrn}", "gene_ranks.csv")
    if not os.path.exists(ranks_file):
        continue
    df = pd.read_csv(ranks_file, index_col=0)
    eas_rows = df.loc[df.index.isin(EAS_POPS)]
    if len(eas_rows) > 0:
        eas_mean = eas_rows.mean(axis=0)  # mean rank across EAS pops per gene
        for gene, val in eas_mean.items():
            all_ranks[gene] = val

print(f"Loaded EAS mean ranks for {len(all_ranks)} genes")

# Convert to series
eas_ranks = pd.Series(all_ranks)
# Convert to percentages
eas_ranks_pct = eas_ranks * 100

# --- Download KEGG human pathways ---
print("Downloading KEGG pathway gene sets...")
try:
    kegg = gp.get_library("KEGG_2021_Human")
except:
    try:
        kegg = gp.get_library("KEGG_2024_Human")
    except:
        kegg = gp.get_library("KEGG_2019_Human")

print(f"Downloaded {len(kegg)} KEGG pathways")

# --- Also get GO Biological Process ---
print("Downloading GO BP gene sets...")
try:
    go_bp = gp.get_library("GO_Biological_Process_2023")
except:
    try:
        go_bp = gp.get_library("GO_Biological_Process_2021")
    except:
        go_bp = {}
        print("  Could not download GO BP")

if go_bp:
    print(f"Downloaded {len(go_bp)} GO BP terms")

# Combine all gene sets
all_genesets = {}
for name, genes in kegg.items():
    all_genesets[f"KEGG:{name}"] = genes
for name, genes in go_bp.items():
    all_genesets[f"GO_BP:{name}"] = genes

print(f"Total gene sets to test: {len(all_genesets)}")

# --- Permutation-based enrichment test ---
N_PERM = 100_000
THRESHOLD_PCT = 10  # genes below this rank are "selected"

results = []
genome_wide_fraction = (eas_ranks_pct < THRESHOLD_PCT).mean()
print(f"Genome-wide fraction below {THRESHOLD_PCT}%: {genome_wide_fraction:.4f}")

all_gene_names = set(eas_ranks_pct.index)

for pathway_name, pathway_genes in all_genesets.items():
    # Intersect with our gene set
    overlap = [g for g in pathway_genes if g in all_gene_names]
    n = len(overlap)
    if n < 5 or n > 500:  # skip tiny or huge pathways
        continue

    pathway_ranks = eas_ranks_pct[overlap].values
    n_below = (pathway_ranks < THRESHOLD_PCT).sum()
    mean_rank = pathway_ranks.mean()

    # Expected under null
    expected_below = n * genome_wide_fraction

    if n_below <= expected_below:
        # No enrichment — skip permutation
        results.append({
            "pathway": pathway_name,
            "n_genes": n,
            "n_below": n_below,
            "expected_below": expected_below,
            "fold_enrichment": n_below / max(expected_below, 0.01),
            "mean_rank_pct": mean_rank,
            "perm_p": 1.0,
        })
        continue

    # Vectorized permutation test
    all_ranks_array = eas_ranks_pct.values
    perm_idx = np.random.randint(0, len(all_ranks_array), size=(N_PERM, n))
    perm_below = (all_ranks_array[perm_idx] < THRESHOLD_PCT).sum(axis=1)
    count_ge = (perm_below >= n_below).sum()
    perm_p = (count_ge + 1) / (N_PERM + 1)

    results.append({
        "pathway": pathway_name,
        "n_genes": n,
        "n_below": n_below,
        "expected_below": expected_below,
        "fold_enrichment": n_below / max(expected_below, 0.01),
        "mean_rank_pct": mean_rank,
        "perm_p": perm_p,
    })

results_df = pd.DataFrame(results)
print(f"\nTested {len(results_df)} pathways (5-500 genes)")

# --- FDR correction ---
reject, qvals, _, _ = multipletests(results_df["perm_p"], method="fdr_bh")
results_df["q_value"] = qvals
results_df["significant_fdr05"] = reject

# Sort by p-value
results_df = results_df.sort_values("perm_p")

# --- Report ---
print(f"\n=== Pathways significant at FDR < 0.05 ({reject.sum()} total) ===\n")
sig = results_df[results_df["significant_fdr05"]]
for _, row in sig.head(30).iterrows():
    print(f"  {row['pathway']}")
    print(f"    {row['n_below']}/{row['n_genes']} below {THRESHOLD_PCT}% "
          f"(expected {row['expected_below']:.1f}, {row['fold_enrichment']:.1f}x)")
    print(f"    mean rank: {row['mean_rank_pct']:.1f}%, p={row['perm_p']:.2e}, q={row['q_value']:.3f}")
    print()

# Check specifically for mucosal/IgA-related pathways
print("=== IgA / mucosal immunity related ===")
for _, row in results_df.iterrows():
    name_lower = row["pathway"].lower()
    if any(k in name_lower for k in ["iga", "mucos", "intestin", "innate", "toll", "nf-kappa",
                                       "nod-like", "dectin", "lectin", "b cell", "hematopoietic"]):
        print(f"  {row['pathway']}: {row['n_below']}/{row['n_genes']}, "
              f"p={row['perm_p']:.2e}, q={row['q_value']:.3f}")

# Save full results
outf = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pathway_enrichment_results.csv")
results_df.to_csv(outf, index=False)
print(f"\nFull results saved to: {outf}")
