#!/usr/bin/env python3
"""Compute FDR statistics for the manuscript.

Outputs:
1. Number of SD-masked genes below 1% in at least one population
2. Pairwise Spearman rank correlation within each continental group
3. Effective number of independent populations (Galwey 2009)
4. Per-candidate replication p-values under independence and correlation-adjusted models
"""
import numpy as np
import pandas as pd
from scipy import stats
from math import comb

REPO = "/vast/projects/smathi/cohort/kkor/tmrca.cu"
RANKS = f"{REPO}/analysis/genome_wide/results/genome_wide_ranks.csv"
SD_FLAG = f"{REPO}/analysis/genome_wide/postprocess/genes_sd_flag.csv"

# Continental groups
GROUPS = {
    "AFR": ["ACB", "ASW", "ESN", "GWD", "LWK", "MSL", "YRI"],
    "EUR": ["CEU", "FIN", "GBR", "IBS", "TSI"],
    "EAS": ["CDX", "CHB", "CHS", "JPT", "KHV"],
    "SAS": ["BEB", "GIH", "ITU", "PJL", "STU"],
    "AMR": ["CLM", "MXL", "PEL", "PUR"],
}

# Novel candidates with their replication patterns
CANDIDATES = [
    # (gene, continental_group, pops_below_1pct, pops_below_5pct, n_pops_in_group)
    ("GRK2", "SAS+EUR", 9, 9, 10),  # 9 of 10 W. Eurasian (5 SAS + 5 EUR)
    ("SLC6A15", "EAS", 5, 5, 5),
    ("CCDC92", "EAS", 1, 3, 5),
    ("CLEC6A", "EAS", 1, 5, 5),  # below 10% in all 5, below 5% needs checking
    ("BPIFA2", "SAS", 1, 5, 5),
]

print("=" * 70)
print("FDR STATISTICS FOR MANUSCRIPT")
print("=" * 70)

# ── Load data ──
df = pd.read_csv(RANKS)
sd = pd.read_csv(SD_FLAG)

# Merge SD flag
sd_genes = set(sd[sd["is_sd"] == True]["gene_name"])
df["is_sd"] = df["gene_name"].isin(sd_genes)

rank_cols = [f"{pop}_rank" for pop in sum(GROUPS.values(), [])]
# Check which rank cols exist
rank_cols = [c for c in rank_cols if c in df.columns]

print(f"\nTotal genes: {len(df)}")
print(f"SD-flagged: {df['is_sd'].sum()}")
print(f"SD-masked (non-SD): {(~df['is_sd']).sum()}")

# ── 1. Gene counts at thresholds ──
df_clean = df[~df["is_sd"]].copy()
n_clean = len(df_clean)

for threshold in [0.01, 0.05, 0.10]:
    # Gene has min rank below threshold in at least one population
    below = (df_clean[rank_cols] < threshold).any(axis=1).sum()
    pct = 100 * below / n_clean
    print(f"\nSD-masked genes below {threshold*100:.0f}% in >= 1 pop: {below} ({pct:.1f}%)")
    # Expected under independence (26 pops)
    n_pops = len(rank_cols)
    p_any = 1 - (1 - threshold) ** n_pops
    expected = n_clean * p_any
    print(f"  Expected under independence ({n_pops} pops): {expected:.0f}")
    # Expected per single population
    expected_single = n_clean * threshold
    print(f"  Expected per single population: {expected_single:.0f}")

# ── 2. Pairwise Spearman correlation within continental groups ──
print("\n" + "=" * 70)
print("WITHIN-GROUP RANK CORRELATIONS")
print("=" * 70)

for group, pops in GROUPS.items():
    cols = [f"{p}_rank" for p in pops if f"{p}_rank" in df_clean.columns]
    if len(cols) < 2:
        continue
    rank_mat = df_clean[cols].dropna()
    
    n = len(cols)
    corrs = []
    for i in range(n):
        for j in range(i+1, n):
            rho, _ = stats.spearmanr(rank_mat.iloc[:, i], rank_mat.iloc[:, j])
            corrs.append(rho)
    
    mean_rho = np.mean(corrs)
    min_rho = np.min(corrs)
    max_rho = np.max(corrs)
    
    # Effective number of independent tests (Galwey 2009 / Li & Ji 2005)
    # n_eff = n / (1 + (n-1) * mean_rho)  [simplified for equal correlations]
    # More precisely, eigenvalue-based: n_eff = sum(I(lambda_i >= 1)) + sum(lambda_i - floor(lambda_i)) for lambda_i < 1
    corr_matrix = rank_mat.corr(method="spearman").values
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = eigenvalues[eigenvalues > 0]  # numerical stability
    # Galwey (2009) formula
    n_eff_galwey = (np.sum(np.sqrt(eigenvalues)))**2 / np.sum(eigenvalues)
    # Simple formula for comparison
    n_eff_simple = n / (1 + (n - 1) * mean_rho)
    
    print(f"\n{group} ({n} populations: {', '.join(pops)}):")
    print(f"  Mean pairwise Spearman rho: {mean_rho:.4f}")
    print(f"  Range: [{min_rho:.4f}, {max_rho:.4f}]")
    print(f"  n_eff (Galwey 2009): {n_eff_galwey:.2f}")
    print(f"  n_eff (simple 1/(1+(n-1)*rho)): {n_eff_simple:.2f}")

# ── 3. Per-candidate replication p-values ──
print("\n" + "=" * 70)
print("PER-CANDIDATE REPLICATION P-VALUES")
print("=" * 70)

def binom_p(k, n, alpha):
    """P(K >= k) under Binomial(n, alpha)"""
    p = 0
    for j in range(k, n+1):
        p += comb(n, j) * alpha**j * (1-alpha)**(n-j)
    return p

# For each candidate, compute actual replication counts from data
print("\nLooking up actual per-population ranks for candidates...")
for gene_name, group, _, _, _ in CANDIDATES:
    row = df_clean[df_clean["gene_name"] == gene_name]
    if len(row) == 0:
        row = df[df["gene_name"] == gene_name]  # might be SD-flagged
    if len(row) == 0:
        print(f"  {gene_name}: NOT FOUND in rank table")
        continue
    row = row.iloc[0]
    
    # Determine populations to check
    if group == "SAS+EUR":
        pops = GROUPS["SAS"] + GROUPS["EUR"]
    else:
        pops = GROUPS[group]
    
    n = len(pops)
    ranks = {}
    below_1 = 0
    below_5 = 0
    below_10 = 0
    for pop in pops:
        col = f"{pop}_rank"
        if col in row.index and pd.notna(row[col]):
            r = row[col]
            ranks[pop] = r
            if r < 0.01:
                below_1 += 1
            if r < 0.05:
                below_5 += 1
            if r < 0.10:
                below_10 += 1
    
    print(f"\n  {gene_name} ({group}, n={n}):")
    # Print individual ranks
    for pop in sorted(ranks, key=lambda p: ranks[p]):
        r = ranks[pop]
        flag = "*" if r < 0.01 else ("+" if r < 0.05 else "")
        print(f"    {pop}: {r*100:.2f}%{flag}")
    
    print(f"    Below 1%: {below_1}/{n}")
    print(f"    Below 5%: {below_5}/{n}")
    print(f"    Below 10%: {below_10}/{n}")
    
    # P-values under independence
    for alpha, k in [(0.01, below_1), (0.05, below_5)]:
        if k > 0:
            p = binom_p(k, n, alpha)
            expected_fp = n_clean * p
            print(f"    P(K>={k} | n={n}, alpha={alpha}) = {p:.2e}  "
                  f"(expected FP genome-wide: {expected_fp:.4f})")
    
    # Correlation-adjusted p-value using n_eff
    if group == "SAS+EUR":
        # Combine SAS and EUR n_eff
        sas_cols = [f"{p}_rank" for p in GROUPS["SAS"] if f"{p}_rank" in df_clean.columns]
        eur_cols = [f"{p}_rank" for p in GROUPS["EUR"] if f"{p}_rank" in df_clean.columns]
        sas_corr = df_clean[sas_cols].dropna().corr(method="spearman").values
        eur_corr = df_clean[eur_cols].dropna().corr(method="spearman").values
        sas_eig = np.linalg.eigvalsh(sas_corr); sas_eig = sas_eig[sas_eig > 0]
        eur_eig = np.linalg.eigvalsh(eur_corr); eur_eig = eur_eig[eur_eig > 0]
        neff_sas = (np.sum(np.sqrt(sas_eig)))**2 / np.sum(sas_eig)
        neff_eur = (np.sum(np.sqrt(eur_eig)))**2 / np.sum(eur_eig)
        neff = neff_sas + neff_eur  # cross-continental, so add
        print(f"    n_eff (SAS={neff_sas:.2f} + EUR={neff_eur:.2f} = {neff:.2f})")
    else:
        cols = [f"{p}_rank" for p in GROUPS[group] if f"{p}_rank" in df_clean.columns]
        corr_mat = df_clean[cols].dropna().corr(method="spearman").values
        eig = np.linalg.eigvalsh(corr_mat); eig = eig[eig > 0]
        neff = (np.sum(np.sqrt(eig)))**2 / np.sum(eig)
        print(f"    n_eff ({group}): {neff:.2f}")
    
    # Conservative p-value: use n_eff, scale k proportionally
    k_adj = max(1, int(np.ceil(below_5 * neff / n)))
    n_adj = max(1, int(np.round(neff)))
    p_adj = binom_p(k_adj, n_adj, 0.05)
    expected_fp_adj = n_clean * p_adj
    print(f"    Adjusted: P(K>={k_adj} | n_eff={n_adj}, alpha=0.05) = {p_adj:.2e}  "
          f"(expected FP: {expected_fp_adj:.4f})")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
