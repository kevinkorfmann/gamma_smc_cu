#!/usr/bin/env python3
"""
Selection scan: within-population gene ranking to separate
gene-specific signal from demographic background.
"""
import numpy as np
import anndata as ad
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import rankdata
import os

rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUT = os.path.dirname(__file__)
RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
adata = ad.read_h5ad(os.path.join(RESULTS, 'chr21_tmrca_genes_25k.h5ad'))

genes = list(adata.var_names)
pops = sorted(adata.obs['population'].unique())
superpops = {p: adata.obs[adata.obs['population']==p]['superpopulation'].iloc[0] for p in pops}
sp_colors = {'AFR':'#e41a1c','EUR':'#377eb8','EAS':'#4daf4a','SAS':'#984ea3','AMR':'#ff7f00'}

n_genes = len(genes)
n_pops = len(pops)

# ── Step 1: Compute per-population mean TMRCA per gene ────────
print("Computing per-population gene means...", flush=True)
pop_gene_mean = np.zeros((n_pops, n_genes))
for pi, pop in enumerate(pops):
    mask = adata.obs['population'] == pop
    pop_gene_mean[pi] = adata[mask].X.mean(axis=0)

# ── Step 2: Rank genes within each population ─────────────────
print("Ranking genes within populations...", flush=True)
# rank 1 = lowest TMRCA (most recent coalescence = sweep candidate)
pop_gene_rank = np.zeros_like(pop_gene_mean)
for pi in range(n_pops):
    pop_gene_rank[pi] = rankdata(pop_gene_mean[pi]) / n_genes  # percentile [0,1]

rank_df = pd.DataFrame(pop_gene_rank, index=pops, columns=genes)

# ── Step 3: Find outlier genes (large rank variance across pops) ─
print("Finding outlier genes...", flush=True)
rank_std = rank_df.std(axis=0)
rank_range = rank_df.max(axis=0) - rank_df.min(axis=0)

# For each gene: which population has the lowest rank (sweep candidate)?
sweep_pop = rank_df.idxmin(axis=0)
sweep_rank = rank_df.min(axis=0)
# Which population has the highest rank (balancing / deep coalescence)?
deep_pop = rank_df.idxmax(axis=0)
deep_rank = rank_df.max(axis=0)

outlier_df = pd.DataFrame({
    'rank_std': rank_std,
    'rank_range': rank_range,
    'sweep_pop': sweep_pop,
    'sweep_percentile': sweep_rank,
    'deep_pop': deep_pop,
    'deep_percentile': deep_rank,
}).sort_values('rank_range', ascending=False)

outlier_df.to_csv(os.path.join(OUT, 'top_sweep_candidates.csv'))
print(f"Top 10 genes by rank range:", flush=True)
print(outlier_df.head(10).to_string(), flush=True)

# ── Plot 1: Heatmap of gene rank percentiles ──────────────────
print("\nPlotting heatmap...", flush=True)
# Sort genes by genomic position
gene_order = list(adata.var.sort_values('start').index)
# Sort populations by superpopulation
pop_order = sorted(pops, key=lambda p: (superpops[p], p))

fig, ax = plt.subplots(figsize=(18, 6))
data = rank_df.loc[pop_order, gene_order].values

im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=0, vmax=1,
               interpolation='nearest')
ax.set_yticks(range(n_pops))
ax.set_yticklabels([f"{p} ({superpops[p]})" for p in pop_order], fontsize=6)
# Only label every 10th gene
label_idx = list(range(0, len(gene_order), 10))
ax.set_xticks(label_idx)
ax.set_xticklabels([gene_order[i] for i in label_idx], fontsize=5, rotation=90)
ax.set_xlabel('Genes (genomic order)')
ax.set_ylabel('Population')
cb = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
cb.set_label('Within-population rank percentile\n(0=recent, 1=deep)', fontsize=7)
ax.set_title('Gene TMRCA rank percentile — demography-normalized', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'gene_rank_heatmap.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# ── Plot 2: Top outlier genes — rank across populations ───────
print("Plotting outliers...", flush=True)
top_genes = list(outlier_df.head(12).index)

fig, axes = plt.subplots(3, 4, figsize=(16, 9))
axes = axes.flatten()

for idx, gene in enumerate(top_genes):
    ax = axes[idx]
    ranks = rank_df[gene]

    # Sort by superpopulation for visual grouping
    sorted_pops = sorted(pops, key=lambda p: (superpops[p], p))
    colors = [sp_colors[superpops[p]] for p in sorted_pops]
    vals = [ranks[p] for p in sorted_pops]

    bars = ax.barh(range(n_pops), vals, color=colors, alpha=0.7, edgecolor='none')

    # Highlight the most extreme population
    min_idx = np.argmin(vals)
    max_idx = np.argmax(vals)
    bars[min_idx].set_alpha(1.0)
    bars[min_idx].set_edgecolor('black')
    bars[min_idx].set_linewidth(1)

    ax.axvline(0.5, color='#cccccc', linewidth=0.5, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_yticks(range(n_pops))
    ax.set_yticklabels(sorted_pops, fontsize=4)
    ax.set_title(f'{gene}\n(range={outlier_df.loc[gene,"rank_range"]:.2f})',
                 fontsize=8, fontweight='bold')
    ax.set_xlabel('Rank percentile', fontsize=6)
    ax.invert_yaxis()

fig.suptitle('Top 12 genes with largest rank variation across populations\n'
             '(demography-normalized: low rank = recent coalescence = sweep candidate)',
             fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'rank_outliers.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# ── Plot 3: Sweep candidates — lowest within-pop rank ─────────
print("Plotting sweep candidates...", flush=True)
# Genes where one population has rank < 0.05 (bottom 5%)
sweep_candidates = outlier_df[outlier_df['sweep_percentile'] < 0.1].head(10)

fig, axes = plt.subplots(2, 5, figsize=(18, 6))
axes = axes.flatten()

for idx, (gene, row) in enumerate(sweep_candidates.iterrows()):
    if idx >= 10:
        break
    ax = axes[idx]
    pop = row['sweep_pop']
    sp = superpops[pop]

    mask_pop = adata.obs['population'] == pop
    vals_pop = adata[mask_pop, gene].X.flatten()
    vals_other = adata[~mask_pop, gene].X.flatten()
    if len(vals_other) > 5000:
        vals_other = np.random.default_rng(42).choice(vals_other, 5000, replace=False)

    bins = np.linspace(min(vals_pop.min(), vals_other.min()),
                        max(vals_pop.max(), vals_other.max()), 35)
    ax.hist(vals_other, bins=bins, density=True, alpha=0.35, color='#999999',
            label='Others', edgecolor='none')
    ax.hist(vals_pop, bins=bins, density=True, alpha=0.8, color=sp_colors[sp],
            label=pop, edgecolor='none')
    ax.set_title(f'{gene}\n{pop} rank={row["sweep_percentile"]:.2f}',
                 fontsize=8, fontweight='bold')
    ax.set_yticks([])
    if idx == 0:
        ax.legend(fontsize=6)

for idx in range(len(sweep_candidates), 10):
    axes[idx].set_visible(False)

fig.suptitle('Sweep candidates: genes with lowest within-population rank\n'
             '(recent coalescence relative to that population\'s genome-wide average)',
             fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'sweep_candidates.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

print("Done!", flush=True)
