#!/usr/bin/env python3
"""Gene-level TMRCA violin plots: top differential + known genes."""
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUT = os.path.join(os.path.dirname(__file__), 'results')
adata = ad.read_h5ad(os.path.join(OUT, 'chr21_tmrca_genes_25k.h5ad'))

sp_order = ['AFR', 'EUR', 'SAS', 'EAS', 'AMR']
sp_colors = {'AFR':'#e41a1c','EUR':'#377eb8','EAS':'#4daf4a','SAS':'#984ea3','AMR':'#ff7f00'}

# ── Panel 1: Top differential genes (one per superpopulation) ─
top_genes = ['BACE2', 'ADAMTS5', 'NRIP1', 'EVA1C', 'RRP1']  # YRI, CEU, CHB, PEL, GIH
known_genes = ['APP', 'SOD1', 'DYRK1A', 'RUNX1', 'CBS', 'ERG']
known_genes = [g for g in known_genes if g in adata.var_names]

fig, axes = plt.subplots(2, max(len(top_genes), len(known_genes)),
                          figsize=(3.2 * max(len(top_genes), len(known_genes)), 7),
                          squeeze=False)

for col, gene in enumerate(top_genes):
    ax = axes[0, col]
    vals = {sp: adata[adata.obs['superpopulation'] == sp, gene].X.flatten()
            for sp in sp_order}
    parts = ax.violinplot([vals[sp] for sp in sp_order], showmedians=True, showextrema=False)
    for i, (pc, sp) in enumerate(zip(parts['bodies'], sp_order)):
        pc.set_facecolor(sp_colors[sp])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    ax.set_xticks(range(1, len(sp_order)+1))
    ax.set_xticklabels(sp_order, fontsize=7)
    ax.set_title(gene, fontsize=10, fontweight='bold')
    if col == 0:
        ax.set_ylabel('log(1 + TMRCA)')

for col, gene in enumerate(known_genes):
    ax = axes[1, col]
    vals = {sp: adata[adata.obs['superpopulation'] == sp, gene].X.flatten()
            for sp in sp_order}
    parts = ax.violinplot([vals[sp] for sp in sp_order], showmedians=True, showextrema=False)
    for i, (pc, sp) in enumerate(zip(parts['bodies'], sp_order)):
        pc.set_facecolor(sp_colors[sp])
        pc.set_alpha(0.7)
    parts['cmedians'].set_color('black')
    ax.set_xticks(range(1, len(sp_order)+1))
    ax.set_xticklabels(sp_order, fontsize=7)
    ax.set_title(gene, fontsize=10, fontweight='bold')
    if col == 0:
        ax.set_ylabel('log(1 + TMRCA)')

# Hide unused axes
for row in range(2):
    n = len(top_genes) if row == 0 else len(known_genes)
    for col in range(n, axes.shape[1]):
        axes[row, col].set_visible(False)

axes[0, 0].text(-0.3, 1.15, 'Top differential genes', transform=axes[0,0].transAxes,
                fontsize=11, fontweight='bold')
axes[1, 0].text(-0.3, 1.15, 'Known chr21 genes', transform=axes[1,0].transAxes,
                fontsize=11, fontweight='bold')

fig.suptitle('Per-gene TMRCA by superpopulation — chr21, 1000 Genomes', fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'gene_violins.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'gene_violins.pdf'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved gene_violins.png", flush=True)

# ── Panel 2: UMAP colored by specific gene TMRCA ─────────────
genes_to_show = ['BACE2', 'APP', 'DYRK1A', 'SOD1', 'NRIP1', 'CBS']
genes_to_show = [g for g in genes_to_show if g in adata.var_names]
n = len(genes_to_show)

fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.2))
umap = adata.obsm['X_umap']

for i, gene in enumerate(genes_to_show):
    ax = axes[i]
    vals = adata[:, gene].X.flatten()
    sc_ = ax.scatter(umap[:, 0], umap[:, 1], c=vals, s=0.3, alpha=0.5,
                      cmap='viridis', rasterized=True)
    ax.set_title(gene, fontsize=10, fontweight='bold')
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2') if i == 0 else None
    ax.set_aspect('equal')
    plt.colorbar(sc_, ax=ax, shrink=0.7, pad=0.02)

fig.suptitle('UMAP colored by gene-level TMRCA', fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'gene_umap_features.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'gene_umap_features.pdf'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved gene_umap_features.png", flush=True)
