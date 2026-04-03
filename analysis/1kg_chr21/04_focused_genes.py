#!/usr/bin/env python3
"""Focused gene plots: highlight the population where each gene differs."""
import numpy as np
import anndata as ad
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

# Top signal per superpopulation
highlights = [
    ('TIAM1',  'AFR', 'Recent coalescence in African populations'),
    ('NRIP1',  'EAS', 'Recent coalescence in East Asian populations'),
    ('RRP1',   'SAS', 'Recent coalescence in South Asian populations'),
    ('ADAMTS5','EUR', 'Recent coalescence in European populations'),
    ('EVA1C',  'AMR', 'Recent coalescence in American populations'),
    ('BACE2',  'AFR', 'Alzheimer-related gene, strong African signal'),
]

sp_colors = {'AFR':'#e41a1c','EUR':'#377eb8','EAS':'#4daf4a','SAS':'#984ea3','AMR':'#ff7f00'}
sp_order = ['AFR', 'EUR', 'SAS', 'EAS', 'AMR']

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for idx, (gene, highlight_sp, desc) in enumerate(highlights):
    ax = axes[idx]

    for si, sp in enumerate(sp_order):
        mask = adata.obs['superpopulation'] == sp
        vals = adata[mask, gene].X.flatten()

        is_highlight = (sp == highlight_sp)
        color = sp_colors[sp]
        alpha = 0.8 if is_highlight else 0.3

        parts = ax.violinplot([vals], positions=[si], showmedians=True, showextrema=False,
                               widths=0.7)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(alpha)
            pc.set_edgecolor(color if is_highlight else '#cccccc')
            pc.set_linewidth(1.5 if is_highlight else 0.5)
        parts['cmedians'].set_color('black' if is_highlight else '#999999')
        parts['cmedians'].set_linewidth(1.5 if is_highlight else 0.5)

    ax.set_xticks(range(len(sp_order)))
    ax.set_xticklabels(sp_order, fontsize=8)
    ax.set_title(gene, fontsize=12, fontweight='bold')
    ax.set_ylabel('log(1 + TMRCA)' if idx % 3 == 0 else '')
    ax.text(0.02, 0.02, desc, transform=ax.transAxes, fontsize=6.5,
            color='#555555', va='bottom')

fig.suptitle('Population-specific TMRCA signals — chr21 protein-coding genes\n'
             '1000 Genomes, 25k within-population pairs',
             fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'gene_highlights.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'gene_highlights.pdf'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved gene_highlights.png")

# Also make UMAP with highlighted population per gene
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()
umap = adata.obsm['X_umap']

for idx, (gene, highlight_sp, desc) in enumerate(highlights):
    ax = axes[idx]
    # Background: all points gray
    ax.scatter(umap[:, 0], umap[:, 1], s=0.2, c='#dddddd', alpha=0.3, rasterized=True)
    # Highlight population colored by gene value
    mask = adata.obs['superpopulation'] == highlight_sp
    vals = adata[mask, gene].X.flatten()
    sc_ = ax.scatter(umap[mask, 0], umap[mask, 1], s=0.5, c=vals,
                      cmap='magma_r', alpha=0.7, rasterized=True)
    ax.set_title(f'{gene} — {highlight_sp}', fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(sc_, ax=ax, shrink=0.6, pad=0.02, label='log(1+TMRCA)')

fig.suptitle('UMAP: population-specific TMRCA at top differential genes', fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'gene_umap_highlights.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'gene_umap_highlights.pdf'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved gene_umap_highlights.png")
