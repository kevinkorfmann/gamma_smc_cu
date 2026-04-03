#!/usr/bin/env python3
"""Supervised UMAP: population-guided embedding of TMRCA landscape."""
import numpy as np
import anndata as ad
import scanpy as sc
import umap
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

sp_colors = {'AFR':'#e41a1c','EUR':'#377eb8','EAS':'#4daf4a','SAS':'#984ea3','AMR':'#ff7f00'}
pop_cmap = plt.cm.tab20(np.linspace(0, 1, 26))

# Encode population labels as integers for UMAP
pops_sorted = sorted(adata.obs['population'].unique())
pop_to_int = {p: i for i, p in enumerate(pops_sorted)}
y = np.array([pop_to_int[p] for p in adata.obs['population']])

# Use PCA components as input
X = adata.obsm['X_pca'][:, :50]

# ── Supervised UMAP at different target_weights ───────────────
weights = [0.0, 0.3, 0.5, 0.8]

fig, axes = plt.subplots(2, len(weights), figsize=(5 * len(weights), 9))

for wi, tw in enumerate(weights):
    print(f"UMAP target_weight={tw}...", flush=True)
    reducer = umap.UMAP(target_metric='categorical', target_weight=tw,
                         n_neighbors=30, min_dist=0.3, random_state=42)
    emb = reducer.fit_transform(X, y=y if tw > 0 else None)

    # Top row: superpopulation
    ax = axes[0, wi]
    for sp in ['AFR', 'EUR', 'EAS', 'SAS', 'AMR']:
        m = adata.obs['superpopulation'] == sp
        ax.scatter(emb[m, 0], emb[m, 1], s=0.5, alpha=0.4,
                   c=sp_colors[sp], label=sp, rasterized=True)
    ax.set_title(f'weight={tw}' + (' (unsupervised)' if tw == 0 else ''),
                 fontsize=9, fontweight='bold')
    ax.set_aspect('equal')
    if wi == 0:
        ax.set_ylabel('Superpopulation')
        ax.legend(markerscale=10, fontsize=7, loc='upper left')

    # Bottom row: population
    ax = axes[1, wi]
    for pi, pop in enumerate(pops_sorted):
        m = adata.obs['population'] == pop
        ax.scatter(emb[m, 0], emb[m, 1], s=0.5, alpha=0.4,
                   c=[pop_cmap[pi]], label=pop, rasterized=True)
    ax.set_aspect('equal')
    if wi == 0:
        ax.set_ylabel('Population')

    # Save the best supervised embedding
    if tw == 0.5:
        adata.obsm['X_umap_supervised'] = emb

fig.suptitle('Supervised UMAP — increasing population guidance\n'
             '1000 Genomes chr21 TMRCA, 214 protein-coding genes',
             fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'umap_supervised_weights.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'umap_supervised_weights.pdf'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved umap_supervised_weights.png", flush=True)

# ── Feature plots on supervised UMAP ──────────────────────────
emb = adata.obsm['X_umap_supervised']
top_genes = ['CFAP298', 'KRTAP21-3', 'SLC37A1', 'TIAM1', 'NRIP1', 'BACE2']
top_genes = [g for g in top_genes if g in adata.var_names]

fig, axes = plt.subplots(1, len(top_genes), figsize=(3.8 * len(top_genes), 3.5))
for i, gene in enumerate(top_genes):
    ax = axes[i]
    vals = adata[:, gene].X.flatten()
    sc_ = ax.scatter(emb[:, 0], emb[:, 1], c=vals, s=0.3, alpha=0.5,
                      cmap='viridis', rasterized=True)
    ax.set_title(gene, fontsize=10, fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(sc_, ax=ax, shrink=0.6, pad=0.02)

fig.suptitle('Supervised UMAP colored by gene TMRCA — selection candidates',
             fontsize=10, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'umap_supervised_features.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'umap_supervised_features.pdf'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved umap_supervised_features.png", flush=True)

# Save
adata.write(os.path.join(OUT, 'chr21_tmrca_genes_25k.h5ad'))
print("Done!", flush=True)
