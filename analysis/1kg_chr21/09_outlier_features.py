#!/usr/bin/env python3
"""Feature plots on the outlier-gene supervised UMAP."""
import numpy as np
import anndata as ad
import pandas as pd
import umap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUT = os.path.join(os.path.dirname(__file__), 'results')
adata = ad.read_h5ad(os.path.join(OUT, 'chr21_tmrca_genes_25k.h5ad'))

outliers = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'selection_scan', 'top_sweep_candidates.csv'),
    index_col=0)

top_genes = [g for g in outliers.head(20).index if g in adata.var_names]
X_outlier = adata[:, top_genes].X.copy()

# Supervised UMAP on outlier genes (superpopulation level)
sp_to_int = {sp: i for i, sp in enumerate(['AFR','EUR','SAS','EAS','AMR'])}
y_sp = np.array([sp_to_int[sp] for sp in adata.obs['superpopulation']])

print("Computing supervised UMAP...", flush=True)
reducer = umap.UMAP(target_metric='categorical', target_weight=0.5,
                     n_neighbors=30, min_dist=0.3, random_state=42)
emb = reducer.fit_transform(X_outlier, y=y_sp)

# Plot top 12 outlier genes on this embedding
show_genes = top_genes[:12]
nrows, ncols = 3, 4
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
axes = axes.flatten()

for i, gene in enumerate(show_genes):
    ax = axes[i]
    vals = adata[:, gene].X.flatten()

    sc_ = ax.scatter(emb[:, 0], emb[:, 1], c=vals, s=0.3, alpha=0.5,
                      cmap='magma_r', rasterized=True)
    ax.set_title(gene, fontsize=9, fontweight='bold')
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(sc_, ax=ax, shrink=0.6, pad=0.02)

    # Annotate which population is the outlier
    info = outliers.loc[gene]
    ax.text(0.02, 0.02, f"sweep: {info['sweep_pop']} ({info['sweep_percentile']:.0%})",
            transform=ax.transAxes, fontsize=5.5, color='white',
            bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))

fig.suptitle('Top 12 rank-outlier genes on supervised UMAP\n'
             'Color = log(1+TMRCA) at each gene — dark = recent coalescence',
             fontsize=10, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'umap_outlier_features_12.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'umap_outlier_features_12.pdf'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved umap_outlier_features_12.png", flush=True)

# Also: reference UMAP with superpop colors for comparison
sp_colors = {'AFR':'#e41a1c','EUR':'#377eb8','EAS':'#4daf4a','SAS':'#984ea3','AMR':'#ff7f00'}

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

# Panel 0: superpop reference
ax = axes[0]
for sp in ['AFR','EUR','EAS','SAS','AMR']:
    m = adata.obs['superpopulation'] == sp
    ax.scatter(emb[m,0], emb[m,1], s=0.5, alpha=0.4, c=sp_colors[sp], label=sp, rasterized=True)
ax.set_title('Superpopulation', fontsize=9, fontweight='bold')
ax.legend(markerscale=10, fontsize=7)
ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

# Panels 1-3: three most interesting genes
for i, gene in enumerate(['CFAP298', 'SLC37A1', 'KRTAP21-3']):
    if gene not in adata.var_names:
        continue
    ax = axes[i+1]
    vals = adata[:, gene].X.flatten()
    sc_ = ax.scatter(emb[:,0], emb[:,1], c=vals, s=0.5, alpha=0.5,
                      cmap='magma_r', rasterized=True)
    info = outliers.loc[gene]
    ax.set_title(f'{gene}\nsweep: {info["sweep_pop"]} (rank {info["sweep_percentile"]:.0%})',
                 fontsize=9, fontweight='bold')
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(sc_, ax=ax, shrink=0.6, pad=0.02)

fig.suptitle('Supervised UMAP — reference + top 3 sweep candidates', fontsize=10, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'umap_top3_sweeps.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'umap_top3_sweeps.pdf'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved umap_top3_sweeps.png", flush=True)
print("Done!", flush=True)
