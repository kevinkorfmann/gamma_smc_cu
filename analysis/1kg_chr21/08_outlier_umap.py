#!/usr/bin/env python3
"""UMAP using only rank-outlier genes as features."""
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
    'font.size': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

OUT = os.path.join(os.path.dirname(__file__), 'results')
adata = ad.read_h5ad(os.path.join(OUT, 'chr21_tmrca_genes_25k.h5ad'))

sp_colors = {'AFR':'#e41a1c','EUR':'#377eb8','EAS':'#4daf4a','SAS':'#984ea3','AMR':'#ff7f00'}

# Load rank outliers
outliers = pd.read_csv(
    os.path.join(os.path.dirname(__file__), 'selection_scan', 'top_sweep_candidates.csv'),
    index_col=0)

# Top 20 genes by rank range
top_genes = list(outliers.head(20).index)
top_genes = [g for g in top_genes if g in adata.var_names]
print(f"Using {len(top_genes)} outlier genes: {top_genes[:5]}...", flush=True)

# Subset to outlier genes only
X_outlier = adata[:, top_genes].X.copy()

# Encode population
pops = sorted(adata.obs['population'].unique())
pop_to_int = {p: i for i, p in enumerate(pops)}
y = np.array([pop_to_int[p] for p in adata.obs['population']])
pop_cmap = plt.cm.tab20(np.linspace(0, 1, len(pops)))

fig, axes = plt.subplots(2, 3, figsize=(17, 10))

configs = [
    ('All 214 genes\nunsupervised', adata.obsm['X_pca'][:, :50], None),
    (f'{len(top_genes)} outlier genes\nunsupervised', X_outlier, None),
    (f'{len(top_genes)} outlier genes\nsupervised (0.5)', X_outlier, y),
]

for col, (title, X, target) in enumerate(configs):
    print(f"UMAP: {title.split(chr(10))[0]}...", flush=True)
    kw = dict(n_neighbors=30, min_dist=0.3, random_state=42)
    if target is not None:
        kw['target_metric'] = 'categorical'
        kw['target_weight'] = 0.5
    reducer = umap.UMAP(**kw)
    emb = reducer.fit_transform(X, y=target)

    # Top: superpopulation
    ax = axes[0, col]
    for sp in ['AFR', 'EUR', 'EAS', 'SAS', 'AMR']:
        m = adata.obs['superpopulation'] == sp
        ax.scatter(emb[m, 0], emb[m, 1], s=0.5, alpha=0.4,
                   c=sp_colors[sp], label=sp, rasterized=True)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_aspect('equal')
    if col == 0:
        ax.legend(markerscale=10, fontsize=7)

    # Bottom: population
    ax = axes[1, col]
    for pi, pop in enumerate(pops):
        m = adata.obs['population'] == pop
        ax.scatter(emb[m, 0], emb[m, 1], s=0.5, alpha=0.4,
                   c=[pop_cmap[pi]], label=pop, rasterized=True)
    ax.set_aspect('equal')
    if col == 2:
        ax.legend(markerscale=10, fontsize=4, ncol=2, loc='upper right')

fig.suptitle('Effect of feature selection on population separation\n'
             'Left: all genes (noise dominates) → Right: outlier genes + supervision (signal emerges)',
             fontsize=10, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'umap_outlier_genes.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'umap_outlier_genes.pdf'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved umap_outlier_genes.png", flush=True)

# ── Superpopulation-level ─────────────────────────────────────
print("\nSuperpopulation analysis...", flush=True)
adata.obs['superpop_label'] = adata.obs['superpopulation']
sp_to_int = {sp: i for i, sp in enumerate(['AFR','EUR','SAS','EAS','AMR'])}
y_sp = np.array([sp_to_int[sp] for sp in adata.obs['superpopulation']])

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

configs_sp = [
    ('All genes, unsupervised', adata.obsm['X_pca'][:, :50], None),
    ('Outlier genes, unsupervised', X_outlier, None),
    ('Outlier genes, supervised\n(5 superpopulations)', X_outlier, y_sp),
]

for col, (title, X, target) in enumerate(configs_sp):
    print(f"  {title.split(chr(10))[0]}...", flush=True)
    kw = dict(n_neighbors=30, min_dist=0.3, random_state=42)
    if target is not None:
        kw['target_metric'] = 'categorical'
        kw['target_weight'] = 0.5
    emb = umap.UMAP(**kw).fit_transform(X, y=target)

    ax = axes[col]
    for sp in ['AFR', 'EUR', 'EAS', 'SAS', 'AMR']:
        m = adata.obs['superpopulation'] == sp
        ax.scatter(emb[m, 0], emb[m, 1], s=0.5, alpha=0.4,
                   c=sp_colors[sp], label=sp, rasterized=True)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_aspect('equal')
    if col == 0:
        ax.legend(markerscale=10, fontsize=8)

fig.suptitle('Superpopulation separation — from noise to signal', fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'umap_superpop.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'umap_superpop.pdf'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved umap_superpop.png", flush=True)
print("Done!", flush=True)
