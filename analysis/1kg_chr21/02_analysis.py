#!/usr/bin/env python3
"""Quick analysis: subsample 25k pairs, PCA 100 comps, UMAP."""
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time

OUT = os.path.join(os.path.dirname(__file__), 'results')
H5AD = os.path.join(OUT, 'chr21_tmrca.h5ad')

print("Loading...", flush=True)
adata = ad.read_h5ad(H5AD)
print(f"  Full: {adata.shape}", flush=True)

# Subsample 25k, stratified by population
print("Subsampling 25k...", flush=True)
rng = np.random.default_rng(42)
pops = adata.obs['population'].unique()
n_per_pop = max(25000 // len(pops), 100)
idx = []
for pop in pops:
    mask = np.where(adata.obs['population'] == pop)[0]
    n = min(n_per_pop, len(mask))
    idx.extend(rng.choice(mask, n, replace=False).tolist())
idx = sorted(idx)
adata = adata[idx].copy()
print(f"  Subsampled: {adata.shape}", flush=True)

# Log transform
adata.X = np.log1p(adata.X)

# PCA
print("PCA...", flush=True)
t0 = time.perf_counter()
sc.pp.pca(adata, n_comps=100)
print(f"  {time.perf_counter()-t0:.1f}s", flush=True)

# Neighbors + UMAP
print("UMAP...", flush=True)
t0 = time.perf_counter()
sc.pp.neighbors(adata, n_pcs=50)
sc.tl.umap(adata)
print(f"  {time.perf_counter()-t0:.1f}s", flush=True)

# Save
adata.write(os.path.join(OUT, 'chr21_tmrca_25k.h5ad'))

# Plot
sp_colors = {'AFR':'#e41a1c','EUR':'#377eb8','EAS':'#4daf4a','SAS':'#984ea3','AMR':'#ff7f00'}
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

umap = adata.obsm['X_umap']
ax = axes[0]
for sp in ['AFR','EUR','EAS','SAS','AMR']:
    m = adata.obs['superpopulation'] == sp
    ax.scatter(umap[m,0], umap[m,1], s=1, alpha=0.5, c=sp_colors[sp], label=sp, rasterized=True)
ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
ax.set_title('Superpopulation'); ax.legend(markerscale=8, fontsize=9)

ax = axes[1]
pops_sorted = sorted(adata.obs['population'].unique())
cm = plt.cm.tab20(np.linspace(0,1,len(pops_sorted)))
for i,p in enumerate(pops_sorted):
    m = adata.obs['population'] == p
    ax.scatter(umap[m,0], umap[m,1], s=1, alpha=0.5, c=[cm[i]], label=p, rasterized=True)
ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
ax.set_title('Population'); ax.legend(markerscale=8, fontsize=5, ncol=2)

fig.suptitle('1000 Genomes chr21 — TMRCA UMAP (25k pairs subsample)', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'umap_tmrca.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'umap_tmrca.pdf'), dpi=200, bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(range(1,21), adata.uns['pca']['variance_ratio'][:20]*100)
ax.set_xlabel('PC'); ax.set_ylabel('Variance explained (%)')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'pca_variance.png'), dpi=200)
plt.close(fig)

print("Done!", flush=True)
