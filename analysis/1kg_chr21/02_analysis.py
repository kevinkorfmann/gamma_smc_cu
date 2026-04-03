#!/usr/bin/env python3
"""
1000 Genomes chr21: TMRCA analysis.
25k subsample → PCA → UMAP → differential TMRCA windows between populations.
"""
import numpy as np
import anndata as ad
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, time

OUT = os.path.join(os.path.dirname(__file__), 'results')
H5AD = os.path.join(OUT, 'chr21_tmrca_genes.h5ad')

# ── Load + subsample ─────────────────────────────────────────
print("Loading...", flush=True)
adata = ad.read_h5ad(H5AD)
print(f"  Full: {adata.shape}", flush=True)

print("Subsampling 25k...", flush=True)
rng = np.random.default_rng(42)
pops = adata.obs['population'].unique()
n_per = max(25000 // len(pops), 100)
idx = []
for pop in pops:
    mask = np.where(adata.obs['population'] == pop)[0]
    idx.extend(rng.choice(mask, min(n_per, len(mask)), replace=False).tolist())
adata = adata[sorted(idx)].copy()
print(f"  {adata.shape}", flush=True)

# ── Preprocess ────────────────────────────────────────────────
adata.X = np.log1p(adata.X)

# ── PCA (50 comps for UMAP) ──────────────────────────────────
print("PCA...", flush=True)
t0 = time.perf_counter()
sc.pp.pca(adata, n_comps=100)
print(f"  {time.perf_counter()-t0:.1f}s", flush=True)

# ── UMAP ──────────────────────────────────────────────────────
print("UMAP...", flush=True)
t0 = time.perf_counter()
sc.pp.neighbors(adata, n_pcs=50)
sc.tl.umap(adata)
print(f"  {time.perf_counter()-t0:.1f}s", flush=True)

# ── Differential TMRCA windows (on raw log-TMRCA, not PCA) ───
print("Differential TMRCA windows...", flush=True)
t0 = time.perf_counter()
sc.tl.rank_genes_groups(adata, groupby='population', method='wilcoxon')
print(f"  {time.perf_counter()-t0:.1f}s", flush=True)

# ── Save ──────────────────────────────────────────────────────
adata.write(os.path.join(OUT, 'chr21_tmrca_genes_25k.h5ad'))

# ── Plots ─────────────────────────────────────────────────────
print("Plotting...", flush=True)
sp_colors = {'AFR':'#e41a1c','EUR':'#377eb8','EAS':'#4daf4a','SAS':'#984ea3','AMR':'#ff7f00'}

# 1. UMAP by superpopulation + population
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
umap = adata.obsm['X_umap']

ax = axes[0]
for sp in ['AFR','EUR','EAS','SAS','AMR']:
    m = adata.obs['superpopulation'] == sp
    ax.scatter(umap[m,0], umap[m,1], s=1, alpha=0.5, c=sp_colors[sp], label=sp, rasterized=True)
ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
ax.set_title('Superpopulation'); ax.legend(markerscale=8)

ax = axes[1]
pops_s = sorted(adata.obs['population'].unique())
cm = plt.cm.tab20(np.linspace(0,1,len(pops_s)))
for i,p in enumerate(pops_s):
    m = adata.obs['population'] == p
    ax.scatter(umap[m,0], umap[m,1], s=1, alpha=0.5, c=[cm[i]], label=p, rasterized=True)
ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
ax.set_title('Population'); ax.legend(markerscale=8, fontsize=5, ncol=2)

fig.suptitle('1000 Genomes chr21 — Pairwise TMRCA UMAP (25k pairs)', fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'umap_tmrca.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'umap_tmrca.pdf'), dpi=200, bbox_inches='tight')
plt.close(fig)

# 2. Top differential windows — dotplot style
sc.pl.rank_genes_groups_dotplot(adata, n_genes=5, show=False, save='_diff.png')

# 3. Top differential windows — genomic Manhattan-style plot
fig, ax = plt.subplots(figsize=(14, 4))
result = adata.uns['rank_genes_groups']
# For each pop, plot the -log10(pval) of top windows along the chromosome
for pi, pop in enumerate(['YRI', 'CEU', 'CHB', 'PEL', 'GIH']):
    pop_idx = list(result['names'].dtype.names).index(pop)
    names = result['names'][pop][:20]
    pvals = result['pvals_adj'][pop][:20]
    # Extract genomic position from window name (chr21:start-end)
    positions = []
    for n in names:
        start = int(adata.var.loc[str(n), 'start'])
        positions.append(start / 1e6)
    logp = -np.log10(np.maximum(pvals, 1e-300))
    ax.scatter(positions, logp, s=20, alpha=0.7, label=pop)

ax.set_xlabel('Position on chr21 (Mb)')
ax.set_ylabel('-log10(adjusted p-value)')
ax.set_title('Top 20 differential TMRCA windows per population')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.1)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'diff_windows_manhattan.png'), dpi=200, bbox_inches='tight')
plt.close(fig)

# 4. PCA variance
fig, ax = plt.subplots(figsize=(6, 3))
ax.bar(range(1, 21), adata.uns['pca']['variance_ratio'][:20] * 100)
ax.set_xlabel('PC'); ax.set_ylabel('Variance explained (%)')
ax.set_title('PCA — top 20 components')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'pca_variance.png'), dpi=200)
plt.close(fig)

# 5. Heatmap of top windows across populations
sc.pl.rank_genes_groups_heatmap(adata, n_genes=3, show=False,
                                       groupby='population', show_gene_labels=True, save='_diff.png')

print("Done!", flush=True)
