#!/usr/bin/env python3
"""Per-population top gene: outlier population vs all others."""
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
    'font.size': 7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
})

OUT = os.path.join(os.path.dirname(__file__), 'results')
adata = ad.read_h5ad(os.path.join(OUT, 'chr21_tmrca_genes_25k.h5ad'))

sp_colors = {'AFR':'#e41a1c','EUR':'#377eb8','EAS':'#4daf4a','SAS':'#984ea3','AMR':'#ff7f00'}
result = adata.uns['rank_genes_groups']
all_pops = sorted(result['names'].dtype.names)

# Get top gene per population
pop_info = []
for pop in all_pops:
    gene = str(result['names'][pop][0])
    score = float(result['scores'][pop][0])
    lfc = float(result['logfoldchanges'][pop][0])
    sp = adata.obs[adata.obs['population'] == pop]['superpopulation'].iloc[0]
    pop_info.append((pop, sp, gene, score, lfc))

# 26 populations → 6x5 grid (with one empty)
nrows, ncols = 6, 5
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.2))
axes_flat = axes.flatten()

for idx, (pop, sp, gene, score, lfc) in enumerate(pop_info):
    ax = axes_flat[idx]

    # Get values for this population and all others
    mask_pop = adata.obs['population'] == pop
    mask_other = ~mask_pop

    vals_pop = adata[mask_pop, gene].X.flatten()
    vals_other = adata[mask_other, gene].X.flatten()

    # Subsample others for speed
    if len(vals_other) > 5000:
        vals_other = np.random.default_rng(42).choice(vals_other, 5000, replace=False)

    color = sp_colors[sp]

    # Histogram: others in gray, population in color
    bins = np.linspace(
        min(vals_pop.min(), vals_other.min()),
        max(vals_pop.max(), vals_other.max()),
        40)

    ax.hist(vals_other, bins=bins, density=True, alpha=0.4, color='#999999',
            label='Other pops', edgecolor='none')
    ax.hist(vals_pop, bins=bins, density=True, alpha=0.75, color=color,
            label=pop, edgecolor='none')

    ax.set_title(f'{pop} ({sp}) — {gene}', fontsize=8, fontweight='bold')
    ax.text(0.97, 0.95, f'score={score:.0f}\nlogFC={lfc:+.2f}',
            transform=ax.transAxes, fontsize=5.5, ha='right', va='top',
            color='#444444')

    if idx == 0:
        ax.legend(fontsize=5.5, loc='upper left')

    ax.set_yticks([])
    if idx >= (nrows - 1) * ncols:
        ax.set_xlabel('log(1+TMRCA)', fontsize=7)

# Hide unused axes
for idx in range(len(pop_info), nrows * ncols):
    axes_flat[idx].set_visible(False)

fig.suptitle('Top differential gene per population — outlier vs rest\n'
             '1000 Genomes chr21, 26 populations',
             fontsize=10, y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'pop_outlier_genes.png'), dpi=200, bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'pop_outlier_genes.pdf'), dpi=200, bbox_inches='tight')
plt.close(fig)
print("Saved pop_outlier_genes.png")
