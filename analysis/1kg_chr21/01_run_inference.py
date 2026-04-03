#!/usr/bin/env python3
"""
1000 Genomes chr21: within-population pairwise TMRCA inference.
Gene-level features from GENCODE protein-coding genes.
"""
import numpy as np
import anndata as ad
import pandas as pd
import scipy.sparse as sp
import h5py
import os, sys, time, gc, re
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
from tmrca_cu import _core

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
CACHE_PATH = os.path.join(DATA_DIR, 'chr21_parsed.npz')
META_PATH = os.path.join(DATA_DIR, 'samples.txt')
GENES_URL = 'https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.basic.annotation.gtf.gz'
GENES_CACHE = os.path.join(DATA_DIR, 'chr21_genes.tsv')
FF = '/sietch_colab/kkor/gamma_smc/resources/default_flow_field.txt'
H5AD_PATH = os.path.join(OUT_DIR, 'chr21_tmrca_genes.h5ad')
MU, RHO, NE = 1.25e-8, 1e-8, 10000
CHUNK_PAIRS = 2000
N_GPUS = min(_core.get_device_count(), 3)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Step 1: Load genotypes ────────────────────────────────────
print("Loading genotypes...", flush=True)
t0 = time.perf_counter()
data = np.load(CACHE_PATH, allow_pickle=True)
G = data['G']; pos = data['positions']
sample_ids = [str(s) for s in data['sample_ids']]
n_haps, n_sites = G.shape
print(f"  {n_haps} haplotypes, {n_sites:,} sites ({time.perf_counter()-t0:.1f}s)", flush=True)

# ── Step 2: Load gene annotations ────────────────────────────
if os.path.exists(GENES_CACHE):
    print("Loading cached gene annotations...", flush=True)
    genes_df = pd.read_csv(GENES_CACHE, sep='\t')
else:
    print("Downloading GENCODE gene annotations...", flush=True)
    import subprocess, gzip, io
    raw = subprocess.check_output(f'curl -sL {GENES_URL}', shell=True)
    text = gzip.decompress(raw).decode()
    records = []
    for line in text.split('\n'):
        if line.startswith('#') or not line.strip():
            continue
        fields = line.split('\t')
        if fields[0] != 'chr21' or fields[2] != 'gene':
            continue
        attrs = fields[8]
        if 'protein_coding' not in attrs:
            continue
        gene_name = re.search(r'gene_name "([^"]+)"', attrs)
        gene_id = re.search(r'gene_id "([^"]+)"', attrs)
        if gene_name:
            records.append({
                'gene_id': gene_id.group(1) if gene_id else '',
                'gene_name': gene_name.group(1),
                'start': int(fields[3]),
                'end': int(fields[4]),
                'strand': fields[6],
            })
    genes_df = pd.DataFrame(records).drop_duplicates(subset='gene_name')
    genes_df = genes_df.sort_values('start').reset_index(drop=True)
    genes_df.to_csv(GENES_CACHE, sep='\t', index=False)

n_genes = len(genes_df)
print(f"  {n_genes} protein-coding genes on chr21", flush=True)

# ── Step 3: Build gene binning matrix ─────────────────────────
print("Building gene binning matrix...", flush=True)
# For each site, find which gene(s) it falls in
# Sites can be in 0 or 1 genes (we skip intergenic sites)
site_gene = np.full(n_sites, -1, dtype=np.int32)
gene_starts = genes_df['start'].values
gene_ends = genes_df['end'].values

for gi in range(n_genes):
    mask = (pos >= gene_starts[gi]) & (pos <= gene_ends[gi])
    site_gene[mask] = gi

# Count sites per gene
gene_counts = np.bincount(site_gene[site_gene >= 0], minlength=n_genes).astype(np.float32)
genes_with_sites = gene_counts > 0
print(f"  {genes_with_sites.sum()} genes with SNPs (of {n_genes})", flush=True)

# Sparse binning matrix: (n_genes, n_sites)
valid_sites = site_gene >= 0
rows = site_gene[valid_sites]
cols = np.where(valid_sites)[0]
bm = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                     shape=(n_genes, n_sites))
bm = bm.multiply(1.0 / np.maximum(gene_counts, 1)[:, None]).tocsr()

# Filter to genes with sites
keep_genes = np.where(genes_with_sites)[0]
bm = bm[keep_genes]
genes_df = genes_df.iloc[keep_genes].reset_index(drop=True)
n_features = len(genes_df)
print(f"  {n_features} gene features", flush=True)

# ── Step 4: Metadata + pairs ─────────────────────────────────
meta = pd.read_csv(META_PATH, sep=r'\s+')
s2p = dict(zip(meta['SampleID'].astype(str), meta['Population'].astype(str)))
s2sp = dict(zip(meta['SampleID'].astype(str), meta['Superpopulation'].astype(str)))
pop_haps = {}
for idx, sid in enumerate(sample_ids):
    p = s2p.get(sid)
    if p:
        pop_haps.setdefault(p, []).extend([2*idx, 2*idx+1])
populations = sorted(pop_haps.keys())

obs_rows = []; pop_pairs = {}
for pop in populations:
    hi = pop_haps[pop]; nh = len(hi)
    pairs = [(hi[i], hi[j]) for i in range(nh) for j in range(i)]
    pop_pairs[pop] = pairs
    for a, b in pairs:
        sa, sb = sample_ids[a//2], sample_ids[b//2]
        obs_rows.append((f"{sa}_{a%2}_{sb}_{b%2}_{pop}", sa, sb, a%2, b%2, pop, s2sp.get(sa,'?')))

total = len(obs_rows)
obs_df = pd.DataFrame(obs_rows, columns=['pid','sample_i','sample_j','hap_i','hap_j','population','superpopulation'])
obs_df.index = pd.Index(obs_df.pop('pid'))
print(f"  {total:,} pairs, {len(populations)} populations", flush=True)

# ── Step 5: Create h5ad ──────────────────────────────────────
print("Creating h5ad...", flush=True)
var_df = pd.DataFrame({
    'gene_id': genes_df['gene_id'].values,
    'start': genes_df['start'].values,
    'end': genes_df['end'].values,
    'strand': genes_df['strand'].values,
    'chrom': 'chr21',
    'n_snps': gene_counts[keep_genes].astype(int),
}, index=genes_df['gene_name'].values)

adata = ad.AnnData(X=sp.csr_matrix((total, n_features), dtype=np.float32),
                    obs=obs_df, var=var_df)
adata.write(H5AD_PATH)
del adata; gc.collect()
with h5py.File(H5AD_PATH, 'r+') as f:
    del f['X']
    f.create_dataset('X', shape=(total, n_features), dtype='float32',
                      chunks=(min(1000, total), n_features), fillvalue=0.0)
print(f"  {total:,} x {n_features}", flush=True)

# ── Step 6: GPU inference ────────────────────────────────────
print(f"GPU inference ({N_GPUS} GPUs)...", flush=True)
h5 = h5py.File(H5AD_PATH, 'r+'); X = h5['X']
row_off = 0; t_all = time.perf_counter()

for pi, pop in enumerate(populations):
    tp = time.perf_counter()
    pairs = pop_pairs[pop]; np_ = len(pairs)
    if np_ == 0: continue

    hi = pop_haps[pop]
    G_pop = np.ascontiguousarray(G[hi])
    h2l = {h: i for i, h in enumerate(hi)}
    lpairs = [(h2l[a], h2l[b]) for a, b in pairs]

    print(f"  [{pi+1}/{len(populations)}] {pop}: {np_:,}...", end=' ', flush=True)

    # One context per GPU, reused across all chunks for this population
    ctxs = []
    for gpu in range(N_GPUS):
        _core.set_device(gpu)
        ctxs.append(_core.FlowContext(G_pop, pos, float(NE), MU, RHO, FF, 0))

    chunks = [(cs, min(cs+CHUNK_PAIRS, np_)) for cs in range(0, np_, CHUNK_PAIRS)]
    gpu_chunks = [[] for _ in range(N_GPUS)]
    for ci, (cs, ce) in enumerate(chunks):
        gpu_chunks[ci % N_GPUS].append((cs, ce))

    def process_gpu(gpu_id, _ctxs=ctxs, _lpairs=lpairs, _gc=gpu_chunks):
        results = []
        for cs, ce in _gc[gpu_id]:
            tmrca = _ctxs[gpu_id].run_fb(_lpairs[cs:ce], mean_only=True)['mean']
            w = (bm @ tmrca).T.astype(np.float32)
            results.append((cs, w))
        return results

    with ThreadPoolExecutor(max_workers=N_GPUS) as pool:
        futures = [pool.submit(process_gpu, g) for g in range(N_GPUS)]
        for f in futures:
            for cs, w in f.result():
                X[row_off+cs:row_off+cs+w.shape[0], :] = w

    del ctxs, G_pop; gc.collect()
    row_off += np_
    el = time.perf_counter()-tp
    print(f"{el:.1f}s ({np_/el:.0f}/s)", flush=True)

h5.close()
tt = time.perf_counter()-t_all
print(f"\nDone! {tt:.0f}s ({total/tt:.0f}/s)\nOutput: {H5AD_PATH}", flush=True)
