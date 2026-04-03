#!/usr/bin/env python3
"""
1000 Genomes chr21: within-population pairwise TMRCA inference.
3 GPUs, one FlowContext per GPU (reused across chunks).
"""
import numpy as np
import anndata as ad
import pandas as pd
import scipy.sparse as sp
import h5py
import os, sys, time, gc
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
from tmrca_cu import _core

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'results')
CACHE_PATH = os.path.join(DATA_DIR, 'chr21_parsed.npz')
META_PATH = os.path.join(DATA_DIR, 'samples.txt')
FF = '/sietch_colab/kkor/gamma_smc/resources/default_flow_field.txt'
H5AD_PATH = os.path.join(OUT_DIR, 'chr21_tmrca.h5ad')
MU, RHO, NE = 1.25e-8, 1e-8, 10000
WINDOW_SIZE = 2000
CHUNK_PAIRS = 2000
N_GPUS = min(_core.get_device_count(), 3)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────
print("Loading genotypes...", flush=True)
t0 = time.perf_counter()
data = np.load(CACHE_PATH, allow_pickle=True)
G = data['G']; pos = data['positions']
sample_ids = [str(s) for s in data['sample_ids']]
n_haps, n_sites = G.shape
print(f"  {n_haps} haplotypes, {n_sites:,} sites ({time.perf_counter()-t0:.1f}s)", flush=True)

# ── Metadata ──────────────────────────────────────────────────
meta = pd.read_csv(META_PATH, sep=r'\s+')
s2p = dict(zip(meta['SampleID'].astype(str), meta['Population'].astype(str)))
s2sp = dict(zip(meta['SampleID'].astype(str), meta['Superpopulation'].astype(str)))
pop_haps = {}
for idx, sid in enumerate(sample_ids):
    p = s2p.get(sid)
    if p:
        pop_haps.setdefault(p, []).extend([2*idx, 2*idx+1])
populations = sorted(pop_haps.keys())
print(f"  {len(populations)} populations, {N_GPUS} GPUs", flush=True)

# ── Windows ───────────────────────────────────────────────────
ws = np.arange(int(pos[0]), int(pos[-1]), WINDOW_SIZE)
we = ws + WINDOW_SIZE; n_win = len(ws)
s2w = np.clip(np.searchsorted(ws, pos, side='right')-1, 0, n_win-1).astype(np.intp)
wc = np.bincount(s2w, minlength=n_win).astype(np.float32)
bm = sp.csr_matrix((np.ones(n_sites, np.float32), (s2w, np.arange(n_sites))), shape=(n_win, n_sites))
bm = bm.multiply(1.0/np.maximum(wc,1)[:,None])
print(f"  {n_win:,} windows", flush=True)

# ── Pairs ─────────────────────────────────────────────────────
print("Building pairs...", flush=True)
obs_rows = []; pop_pairs = {}
for pop in populations:
    hi = pop_haps[pop]; nh = len(hi)
    pairs = [(hi[i], hi[j]) for i in range(nh) for j in range(i)]
    pop_pairs[pop] = pairs
    for a,b in pairs:
        sa,sb = sample_ids[a//2], sample_ids[b//2]
        obs_rows.append((f"{sa}_{a%2}_{sb}_{b%2}_{pop}", sa, sb, a%2, b%2, pop, s2sp.get(sa,'?')))
total = len(obs_rows)
obs_df = pd.DataFrame(obs_rows, columns=['pid','sample_i','sample_j','hap_i','hap_j','population','superpopulation'])
obs_df.index = pd.Index(obs_df.pop('pid'))
print(f"  {total:,} pairs", flush=True)

# ── h5ad ──────────────────────────────────────────────────────
print("Creating h5ad...", flush=True)
var_df = pd.DataFrame({'start':ws,'end':we,'chrom':'chr21'},
                        index=[f"chr21:{s}-{e}" for s,e in zip(ws,we)])
ad.AnnData(X=sp.csr_matrix((total,n_win),dtype=np.float32), obs=obs_df, var=var_df).write(H5AD_PATH)
with h5py.File(H5AD_PATH,'r+') as f:
    del f['X']
    f.create_dataset('X', shape=(total,n_win), dtype='float32',
                      chunks=(min(1000,total),n_win), fillvalue=0.0)

# ── Inference ─────────────────────────────────────────────────
print(f"Running inference...", flush=True)
h5 = h5py.File(H5AD_PATH, 'r+'); X = h5['X']
row_off = 0; t_all = time.perf_counter()

for pi, pop in enumerate(populations):
    tp = time.perf_counter()
    pairs = pop_pairs[pop]; np_ = len(pairs)
    if np_ == 0: continue

    hi = pop_haps[pop]
    G_pop = np.ascontiguousarray(G[hi])
    h2l = {h:i for i,h in enumerate(hi)}
    lpairs = [(h2l[a], h2l[b]) for a,b in pairs]

    print(f"  [{pi+1}/{len(populations)}] {pop}: {np_:,} pairs...", end=' ', flush=True)

    # Create one context per GPU for this population
    ctxs = []
    for gpu in range(N_GPUS):
        _core.set_device(gpu)
        ctxs.append(_core.FlowContext(G_pop, pos, float(NE), MU, RHO, FF, 0))

    # Split all chunks across GPUs
    chunks = [(cs, min(cs+CHUNK_PAIRS, np_)) for cs in range(0, np_, CHUNK_PAIRS)]

    # Assign chunks to GPUs round-robin
    gpu_chunks = [[] for _ in range(N_GPUS)]
    for ci, (cs, ce) in enumerate(chunks):
        gpu_chunks[ci % N_GPUS].append((cs, ce))

    def run_gpu(gpu_id):
        results = []
        for cs, ce in gpu_chunks[gpu_id]:
            chunk = lpairs[cs:ce]
            tmrca = ctxs[gpu_id].run_fb(chunk, mean_only=True)['mean']
            w = (bm @ tmrca).T.astype(np.float32)
            results.append((cs, w))
        return results

    with ThreadPoolExecutor(max_workers=N_GPUS) as pool:
        futures = [pool.submit(run_gpu, g) for g in range(N_GPUS)]
        for f in futures:
            for cs, w in f.result():
                X[row_off+cs:row_off+cs+w.shape[0], :] = w

    del ctxs, G_pop; gc.collect()
    row_off += np_
    el = time.perf_counter()-tp
    print(f"{el:.1f}s ({np_/el:.0f} pairs/s)", flush=True)

h5.close()
tt = time.perf_counter()-t_all
print(f"\nDone! {tt:.0f}s ({total/tt:.0f} pairs/s)\nOutput: {H5AD_PATH}", flush=True)
