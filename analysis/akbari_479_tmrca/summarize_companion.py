"""Descriptive companion panels; run via Slurm, with no background enrichment claim."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

p = argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
m = pd.read_csv(a.input/'heatmap_matrix_kya.csv', index_col='key')
d = pd.read_csv(a.input/'heatmap_row_order.csv', index_col='key')
assert m.shape == (474, 26) and m.index.equals(d.index)
assert np.isfinite(m.values).all() and (m.values > 0).all()
groups = {'AFR':['YRI','LWK','GWD','MSL','ESN','ACB','ASW'],
          'EUR':['CEU','TSI','FIN','GBR','IBS'], 'SAS':['GIH','PJL','BEB','STU','ITU'],
          'EAS':['CHB','JPT','CHS','CDX','KHV'], 'AMR':['MXL','PUR','CLM','PEL']}
for g, pops in groups.items():
    d[g+'_kya'] = m[pops].median(axis=1)
d['abs_X'] = d.akbari_X.abs()
rng = np.random.default_rng(20260927)
chroms = np.unique(d.chrom)
blocks = [np.flatnonzero(d.chrom.values == c) for c in chroms]
draws = [np.concatenate([blocks[j] for j in rng.integers(len(blocks), size=len(blocks))]) for _ in range(2000)]
rows = []
for threshold in (50, 100, 200):
    indicator = m.values < threshold
    boot = np.array([indicator[ix].mean(axis=0)*100 for ix in draws])
    lo, hi = np.quantile(boot, [.025,.975], axis=0)
    for j, pop in enumerate(m.columns):
        rows.append(dict(population=pop, group=next(g for g,v in groups.items() if pop in v),
                         cutoff_kya=threshold, n=int(indicator[:,j].sum()), total=len(m),
                         percent=indicator[:,j].mean()*100, low=lo[j], high=hi[j]))
pd.DataFrame(rows).to_csv(a.output/'shallow_fractions.csv', index=False)
stats = {}
for name, x, y in [('ancient_vs_EUR', d.abs_X.values,d.EUR_kya.values),
                   ('EUR_vs_SAS',d.EUR_kya.values,d.SAS_kya.values)]:
    boot = [spearmanr(x[ix],y[ix]).statistic for ix in draws]
    stats[name] = dict(spearman_rho=float(spearmanr(x,y).statistic),
                       ci95=np.quantile(boot,[.025,.975]).tolist())
d.to_csv(a.output/'locus_summaries.csv')
manifest = dict(n=474, groups=groups, statistics=stats, bootstrap='2000 chromosome-block resamples; seed 20260927',
    interpretation='Descriptive fractions at absolute cutoffs, not enrichment. No matched background exists in these inputs. Group summaries are medians across population window summaries, not pooled individuals. TMRCA does not date selection.',
    hashes={f:hashlib.sha256((a.input/f).read_bytes()).hexdigest() for f in ['heatmap_matrix_kya.csv','heatmap_row_order.csv']},
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(a.output/'summary.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(stats))
