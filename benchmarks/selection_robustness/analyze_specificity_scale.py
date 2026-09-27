"""Post-hoc spatial-averaging and calibration-span diagnostics for the fixed bank.

Reuses every fixed-sweep and neutral replicate; no new simulations or outcome
selection beyond the already reported fixation stratum. Other genome-wide
data remain fixed, so this is not a widespread-selection experiment.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tskit

from analyze_specificity import neutralized_tree
from bprime_refit import BprimeFit
from common import digest, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--published',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    with (a.analysis/'replicates.csv').open() as f:old=list(csv.DictReader(f))
    keep=[r for r in old if r['regime']=='neutral' or r['outcome']=='fixed']
    cfg=json.loads((a.bank/keep[0]['run']/'config.json').read_text())
    origin=cfg['region']['left'];length=cfg['region']['length'];end=origin+length;focal=origin+cfg['focal_position0']
    z=np.load(a.published);m=BprimeFit(z);base=m.fit(z['Y'])
    target=np.flatnonzero((z['chrom']=='chr6')&(z['start']>=origin)&(z['end']<=end))
    gene=int(np.flatnonzero((z['chrom']=='chr6')&(z['start']==41100000))[0])
    gi=int(np.flatnonzero(target==gene)[0]);counts=[1,3,5,11,21,29]
    subsets=[target[max(0,gi-n//2):min(len(target),gi+n//2+1)] for n in counts]
    if [len(s) for s in subsets]!=counts:raise ValueError('Unexpected regional fitting-bin geometry')
    # The original 100-kb estimator bin is reported separately from concentric windows.
    windows=[('TREM2_body',41158505,41163186)]
    windows += [(f'centered_{w}',focal-w//2,focal+w//2) for w in [10000,50000,100000,200000,500000,1000000,2000000]]
    windows += [('full_region',origin,end),('left_edge_250kb',origin,origin+250000),
                ('right_edge_250kb',end-250000,end),('native_focal_100kb',41100000,41200000)]
    boundaries=np.unique(np.r_[origin,end,z['start'][target],z['end'][target],
                                [v for _,lo,hi in windows for v in [lo,hi]]])
    widths=np.diff(boundaries)
    native_index=np.searchsorted(boundaries,z['start'][target])
    labels=[];pis=[];times=[];Bs=[];fit_rows=[]
    for k,row in enumerate(keep):
        directory=a.bank/row['run'];ts=tskit.load(directory/'truth.trees')
        qc=json.loads((directory/'qc.json').read_text())
        if digest(directory/'truth.trees')!=qc['tree_sha256']:raise ValueError('Tree checksum mismatch')
        ts=neutralized_tree(ts,cfg['focal_position0'])
        pi=ts.diversity(windows=boundaries-origin);T=ts.diversity(windows=boundaries-origin,mode='branch')/2
        values=[];tm=[];native=[]
        for _,lo,hi in windows:
            mask=(boundaries[:-1]>=lo)&(boundaries[1:]<=hi)
            if widths[mask].sum()!=hi-lo:raise ValueError('Incomplete interval')
            values.append(np.average(pi[mask],weights=widths[mask])/(4*cfg['Ne']*cfg['mu']))
            tm.append(np.average(T[mask],weights=widths[mask])/(2*cfg['Ne']))
        for lo,hi in zip(z['start'][target],z['end'][target]):
            mask=(boundaries[:-1]>=lo)&(boundaries[1:]<=hi)
            native.append(np.average(pi[mask],weights=widths[mask])/(4*cfg['Ne']*cfg['mu']))
        native=np.array(native);predicted=z['theta'][0]*native
        fitvals=[]
        for n,subset in zip(counts,subsets):
            j=np.searchsorted(target,subset);Y=z['Y'].astype(float).copy();total=Y[subset].sum(axis=1)
            Y[subset,1]=total*predicted[j];Y[subset,0]=total-Y[subset,1]
            fit=m.fit(Y,start=base['x']);fitvals.append(fit['B'][gene])
            fit_rows.append(dict(run=row['run'],group='neutral' if row['regime']=='neutral' else 'fixed',
                calibration_bins=n,focal_B=float(fit['B'][gene]),pi0=fit['pi0'],mu=fit['mu']))
        np.testing.assert_allclose(fitvals[-1],float(row['focal_B']),rtol=0,atol=2e-7)
        labels.append('neutral' if row['regime']=='neutral' else 'fixed');pis.append(values);times.append(tm);Bs.append(fitvals)
        print(json.dumps(dict(done=k+1,total=len(keep),run=row['run'])),flush=True)
    labels=np.array(labels);pis=np.array(pis);times=np.array(times);Bs=np.array(Bs)
    rng=np.random.default_rng(4381650);sel=labels=='fixed';neu=labels=='neutral'
    si=rng.integers(sel.sum(),size=(10000,sel.sum()));ni=rng.integers(neu.sum(),size=(10000,neu.sum()))
    def contrast(values,ratio):
        aa=values[sel][si].mean(axis=1);bb=values[neu][ni].mean(axis=1)
        boot=aa/bb if ratio else aa-bb
        mean=values[sel].mean(axis=0)/values[neu].mean(axis=0) if ratio else values[sel].mean(axis=0)-values[neu].mean(axis=0)
        return mean,np.quantile(boot,[.025,.975],axis=0)
    pr,pc=contrast(pis,True);tr,tc=contrast(times,True);br,bc=contrast(Bs,False)
    window_results=[dict(name=name,left=lo,right=hi,bp=hi-lo,pi_ratio=float(pr[j]),pi_ci95=pc[:,j].tolist(),
                         tmrca_ratio=float(tr[j]),tmrca_ci95=tc[:,j].tolist()) for j,(name,lo,hi) in enumerate(windows)]
    fit_results=[dict(bins=n,bp=int(np.sum(z['end'][subset]-z['start'][subset])),
        pair_weight_fraction=float(z['Y'][subset].sum()/z['Y'].sum()),delta_B=float(br[j]),ci95=bc[:,j].tolist())
        for j,(n,subset) in enumerate(zip(counts,subsets))]
    a.out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(a.out/'scale_replicates.npz',labels=labels,pi=pis,tmrca=times,Bprime=Bs,
        window_names=np.array([w[0] for w in windows]),window_bounds=np.array([[w[1],w[2]] for w in windows]),
        calibration_bins=counts)
    with (a.out/'scale_refits.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fit_rows[0].keys());writer.writeheader();writer.writerows(fit_rows)
    write_json(a.out/'scale_summary.json',dict(post_hoc=True,n_fixed=int(sel.sum()),n_neutral=int(neu.sum()),
        averaging_windows=window_results,calibration_spans=fit_results,source_sha256=digest(__file__),
        original_analysis_sha256=digest(a.analysis/'replicates.csv'),
        caveat='Reuses the original independent replicates across all diagnostic widths. Intervals are conditional on the unchanged empirical genome. No region-length simulation convergence or widespread positive-selection model has been tested.'))


if __name__=='__main__':main()
