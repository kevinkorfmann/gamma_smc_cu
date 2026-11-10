"""Frozen-normalisation, held-out detection and localisation summaries."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd
from metrics import METHODS,summaries,threshold,detect,wilson,localisation

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_json(p,j):p.write_text(json.dumps(j,indent=2)+'\n')

def run(a):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm execution required')
    manifest=json.loads((a.root/'manifest.json').read_text());tasks=manifest['tasks']
    assert len({t['seed'] for t in tasks})==len(tasks)
    a.out.mkdir(parents=True,exist_ok=False)
    def read(t):
        folder=a.root/'scores'/t['id'];r=json.loads((folder/'COMPLETE.json').read_text())
        assert sha(folder/'raw_scores.npz')==r['scores_sha256']
        for k in ['seed','split','regime','replicate']:assert r['simulation'][k]==t[k]
        return dict(np.load(folder/'raw_scores.npz'))
    normal=[t for t in tasks if t['split']=='normalisation'];assert all(t['regime']=='neutral' for t in normal)
    iv=[[] for _ in range(20)];xv=[]
    for t in normal:
        z=read(t);bins=np.minimum((z['derived_frequency']*20).astype(int),19)
        for i in range(20):iv[i].extend(z['ihs'][(bins==i)&np.isfinite(z['ihs'])])
        xv.extend(z['xp'][np.isfinite(z['xp'])])
    means=[float(np.mean(v)) if len(v)>=50 else np.nan for v in iv]
    sds=[float(np.std(v)) if len(v)>=50 else np.nan for v in iv]
    norm=dict(ihs_mean=means,ihs_sd=sds,ihs_n=[len(v) for v in iv],xp_mean=float(np.mean(xv)),xp_sd=float(np.std(xv)),xp_n=len(xv),normalisation_ids=[t['id'] for t in normal])
    assert all(np.isfinite(sds[i]) and sds[i]>0 for i in range(1,20)) and norm['xp_sd']>0
    write_json(a.out/'normalisation.json',norm)
    records=[]
    for t in tasks:
        if t['split']=='normalisation':continue
        z=read(t);target=int(z['selected_position0'])
        for width in [5000,20000]:
            target_scores=summaries(z,norm,target,width)
            centers=np.arange(250000+width/2,750000-width/2+1,5000)
            scan=np.array([[x[0] for x in summaries(z,norm,c,width)] for c in centers])
            for i,method in enumerate(METHODS):
                loc=localisation(scan[:,i],centers,target)
                records.append(dict(**t,width=width,method=method,target_score=target_scores[i][0],
                                    n_finite_target=target_scores[i][1],**loc))
    frame=pd.DataFrame(records);frame.to_csv(a.out/'per_replicate.csv',index=False)
    summary=[];cuts=[];differences=[]
    rng=np.random.default_rng(20260927)
    for (width,method),block in frame.groupby(['width','method'],sort=False):
        cal=block[block.split=='calibration'];assert len(cal)==manifest['calibration_replicates']
        for endpoint,scorekey in [('target','target_score'),('scan','scan_score')]:
            cutoff=threshold(cal[scorekey].to_numpy())
            cuts.append(dict(width=width,method=method,endpoint=endpoint,threshold=cutoff,n=len(cal),calibration_calls=int(detect(cal[scorekey],cutoff).sum())))
            for regime,test in block[block.split=='evaluation'].groupby('regime',sort=False):
                values=test[scorekey].to_numpy();finite=np.isfinite(values);called=detect(values,cutoff);n=len(test);k=int(called.sum())
                lo,hi=wilson(k,n);covlo,covhi=wilson(int(finite.sum()),n)
                clo,chi=wilson(k,int(finite.sum()))
                err=test.error.to_numpy();good=np.isfinite(err)
                q=np.quantile(err[good],[.25,.5,.75]) if good.any() else [np.nan]*3
                row=dict(width=width,method=method,endpoint=endpoint,regime=regime,n=n,computable=int(finite.sum()),detected=k,
                   rate=k/n,ci_low=lo,ci_high=hi,coverage=float(finite.mean()),coverage_low=covlo,coverage_high=covhi,
                   conditional_rate=k/finite.sum() if finite.any() else np.nan,conditional_low=clo,conditional_high=chi,
                   localisation_median=q[1],localisation_q25=q[0],localisation_q75=q[2],
                   localisation_detected_median=float(np.median(err[good&called])) if (good&called).any() else np.nan,
                   tied_best_fraction=float((test.ties>1).mean()),median_tie_span=float(test.tie_span.median()))
                summary.append(row)
    cuts=pd.DataFrame(cuts);cuts.to_csv(a.out/'thresholds.csv',index=False)
    pd.DataFrame(summary).to_csv(a.out/'performance.csv',index=False)
    cutlookup={(r.width,r.method,r.endpoint):r.threshold for r in cuts.itertuples()}
    for (width,regime),block in frame[frame.split=='evaluation'].groupby(['width','regime']):
        for endpoint,scorekey in [('target','target_score'),('scan','scan_score')]:
            pivot=block.pivot(index='id',columns='method',values=scorekey)
            baseline=detect(pivot['Gamma'],cutlookup[width,'Gamma',endpoint]).astype(float)
            for method in METHODS[1:]:
                other=detect(pivot[method],cutlookup[width,method,endpoint]).astype(float)
                d=baseline-other;n=len(d)
                boot=np.mean(d[rng.integers(0,n,size=(2000,n))],axis=1)
                lo,hi=np.quantile(boot,[.025,.975])
                differences.append(dict(width=width,regime=regime,endpoint=endpoint,comparison='Gamma minus '+method,n=n,difference=float(d.mean()),ci_low=float(lo),ci_high=float(hi)))
    pd.DataFrame(differences).to_csv(a.out/'paired_differences.csv',index=False)
    write_json(a.out/'verification.json',dict(status='passed',independent_replicates=len(tasks),normalisation=len(normal),
       source_sha256=sha(__file__),metrics_sha256=sha(Path(__file__).with_name('metrics.py')),
       manifest_sha256=sha(a.root/'manifest.json'),outputs={p.name:sha(p) for p in a.out.iterdir() if p.is_file()},
       protocol='Nominal 5% neutral tails, strict ties; Wilson intervals; paired independent-replicate bootstrap',
       limitations='One no-migration demographic model; hard sweeps; known fixed rates; no empirical BGS calibration'))
    print(pd.DataFrame(summary).query("width==5000 and endpoint=='target'")[['method','regime','n','coverage','rate']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
