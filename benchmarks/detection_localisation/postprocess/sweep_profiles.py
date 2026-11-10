"""Target-aligned true/inferred TMRCA and EHH profiles over every evaluation replicate.

Post-evaluation visual augmentation, requested after the detection experiment.
This does not change its inference, normalisation, thresholds or power estimates.
"""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd
import tskit

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def unit(root,task,validate):
    sim=root/'simulations'/task['id'];score=root/'scores'/task['id']
    receipt=json.loads((sim/'COMPLETE.json').read_text())
    assert sha(sim/'variants.trees')==receipt['tree_sha256']
    ts=tskit.load(sim/'variants.trees');samples=ts.samples()
    focal=[int(n) for n in samples if ts.node(int(n)).population==0];assert len(focal)==100
    q=np.load(score/'raw_scores.npz');gpos=q['gpos'];truth=np.full(len(gpos),np.nan)
    checked=0
    for tree in ts.trees(tracked_samples=focal):
        lo,hi=np.searchsorted(gpos,[tree.interval.left,tree.interval.right])
        if hi==lo:continue
        weight=0;total=0.
        for node in tree.nodes():
            n=tree.num_tracked_samples(node)
            if n<2:continue
            pairs=n*(n-1)//2
            for child in tree.children(node):
                nc=tree.num_tracked_samples(child);pairs-=nc*(nc-1)//2
            if pairs:
                assert tree.time(node)>0
                total+=pairs*np.log(tree.time(node));weight+=pairs
        assert weight==4950
        mean=total/weight;truth[lo:hi]=mean
        if validate and checked<5:
            direct=np.mean([np.log(tree.time(tree.mrca(x,y))) for i,x in enumerate(focal) for y in focal[i+1:]])
            assert np.isclose(mean,direct,rtol=1e-12,atol=1e-12)
            checked+=1
    assert np.all(np.isfinite(truth))
    norm=json.loads((root/'summary_v1/normalisation.json').read_text())
    bins=np.minimum((q['derived_frequency']*20).astype(int),19)
    ihs=np.abs((q['ihs']-np.array(norm['ihs_mean'])[bins])/np.array(norm['ihs_sd'])[bins])
    xp=(q['xp']-norm['xp_mean'])/norm['xp_sd'];target=int(q['selected_position0'])
    rows=[]
    for offset in range(-250000,250001,5000):
        lo=target+offset-10000;hi=target+offset+10000
        g=(gpos>=lo)&(gpos<hi);iv=ihs[(q['ipos']>=lo)&(q['ipos']<hi)&np.isfinite(ihs)]
        xv=xp[(q['xpos']>=lo)&(q['xpos']<hi)&np.isfinite(xp)]
        rows.append(dict(id=task['id'],regime=task['regime'],offset_bp=offset,width_bp=20000,
            n_tmrca=int(g.sum()),n_ihs=len(iv),n_xp=len(xv),
            true_tmrca=float(np.exp(truth[g].mean())) if g.sum()>=2 else np.nan,
            inferred_tmrca=float(np.exp(q['logtmrca'][g].mean())) if g.sum()>=2 else np.nan,
            ihs_max=float(iv.max()) if len(iv) else np.nan,
            xp_max=float(xv.max()) if len(xv) else np.nan))
    return rows,checked

def run(a):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    tasks=[t for t in json.loads((a.root/'manifest.json').read_text())['tasks'] if t['split']=='evaluation']
    assert len(tasks)==600
    a.out.mkdir(parents=True,exist_ok=True)
    if a.index is not None:
        dest=a.out/f'part_{a.index:02d}.csv';assert not dest.exists()
        rows=[];checked=0
        for i,task in enumerate(tasks[a.index*20:(a.index+1)*20]):
            r,c=unit(a.root,task,i==0);rows.extend(r);checked+=c
        assert len(rows)==20*101
        pd.DataFrame(rows).to_csv(dest,index=False)
        (a.out/f'part_{a.index:02d}.json').write_text(json.dumps(dict(source_sha256=sha(__file__),rows=len(rows),direct_pair_checks=checked,csv_sha256=sha(dest)))+'\n')
    else:
        frames=[];checks=0
        for i in range(30):
            path=a.out/f'part_{i:02d}.csv';receipt=json.loads(path.with_suffix('.json').read_text())
            assert receipt['source_sha256']==sha(__file__) and receipt['csv_sha256']==sha(path)
            checks+=receipt['direct_pair_checks'];frames.append(pd.read_csv(path))
        f=pd.concat(frames,ignore_index=True);assert len(f)==60600 and not f.duplicated(['id','offset_bp']).any()
        f.to_csv(a.out/'profiles_per_replicate.csv.gz',index=False,compression='gzip')
        metrics=['true_tmrca','inferred_tmrca','ihs_max','xp_max'];out=[]
        for (regime,offset),g in f.groupby(['regime','offset_bp'],sort=False):
            assert len(g)==(200 if regime=='neutral' else 100)
            row=dict(regime=regime,offset_bp=offset,n=len(g))
            for metric in metrics:
                v=g[metric].dropna();row[metric+'_n']=len(v)
                for key,quantile in [('q25',.25),('median',.5),('q75',.75)]:row[metric+'_'+key]=v.quantile(quantile) if len(v) else np.nan
            out.append(row)
        pd.DataFrame(out).to_csv(a.out/'profiles_summary.csv',index=False)
        perf=pd.read_csv(a.root/'summary_v1/performance.csv')
        for regime,g in f[f.offset_bp==0].groupby('regime'):
            for metric,method in [('true_tmrca','Gamma'),('inferred_tmrca','Gamma'),('ihs_max','iHS maximum'),('xp_max','XP-EHH maximum')]:
                r=perf[(perf.width==20000)&(perf.endpoint=='target')&(perf.method==method)&(perf.regime==regime)].iloc[0]
                assert g[metric].notna().sum()==r.computable
        (a.out/'verification.json').write_text(json.dumps(dict(status='passed',replicates=600,rows=len(f),
            direct_tree_pair_enumerations=checks,source_sha256=sha(__file__),
            definition='Per-tree mean log TMRCA across all 4950 focal pairs; same retained sites and 20 kb windows as inferred profiles; 5 kb display steps; all evaluation replicates',
            scope='Post-evaluation visual augmentation; original thresholds and performance unchanged',
            outputs={n:sha(a.out/n) for n in ['profiles_per_replicate.csv.gz','profiles_summary.csv']}),indent=2)+'\n')
        print('Profile verification passed:',len(f),'rows;',checks,'direct pair-enumeration checks')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--index',type=int);run(p.parse_args())
