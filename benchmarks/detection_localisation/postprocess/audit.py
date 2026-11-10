"""Independent accounting and source checks for the frozen benchmark outputs."""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(a):
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Slurm required')
    m=json.loads((a.root/'manifest.json').read_text())
    for name,h in m['source_sha256'].items(): assert sha(a.root/'source'/name)==h
    outcomes=[]; receipts=[]
    for task in m['tasks']:
        sim=a.root/'simulations'/task['id']; score=a.root/'scores'/task['id']
        sr=json.loads((sim/'COMPLETE.json').read_text()); cr=json.loads((score/'COMPLETE.json').read_text())
        assert sr['source_sha256']==m['source_sha256']['simulate.py']
        assert cr['source_sha256']==m['source_sha256']['score.py']
        assert sr['input_sha256']==cr['input_sha256']==sha(sim/'input.npz')
        assert sha(score/'raw_scores.npz')==cr['scores_sha256']
        assert cr['backend']=='cpu'
        for key in ['seed','split','regime','replicate']:
            assert task[key]==sr['config'][key]==cr['simulation'][key]
        z=np.load(sim/'input.npz'); q=np.load(score/'raw_scores.npz')
        target=int(z['selected_position0']); assert target==int(q['selected_position0'])
        assert not np.any(z['positions']==target)
        assert np.array_equal(z['positions'],q['xpos'])
        assert (z['population']==0).sum()==(z['population']==1).sum()==100
        ac=z['G'][z['population']==0].sum(axis=0)
        assert np.array_equal(z['positions'][(ac>0)&(ac<100)],q['gpos'])
        assert np.array_equal(z['positions'][(ac>=5)&(ac<=95)],q['ipos'])
        assert np.array_equal(ac[(ac>=5)&(ac<=95)]/100,q['derived_frequency'])
        assert np.all(np.isfinite(q['logtmrca']))
        outcomes.append(dict(**task,position0=target,**sr['outcome'],**cr['counts'],simulation_seconds=sr['seconds'],scoring_seconds=cr['seconds']))
        receipts.append(dict(id=task['id'],input_sha256=sr['input_sha256'],scores_sha256=cr['scores_sha256'],simulation_job=sr['config']['slurm_job'],score_job=cr['slurm_job']))
    a.out.mkdir(parents=True,exist_ok=False)
    f=pd.DataFrame(outcomes);f.to_csv(a.out/'replicate_metadata.csv',index=False)
    pd.DataFrame(receipts).to_csv(a.out/'input_score_hashes.csv',index=False)
    trajectories=[]
    for regime,b in f[f.regime!='neutral'].groupby('regime',sort=False):
        assert len(b)==100
        assert b.sample_frequency.notna().all() and b.sample_frequency.between(0,1).all()
        if regime=='incomplete':
            assert b.final_population_frequency.between(.25,.75).all() and b.fixation_age.isna().all()
        elif regime=='near_fixed':
            assert b.final_population_frequency.between(.8,.99).all() and b.fixation_age.isna().all()
        elif regime=='completed':
            assert (b.final_population_frequency==1).all() and b.fixation_age.between(0,2000).all()
        elif regime=='older_completed':
            assert (b.final_population_frequency==1).all() and b.fixation_age.between(2000,4000).all()
        for field in ['final_population_frequency','sample_frequency','fixation_age','observed_restart_count']:
            v=b[field].dropna()
            if len(v):
                trajectories.append(dict(regime=regime,field=field,n=len(v),minimum=v.min(),q25=v.quantile(.25),median=v.median(),q75=v.quantile(.75),maximum=v.max()))
    pd.DataFrame(trajectories).to_csv(a.out/'trajectory_summary.csv',index=False)
    pr=pd.read_csv(a.summary/'per_replicate.csv');perf=pd.read_csv(a.summary/'performance.csv');cuts=pd.read_csv(a.summary/'thresholds.csv')
    assert len(pr)==800*2*7 and not pr.duplicated(['id','width','method']).any()
    checked=0
    for cut in cuts.itertuples():
        block=pr[(pr.width==cut.width)&(pr.method==cut.method)]
        key='target_score' if cut.endpoint=='target' else 'scan_score'
        neutral=block[block.split=='calibration'][key].to_numpy()
        assert len(neutral)==200
        ordered=sorted(float(v) if np.isfinite(v) else -float('inf') for v in neutral)
        assert np.isclose(cut.threshold,ordered[190])
        for regime,b in block[block.split=='evaluation'].groupby('regime'):
            row=perf[(perf.width==cut.width)&(perf.method==cut.method)&(perf.endpoint==cut.endpoint)&(perf.regime==regime)].iloc[0]
            ok=b[key].notna();k=int((ok&(b[key]>cut.threshold)).sum())
            assert row.n==len(b) and row.detected==k and row.computable==ok.sum()
            assert np.isclose(row.rate,k/len(b)) and np.isclose(row.coverage,ok.mean())
            assert np.isclose(row.localisation_median,b.error.median(),equal_nan=True)
            checked+=1
    result=dict(status='passed',replicates=len(f),summary_rows_checked=checked,
                checks=['frozen source','disjoint manifest seeds','all input and score hashes','shared population and marker identities','causal-site removal','complete per-replicate accounting','conditioned population frequencies and fixation ages','independent threshold order statistics','detection and coverage denominators','localisation medians'],
                manifest_sha256=sha(a.root/'manifest.json'),audit_source_sha256=sha(__file__),
                outputs={p.name:sha(p) for p in a.out.iterdir() if p.is_file()})
    assert len(set(t['seed'] for t in m['tasks']))==1000
    (a.out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
