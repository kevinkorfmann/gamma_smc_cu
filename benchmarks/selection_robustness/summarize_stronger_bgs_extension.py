"""Paired 20Ne/10Ne check for the prespecified stronger-BGS burn-in extension."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import digest, write_json
from summarize_stronger_bgs import read_snapshot


def main():
    p=argparse.ArgumentParser()
    for name in ['bank','summary','out']: p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();m=json.loads((a.bank/'manifest.json').read_text())
    assert m['mode']=='extension' and m['families']==6
    assert digest(a.summary)==m['calibration_summary_sha256']
    completion=json.loads((a.bank/'completion.json').read_text())
    assert len(completion['results'])==6 and all(r['status']=='completed' for r in completion['results'])
    model=m['model']['name'];cal=Path(m['calibration']);baseline=Path(m['baseline'])
    assert digest(baseline/'manifest.json')==m['baseline_manifest_sha256']
    records=[]
    for i in range(6):
        fam=a.bank/model/f'family_{i:03d}';old=cal/model/f'family_{i:03d}/founder'
        assert digest(old/'final.raw.trees')==m['parent_sha256'][str(i)]
        for kind,tick,parent in [('extension',200001,old/'final.raw.trees'),('control',202501,fam/'extension/final.raw.trees')]:
            cfg=json.loads((fam/kind/'config.json').read_text())
            assert cfg['end_tick']==tick and cfg['parent_sha256']==digest(parent)
            assert cfg['bgs'] and not cfg['sweep'] and cfg['Ne']==10000 and cfg['scaling_factor']==1
            assert cfg['source_sha256']==m['sources']['stronger_bgs.py']
            assert cfg['region']['selection_model']==m['model']['model']
        records.append(dict(family=i,ten=read_snapshot(old,'final')[0],
            twenty=read_snapshot(fam/'extension','final')[0],control=read_snapshot(fam/'control','final')[0]))
    metrics=[k for k in records[0]['ten'] if k!='uncoalesced_span_fraction']
    old=np.array([[r['ten'][k] for k in metrics] for r in records])
    new=np.array([[r['twenty'][k] for k in metrics] for r in records])
    control=np.array([[r['control'][k] for k in metrics] for r in records])
    refs=[read_snapshot(f/'control','final')[0] for f in sorted(baseline.glob('neutral_family_*'))]
    assert len(refs)==30
    neutral=np.array([[r[k] for k in metrics] for r in refs])
    rng=np.random.default_rng(260928);B=4000
    w=rng.multinomial(6,np.ones(6)/6,size=B);wr=rng.multinomial(30,np.ones(30)/30,size=B)
    burn_draw=(w@new)/(w@old);ratio_draw=(w@control/6)/(wr@neutral/30)
    def record(point,draw):return dict(ratio=float(point),ci95=np.quantile(draw,[.025,.975]).tolist())
    burn={k:record(new[:,j].mean()/old[:,j].mean(),burn_draw[:,j]) for j,k in enumerate(metrics)}
    effect={k:record(control[:,j].mean()/neutral[:,j].mean(),ratio_draw[:,j]) for j,k in enumerate(metrics)}
    root=float(np.mean([r['twenty']['uncoalesced_span_fraction'] for r in records]))
    passed=abs(burn['genome_tmrca']['ratio']-1)<=.10 and abs(burn['focal100kb_tmrca']['ratio']-1)<=.20 and root<.05
    ratio=effect['focal100kb_tmrca']['ratio']
    summary=dict(model=model,calibration_summary_sha256=digest(a.summary),extension_manifest_sha256=digest(a.bank/'manifest.json'),
        source_sha256=digest(__file__),families=6,bootstrap=B,seed=260928,burn_in_20Ne_over_10Ne=burn,
        effects=effect,mean_uncoalesced_span=root,burn_gate_passed=bool(passed),
        eligible_for_confirmation=bool(passed and .10<=ratio<=.35),confirmation_burn_Ne=20,
        synthetic=True,production_complete=False,
        interpretation='Exploratory BGS-only extension; paired founder uncertainty; 30 independent neutral reference families. No sweep or Gamma outcomes used.')
    a.out.mkdir(parents=True,exist_ok=True)
    write_json(a.out/'summary.json',summary);write_json(a.out/'measurements.json',records)
    print(json.dumps(dict(model=model,eligible=summary['eligible_for_confirmation'],focal_ratio=ratio,burn_gate_passed=bool(passed))))


if __name__=='__main__':main()
