"""Summarize every completed BGS-only model, before seeing any sweep outcome."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import digest,write_json


def read_snapshot(run,stage):
    cfg=json.loads((run/'config.json').read_text());focal=cfg['focal_position0']
    processed=json.loads((run/'processed.json').read_text())
    item=next(r for r in processed['results'] if r['stage']==stage)
    with np.load(run/(stage+'.npz')) as z:
        centers=(z['edges'][:-1]+z['edges'][1:])/2;values={}
        for name,width in [('focal100kb',100000),('focal1Mb',1000000),('genome',cfg['region']['length'])]:
            mask=abs(centers-focal)<width/2
            for measure in ['tmrca','pi']:
                values[name+'_'+measure]=float(np.average(z[measure][mask],weights=np.diff(z['edges'])[mask]))
        profile=z['tmrca'].reshape(-1,5).mean(axis=1)
    return dict(**values,uncoalesced_span_fraction=item['uncoalesced_span_fraction']),profile


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.bank/'manifest.json').read_text())
    completion=json.loads((a.bank/'completion.json').read_text())
    assert len(completion['results'])==len(manifest['models'])*manifest['families_per_model']
    assert all(r['status']=='completed' for r in completion['results'])
    assert digest(a.baseline/'manifest.json')==manifest['baseline_manifest_sha256']
    rng=np.random.default_rng(260927);B=4000
    refs=[read_snapshot(f/'control','final') for f in sorted(a.baseline.glob('neutral_family_*'))]
    assert len(refs)==30
    metrics=[m for m in refs[0][0] if m!='uncoalesced_span_fraction']
    ref=np.array([[r[0][m] for m in metrics] for r in refs])
    wb=rng.multinomial(len(ref),np.ones(len(ref))/len(ref),size=B)
    ref_draw=wb@ref/len(ref);ref_mean=ref.mean(axis=0)
    profiles={'neutral_control_mean':np.mean([r[1] for r in refs],axis=0)}
    results=[];all_records=[]
    for model in manifest['models']:
        families=sorted((a.bank/model['name']).glob('family_*'))
        assert len(families)==manifest['families_per_model']
        control=[];five=[];ten=[];roots=[];curves=[]
        for fam in families:
            status=json.loads((fam/'family_status.json').read_text());assert status['status']=='completed'
            for kind,stages in [('founder',['burn5Ne','final']),('control',['control_t100','control_t300','control_t1000','final'])]:
                cfg=json.loads((fam/kind/'config.json').read_text())
                assert cfg['Ne']==manifest['Ne'] and cfg['scaling_factor']==1 and not cfg['sweep']
                assert cfg['source_sha256']==manifest['sources']['stronger_bgs.py']
                assert cfg['region']['selection_model']==model['model']
                for stage in stages:
                    vals,profile=read_snapshot(fam/kind,stage)
                    all_records.append(dict(model=model['name'],family=fam.name,kind=kind,stage=stage,**vals))
                    if kind=='founder':
                        (five if stage=='burn5Ne' else ten).append([vals[m] for m in metrics])
                        if stage=='final':roots.append(vals['uncoalesced_span_fraction'])
                    elif stage=='final':control.append([vals[m] for m in metrics]);curves.append(profile)
        control=np.array(control);five=np.array(five);ten=np.array(ten);n=len(control)
        weights=rng.multinomial(n,np.ones(n)/n,size=B)
        cd=weights@control/n;ratios=cd/ref_draw
        burn_draw=(weights@ten)/(weights@five);burn_point=ten.mean(axis=0)/five.mean(axis=0)
        effects={metric:dict(ratio=float(control[:,i].mean()/ref_mean[i]),
            ci95=np.quantile(ratios[:,i],[.025,.975]).tolist(),mean=float(control[:,i].mean()),neutral_mean=float(ref_mean[i]))
            for i,metric in enumerate(metrics)}
        burn={metric:dict(ratio=float(burn_point[i]),ci95=np.quantile(burn_draw[:,i],[.025,.975]).tolist()) for i,metric in enumerate(metrics)}
        gate=abs(burn['genome_tmrca']['ratio']-1)<=.10 and abs(burn['focal100kb_tmrca']['ratio']-1)<=.20 and np.mean(roots)<.05
        results.append(dict(model=model['name'],settings=model['model'],families=n,effects=effects,
            burn_in=burn,burn_gate_passed=bool(gate),mean_uncoalesced_span=float(np.mean(roots)),max_uncoalesced_span=float(max(roots))))
        profiles[model['name']+'_mean']=np.mean(curves,axis=0)
    candidates=[r for r in results if r['burn_gate_passed'] and r['effects']['focal100kb_tmrca']['ratio']<.7]
    selected=[]
    unmet=[]
    for target,low,high in [(.5,.35,.65),(.2,.10,.35)]:
        available=[r for r in candidates if r['model'] not in [v['model'] for v in selected]
            and low<=r['effects']['focal100kb_tmrca']['ratio']<=high]
        if available:
            best=min(available,key=lambda r:abs(r['effects']['focal100kb_tmrca']['ratio']-target))
            selected.append(dict(model=best['model'],target=target,observed_ratio=best['effects']['focal100kb_tmrca']['ratio'],
                synthetic=best['settings']['synthetic'],confirmation_required=True))
        else:unmet.append(dict(target=target,acceptable_range=[low,high]))
    summary=dict(calibration_complete=True,production_complete=False,source_sha256=digest(__file__),
        bank_manifest_sha256=digest(a.bank/'manifest.json'),bootstrap=4000,seed=260927,
        independent_neutral_families=30,results=results,candidates_for_confirmation=selected,unmet_targets=unmet,
        interpretation='BGS-only screen; model selection must be independently confirmed; synthetic density is not a human annotation')
    write_json(a.out/'summary.json',summary);write_json(a.out/'snapshot_measurements.json',all_records)
    np.savez_compressed(a.out/'profiles.npz',**profiles)
    lines=['# Stronger-BGS calibration (BGS only)','',
        'All six prespecified settings are reported. No sweep outcome or Gamma estimate entered model selection. '
        'Ratios compare independent forward simulations with the completed 30-family neutral reference. '
        'Intervals resample founders, not pairs. These are exploratory calibration results requiring independent confirmation.','',
        '| Model | Target fraction | Focal TMRCA / neutral (95% CI) | 10-Mb TMRCA / neutral | Burn-in gate |','|---|---:|---:|---:|---|']
    for r in results:
        e=r['effects']['focal100kb_tmrca']
        lines.append(f"| {r['model']} | {r['settings']['target_fraction']:.1%} | {e['ratio']:.3f} ({e['ci95'][0]:.3f}–{e['ci95'][1]:.3f}) | {r['effects']['genome_tmrca']['ratio']:.3f} | {r['burn_gate_passed']} |")
    lines+=['','Candidates for independent confirmation: '+', '.join(r['model'] for r in selected)+'.',
        'A setting failing burn-in must be extended before confirmation; uncertainty in all gates is retained in summary.json. '
        'Target fractions of 15% and 30% are synthetic robustness tests and are not calibrated human models. '
        'The Gamma_K17 transfer to conserved noncoding sequence is also a DFE assumption, not a fitted regional DFE.']
    (a.out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(calibration_complete=True,candidates=selected)))


if __name__=='__main__':main()
