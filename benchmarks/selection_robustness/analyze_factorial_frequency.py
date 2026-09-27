"""Founder-cluster summaries; stages are conditional on natural attainment."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from common import digest, write_json

STAGES=['control','f25','f50','f75','f100','fixed_plus250','fixed_plus1000']
STAGES += [s+'_fixedcohort' for s in ['f25','f50','f75','f100']]
METHODS=['truth','auto','fixed']


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--partial',action='store_true')
    p.add_argument('--bootstrap',type=int,default=4000);a=p.parse_args()
    manifest=json.loads((a.bank/'manifest.json').read_text())
    complete=a.bank/'completion.json'
    if not a.partial:
        if not complete.exists():raise ValueError('Simulation bank is still running')
        if any(x['status']!='completed' for x in json.loads(complete.read_text())['results']):
            raise ValueError('Failed families must be resolved or reported before final analysis')
    a.out.mkdir(exist_ok=True,parents=True)
    effects=[];counts=[];diagnostics=[];profile={};burn=[];edges=None
    focal_records=[];matched_effects=[];control_draws={};control_points={}
    region=manifest['region'];focal=41160846-region['left']
    summary=dict(complete=not a.partial,manifest_sha256=digest(a.bank/'manifest.json'),
        source_sha256=digest(__file__),bootstrap_replicates=a.bootstrap,
        uncertainty='95% pointwise percentile bootstrap, resampling independent founder families',
        conditioning='each stage includes all introductions reaching it; no forced fixation',
        region=region,counts=counts,effects=effects,burn_in=burn)
    rng=np.random.default_rng(804301)
    for bg in ['neutral','bgs']:
        families=[f for f in sorted(a.bank.glob(f'{bg}_family_*'))
            if (f/'family_status.json').exists() and json.loads((f/'family_status.json').read_text())['status']=='completed']
        if not families:continue
        n=len(families);weights=rng.multinomial(n,np.full(n,1/n),size=a.bootstrap)
        group=defaultdict(lambda:[[] for _ in families]);metadata=defaultdict(list)
        outcomes=defaultdict(int)
        for fi,fam in enumerate(families):
            for run in [fam/'control',*sorted(r for r in fam.glob('sweep_*') if r.is_dir())]:
                if run.name.startswith('sweep_'):
                    tr=np.loadtxt(run/'trajectory.tsv',ndmin=2)
                    outcomes['fixed' if tr[-1,2]==1 else 'lost' if tr[-1,2]==0 else 'segregating']+=1
                snaps={}
                sp=run/'snapshots.tsv'
                if sp.exists():
                    for line in sp.read_text().splitlines():
                        name,tick,freq=line.split('\t');snaps[name]=(int(tick),float(freq))
                cfg=json.loads((run/'config.json').read_text())
                for stage in STAGES:
                    key='final' if stage=='control' else stage.removesuffix('_fixedcohort')
                    if (stage=='control') != (run.name=='control'):continue
                    if stage.endswith('_fixedcohort') and not (run/'f100.variants.trees').exists():continue
                    path=run/'gamma'/key/'predictions.npz'
                    if not path.exists():continue
                    z=np.load(path);new_edges=z['edges']
                    if edges is None:edges=new_edges
                    if not np.array_equal(edges,new_edges):raise ValueError('Mismatched genomic grids')
                    focal_record=dict(background=bg,family=fam.name,run=run.name,stage=stage)
                    for method in METHODS:
                        values=z[method].mean(axis=1)
                        # Only plotting grid ends can lack SNP projection support.
                        if method!='truth':values[z[method+'_coverage']<.999]=np.nan
                        group[(stage,method)][fi].append(values)
                        focal_record[method]=float(np.mean(values[np.abs((new_edges[:-1]+new_edges[1:])/2-focal)<50000]))
                    if not stage.endswith('_fixedcohort'):focal_records.append(focal_record)
                    m=json.loads((path.parent/'metrics.json').read_text())
                    diagnostics.append(dict(background=bg,family=fam.name,stage=stage,
                        snps=m['total_snps'],focal_snps=m['focal100kb_snps'],
                        auto=m['auto'],fixed=m['fixed'],context=m.get('context_sensitivity',{})))
                    if key in snaps:
                        tick,freq=snaps[key]
                        metadata[stage].append(dict(family=fam.name,frequency=freq,
                            age=tick-cfg['start_tick']-1))
            if bg=='bgs':
                for stage in ['burn5Ne','final']:
                    z=np.load(fam/'founder'/f'{stage}.npz')
                    centers=(z['edges'][:-1]+z['edges'][1:])/2
                    local=np.abs(centers-focal)<50000
                    burn.append(dict(family=fam.name,stage=stage,
                        genome_tmrca=float(np.average(z['tmrca'],weights=np.diff(z['edges']))),
                        focal100kb_tmrca=float(np.average(z['tmrca'][local],weights=np.diff(z['edges'])[local]))))
        centers=(edges[:-1]+edges[1:])/2;local=np.abs(centers-focal)<50000
        draws={}
        for stage in STAGES:
            for method in METHODS:
                rows=group[(stage,method)]
                numbers=np.array([len(v) for v in rows])
                if not numbers.sum():continue
                sums=np.array([np.sum(v,axis=0) if len(v) else np.zeros(len(centers)) for v in rows])
                mean=sums.sum(axis=0)/numbers.sum()
                denom=weights@numbers
                boot=np.divide(weights@sums,denom[:,None],
                    out=np.full((a.bootstrap,len(centers)),np.nan),where=denom[:,None]>0)
                lo,hi=np.nanquantile(boot,[.025,.975],axis=0)
                for name,val in [('mean',mean),('lo',lo),('hi',hi)]:
                    profile[f'{bg}_{stage}_{method}_{name}']=val
                draws[(stage,method)]=np.mean(boot[:,local],axis=1)
                if stage=='control':
                    control_draws[(bg,method)]=draws[(stage,method)]
                    control_points[(bg,method)]=float(np.mean(mean[local]))
                else:
                    # Sensitivity to conditioning on successful founders:
                    # weight each founder's control by its attained stages.
                    controls=np.array([np.mean(v[0][local]) for v in group[('control',method)]])
                    stage_sums=np.mean(sums[:,local],axis=1)
                    matched_ratio=(weights@stage_sums)/(weights@(numbers*controls))
                    matched_effects.append(dict(background=bg,stage=stage,method=method,
                        focal100kb_retained_tmrca=float(stage_sums.sum()/np.sum(numbers*controls)),
                        ci95=np.nanquantile(matched_ratio,[.025,.975]).tolist()))
                if method=='truth':
                    md=metadata[stage]
                    counts.append(dict(background=bg,stage=stage,snapshots=int(numbers.sum()),
                        contributing_families=int((numbers>0).sum()),total_families=n,
                        mean_frequency=float(np.mean([v['frequency'] for v in md])) if md else None,
                        mean_age_generations=float(np.mean([v['age'] for v in md])) if md else None))
        for stage in STAGES[1:]:
            for method in METHODS:
                if (stage,method) not in draws:continue
                ratio=draws[(stage,method)]/draws[('control',method)]
                actual=np.mean(profile[f'{bg}_{stage}_{method}_mean'][local])/np.mean(profile[f'{bg}_control_{method}_mean'][local])
                lo,hi=np.nanquantile(ratio,[.025,.975])
                effects.append(dict(background=bg,stage=stage,method=method,
                    focal100kb_retained_tmrca=float(actual),ci95=[float(lo),float(hi)]))
        summary[bg]=dict(completed_families=n,outcomes=dict(outcomes))
    if edges is None:raise ValueError('No fully completed families yet')
    summary['matched_founder_effects']=matched_effects
    summary['background_control_contrasts']=[]
    for method in METHODS:
        ratio=control_draws[('bgs',method)]/control_draws[('neutral',method)]
        summary['background_control_contrasts'].append(dict(method=method,
            focal100kb_retained_tmrca=control_points[('bgs',method)]/control_points[('neutral',method)],
            ci95=np.nanquantile(ratio,[.025,.975]).tolist()))
    write_json(a.out/'focal_replicates.json',focal_records)
    profile['edges']=edges
    np.savez_compressed(a.out/'profiles.npz',**profile)
    # Paired burn-in difference, with family resampling rather than treating
    # the two snapshots from a founder as independent observations.
    if burn:
        paired={r['family']:{} for r in burn}
        for r in burn:paired[r['family']][r['stage']]=r
        summary['burn_in_contrast']={}
        for metric in ['genome_tmrca','focal100kb_tmrca']:
            x=np.array([[v['burn5Ne'][metric],v['final'][metric]] for v in paired.values()])
            w=rng.multinomial(len(x),np.full(len(x),1/len(x)),size=a.bootstrap)
            bx=w@x
            ratios=bx[:,1]/bx[:,0]
            summary['burn_in_contrast'][metric]=dict(ratio_10Ne_over_5Ne=float(x[:,1].mean()/x[:,0].mean()),
                ci95=np.quantile(ratios,[.025,.975]).tolist(),families=len(x))
    write_json(a.out/'summary.json',summary)
    write_json(a.out/'inference_diagnostics.json',diagnostics)
    lines=['# Matched-map BGS × sweep frequency experiment',
        '', '**Completed simulation bank.**' if not a.partial else '**Preliminary: completed families only; the bank is still running.**',
        '', 'All arms use the same 10-Mb deCODE map, Ne = 10,000 and mutation rate. Gamma-SMC-CU uses 50 haplotypes and all 1,225 pairs; these pairs are not independent simulation replicates.',
        '', 'Confidence intervals resample founder families. Sweep stages are conditional on reaching that frequency. Realized diversity/TMRCA reduction is distinct from the published B′ map.',
        '', '| Background | Stage | Method | Retained focal TMRCA | 95% CI |',
        '|---|---|---|---:|---:|']
    for r in effects:
        lines.append(f"| {r['background']} | {r['stage']} | {r['method']} | {r['focal100kb_retained_tmrca']:.3f} | {r['ci95'][0]:.3f}–{r['ci95'][1]:.3f} |")
    lines+=['','Burn-in contrasts and context-sensitivity results must be reviewed before treating these figures as final SI evidence. This report does not establish a selection model for empirical TREM2 or demonstrate bias in the published B′ map.']
    (a.out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
