"""Controlled positive-only SLiM experiment and published B-prime refitting.

The genome outside the simulated interval is held fixed. Inside fully contained
100-kb fitting bins, expected pair counts encode pi0 * pi_sim/(4 Ne mu).
This is a conditional likelihood perturbation, not a synthetic human genome or
an empirical selection posterior. All attempted sweeps must be accounted for.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import tskit

from bprime_refit import BprimeFit
from common import digest, write_json


def bootstrap_difference(selected, neutral, rng, n=10000):
    selected=np.asarray(selected);neutral=np.asarray(neutral)
    if len(selected)<2 or len(neutral)<2:
        return dict(estimate=float(selected.mean()-neutral.mean()) if len(selected) else None,
                    ci95=None,n_selected=len(selected),n_neutral=len(neutral))
    sims=selected[rng.integers(len(selected),size=(n,len(selected)))].mean(axis=1)
    sims-=neutral[rng.integers(len(neutral),size=(n,len(neutral)))].mean(axis=1)
    return dict(estimate=float(selected.mean()-neutral.mean()),
                ci95=np.quantile(sims,[.025,.975]).tolist(),
                n_selected=len(selected),n_neutral=len(neutral))


def neutralized_tree(ts, focal):
    # Diversity must exclude the intentionally introduced selected mutation.
    sites=np.flatnonzero(ts.tables.sites.position==focal)
    return ts.delete_sites(sites) if len(sites) else ts


def analyze(bank, model, out):
    manifest=json.loads((bank/'manifest.json').read_text())
    completion=json.loads((bank/'completion.json').read_text())
    if len(completion['results'])!=len(manifest['jobs']) or any(
            r['status'] not in ['completed','already_completed'] for r in completion['results']):
        raise ValueError('Bank incomplete or contains failed attempts')
    first=bank/f"{manifest['jobs'][0]['regime']}_{manifest['jobs'][0]['replicate']:04d}"
    config=json.loads((first/'config.json').read_text());reg=config['region']
    z=model.data;origin=reg['left'];end=origin+reg['length'];focal=config['focal_position0']
    target=np.flatnonzero((z['chrom']=='chr6')&(z['start']>=origin)&(z['end']<=end))
    gene=np.flatnonzero((z['chrom']=='chr6')&(z['start']<=origin+focal)&(z['end']>origin+focal))
    if len(gene)!=1 or not set(gene).issubset(set(target)):
        raise ValueError('Focal bin absent from model')
    gene=int(gene[0]);gene_in_target=int(np.flatnonzero(target==gene)[0])
    distance=np.abs((z['start'][target]+z['end'][target])/2-(origin+focal))
    flanks=target[(distance>=250000)&(distance<=500000)]
    outside=np.ones(len(z['Y']),dtype=bool);outside[target]=False
    if len(flanks)<2:raise ValueError('Missing comparison flanks')
    edges=np.unique(np.r_[0,z['start'][target]-origin,z['end'][target]-origin,reg['length']])
    indices=np.searchsorted(edges,z['start'][target]-origin)
    if not np.array_equal(edges[indices+1],z['end'][target]-origin):
        raise ValueError('Model bins not elementary intervals')
    plot_edges=np.unique(np.r_[np.arange(0,reg['length'],20000),reg['length']])
    baseline=model.fit(z['Y']);baseB=baseline['B'];baseX=baseline['x']
    rows=[];profiles=[];branch_profiles=[];Bprofiles=[];xs=[];audit=[]
    for job in manifest['jobs']:
        path=bank/f"{job['regime']}_{job['replicate']:04d}"
        cfg=json.loads((path/'config.json').read_text());qc=json.loads((path/'qc.json').read_text())
        if cfg['region']['maps_sha256']!=reg['maps_sha256'] or qc['time_units']!='generations':
            raise ValueError('Unexpected map or timescale')
        if any(cfg[k]!=job[k] for k in job):raise ValueError('Run differs from manifest')
        ts=tskit.load(path/'truth.trees')
        if digest(path/'truth.trees')!=qc['tree_sha256']:raise ValueError('Tree checksum mismatch')
        ts=neutralized_tree(ts,focal)
        pi=ts.diversity(windows=edges)[indices];T=ts.diversity(windows=edges,mode='branch')[indices]/2
        ratio=pi/(4*cfg['Ne']*cfg['mu'])
        Y=z['Y'].astype(float).copy();total=Y[target].sum(axis=1)
        # Use published pi0, not a fitted value for each simulated replicate.
        p=z['theta'][0]*ratio
        if np.any((p<=0)|(p>=1)):raise ValueError('Invalid replacement diversity')
        Y[target,1]=total*p;Y[target,0]=total*(1-p)
        fit=model.fit(Y,start=baseX)
        if job['replicate'] in [0,1] or (job['regime']=='positive' and qc['population_outcome']=='fixed' and len(audit)<5):
            alternate=baseX.copy();alternate[0]+=np.log(1.1)
            alternate[1:]=alternate[1:][::-1]
            check=model.fit(Y,start=alternate)
            delta=float(np.max(np.abs(fit['B']-check['B'])))
            focal_delta=float(abs(fit['B'][gene]-check['B'][gene]))
            audit.append(dict(run=path.name,max_B_difference=delta,focal_B_difference=focal_delta,
                              nll_difference=fit['nll']-check['nll']))
            if delta>2e-5:raise ValueError(f'Optimizer start sensitivity {delta}: {path}')
            if focal_delta>5e-7:raise ValueError(f'Focal optimizer start sensitivity {focal_delta}: {path}')
        row=dict(run=path.name,regime=job['regime'],replicate=job['replicate'],seed=cfg['seed'],
            outcome=qc['population_outcome'],fixation_age=qc['fixation_age_generations'],
            focal_pi_ratio=float(ratio[gene_in_target]),focal_tmrca_ratio=float(T[gene_in_target]/(2*cfg['Ne'])),
            focal_B=float(fit['B'][gene]),focal_B_minus_published=float(fit['B'][gene]-z['Bprime'][gene]),
            focal_minus_flanks_B=float(fit['B'][gene]-fit['B'][flanks].mean()),
            outside_region_mean_B=float(fit['B'][outside].mean()),
            refit_pi0=fit['pi0'],refit_mu=fit['mu'],iterations=fit['iterations'],nll=fit['nll'])
        rows.append(row);xs.append(fit['x']);Bprofiles.append(fit['B'][target])
        profiles.append(ts.diversity(windows=plot_edges)/(4*cfg['Ne']*cfg['mu']))
        branch_profiles.append(ts.diversity(windows=plot_edges,mode='branch')/(4*cfg['Ne']))
        print(json.dumps(dict(bank=str(bank),done=len(rows),total=len(manifest['jobs']),run=path.name)),flush=True)
    labels=np.array([r['outcome'] if r['regime']=='positive' else 'neutral' for r in rows])
    rng=np.random.default_rng(4381635);contrasts={}
    for group in ['all_positive','fixed','lost','segregating']:
        mask=labels!='neutral' if group=='all_positive' else labels==group
        contrasts[group]={key:bootstrap_difference([r[key] for r,m in zip(rows,mask) if m],
                         [r[key] for r,m in zip(rows,labels=='neutral') if m],rng)
                         for key in ['focal_pi_ratio','focal_tmrca_ratio','focal_B',
                                     'focal_minus_flanks_B','outside_region_mean_B']}
    out.mkdir(parents=True,exist_ok=False)
    with (out/'replicates.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    np.savez_compressed(out/'profiles.npz',labels=labels,edges=plot_edges+origin,
        pi=profiles,tmrca=branch_profiles,Bprime=Bprofiles,fit_x=xs,
        model_start=z['start'][target],model_end=z['end'][target],published_B=z['Bprime'][target])
    summary=dict(bank=str(bank),n=len(rows),outcomes=dict(Counter(labels)),
        neutral_means={key:float(np.mean([r[key] for r in rows if r['regime']=='neutral']))
                       for key in ['focal_pi_ratio','focal_tmrca_ratio','focal_B']},
        contrasts=contrasts,published_focal_B=float(z['Bprime'][gene]),
        refitted_original_focal_B=float(baseB[gene]),fixed_input_B_change=0.0,
        target_bins=len(target),target_bp=int(np.sum(z['end'][target]-z['start'][target])),
        focal_bin=[int(z['start'][gene]),int(z['end'][gene])],
        flank_bins=np.c_[z['start'][flanks],z['end'][flanks]].tolist(),
        target_fraction_of_genome_pair_counts=float(z['Y'][target].sum()/z['Y'].sum()),
        optimizer_checks=audit,bootstrap_replicates=10000,
        map_sha256=reg['maps_sha256'],manifest_sha256=digest(bank/'manifest.json'),
        published_fit_sha256=digest(Path(model.input_path)),source_sha256=digest(__file__),
        interpretation='Independent SLiM replicates; global B-prime refits conditional on a fixed empirical genome outside 29 fully contained regional bins. No BGS in either simulated arm. No empirical ABC posterior.',
        masks='Simulated bins are fully callable; published per-bin pair counts provide weights. No within-bin empirical mask or mutation-map correction is reconstructed.')
    write_json(out/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--published',type=Path,required=True)
    p.add_argument('--bank',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();model=BprimeFit(np.load(a.published));model.input_path=a.published
    summary=analyze(a.bank,model,a.out);print(json.dumps(summary))


if __name__=='__main__':main()
