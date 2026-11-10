"""Explicit recurrent negative selection in SLiM; frozen Gamma calibration."""
import argparse,fcntl,importlib.metadata,importlib.util,json,os,subprocess,time
from pathlib import Path
import numpy as np
from metrics import bin_piecewise,bin_snps,digest,seed_for,write_json


def selected_intervals(config,length,annotation):
    if config['target']=='annotated':return np.asarray(annotation,dtype=np.int64)
    if config['target']!='block':raise ValueError('Unknown selected-target model')
    center=length//2+int(config['offset']);width=int(config['width'])
    lo,hi=center-width//2,center+(width-width//2)
    if lo<0 or hi>length or width<1:raise ValueError('Selected block outside context')
    return np.array([[lo,hi]],dtype=np.int64)


def validate_config(c,samples):
    if c['model'] not in ['negative','neutral']:raise ValueError('Only negative selection or neutral allowed')
    if c['Ne']/c['Q']<samples//2:raise ValueError('Scaled population smaller than sample')
    if c['burn_Ne']<10 or c['Q']<1:raise ValueError('Insufficient burn or invalid scaling')
    if c['model']=='negative':
        if not -1<c['s']<0 or c['s']*c['Q']<=-1:raise ValueError('Invalid negative homozygous selection coefficient')
        if not 0<=c['h']<=1 or not 0<c['selected_fraction']<=1:raise ValueError('Invalid h or selected fraction')


def run(a):
    import msprime,stdpopsim,tskit
    import gamma_smc_cu
    from gamma_smc_cu import _core
    started=time.monotonic();meta=json.loads((a.target/'manifest.json').read_text())
    c=json.loads(a.config.read_text());validate_config(c,meta['n_haplotypes'])
    for name in ['target','map']:
        if digest(a.target/(name+'.npz'))!=meta[name+'_sha256']:raise ValueError('Frozen target changed')
    annotation_meta=json.loads((a.annotation/'manifest.json').read_text())
    if digest(a.annotation/'annotation.npz')!=annotation_meta['annotation_sha256']:raise ValueError('Annotation changed')
    if digest(a.target/'manifest.json')!=annotation_meta['target_manifest_sha256']:raise ValueError('Annotation/target mismatch')
    target=dict(np.load(a.target/'target.npz'));r=dict(np.load(a.target/'map.npz'))
    length=meta['right0']-meta['left0'];cal=meta['calibration'];mu=cal['physical_mu']
    rate=msprime.RateMap(position=r['position'],rate=r['rate'])
    ann=np.load(a.annotation/'annotation.npz')['selected_intervals']
    intervals=selected_intervals(c,length,ann) if c['model']=='negative' else np.empty((0,2),dtype=np.int64)
    a.out.mkdir(parents=True,exist_ok=False)
    write_json(a.out/'started.json',dict(seed=a.seed,config=c,source_sha256=digest(__file__)))
    selected_coefficients=[];raw_diagnostics={}
    if c['model']=='neutral':
        dem=msprime.Demography();dem.add_population(name='pop',initial_size=c['Ne'])
        if 'ancient_Ne' in c:dem.add_population_parameters_change(time=c['change_time'],initial_size=c['ancient_Ne'])
        ts=msprime.sim_ancestry(samples=meta['n_haplotypes']//2,ploidy=2,demography=dem,
            recombination_rate=rate,random_seed=a.seed)
        real=msprime.sim_mutations(ts,rate=mu,model=msprime.BinaryMutationModel(),random_seed=seed_for(a.seed,'variants'))
        reporter=real
    else:
        contig=stdpopsim.Contig.basic_contig(length=length,mutation_rate=mu,recombination_rate=rate.mean_rate)
        contig.recombination_map=rate
        dfe=stdpopsim.DFE(id='negative_fit',description='Recurrent deleterious mutations',
            long_description='Exploratory fixed deleterious DFE; no beneficial mutation or conditioned fixation',
            mutation_types=[stdpopsim.MutationType(dominance_coeff=c['h'],distribution_type='f',distribution_args=[c['s']]),
                            stdpopsim.MutationType(distribution_type='f',distribution_args=[0.])],
            proportions=[c['selected_fraction'],1-c['selected_fraction']])
        contig.add_dfe(intervals=intervals,DFE=dfe)
        engine=stdpopsim.get_engine('slim');original=engine._run_slim
        def logged(script_file,**kwargs):
            text=Path(script_file).read_text();(a.out/'executed.slim').write_text(text)
            original(script_file,**kwargs)
        engine._run_slim=logged
        recap_original=engine._recap_and_rescale
        def audit_raw(*args,**kwargs):
            raw=args[0] if args else kwargs['ts']
            sampled=engine._simplify_remembered(raw)
            raw_diagnostics.update(uncoalesced_fraction=sum(t.span for t in sampled.trees() if t.num_roots>1)/length,
                slim_tick=raw.metadata['SLiM']['tick'],Q=c['Q'])
            return recap_original(*args,**kwargs)
        engine._recap_and_rescale=audit_raw
        try:
            real=engine.simulate(stdpopsim.PiecewiseConstantSize(c['Ne']),contig,
                samples={'pop_0':meta['n_haplotypes']//2},seed=a.seed,slim_path=str(a.slim),
                slim_scaling_factor=c['Q'],slim_burn_in=c['burn_Ne'],verbosity=0,
                keep_mutation_ids_as_alleles=True)
        finally:engine._run_slim=original;engine._recap_and_rescale=recap_original
        for mut in real.mutations():
            for item in mut.metadata.get('mutation_list',[]):
                s=item['selection_coeff']
                if s>0:raise ValueError('Positive mutation in negative-only simulation')
                if s<0:selected_coefficients.append(s)
        tables=real.dump_tables();tables.sites.clear();tables.mutations.clear();ts=tables.tree_sequence()
        reporter=msprime.sim_mutations(ts,rate=mu,model=msprime.BinaryMutationModel(),random_seed=seed_for(a.seed,'reporters'))
    if ts.num_samples!=meta['n_haplotypes'] or any(t.num_roots!=1 for t in ts.trees()):raise ValueError('Incomplete ancestry or wrong sample count')
    # Preserve real selected + neutral variants, alongside a neutral reporter assay.
    real.dump(a.out/'variants.trees');ts.dump(a.out/'ancestry.trees')
    pairs=target['pairs'].astype(int);edges=target['edges']-meta['left0']
    truth=bin_piecewise(ts.breakpoints(as_array=True),[np.mean([t.tmrca(int(i),int(j)) for i,j in pairs]) for t in ts.trees()],edges)
    spec=importlib.util.spec_from_file_location('frozen_gamma_api',a.api);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    def infer(variants):
        G=variants.genotype_matrix().T;pos=variants.tables.sites.position
        # Only true biallelic, polymorphic sites. Selected SLiM mutation IDs remain alleles.
        use=(G.max(axis=0)==1)&(G.min(axis=0)==0)
        G=np.ascontiguousarray(G[:,use],dtype=np.uint8);pos=pos[use]
        with a.gamma_lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            free,_=_core.cuda_mem_info()
            if free<4*1024**3:raise RuntimeError('Less than 4 GiB on assigned GPU')
            pred=api.infer(G,pos,pairs=pairs.tolist(),Ne=cal['calibrated_Ne'],mu=mu,rho=cal['physical_rho'],physical_mu=mu,
                auto_estimate_theta=False,mean_only=True,flow_field_path=str(Path(gamma_smc_cu.__file__).parent/'default_flow_field.txt'))
        if pred['metadata']['time_units']!='generations':raise ValueError('Incorrect Gamma time units')
        return bin_snps(pred['positions'],pred['mean'].mean(axis=1),edges),dict(sites=len(pos),excluded_sites=variants.num_sites-len(pos))
    inferred,counts=infer(real)
    if c['model']=='neutral':reported=inferred;reporter_counts=counts
    else:reported,reporter_counts=infer(reporter)
    np.savez_compressed(a.out/'profile.npz',inferred=inferred,truth=truth,reporter=reported,edges=target['edges'])
    write_json(a.out/'complete.json',dict(seed=a.seed,config=c,seconds=time.monotonic()-started,
        target_manifest_sha256=digest(a.target/'manifest.json'),annotation_sha256=annotation_meta['annotation_sha256'],
        config_sha256=digest(a.config),source_sha256=digest(__file__),api_sha256=digest(a.api),
        profile_sha256=digest(a.out/'profile.npz'),ancestry_sha256=digest(a.out/'ancestry.trees'),variants_sha256=digest(a.out/'variants.trees'),
        selected_intervals=intervals.tolist(),selected_bp=int(np.diff(intervals,axis=1).sum()),
        negative_mutation_entries=len(selected_coefficients),negative_s_range=[float(min(selected_coefficients)),float(max(selected_coefficients))] if selected_coefficients else [],
        raw_diagnostics=raw_diagnostics,actual_variant_counts=counts,reporter_counts=reporter_counts,
        calibration=cal,versions={n:importlib.metadata.version(n) for n in ['stdpopsim','msprime','pyslim','tskit']},
        Q_interpretation='Approximate SLiM rescaling; Ne/Q individuals, Q*s selection; times returned in physical generations',
        primary_inference='Actual selected and neutral biallelic variants; no duplicate neutral overlay at selected sites',
        reporter_inference='Independent neutral reporter overlay on the same selected genealogy; diagnostic only'))
    print(json.dumps(dict(seconds=time.monotonic()-started,config=c,raw_diagnostics=raw_diagnostics)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['target','annotation','config','api','out','gamma-lock','slim']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    run(p.parse_args())
