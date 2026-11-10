"""Independent two-population hard-sweep simulations; run only in Slurm."""
import argparse, hashlib, importlib.metadata, json, os, time
from pathlib import Path
import numpy as np
import msprime, stdpopsim

REGIMES = {'neutral':None, 'incomplete':(800,0,.25,.75),
           'near_fixed':(1200,0,.8,.99), 'completed':(2000,0,1.,None),
           'older_completed':(4000,2000,1.,None)}

def seed_for(*parts):
    return int(hashlib.sha256('|'.join(map(str,parts)).encode()).hexdigest()[:12],16)%(2**31-2)+1

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(a):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm execution required')
    start=time.monotonic();a.out.mkdir(parents=True,exist_ok=False)
    seed=seed_for('detection-localisation-v1',a.split,a.regime,a.replicate)
    focal=int(np.random.default_rng(seed_for(seed,'target')).integers(400000,600000))
    dem=msprime.Demography()
    for name in ['focal','reference','ancestor']:dem.add_population(name=name,initial_size=10000)
    dem.add_population_split(time=6000,derived=['focal','reference'],ancestral='ancestor')
    model=stdpopsim.DemographicModel(id='TwoPopConstant',description='Two populations split 6000 generations ago',long_description='Controlled no-migration split; constant diploid Ne=10000',generation_time=1,model=dem)
    contig=stdpopsim.Contig.basic_contig(length=1000000,mutation_rate=1.25e-8,recombination_rate=1e-8)
    contig.add_single_site('target',coordinate=focal)
    events=[];spec=REGIMES[a.regime]
    if spec:
        age,end,lo,hi=spec
        events=stdpopsim.selective_sweep(single_site_id='target',population='focal',mutation_generation_ago=age,
                selection_coeff=.02,dominance_coeff=.5,end_generation_ago=end,min_freq_at_end=lo,globally_adaptive=False)
        if hi is not None:
            events.append(stdpopsim.ConditionOnAlleleFrequency(single_site_id='target',population='focal',start_time=0,end_time=0,op='<=',allele_frequency=hi))
    config=dict(regime=a.regime,split=a.split,replicate=a.replicate,seed=seed,selected_position0=focal,
                sequence_length=1000000,Ne=10000,split_age=6000,mu=1.25e-8,r=1e-8,scale=1,burn=.1,
                sampled_diploids_per_population=50,selection_spec=spec,source_sha256=sha(__file__),
                protocol_sha256=sha(Path(__file__).with_name('PROTOCOL.md')),slurm_job=os.environ['SLURM_JOB_ID'])
    (a.out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    engine=stdpopsim.get_engine('slim');original=engine._run_slim
    def logged(script_filename,**kwargs):
        p=Path(script_filename);s=p.read_text()
        if spec:
            observer=f'''
function (void)observe_target(void) {{
 if (community.tick < time_to_tick({spec[0]})) return;
 muts=sim.mutations[sim.mutations.position == {focal}];
 freq=0.0;
 if (size(muts)>0) freq=sum(sim.mutationFrequencies(p0,muts));
 writeFile({json.dumps(str((a.out/'trajectory.tsv').resolve()))},paste(c(community.tick,(G0-community.tick)*Q,freq),sep="\\t"),append=T);
}}
'''
            sentinel='function (void)end(void) {'
            assert s.count(sentinel)==1
            s=s.replace(sentinel,observer+sentinel+'\n observe_target();')
            s+='\n1: late() { observe_target(); }\n'
        p.write_text(s);(a.out/'executed.slim').write_text(s)
        return original(script_filename,**kwargs)
    engine._run_slim=logged
    try:
        ts=engine.simulate(model,contig,samples={'focal':50,'reference':50},seed=seed,
             extended_events=events,slim_path=str(a.slim),slim_scaling_factor=1,slim_burn_in=.1,
             keep_mutation_ids_as_alleles=True,verbosity=0)
    finally:engine._run_slim=original
    samples=ts.samples();labels=np.array([ts.node(int(n)).population for n in samples])
    assert len(samples)==200 and np.sum(labels==0)==100 and np.sum(labels==1)==100
    assert all(t.num_roots==1 for t in ts.trees())
    G=ts.genotype_matrix().T;pos=ts.tables.sites.position.astype(np.int64)
    target_index=np.flatnonzero(pos==focal)
    sample_frequency=float(G[labels==0,target_index[0]].mean()) if len(target_index)==1 else None
    # One genotype preparation for all three methods; exclude multiallelic and causal sites.
    keep=(G.min(axis=0)==0)&(G.max(axis=0)==1)&(pos!=focal)
    G=np.ascontiguousarray(G[:,keep],dtype=np.uint8);pos=pos[keep]
    assert len(np.unique(pos))==len(pos) and np.all(np.diff(pos)>0)
    outcome={}
    if spec:
        t=np.loadtxt(a.out/'trajectory.tsv',ndmin=2)
        assert np.all((t[:,2]>=0)&(t[:,2]<=1))
        # Rejection restarts can move the clock backwards; retain the accepted final segment.
        resets=np.flatnonzero(np.diff(t[:,0])<0)
        accepted=t[(resets[-1]+1 if len(resets) else 0):]
        assert accepted[-1,1]==0
        fixed=accepted[accepted[:,2]==1]
        outcome=dict(final_population_frequency=float(accepted[-1,2]),sample_frequency=sample_frequency,fixation_age=float(fixed[0,1]) if len(fixed) else None,
                     observed_restart_count=int(len(resets)))
        if a.regime in ['completed','older_completed']:assert outcome['final_population_frequency']==1
        else:assert spec[2]<=outcome['final_population_frequency']<=spec[3]
        np.savetxt(a.out/'accepted_trajectory.tsv',accepted,delimiter='\t')
    np.savez_compressed(a.out/'input.npz',G=G,positions=pos,population=labels,selected_position0=focal)
    ts.dump(a.out/'variants.trees')
    record=dict(config=config,seconds=time.monotonic()-start,outcome=outcome,retained_sites=len(pos),
       source_sha256=sha(__file__),input_sha256=sha(a.out/'input.npz'),tree_sha256=sha(a.out/'variants.trees'),
       versions={m:importlib.metadata.version(m) for m in ['stdpopsim','msprime','pyslim','tskit','numpy']})
    (a.out/'COMPLETE.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--regime',choices=REGIMES,required=True);p.add_argument('--split',required=True)
    p.add_argument('--replicate',type=int,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--slim',type=Path,required=True)
    run(p.parse_args())
