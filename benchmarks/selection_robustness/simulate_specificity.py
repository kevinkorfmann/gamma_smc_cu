"""Controlled regional SLiM experiment with an audited population trajectory.

Separate from the broad production benchmark: constant Ne, no ascertainment,
and exact generating maps. Logs do not draw random numbers or condition on
survival. Neutral recapitation is only older than the forward selected phase.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import msprime
import numpy as np
import stdpopsim

from common import digest, extract_inputs, seed_for, write_json


def instrument(script, path, focal, age):
    """Add observational logging to the exact script executed by SLiM 4.3."""
    sentinel = "function (void)end(void) {"
    if script.count(sentinel) != 1:
        raise ValueError("Unexpected stdpopsim script version")
    logger = f'''
function (void)record_focal(void) {{
    if (community.tick < time_to_tick({age})) return;
    muts = sim.mutations[sim.mutations.position == {focal}];
    freq = 0.0;
    if (size(muts) > 0) freq = sum(sim.mutationFrequencies(NULL, muts));
    writeFile({json.dumps(str(path.resolve()))}, paste(c(community.tick,
        (G0-community.tick)*Q, freq, sum(sim.subpopulations.individualCount)),
        sep="\\t"), append=T);
}}
'''
    script = script.replace(sentinel, logger + sentinel + "\n    record_focal();")
    draw = "targets.addNewDrawnMutation(mut_type, pos);"
    if script.count(draw) != 1:
        raise ValueError("Unexpected mutation introduction code")
    script = script.replace(draw, draw + "\n   record_focal();")
    return script + "\n1: late() { record_focal(); }\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--region", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--regime", choices=["neutral","positive","bgs","bgs_positive","balancing","bgs_balancing"], required=True)
    p.add_argument("--replicate", type=int, required=True)
    p.add_argument("--Ne", type=int, default=10000)
    p.add_argument("--scale", type=float, default=1)
    p.add_argument("--burn", type=float, default=1)
    p.add_argument("--age", type=int, default=1000)
    p.add_argument("--s", type=float, default=.05)
    p.add_argument("--mu", type=float, default=1.29e-8)
    p.add_argument("--diploids", type=int, default=25)
    p.add_argument("--slim", required=True)
    a = p.parse_args()
    region = json.loads((a.region/'region.json').read_text())
    if digest(a.region/'maps.npz') != region['maps_sha256']:
        raise ValueError("Region hash mismatch")
    if a.age >= a.Ne*a.burn or a.Ne/a.scale < a.diploids or a.s*a.scale >= 1:
        raise ValueError("Invalid history, population or fitness")
    if a.regime.startswith('bgs') and a.burn < 5:
        raise ValueError("BGS requires at least 5 Ne forward burn-in; convergence must be checked")
    a.out.mkdir(parents=True, exist_ok=False)
    maps=np.load(a.region/'maps.npz')
    L=region['length']; focal=41160846-region['left']
    if not 0 <= focal < L:
        raise ValueError("This controlled experiment targets the TREM2 neighborhood")
    rmap=msprime.RateMap(position=maps['map_position'],rate=maps['map_rate'])
    contig=stdpopsim.Contig.basic_contig(length=L,mutation_rate=a.mu,recombination_rate=rmap.mean_rate)
    contig.recombination_map=rmap
    if a.regime.startswith('bgs'):
        contig.add_dfe(intervals=maps['exons'],DFE=stdpopsim.get_species('HomSap').get_dfe('Gamma_K17'))
    selected=('positive' in a.regime or 'balancing' in a.regime)
    events=[]
    if selected:
        contig.add_single_site('focal',coordinate=focal)
        events=[stdpopsim.DrawMutation(time=a.age,single_site_id='focal',population='pop_0'),
            stdpopsim.ChangeMutationFitness(start_time=a.age,end_time=0,single_site_id='focal',population='pop_0',
                selection_coeff=-a.s if 'balancing' in a.regime else a.s,
                dominance_coeff=-1 if 'balancing' in a.regime else .5)]
    seed=seed_for('specificity-v1',region['maps_sha256'],a.regime,a.replicate,a.Ne,a.scale,a.burn,a.age,a.s)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config.update(seed=seed,region=region,focal_position0=focal,source_sha256=digest(__file__),
                  versions={x:importlib.metadata.version(x) for x in ['stdpopsim','msprime','pyslim','tskit']},
                  purpose='controlled_specificity_experiment_not_empirical_ABC')
    write_json(a.out/'config.json',config)
    engine=stdpopsim.get_engine('slim'); original=engine._run_slim
    def logged_run(script_filename, **kwargs):
        f=Path(script_filename);script=f.read_text()
        if selected:script=instrument(script,a.out/'trajectory.tsv',focal,a.age)
        f.write_text(script);(a.out/'executed.slim').write_text(script)
        return original(script_filename,**kwargs)
    engine._run_slim=logged_run
    started=time.perf_counter()
    try:
        ts=engine.simulate(stdpopsim.PiecewiseConstantSize(a.Ne),contig,samples={'pop_0':a.diploids},
            seed=seed,extended_events=events,slim_path=a.slim,slim_scaling_factor=a.scale,
            slim_burn_in=a.burn,verbosity=0)
    finally:engine._run_slim=original
    elapsed=time.perf_counter()-started
    ts.dump(a.out/'truth.trees')
    G,pos,dropped=extract_inputs(ts)
    edges=np.arange(0,L+1,2000)
    if edges[-1] != L:edges=np.r_[edges,L]
    pi=ts.diversity(windows=edges)
    branch=ts.diversity(windows=edges,mode='branch')/2
    np.savez_compressed(a.out/'input.npz',G=G,positions=pos,edges=edges,pi=pi,
                        mean_pair_tmrca=branch,map_position=rmap.position,map_rate=rmap.rate)
    outcome='not_applicable';final_freq=None;max_freq=None;fix_age=None
    if selected:
        trajectory=np.loadtxt(a.out/'trajectory.tsv',ndmin=2)
        if not np.all((trajectory[:,2]>=0)&(trajectory[:,2]<=1)):
            raise ValueError('Invalid population frequencies')
        if not np.isclose(trajectory[:,2].max(),1/(2*a.Ne/a.scale)) and trajectory[:,2].max()<1/(2*a.Ne/a.scale):
            raise ValueError('Mutation introduction not logged')
        if trajectory[-1,1]!=0:raise ValueError('Missing final population observation')
        final_freq=float(trajectory[-1,2]);max_freq=float(trajectory[:,2].max())
        outcome='lost' if final_freq==0 else 'fixed' if final_freq==1 else 'segregating'
        fixed=trajectory[trajectory[:,2]==1]
        fix_age=float(fixed[0,1]) if len(fixed) else None
    write_json(a.out/'qc.json',dict(status='completed_controlled_simulation',seconds=elapsed,
        population_outcome=outcome,population_final_frequency=final_freq,max_population_frequency=max_freq,
        fixation_age_generations=fix_age,samples=ts.num_samples,sites=ts.num_sites,
        dropped=dropped,mean_diversity=ts.diversity(),neutral_expectation=4*a.Ne*a.mu,
        time_units=ts.time_units,source_sha256=digest(__file__),tree_sha256=digest(a.out/'truth.trees')))
    print(json.dumps(dict(regime=a.regime,replicate=a.replicate,seconds=elapsed,outcome=outcome)))


if __name__=='__main__':main()
