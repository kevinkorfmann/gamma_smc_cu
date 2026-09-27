"""Stronger-BGS extension of the frozen matched-map frequency simulator.

Burn-in founders are independent units. Branches from one founder are clustered
replicates, not independent burn-ins. Q is deliberately fixed at one. Raw SLiM
checkpoints are sampled offline so stage collection never consumes SLiM RNG.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import re
import subprocess
import time

import msprime
import numpy as np
import pyslim
import stdpopsim
import tskit

from common import digest, seed_for, write_json


def setup(region_path, bgs, Ne, mu):
    region = json.loads((region_path / 'region.json').read_text())
    if digest(region_path / 'maps.npz') != region['maps_sha256']:
        raise ValueError('Regional map checksum mismatch')
    arrays = np.load(region_path / 'maps.npz')
    rmap = msprime.RateMap(position=arrays['map_position'], rate=arrays['map_rate'])
    contig = stdpopsim.Contig.basic_contig(length=region['length'],
        mutation_rate=mu, recombination_rate=rmap.mean_rate)
    contig.recombination_map = rmap
    if bgs:
        selection=region['selection_model']
        if selection['dfe']=='Gamma_K17':
            dfe=stdpopsim.get_species('HomSap').get_dfe('Gamma_K17')
        else:
            if not -1<selection['s']<0:raise ValueError('Expected deleterious homozygous coefficient')
            if selection['neutral_fraction']!=0:raise ValueError('Fixed-DFE model must have zero neutral target fraction')
            dfe=stdpopsim.DFE(id=selection['id'],description='Prespecified deleterious sensitivity',
                long_description=selection['interpretation'],mutation_types=[stdpopsim.MutationType(
                    dominance_coeff=selection['h'],distribution_type='f',distribution_args=[selection['s']])],proportions=[1.])
        contig.add_dfe(intervals=arrays['selected_intervals'],DFE=dfe)
    focal = 41160846 - region['left']
    contig.add_single_site('focal', coordinate=focal)
    return region, arrays, contig, stdpopsim.PiecewiseConstantSize(Ne), focal


def generate(a):
    region, arrays, contig, model, focal = setup(a.region, a.bgs, a.Ne, a.mu)
    branch = a.mode == 'branch'
    if not branch and a.bgs and a.burn < 5:
        raise ValueError('At least 5 Ne is required; convergence still needs measurement')
    if branch:
        raw = tskit.load(a.parent)
        start = int(raw.metadata['SLiM']['tick'])
        if raw.metadata['SLiM']['stage'] != 'late':
            raise ValueError('Only late-stage checkpoints are supported')
        parent_config = json.loads((a.parent.parent / 'config.json').read_text())
        for key, value in [('bgs', a.bgs), ('Ne', a.Ne), ('mu', a.mu)]:
            if parent_config[key] != value:
                raise ValueError(f'Parent mismatch: {key}')
        if parent_config['region']['maps_sha256'] != region['maps_sha256']:
            raise ValueError('Parent map mismatch')
        if a.bgs and start < 1 + 5 * a.Ne:
            raise ValueError('BGS founder has insufficient burn-in')
        end = start + a.duration
        burn = (end - 1) / a.Ne
    else:
        start = 1
        end = 1 + round(a.burn * a.Ne)
        burn = a.burn
    if not 0 < a.s < 1:
        raise ValueError('Invalid beneficial homozygous fitness increment')
    events = []
    if branch and a.sweep:
        # Introduce in the first new generation after restoring the late checkpoint.
        events = [stdpopsim.DrawMutation(time=a.duration-1, single_site_id='focal', population='pop_0'),
            stdpopsim.ChangeMutationFitness(start_time=a.duration-1, end_time=0,
                single_site_id='focal', population='pop_0', selection_coeff=a.s, dominance_coeff=.5)]
    seed = seed_for('stronger-bgs-v1', region['maps_sha256'],region['selection_model'], a.mode,
        a.bgs, a.sweep, a.replicate, start, end, a.Ne, a.mu, a.s,
        digest(a.parent) if branch else None)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        stdpopsim.get_engine('slim').simulate(model, contig, samples={'pop_0':25},
            extended_events=events, seed=seed, slim_scaling_factor=1,
            slim_burn_in=burn, slim_script=True)
    script = buffer.getvalue()
    out = a.out.resolve()
    script, n = re.subn(r'defineConstant\("trees_file", "[^"]+"\);',
        f'defineConstant("trees_file", {json.dumps(str(out / "final.raw.trees"))});', script)
    if n != 1:
        raise ValueError('Unexpected generated script')
    # No stochastic operations here: population samples are selected in Python.
    observer = f'''
function (void)factorial_snapshot(string$ label, float$ freq) {{
    key = "snapshot_" + label;
    if (!isNULL(sim.getValue(key))) return;
    sim.setValue(key, T);
    sim.treeSeqOutput({json.dumps(str(out))} + "/" + label + ".raw.trees", metadata=metadata);
    writeFile({json.dumps(str(out / 'snapshots.tsv'))},
        paste(c(label, community.tick, freq), sep="\\t"), append=T);
}}
function (void)factorial_observe(void) {{
    if (community.tick % 1000 == 1)
        writeFile({json.dumps(str(out / 'progress.tsv'))},
            paste(c(community.tick, size(sim.mutations)), sep="\\t"), append=T);
'''
    if branch:
        observer += f'''
    if (community.tick < {start}) return;
    muts = sim.mutations[sim.mutations.position == {focal}];
    freq = 0.0;
    if (size(muts) > 0) freq = sum(sim.mutationFrequencies(NULL, muts));
    if (community.tick > {start})
        writeFile({json.dumps(str(out / 'trajectory.tsv'))},
            paste(c(community.tick, community.tick-{start}, freq), sep="\\t"), append=T);
'''
        if a.sweep:
            observer += '''
    levels = c(0.25, 0.50, 0.75, 1.0);
    names = c("f25", "f50", "f75", "f100");
    for (i in seqAlong(levels)) {
        if (freq >= levels[i]) factorial_snapshot(names[i], freq);
    }
    if (freq == 1.0 & isNULL(sim.getValue("fix_tick")))
        sim.setValue("fix_tick", community.tick);
    fixed = sim.getValue("fix_tick");
    if (!isNULL(fixed)) {
        if (community.tick == fixed + 250) factorial_snapshot("fixed_plus250", freq);
        if (community.tick == fixed + 1000) factorial_snapshot("fixed_plus1000", freq);
    }
'''
        else:
            observer += f'''
    if (community.tick == {start}+100) factorial_snapshot("control_t100", 0.0);
    if (community.tick == {start}+300) factorial_snapshot("control_t300", 0.0);
    if (community.tick == {start}+1000) factorial_snapshot("control_t1000", 0.0);
'''
    else:
        for t in [5, 10]:
            tick = 1 + t*a.Ne
            if tick < end:
                observer += f'    if (community.tick == {tick}) factorial_snapshot("burn{t}Ne", 0.0);\n'
    observer += '\n}\n'
    sentinel = 'function (void)end(void) {'
    if script.count(sentinel) != 1:
        raise ValueError('Unexpected end callback')
    script = script.replace(sentinel, observer + sentinel + '\n    factorial_observe();')
    if branch and a.sweep:
        draw = 'targets.addNewDrawnMutation(mut_type, pos);'
        if script.count(draw) != 1: raise ValueError('Unexpected draw callback')
        script = script.replace(draw, draw + '\n    factorial_observe();')
    if branch:
        # Late -> late restore is required for consistent SLiM/tskit times.
        # This block precedes the recurring observer; future scheduled events
        # were registered in 1 early() before the restore changes community.tick.
        script += f'''\n1 late() {{
    sim.readFromPopulationFile({json.dumps(str(a.parent.resolve()))});
    setSeed({seed});
}}\n'''
    script += '\n1: late() { factorial_observe(); }\n'
    config = {k:str(v.resolve()) if isinstance(v, Path) else v for k,v in vars(a).items()}
    config.update(region=region, seed=seed, start_tick=start, end_tick=end,
        source_sha256=digest(__file__), parent_sha256=digest(a.parent) if branch else None,
        conditioning='natural first upward passage; all introduction attempts retained',
        dfe=region['selection_model'] if a.bgs else None,
        focal_position0=focal, scaling_factor=1)
    return script, config


def run(a):
    a.out.mkdir(parents=True, exist_ok=False)
    script, config = generate(a)
    write_json(a.out / 'config.json', config)
    path = a.out / 'executed.slim'
    path.write_text(script)
    start = time.monotonic()
    with (a.out / 'slim.log').open('w') as log:
        subprocess.run([a.slim, '-s', str(config['seed']), '-d', 'verbosity=0', str(path)],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    raw = tskit.load(a.out / 'final.raw.trees')
    if raw.metadata['SLiM']['tick'] != config['end_tick']:
        raise ValueError('Incorrect final tick after restore')
    write_json(a.out / 'simulation_complete.json', dict(seconds=time.monotonic()-start,
        final_tick=raw.metadata['SLiM']['tick'], final_sha256=digest(a.out / 'final.raw.trees')))


def process_one(path, config, region, contig, exons, diploids):
    raw = pyslim.update(tskit.load(path))
    alive = pyslim.individuals_alive_at(raw, 0)
    if len(alive) != config['Ne']:
        raise ValueError(f'Wrong living population size: {len(alive)}')
    seed = seed_for('factorial-offline-sample', config['seed'], path.name, diploids)
    rng = np.random.default_rng(seed)
    inds = rng.choice(alive, diploids, replace=False)
    nodes = np.concatenate([raw.individual(int(i)).nodes for i in inds])
    if len(nodes) != diploids * 2 or np.any(raw.nodes_time[nodes] != 0):
        raise ValueError('Incorrect diploid samples or sample times')
    ts = raw.simplify(nodes, keep_input_roots=True)
    # Retain a pre-recapitation diagnostic: selected histories must become
    # insensitive to adding more forward burn-in before scientific inference.
    uncoalesced = sum(t.span for t in ts.trees() if t.num_roots > 1) / ts.sequence_length
    ts = pyslim.recapitate(ts, ancestral_Ne=config['Ne'],
        recombination_rate=contig.recombination_map, random_seed=seed_for(seed,'recap'))
    ts = ts.simplify()
    if any(t.num_roots != 1 for t in ts.trees()):
        raise ValueError('Incomplete recapitation')
    # Actual polymorphisms for inference: preserve the selected mutations,
    # overlay only the neutral fraction during the forward epoch, then the
    # full mutation rate in the older neutral recapitated ancestry.
    real = ts
    focal = config['focal_position0']
    breaks = np.unique(np.r_[0,region['length'],exons.ravel(),focal,focal+1])
    rate = np.full(len(breaks)-1,config['mu'])
    if config['bgs']:
        for lo,hi in exons: rate[(breaks[:-1]>=lo)&(breaks[1:]<=hi)] *= region['selection_model']['neutral_fraction']
    rate[(breaks[:-1]>=focal)&(breaks[1:]<=focal+1)] = 0
    for label,rates,start_time,end_time in [
        ('forward',msprime.RateMap(position=breaks,rate=rate),None,raw.metadata['SLiM']['tick']),
        ('ancestral',config['mu'],raw.metadata['SLiM']['tick'],None)]:
        next_id = 1+max((int(i) for m in real.mutations() for i in m.derived_state.split(',')),default=-1)
        real = msprime.sim_mutations(real,rate=rates,keep=True,
            start_time=start_time,end_time=end_time,
            model=msprime.SLiMMutationModel(type=99,next_id=next_id),
            random_seed=seed_for(seed,'real-variants',label))
    real_path = path.with_name(path.name.replace('.raw.trees','.variants.trees'))
    real.dump(real_path)
    # Independent neutral reporter process everywhere; it measures linked
    # effects, not directly selected polymorphism. Exon-masked summaries below
    # are the primary neutral-diversity assay.
    tables = ts.dump_tables(); tables.mutations.clear(); tables.sites.clear()
    ts = msprime.sim_mutations(tables.tree_sequence(), rate=config['mu'],
        random_seed=seed_for(seed,'neutral-reporters'), model=msprime.BinaryMutationModel())
    edges = np.arange(0, region['length']+1, 2000)
    if edges[-1] != region['length']: edges = np.r_[edges,region['length']]
    target = path.with_name(path.name.replace('.raw.trees', '.sample.trees'))
    ts.dump(target)
    accessible = np.diff(edges).astype(float)
    for lo,hi in exons:
        accessible -= np.maximum(0, np.minimum(edges[1:],hi)-np.maximum(edges[:-1],lo))
    masked = ts.delete_intervals(exons, simplify=False)
    pi_sum = masked.diversity(windows=edges,span_normalise=False)
    neutral_pi = np.divide(pi_sum,accessible,out=np.full(len(accessible),np.nan),where=accessible>0)
    np.savez_compressed(path.with_name(path.name.replace('.raw.trees','.npz')),
        edges=edges, tmrca=ts.diversity(windows=edges,mode='branch')/2,
        pi=ts.diversity(windows=edges), neutral_pi=neutral_pi, accessible_bp=accessible)
    return dict(stage=path.name.removesuffix('.raw.trees'), tick=raw.metadata['SLiM']['tick'],
        Ne=config['Ne'], sample_size=len(nodes), uncoalesced_span_fraction=uncoalesced,
        sample_sha256=digest(target), raw_sha256=digest(path),
        variants_sha256=digest(real_path), variants_sites=real.num_sites,
        mean_tmrca=ts.diversity(mode='branch')/2, mean_reporter_pi=ts.diversity())


def process(a):
    config = json.loads((a.run / 'config.json').read_text())
    region, arrays, contig, model, focal = setup(a.region, config['bgs'], config['Ne'], config['mu'])
    if region['maps_sha256'] != config['region']['maps_sha256']:
        raise ValueError('Processing map mismatch')
    results = [process_one(p,config,region,contig,arrays['selected_intervals'],a.diploids)
        for p in sorted(a.run.glob('*.raw.trees'))]
    if not results: raise ValueError('No raw snapshots')
    write_json(a.run / 'processed.json', dict(results=results, processor_sha256=digest(__file__)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['burnin','branch','process'])
    p.add_argument('--region',type=Path,required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--run',type=Path)
    p.add_argument('--parent',type=Path)
    p.add_argument('--bgs',action='store_true')
    p.add_argument('--sweep',action='store_true')
    p.add_argument('--replicate',type=int,default=0)
    p.add_argument('--Ne',type=int,default=10000)
    p.add_argument('--mu',type=float,default=1.29e-8)
    p.add_argument('--burn',type=float,default=10)
    p.add_argument('--duration',type=int,default=2500)
    p.add_argument('--s',type=float,default=.1)
    p.add_argument('--diploids',type=int,default=25)
    p.add_argument('--slim',default='slim')
    a=p.parse_args()
    if a.mode=='process': process(a)
    else: run(a)


if __name__=='__main__': main()
