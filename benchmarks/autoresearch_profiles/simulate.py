"""One replicate: conditional sweep ancestry, mutations, matched Gamma inference."""
import argparse
import fcntl
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from metrics import bin_piecewise, bin_snps, digest, write_json


def run(a):
    import msprime
    import gamma_smc_cu
    from gamma_smc_cu import _core
    start = time.monotonic()
    meta = json.loads((a.target / 'manifest.json').read_text())
    for name in ['target', 'map']:
        if digest(a.target / (name + '.npz')) != meta[name + '_sha256']:
            raise ValueError('Frozen input changed')
    z, r = np.load(a.target / 'target.npz'), np.load(a.target / 'map.npz')
    c = json.loads(a.config.read_text())
    n = int(meta['n_haplotypes'])
    rate = msprime.RateMap(position=r['position'], rate=r['rate'])
    model = []
    if c['model'] != 'neutral':
        if c['age'] > 0:
            model.append(msprime.StandardCoalescent(duration=c['age']))
        sweep_Ne = c['ancient_Ne'] if 'ancient_Ne' in c and c['age'] >= c['change_time'] else c['Ne']
        start_frequency = 1 / (2*sweep_Ne) if c['model'] == 'hard' else c['start_frequency']
        end_frequency = min(c['end_frequency'], 1 - 1/(2*sweep_Ne))
        model.append(msprime.SweepGenicSelection(position=1500000 + c['offset'],
            start_frequency=start_frequency, end_frequency=end_frequency, s=c['s'], dt=1e-6))
    model.append(msprime.StandardCoalescent())
    demography = msprime.Demography()
    demography.add_population(name='pop', initial_size=c['Ne'])
    if 'ancient_Ne' in c:
        demography.add_population_parameters_change(time=c['change_time'], initial_size=c['ancient_Ne'])
    ts = msprime.sim_ancestry(samples=n//2, ploidy=2, demography=demography,
        recombination_rate=rate, model=model, random_seed=a.seed)
    pairs = z['pairs'].astype(int)
    edges = z['edges'] - meta['left0']
    truth_values = []
    for tree in ts.trees():
        truth_values.append(np.mean([tree.tmrca(int(i), int(j)) for i, j in pairs]))
    truth = bin_piecewise(ts.breakpoints(as_array=True), truth_values, edges)
    mu = meta['calibration']['physical_mu']
    mutated = msprime.sim_mutations(ts, rate=mu, random_seed=(a.seed+104729) % (2**31-2)+1,
        model=msprime.BinaryMutationModel())
    G = mutated.genotype_matrix().T
    pos = mutated.tables.sites.position
    use = (G.max(axis=0) > G.min(axis=0))
    G, pos = np.ascontiguousarray(G[:, use], dtype=np.uint8), pos[use]
    spec = importlib.util.spec_from_file_location('frozen_gamma_api', a.api)
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    # Fix the same chromosome-calibrated HMM prior as the observations. Do not
    # learn a depressed regional theta or rescale time to improve curve fit.
    cal = meta['calibration']
    with a.gamma_lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        free, total = _core.cuda_mem_info()
        if free < 4*1024**3:
            raise RuntimeError('Less than 4 GiB free on assigned GPU')
        prediction = api.infer(G, pos, pairs=pairs.tolist(), Ne=cal['calibrated_Ne'],
            mu=mu, rho=cal['physical_rho'], physical_mu=mu, auto_estimate_theta=False,
            mean_only=True, flow_field_path=str(Path(gamma_smc_cu.__file__).parent / 'default_flow_field.txt'))
    if prediction['metadata']['time_units'] != 'generations':
        raise ValueError('Inference must return physical generations')
    curve = bin_snps(prediction['positions'], prediction['mean'].mean(axis=1), edges)
    a.out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(a.out / 'profile.npz', inferred=curve, truth=truth, edges=z['edges'])
    ts.dump(a.out / 'ancestry.trees')
    write_json(a.out / 'complete.json', dict(seed=a.seed, config=c, sites=len(pos),
        haplotypes=n, pairs=len(pairs), seconds=time.monotonic()-start,
        target_manifest_sha256=digest(a.target / 'manifest.json'), config_sha256=digest(a.config),
        source_sha256=digest(__file__), api_sha256=digest(a.api),
        profile_sha256=digest(a.out / 'profile.npz'), ancestry_sha256=digest(a.out / 'ancestry.trees'),
        calibration=prediction['metadata'], msprime_version=msprime.__version__,
        conditioning='Sweep trajectories conditioned on starting/ending frequencies; not establishment probability',
        selected_site='Selected mutation is not added to neutral mutation overlay'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['target', 'config', 'api', 'out', 'gamma-lock']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--seed', type=int, required=True)
    run(p.parse_args())
