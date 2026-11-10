"""Bounded autoresearch-style propose/evaluate/keep loop, with an immutable metric.

The proposals are seeded parameter search, not an LLM inside the simulator.
The supervising agent can assess this completed campaign and launch a new one.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from metrics import digest, loss, scores, seed_for, write_json


def propose(index, rng, best, baseline_Ne, key, Q, burn):
    base=dict(Q=Q,burn_Ne=burn)
    if index==0:return dict(model='neutral',Ne=baseline_Ne,**base)
    if index==1:
        return (dict(model='neutral',Ne=3341.698377687549,**base) if key=='GRK2_GIH'
                else dict(model='neutral',Ne=3000,ancient_Ne=25000,change_time=3000,**base))
    width=500000 if key=='GRK2_GIH' else 200000
    default=dict(model='negative',Ne=16000,s=-.001,h=.5,target='block',width=width,
                 offset=-50000,selected_fraction=.7,**base)
    initial=[dict(default,target='annotated',Ne=baseline_Ne),
             dict(default,target='annotated',Ne=baseline_Ne,s=-.01),
             default,dict(default,Ne=25000),dict(default,Ne=25000,s=-.005),
             dict(default,s=-.0003,selected_fraction=1.),
             dict(default,s=-.003,selected_fraction=1.),
             dict(default,Ne=8000,selected_fraction=.5)]
    if index-2<len(initial):return initial[index-2]
    if index%5==0 or best is None:
        return dict(default,Ne=int(rng.integers(4000,35001)//Q*Q),
            s=-float(np.exp(rng.uniform(np.log(.0001),np.log(.02)))),
            width=int(rng.integers(100000,700001)//10000*10000),
            offset=int(rng.integers(-150000,100001)//10000*10000),
            selected_fraction=float(rng.choice([.2,.4,.7,1.])))
    c=dict(best)
    field=str(rng.choice(['Ne','s','width','offset','selected_fraction']))
    if field=='Ne':c[field]=int(np.clip(c[field]*np.exp(rng.normal(0,.35)),3000,40000)//Q*Q)
    elif field=='s':c[field]=-float(np.clip(abs(c[field])*np.exp(rng.normal(0,.6)),.0001,.03))
    elif field=='width':c.update(target='block',width=int(np.clip(c['width']*np.exp(rng.normal(0,.35)),60000,800000)//1000*1000))
    elif field=='offset':c.update(target='block',offset=int(np.clip(c['offset']+rng.normal(0,35000),-200000,150000)))
    else:c[field]=float(np.clip(c[field]+rng.normal(0,.2),.1,1.))
    return c


def run_replicates(a, key, folder, config, seeds):
    folder.mkdir(parents=True, exist_ok=True)
    cp = folder / 'config.json'
    if cp.exists() and json.loads(cp.read_text()) != config:
        raise ValueError('Attempt to change an existing candidate')
    write_json(cp, config)
    def one(item):
        i, seed = item
        out = folder / f'rep_{i:03d}'
        done = out / 'complete.json'
        if not done.exists():
            if out.exists():
                raise ValueError('Incomplete replicate preserved; use a new campaign to retry')
            cmd = [sys.executable, str(Path(__file__).with_name('simulate.py')),
                '--target', str(a.targets/key), '--annotation', str(a.annotations/key), '--slim', str(a.slim), '--config', str(cp), '--seed', str(seed),
                '--api', str(a.api), '--gamma-lock', str(a.gamma_lock), '--out', str(out)]
            with (folder / f'rep_{i:03d}.log').open('w') as f:
                process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    status = process.wait(timeout=a.timeout)
                    if status: raise subprocess.CalledProcessError(status, cmd)
                except subprocess.TimeoutExpired:
                    import os, signal
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
                    raise
        meta = json.loads(done.read_text())
        if (meta['seed'] != seed or meta['config'] != config
                or meta['profile_sha256'] != digest(out/'profile.npz')
                or meta['api_sha256'] != digest(a.api)
                or meta['source_sha256'] != digest(Path(__file__).with_name('simulate.py'))):
            raise ValueError('Replicate provenance mismatch')
        z = np.load(out / 'profile.npz')
        return z['inferred'], z['truth'], z['reporter']
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = list(pool.map(one, enumerate(seeds)))
    return tuple(np.array([r[j] for r in results]) for j in range(3))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['targets', 'annotations', 'slim', 'out', 'api', 'gamma-lock']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--keys', nargs='+', default=['GRK2_GIH', 'TREM2_IBS'])
    p.add_argument('--candidates', type=int, default=20)
    p.add_argument('--replicates', type=int, default=3)
    p.add_argument('--validation-replicates', type=int, default=24)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--workers', type=int, default=1, help='Independent CPU simulations; GPU inference remains locked')
    p.add_argument('--seed', type=int, default=20260927)
    p.add_argument('--fit-bins', choices=['train', 'all_bins'], default='all_bins',
                   help='All-bin fitting reserves only independent simulation seeds for validation')
    p.add_argument('--initial-configs', type=Path,
                   help='Optional frozen JSON mapping target keys to prior candidate configurations')
    p.add_argument('--Q', type=int, default=10)
    p.add_argument('--burn-Ne', type=int, default=10)
    a = p.parse_args()
    if a.candidates < 1 or a.replicates < 2 or a.validation_replicates < 2 or not 1 <= a.workers <= 4:
        raise ValueError('At least two independent replicates per stage required')
    a.out.mkdir(parents=True, exist_ok=True)
    lock = (a.out / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import gamma_smc_cu
    from gamma_smc_cu import _core
    sources = {f.name:digest(f) for f in Path(__file__).parent.glob('*.py')}
    initial = json.loads(a.initial_configs.read_text()) if a.initial_configs else {}
    manifest = dict(settings={k:str(v.resolve()) if isinstance(v, Path) else v for k,v in vars(a).items()},
        initial_configurations=initial,
        initial_sha256=digest(a.initial_configs) if a.initial_configs else None,
        annotations={k:digest(a.annotations/k/'manifest.json') for k in a.keys},
        slim_sha256=digest(a.slim), sources=sources, api_sha256=digest(a.api), core_sha256=digest(_core.__file__),
        flow_sha256=digest(Path(gamma_smc_cu.__file__).parent/'default_flow_field.txt'),
        targets={k:digest(a.targets/k/'manifest.json') for k in a.keys},
        training_seeds={k:[seed_for('train', a.seed, k, r) for r in range(a.replicates)] for k in a.keys},
        validation_seeds={k:[seed_for('validation', a.seed, k, r) for r in range(a.validation_replicates)] for k in a.keys},
        selection=f"Training random seeds select candidates using {a.fit_bins}; all-bin fitting has no spatial holdout",
        purpose='Negative-only SLiM profile optimization; scaling is approximate; synthetic blocks are not human annotations')
    mp = a.out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text()) != manifest:
        raise ValueError('Frozen campaign changed; create a new output directory')
    write_json(mp, manifest)
    all_results = {}
    for key in a.keys:
        target = dict(np.load(a.targets/key/'target.npz'))
        tm = json.loads((a.targets/key/'manifest.json').read_text())
        rng = np.random.default_rng(seed_for('proposal', a.seed, key))
        baseline_Ne = tm['calibration']['calibrated_Ne']
        best, best_loss, best_index = None, float('inf'), None
        rows = []; root = a.out/key; root.mkdir(exist_ok=True)
        for i in range(a.candidates):
            start = time.monotonic()
            if 1 <= i <= len(initial.get(key, [])):
                c = dict(initial[key][i-1])
            else:
                c = propose(i, rng, best, baseline_Ne, key, a.Q, a.burn_Ne)
            folder = root/f'candidate_{i:04d}'
            # A saved outcome includes discarded/crashed candidates too, so a
            # restart reproduces the proposal stream without retrying failures.
            rp = folder/'result.json'
            if rp.exists():
                row = json.loads(rp.read_text())
                if row['config'] != c:
                    raise ValueError('Replay differs from frozen history')
            else:
                try:
                    values, truth, reporter = run_replicates(a, key, folder, c, manifest['training_seeds'][key])
                    train_loss = loss(values.mean(axis=0), target['observed'], target[a.fit_bins], target['focal'])
                    np.savez_compressed(folder/'ensemble.npz', inferred=values, truth=truth, reporter=reporter)
                    row = dict(candidate=i, config=c, train_loss=train_loss,
                        status=('control' if c['model']=='neutral' else 'keep' if train_loss < best_loss else 'discard'), seconds=time.monotonic()-start)
                except (subprocess.SubprocessError, ValueError, RuntimeError) as exc:
                    row = dict(candidate=i, config=c, train_loss=None, status='crash',
                               error=str(exc), seconds=time.monotonic()-start)
                folder.mkdir(exist_ok=True)
                write_json(rp, row)
            if row['status'] == 'keep':
                best, best_loss, best_index = c, row['train_loss'], i
            rows.append(row)
            write_json(root/'history.json', rows)
            (root/'results.tsv').write_text('candidate\ttrain_loss\tstatus\tdescription\n' + ''.join(
                f"{r['candidate']}\t{r['train_loss']}\t{r['status']}\t{json.dumps(r['config'], sort_keys=True)}\n" for r in rows))
            write_json(a.out/'progress.json', dict(key=key, completed_candidates=i+1,
                total_candidates=a.candidates, best_index=best_index, best_loss=None if best_index is None else best_loss))
            print(json.dumps(dict(key=key, **row)), flush=True)
        if best_index is None:
            raise RuntimeError(f'No successful simulations for {key}; inspect replicate logs')
        # Freeze choice BEFORE evaluating any validation seed or held-out block.
        chosen = dict(index=best_index, config=best, train_loss=best_loss)
        write_json(root/'chosen.json', chosen)
        neutral_rows = [r for r in rows if r['config']['model'] == 'neutral' and r['train_loss'] is not None]
        best_neutral = min(neutral_rows, key=lambda r:r['train_loss'])['config'] if neutral_rows else None
        validation = {}
        comparisons = [('neutral', {'model':'neutral', 'Ne':baseline_Ne, 'Q':a.Q, 'burn_Ne':a.burn_Ne}), ('best', best)]
        if best_neutral is not None:
            comparisons.append(('tuned_neutral', best_neutral))
        for name, c in comparisons:
            vals, truth, reporter = run_replicates(a, key, root/'validation'/name, c, manifest['validation_seeds'][key])
            np.savez_compressed(root/f'validation_{name}.npz', inferred=vals, truth=truth, reporter=reporter)
            validation[name] = scores(vals.mean(axis=0), target)
            if name=='best':
                validation['neutral_reporter']=scores(reporter.mean(axis=0),target)
                validation['genealogical_truth']=scores(truth.mean(axis=0),target)
        result = dict(chosen=chosen, best_neutral=best_neutral, validation=validation,
            candidates=a.candidates, successful=sum(r['status']!='crash' for r in rows),
            training_replicates=a.replicates, validation_replicates=a.validation_replicates,
            fractional_validation_improvement=1-validation['best']['all_bins']/validation['neutral']['all_bins'])
        write_json(root/'summary.json', result); all_results[key]=result
    write_json(a.out/'completion.json', dict(complete=True, results=all_results))


if __name__ == '__main__':
    main()
