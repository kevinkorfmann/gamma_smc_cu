"""Frozen-winner fresh-seed checks of SLiM scaling and forward burn-in."""
import argparse,fcntl,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from metrics import digest,write_json,seed_for,scores,loss
from search import run_replicates


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['campaign','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--replicates',type=int,default=4)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    lock=(a.out/'scheduler.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    base=json.loads((a.campaign/'manifest.json').read_text());settings=base['settings']
    if not (a.campaign/'completion.json').exists():raise ValueError('Main campaign must finish before sensitivity checks')
    tasks=SimpleNamespace(targets=Path(settings['targets']),annotations=Path(settings['annotations']),
        slim=Path(settings['slim']),api=Path(settings['api']),gamma_lock=Path(settings['gamma_lock']),
        timeout=7200,workers=2)
    spec=dict(campaign_manifest_sha256=digest(a.campaign/'manifest.json'),replicates=a.replicates,
        sources={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        changes=[dict(name='Q5',Q=5,burn_Ne=10),dict(name='burn20',Q=10,burn_Ne=20)],
        purpose='Sensitivity of a frozen winner, not a second optimization or a proof of Q1 equivalence')
    mp=a.out/'manifest.json'
    if mp.exists() and json.loads(mp.read_text())!=spec:raise ValueError('Frozen sensitivity design changed')
    write_json(mp,spec);report={}
    for key in settings['keys']:
        summary=json.loads((a.campaign/key/'summary.json').read_text())
        config=summary['chosen']['config'];target=dict(np.load(tasks.targets/key/'target.npz'))
        baseline=np.load(a.campaign/key/'validation_best.npz')['inferred']
        report[key]={}
        for arm in spec['changes']:
            cfg=dict(config,Q=arm['Q'],burn_Ne=arm['burn_Ne'])
            seeds=[seed_for('negative-sensitivity',settings['seed'],key,i) for i in range(a.replicates)]
            folder=a.out/key/arm['name']
            inferred,truth,reporter=run_replicates(tasks,key,folder,cfg,seeds)
            np.savez_compressed(folder/'ensemble.npz',inferred=inferred,truth=truth,reporter=reporter)
            raw=[json.loads((folder/f'rep_{i:03d}'/'complete.json').read_text()) for i in range(a.replicates)]
            stats=dict(config=cfg,replicates=a.replicates,seeds=seeds,
                error=scores(inferred.mean(0),target)['all_bins'],
                profile_change=loss(inferred.mean(0),baseline.mean(0),target['all_bins'],target['focal']),
                neutral_reporter_error=scores(reporter.mean(0),target)['all_bins'],
                true_tmrca_error=scores(truth.mean(0),target)['all_bins'],
                max_uncoalesced_fraction=max(r['raw_diagnostics']['uncoalesced_fraction'] for r in raw))
            report[key][arm['name']]=stats;write_json(a.out/'progress.json',report)
            print(json.dumps(dict(key=key,arm=arm['name'],**stats)),flush=True)
    write_json(a.out/'completion.json',dict(complete=True,results=report))

if __name__=='__main__':main()
