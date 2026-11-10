"""Independent confirmation or paired 10Ne-to-20Ne extension; immutable inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

from common import digest, write_json


def decision(summary, model, mode, extension=None):
    if not summary['calibration_complete']:
        raise ValueError('Calibration must finish before follow-up')
    row = next(r for r in summary['results'] if r['model'] == model)
    ratio = row['effects']['focal100kb_tmrca']['ratio']
    if mode == 'extension':
        if row['burn_gate_passed'] or not .10 <= ratio <= .65:
            raise ValueError('Extension requires an in-range model failing the burn-in gate')
        return dict(model=model, burn_Ne=20, purpose='Paired extension of existing calibration founders')
    candidates = summary['candidates_for_confirmation']
    if extension is not None:
        if extension['model'] != model or not extension['eligible_for_confirmation']:
            raise ValueError('Extended model has not passed the prespecified strength and burn-in gates')
        return dict(model=model, burn_Ne=20, target=.2, synthetic=row['settings']['synthetic'])
    match = next((r for r in candidates if r['model'] == model), None)
    if match is None or not row['burn_gate_passed']:
        raise ValueError('Model not selected by the frozen BGS-only rule')
    return dict(match, burn_Ne=10)


def build_tasks(mode, model, bank):
    if mode == 'extension':
        return [(i, bank / model / f'family_{i:03d}') for i in range(6)]
    return [(i, bank / model / f'family_{i:04d}') for i in range(1000, 1030)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['extension', 'confirmation'], required=True)
    p.add_argument('--model', required=True)
    for name in ['calibration', 'summary', 'baseline', 'bank']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--extension-summary', type=Path)
    p.add_argument('--slim', required=True)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--gamma-lock', type=Path, required=True)
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    if not 1 <= a.workers <= 4:
        raise ValueError('One to four worker slots only; check other banks before launch')
    a.bank.mkdir(parents=True, exist_ok=True)
    lock = (a.bank / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    study = Path(__file__).resolve().parent
    cal = json.loads((a.calibration / 'manifest.json').read_text())
    summary = json.loads(a.summary.read_text())
    complete = json.loads((a.calibration / 'completion.json').read_text())
    assert len(complete['results']) == 36 and all(r['status'] == 'completed' for r in complete['results'])
    assert summary['bank_manifest_sha256'] == digest(a.calibration / 'manifest.json')
    assert digest(a.baseline / 'manifest.json') == cal['baseline_manifest_sha256']
    baseline = json.loads((a.baseline / 'manifest.json').read_text())
    for name in ['stronger_bgs.py', 'common.py']:
        assert digest(study / name) == cal['sources'][name], 'Frozen simulator changed'
    assert digest(study / 'infer_factorial_gamma.py') == baseline['sources']['infer_factorial_gamma.py']
    ext = json.loads(a.extension_summary.read_text()) if a.extension_summary else None
    if ext is not None:
        assert ext['calibration_summary_sha256'] == digest(a.summary)
    selected = decision(summary, a.model, a.mode, ext)
    model = next(m for m in cal['models'] if m['name'] == a.model)
    region = Path(model['region'])
    cfg = json.loads((region / 'region.json').read_text())
    assert cfg['selection_model'] == model['model']
    assert digest(region / 'maps.npz') == cfg['maps_sha256']
    source_names = ['run_stronger_bgs_followup.py', 'stronger_bgs.py', 'common.py',
                    'infer_factorial_gamma.py', 'summarize_stronger_bgs_extension.py', 'summarize_stronger_bgs.py']
    manifest = dict(mode=a.mode, model=model, decision=selected,
        calibration=str(a.calibration.resolve()), calibration_manifest_sha256=digest(a.calibration/'manifest.json'),
        calibration_summary_sha256=digest(a.summary), baseline=str(a.baseline.resolve()),
        baseline_manifest_sha256=digest(a.baseline/'manifest.json'),
        extension_summary_sha256=digest(a.extension_summary) if a.extension_summary else None,
        sources={name: digest(study/name) for name in source_names},
        Ne=10000, mu=1.29e-8, Q=1, diploids=25, s=.1, h=.5, duration=2500,
        families=6 if a.mode=='extension' else 30,
        founder_indices=list(range(6)) if a.mode=='extension' else list(range(1000,1030)),
        sweep_attempts_per_family=0 if a.mode=='extension' else 12,
        gamma=a.mode=='confirmation', gamma_crops_on_branches=True,
        gamma_lock=str(a.gamma_lock.resolve()), uncertainty_unit='founder family',
        resource_limit=dict(max_workers=4, slim_threads=1, concurrent_gpu_calls=1))
    if a.mode == 'extension':
        manifest['parent_sha256'] = {str(i): digest(a.calibration/a.model/f'family_{i:03d}/founder/final.raw.trees') for i in range(6)}
    mp = a.bank/'manifest.json'
    if mp.exists():
        assert json.loads(mp.read_text()) == manifest, 'Immutable follow-up manifest mismatch'
    else:
        write_json(mp, manifest)
    if a.plan_only:
        print(json.dumps(manifest)); return
    a.gamma_lock.parent.mkdir(parents=True, exist_ok=True)
    def command(args, log):
        with log.open('a') as f:
            subprocess.run(args, stdout=f, stderr=subprocess.STDOUT, check=True)
    def simulate(out, rep, parent=None, sweep=False, duration=2500):
        cmd = [sys.executable, str(study/'stronger_bgs.py'), 'branch' if parent else 'burnin',
            '--region', str(region), '--out', str(out), '--bgs', '--replicate', str(rep),
            '--slim', a.slim, '--burn', str(selected['burn_Ne']), '--Ne', '10000', '--mu', '1.29e-8', '--s', '.1']
        if parent:
            cmd += ['--parent', str(parent), '--duration', str(duration)]
        if sweep:
            cmd += ['--sweep']
        if not (out/'simulation_complete.json').exists():
            if out.exists():
                raise ValueError(f'Incomplete run preserved for explicit repair: {out}')
            command(cmd, out.parent/(out.name+'.driver.log'))
        if not (out/'processed.json').exists():
            command([sys.executable, str(study/'stronger_bgs.py'), 'process',
                '--region', str(region), '--run', str(out), '--diploids', '25'], out/'processing.log')
        if a.mode == 'confirmation':
            cmd = [sys.executable, str(study/'infer_factorial_gamma.py'),
                '--run', str(out), '--lock', str(a.gamma_lock)]
            if parent: cmd += ['--crops']
            command(cmd, out/'gamma.log')
    def family(task):
        i, folder = task; folder.mkdir(parents=True, exist_ok=True)
        status = folder/'family_status.json'
        if status.exists(): return json.loads(status.read_text())
        start = time.monotonic()
        try:
            if a.mode == 'extension':
                old = a.calibration/a.model/f'family_{i:03d}/founder'
                old_cfg = json.loads((old/'config.json').read_text())
                assert old_cfg['end_tick'] == 100001 and not old_cfg['sweep']
                founder = folder/'extension'
                simulate(founder, i, old/'final.raw.trees', duration=100000)
            else:
                founder = folder/'founder'; simulate(founder, i)
            simulate(folder/'control', 0, founder/'final.raw.trees')
            if a.mode == 'confirmation':
                for j in range(12):
                    simulate(folder/f'sweep_{j:03d}', j, founder/'final.raw.trees', sweep=True)
            result = dict(model=a.model, family=i, status='completed', seconds=time.monotonic()-start)
        except Exception as exc:
            result = dict(model=a.model, family=i, status='failed', error=repr(exc), seconds=time.monotonic()-start)
        write_json(status, result); return result
    results=[]; start=time.monotonic()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(family,t) for t in build_tasks(a.mode,a.model,a.bank)]):
            results.append(f.result()); write_json(a.bank/'progress.json',results)
            print(json.dumps(dict(done=len(results), total=manifest['families'], **results[-1])),flush=True)
    write_json(a.bank/'completion.json',dict(results=results,seconds=time.monotonic()-start))
    if any(r['status']!='completed' for r in results): raise SystemExit(1)
    if a.mode=='extension':
        subprocess.run([sys.executable,str(study/'summarize_stronger_bgs_extension.py'),
            '--bank',str(a.bank),'--summary',str(a.summary),'--out',str(a.bank/'analysis')],check=True)


if __name__ == '__main__':
    main()
