"""Bounded, restart-safe BGS-only screen; never selects on sweep outcomes."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from common import digest,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--regions',type=Path,required=True)
    p.add_argument('--bank',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--slim',required=True)
    p.add_argument('--families',type=int,default=6);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();a.bank.mkdir(parents=True,exist_ok=True);study=Path(__file__).resolve().parent
    lock=(a.bank/'scheduler.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    models=json.loads((a.regions/'models.json').read_text())['models']
    manifest=dict(stage='BGS-only calibration; not confirmation',models=models,
        sources={n:digest(study/n) for n in ['stronger_bgs.py','common.py','run_stronger_bgs_calibration.py','summarize_stronger_bgs.py']},
        families_per_model=a.families,Ne=10000,mu=1.29e-8,Q=1,burn=10,
        baseline=str(a.baseline.resolve()),baseline_manifest_sha256=digest(a.baseline/'manifest.json'),
        controls_per_family=1,control_duration=2500,sweep_introductions=0,
        selection_rule='Choose closest distinct models to retained focal TMRCA 0.5 within [0.35,0.65] and 0.2 within [0.10,0.35], using BGS-only means; require burn-in gate; confirm independently',
        burn_gate='10Ne/5Ne mean ratio within 10% genome and 20% focal; mean uncoalesced span <0.05',
        resources=dict(workers=a.workers,slim_threads=1,gpu=False))
    mp=a.bank/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Immutable calibration manifest mismatch'
    else:write_json(mp,manifest)
    def run_command(cmd,log):
        with log.open('a') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    def family(model,i):
        start=time.monotonic();folder=a.bank/model['name']/f'family_{i:03d}'
        folder.mkdir(parents=True,exist_ok=True)
        status=folder/'family_status.json'
        if status.exists():return json.loads(status.read_text())
        try:
            for kind in ['founder','control']:
                out=folder/kind
                cmd=[sys.executable,str(study/'stronger_bgs.py'),'burnin' if kind=='founder' else 'branch',
                    '--region',model['region'],'--out',str(out),'--bgs','--replicate',str(i),
                    '--slim',a.slim,'--burn','10']
                if kind=='control':cmd+=['--parent',str(folder/'founder/final.raw.trees'),'--duration','2500']
                if not (out/'simulation_complete.json').exists():run_command(cmd,folder/(kind+'.driver.log'))
                if not (out/'processed.json').exists():run_command([sys.executable,str(study/'stronger_bgs.py'),
                    'process','--region',model['region'],'--run',str(out)],folder/(kind+'.processing.log'))
            r=dict(model=model['name'],family=i,status='completed',seconds=time.monotonic()-start)
        except Exception as exc:r=dict(model=model['name'],family=i,status='failed',error=repr(exc),seconds=time.monotonic()-start)
        write_json(status,r);return r
    tasks=[(m,i) for i in range(a.families) for m in models];results=[];start=time.monotonic()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(family,*task) for task in tasks]
        for f in as_completed(futures):
            results.append(f.result());write_json(a.bank/'progress.json',results)
            print(json.dumps(dict(done=len(results),total=len(tasks),**results[-1])),flush=True)
    write_json(a.bank/'completion.json',dict(results=results,seconds=time.monotonic()-start))
    if any(r['status']!='completed' for r in results):raise SystemExit(1)
    subprocess.run([sys.executable,str(study/'summarize_stronger_bgs.py'),'--bank',str(a.bank),
        '--baseline',str(a.baseline),'--out',str(a.out)],check=True)


if __name__=='__main__':main()
