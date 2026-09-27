"""Bounded family scheduler for the matched-map sweep-frequency experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import subprocess
import sys
import time
from common import digest,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--region',type=Path,required=True);p.add_argument('--slim',required=True)
    p.add_argument('--families',type=int,default=30);p.add_argument('--branches',type=int,default=12)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--gamma',action='store_true')
    a=p.parse_args();a.bank.mkdir(parents=True,exist_ok=True)
    study=Path(__file__).resolve().parent
    sources={s:digest(study/s) for s in ['factorial_frequency.py','infer_factorial_gamma.py','common.py']}
    manifest=dict(families_per_background=a.families,sweep_attempts_per_family=a.branches,
        controls_per_family=1,region=json.loads((a.region/'region.json').read_text()),
        Ne=10000,mu=1.29e-8,s=.1,h=.5,duration=2500,Q=1,
        bgs_burn=10,neutral_burn=.2,sources=sources,
        uncertainty_unit='founder family',gamma=a.gamma)
    mp=a.bank/'manifest.json'
    if mp.exists():
        if json.loads(mp.read_text())!=manifest:raise ValueError('Immutable manifest mismatch')
    else:write_json(mp,manifest)
    def command(cmd,log):
        with log.open('a') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    def simulate(out,mode,bgs,rep,parent=None,sweep=False):
        cmd=[sys.executable,str(study/'factorial_frequency.py'),mode,'--region',str(a.region),
             '--out',str(out),'--replicate',str(rep),'--slim',a.slim]
        if bgs:cmd+=['--bgs']
        if sweep:cmd+=['--sweep']
        if parent:cmd+=['--parent',str(parent),'--duration','2500']
        else:cmd+=['--burn','10' if bgs else '.2']
        if not (out/'simulation_complete.json').exists():command(cmd,out.parent/(out.name+'.driver.log'))
        if not (out/'processed.json').exists():
            command([sys.executable,str(study/'factorial_frequency.py'),'process',
                '--region',str(a.region),'--run',str(out)],out/'processing.log')
        if a.gamma:
            cmd=[sys.executable,str(study/'infer_factorial_gamma.py'),'--run',str(out),'--lock',str(a.bank/'gamma.lock')]
            # Full versus 6-/3-Mb context sensitivity for every control and
            # every actually reached sweep stage, not only favourable examples.
            if parent:cmd+=['--crops']
            command(cmd,out/'gamma.log')
    def family(bgs,i):
        start=time.monotonic();folder=a.bank/f'{"bgs" if bgs else "neutral"}_family_{i:03d}'
        folder.mkdir(exist_ok=True)
        try:
            founder=folder/'founder';simulate(founder,'burnin',bgs,i)
            simulate(folder/'control','branch',bgs,0,founder/'final.raw.trees')
            for j in range(a.branches):
                simulate(folder/f'sweep_{j:03d}','branch',bgs,j,founder/'final.raw.trees',True)
            result=dict(family=folder.name,status='completed',seconds=time.monotonic()-start)
        except Exception as exc:
            result=dict(family=folder.name,status='failed',error=repr(exc),seconds=time.monotonic()-start)
        write_json(folder/'family_status.json',result)
        return result
    tasks=[(bgs,i) for i in range(a.families) for bgs in [True,False]]
    start=time.monotonic();results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(family,*task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result());write_json(a.bank/'progress.json',results)
            print(json.dumps(dict(done=len(results),total=len(tasks),**results[-1])),flush=True)
    write_json(a.bank/'completion.json',dict(results=results,seconds=time.monotonic()-start))
    if any(r['status']=='failed' for r in results):raise SystemExit(1)


if __name__=='__main__':main()
