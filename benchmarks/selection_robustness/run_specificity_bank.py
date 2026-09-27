"""Execute an immutable controlled simulation manifest with bounded workers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--region',type=Path,required=True);p.add_argument('--slim',required=True)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    jobs=json.loads((a.bank/'manifest.json').read_text())['jobs']
    def run(job):
        out=a.bank/f"{job['regime']}_{job['replicate']:04d}"
        if (out/'qc.json').exists():return dict(directory=str(out),status='already_completed')
        cmd=[sys.executable,str(Path(__file__).with_name('simulate_specificity.py')),
             '--region',str(a.region),'--out',str(out),'--slim',a.slim]
        for key,value in job.items():cmd += ['--'+key,str(value)]
        with (a.bank/(out.name+'.log')).open('x') as log:
            ret=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        return dict(directory=str(out),returncode=ret.returncode,status='completed' if ret.returncode==0 else 'failed')
    start=time.monotonic();results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        for future in as_completed([executor.submit(run,j) for j in jobs]):
            result=future.result();results.append(result)
            print(json.dumps(dict(done=len(results),total=len(jobs),seconds=time.monotonic()-start,**result)),flush=True)
            (a.bank/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
    (a.bank/'completion.json').write_text(json.dumps(dict(results=results,seconds=time.monotonic()-start),indent=2)+'\n')
    if any(x['status']=='failed' for x in results):raise SystemExit(1)


if __name__=='__main__':main()
