"""Read-only Gamma inference on all completed BGS calibration controls."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
from common import digest,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
    a.out.mkdir(parents=True,exist_ok=True)
    scheduler=(a.out/'scheduler.lock').open('a');fcntl.flock(scheduler,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cal=root/'runs/stronger_bgs_v1/calibration_v1'
    summary=root/'results/stronger_bgs_v1/calibration/summary.json'
    m=json.loads((cal/'manifest.json').read_text());s=json.loads(summary.read_text())
    assert s['calibration_complete'] and s['bank_manifest_sha256']==digest(cal/'manifest.json')
    inference=root/'study/stronger_bgs_followup_v1/infer_factorial_gamma.py'
    old=json.loads((root/'runs/factorial_frequency_v1/production_10Mb/manifest.json').read_text())
    assert digest(inference)==old['sources']['infer_factorial_gamma.py']
    jobs=[]
    for model in m['models']:
        for i in range(6):
            fam=cal/model['name']/f'family_{i:03d}'
            assert json.loads((fam/'family_status.json').read_text())['status']=='completed'
            run=fam/'control';pr=json.loads((run/'processed.json').read_text())
            row=next(r for r in pr['results'] if r['stage']=='final')
            assert digest(run/'final.variants.trees')==row['variants_sha256']
            jobs.append(dict(model=model['name'],family=i,source=str(run),tree_sha256=row['variants_sha256']))
    manifest=dict(calibration_summary_sha256=digest(summary),calibration_manifest_sha256=digest(cal/'manifest.json'),
        inference_sha256=digest(inference),source_sha256=digest(__file__),jobs=jobs,
        pairs=1225,Ne=10000,full_10Mb=True,crops=False,
        model_selection_already_frozen=True,new_forward_simulations=0,
        gamma_lock=str(root/'runs/stronger_bgs_v1/gamma.lock'))
    mp=a.out/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest
    else:write_json(mp,manifest)
    results=[]
    for job in jobs:
        d=a.out/job['model']/f"family_{job['family']:03d}";d.mkdir(parents=True,exist_ok=True)
        source=Path(job['source']);cfg=d/'config.json'
        if cfg.exists():assert cfg.read_bytes()==(source/'config.json').read_bytes()
        else:cfg.write_bytes((source/'config.json').read_bytes())
        link=d/'final.variants.trees'
        if link.exists():assert link.resolve()==(source/'final.variants.trees').resolve()
        else:link.symlink_to(source/'final.variants.trees')
        start=time.monotonic()
        with (d/'inference.log').open('a') as log:
            subprocess.run(['bash',str(root/'study/run_python.sh'),str(inference),
                '--run',str(d.resolve()),'--lock',manifest['gamma_lock']],stdout=log,stderr=subprocess.STDOUT,check=True)
        metrics=json.loads((d/'gamma/final/metrics.json').read_text())
        assert metrics['tree_sha256']==job['tree_sha256'] and metrics['source_sha256']==manifest['inference_sha256']
        assert metrics['pairs']==1225 and metrics['genotype_shape'][0]==50
        results.append(dict(model=job['model'],family=job['family'],seconds=time.monotonic()-start,status='completed'))
        write_json(a.out/'progress.json',results)
        print(json.dumps(dict(done=len(results),total=len(jobs),**results[-1])),flush=True)
    write_json(a.out/'completion.json',dict(results=results,complete=True))


if __name__=='__main__':main()
