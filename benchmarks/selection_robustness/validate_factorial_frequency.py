"""Same-seed replay without observers, plus natural-stage and restore checks."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import tskit

from common import digest, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--slim',required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True)
    config=json.loads((a.run/'config.json').read_text())
    text=(a.run/'executed.slim').read_text()
    if 'factorial_observe();' not in text:raise ValueError('No observer to validate')
    text=text.replace('factorial_observe();','')
    text=text.replace(str(a.run.resolve()),str(a.out.resolve()))
    path=a.out/'no_observers.slim';path.write_text(text)
    with (a.out/'replay.log').open('w') as log:
        subprocess.run([a.slim,'-s',str(config['seed']),'-d','verbosity=0',str(path)],
            stdout=log,stderr=subprocess.STDOUT,check=True)
    x=tskit.load(a.run/'final.raw.trees');y=tskit.load(a.out/'final.raw.trees')
    checks={name:getattr(x.tables,name).equals(getattr(y.tables,name))
        for name in ['nodes','edges','sites','mutations','individuals','populations','migrations']}
    if not all(checks.values()):raise ValueError(f'Observer altered tables: {checks}')
    stages={}
    for line in (a.run/'snapshots.tsv').read_text().splitlines():
        name,tick,freq=line.split('\t');stages[name]=(int(tick),float(freq))
    for name,target in [('f25',.25),('f50',.5),('f75',.75),('f100',1.)]:
        tick,freq=stages[name]
        assert freq>=target
        raw=tskit.load(a.run/f'{name}.raw.trees')
        assert raw.metadata['SLiM']['tick']==tick
        assert raw.metadata['SLiM']['stage']=='late'
        assert raw.metadata['SLiM']['tick']>config['start_tick']
    assert stages['fixed_plus250'][0]-stages['f100'][0]==250
    assert stages['fixed_plus1000'][0]-stages['f100'][0]==1000
    assert x.metadata['SLiM']['tick']==config['end_tick']
    trajectory=np.loadtxt(a.run/'trajectory.tsv')
    intro=trajectory[trajectory[:,0]==config['start_tick']+1,2]
    assert np.isclose(intro.max(),1/(2*config['Ne']))
    for name,target in [('f25',.25),('f50',.5),('f75',.75),('f100',1.)]:
        first=trajectory[trajectory[:,2]>=target,0].min()
        assert first==stages[name][0]
    write_json(a.out/'validation.json',dict(biological_tables_exactly_equal=checks,
        natural_first_passage=True, single_copy_introduction=True,
        late_stage_restore=True, final_tick=config['end_tick'], stages=stages,
        simulator_sha256=config['source_sha256'], validator_sha256=digest(__file__),
        fixture_note='Ne=100, s=0.9 technical validation only; excluded from biological estimates'))


if __name__=='__main__':main()
