"""Replay SLiM with/without observational logging and compare biological tables."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess

import tskit

from common import digest, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--script',type=Path,required=True)
    p.add_argument('--slim',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(exist_ok=False,parents=True);script=a.script.read_text()
    def run(logging):
        name='logging' if logging else 'no_logging'
        dest=(a.out/(name+'.trees')).resolve()
        s,n=re.subn(r'defineConstant\("trees_file", "[^"]+"\);',
                    f'defineConstant("trees_file", {json.dumps(str(dest))});',script)
        if n!=1:raise ValueError('Unexpected tree output definition')
        s,n=re.subn(r'writeFile\("[^"]+/trajectory.tsv",',
                    f'writeFile({json.dumps(str((a.out/"trajectory.tsv").resolve()))},',s)
        if n!=1:raise ValueError('Unexpected trajectory writer')
        if not logging:s=s.replace('record_focal();','')
        path=a.out/(name+'.slim');path.write_text(s)
        with (a.out/(name+'.log')).open('w') as f:
            subprocess.run([a.slim,'-s','938164','-d','verbosity=0',str(path)],stdout=f,stderr=subprocess.STDOUT,check=True)
        return tskit.load(dest)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a_ts,b_ts=list(pool.map(run,[True,False]))
    checks={}
    for table in ['nodes','edges','sites','mutations','individuals','populations','migrations']:
        checks[table]=getattr(a_ts.tables,table).equals(getattr(b_ts.tables,table))
    if not all(checks.values()):raise ValueError(f'Logger changed biological tables: {checks}')
    write_json(a.out/'validation.json',dict(tables_exactly_equal=checks,seed=938164,
        script_sha256=digest(a.script),validator_sha256=digest(__file__),
        note='Provenance command and output paths differ by design; every biological table matches exactly.'))


if __name__=='__main__':main()
