"""Create an immutable production manifest without looking at score outcomes."""
import argparse,hashlib,json
from pathlib import Path

def seed_for(*p):return int(hashlib.sha256('|'.join(map(str,p)).encode()).hexdigest()[:12],16)%(2**31-2)+1

def run(a):
    a.out.mkdir(parents=True,exist_ok=False);tasks=[]
    for split,regimes,n in [('normalisation',['neutral'],200),('calibration',['neutral'],200),('evaluation',['neutral'],200),('evaluation',['incomplete','near_fixed','completed','older_completed'],100)]:
        for regime in regimes:
            for rep in range(n):
                tasks.append(dict(id=f'{split}_{regime}_{rep:04d}',split=split,regime=regime,replicate=rep,
                    seed=seed_for('detection-localisation-v1',split,regime,rep)))
    assert len(tasks)==len({t['seed'] for t in tasks})==1000
    source=Path(__file__).parent
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.suffix in ['.py','.md','.sh']}
    (a.out/'manifest.json').write_text(json.dumps(dict(protocol_id='detection-localisation-v1',date='2026-09-27',
        calibration_replicates=200,evaluation_neutral_replicates=200,evaluation_selected_replicates=100,
        source_sha256=hashes,pilot_excluded=True,tasks=tasks),indent=2)+'\n')
    print(json.dumps({'replicates':len(tasks),'manifest':str(a.out/'manifest.json')}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args())
