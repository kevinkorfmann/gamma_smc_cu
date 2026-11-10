"""Execute one hash-checked production unit from its frozen manifest."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

def run(a):
 m=json.loads((a.root/'manifest.json').read_text());source=a.root/'source'
 for name,digest in m['source_sha256'].items():
  assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
 t=m['tasks'][a.index]
 if a.mode in ['simulate','both']:
  cmd=[sys.executable,str(source/'simulate.py'),'--out',str(a.root/'simulations'/t['id']),
       '--split',t['split'],'--regime',t['regime'],'--replicate',str(t['replicate']),'--slim',str(a.slim)]
 else:cmd=[sys.executable,str(source/'score.py'),'--input',str(a.root/'simulations'/t['id']),'--out',str(a.root/'scores'/t['id']),'--backend','cpu']
 subprocess.run(cmd,check=True)
 if a.mode=='both':subprocess.run([sys.executable,str(source/'score.py'),'--input',str(a.root/'simulations'/t['id']),'--out',str(a.root/'scores'/t['id']),'--backend','cpu'],check=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int,required=True);p.add_argument('--mode',choices=['simulate','score','both'],required=True);p.add_argument('--slim',type=Path);run(p.parse_args())
