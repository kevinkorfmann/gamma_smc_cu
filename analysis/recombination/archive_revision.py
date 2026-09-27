#!/usr/bin/env python3
"""Archive exactly the final requested maps and focal checks into the revision."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--revision',type=Path,required=True);a=ap.parse_args()
    dest=a.revision/'data/recombination_2026';dest.mkdir(parents=True,exist_ok=True)
    for name in ['inputs','results','bgs','scripts','run_logs']:(dest/name).mkdir(exist_ok=True)
    (dest/'maps').mkdir(exist_ok=True)
    targets=pd.read_csv(a.root/'inputs/targets.tsv',sep='\t')
    expected=[]
    for t in targets.itertuples():
        for pop in t.populations.split(','):
            expected.append(f'{t.gene}.{pop}')
            if t.gene in ['GRK2','TREM2']:expected.extend(f'{t.gene}.{pop}_half{k}' for k in [1,2])
    for prefix in expected:
        for suffix in ['.json','.windows.tsv','.intervals.npz','.bed.gz']:
            src=a.root/'maps'/(prefix+suffix);assert src.exists(),src
            shutil.copy2(src,dest/'maps'/src.name)
    for dirname in ['inputs','results','bgs']:
        for src in (a.root/dirname).iterdir():
            if src.is_file():shutil.copy2(src,dest/dirname/src.name)
    for src in a.root.glob('*.py'):shutil.copy2(src,dest/'scripts'/src.name)
    shutil.copy2(a.root/'README.md',dest/'scripts/README.md')
    for pattern in ['complete.*.json','environment.*.json','control_update.complete.json']:
        for src in (a.root/'logs').glob(pattern):shutil.copy2(src,dest/'run_logs'/src.name)
    (dest/'.gitignore').write_text('maps/*.intervals.npz\nmaps/*.bed.gz\n__pycache__/\n')
    checksums={}
    for p in sorted(dest.rglob('*')):
        if p.is_file() and p.name!='archive_manifest.json':
            with p.open('rb') as f:checksums[str(p.relative_to(dest))]=hashlib.file_digest(f,'sha256').hexdigest()
    (dest/'archive_manifest.json').write_text(json.dumps({'primary_maps':len(expected)-8,'split_maps':8,
          'sha256':checksums},indent=2)+'\n')
    figures=a.revision/'figures/recombination_exploratory';figures.mkdir(exist_ok=True)
    for name in ['highlighted_gene_recombination_atlas.pdf','focal_recombination_comparison.pdf',
                 'focal_recombination_comparison.png','figure_legends.md']:
        shutil.copy2(a.root/'results'/name,figures/name)
    print(f'Archived {len(expected)} maps in {dest}')

if __name__=='__main__':main()
