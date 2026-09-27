"""Fixed-map annotation extension and explicitly synthetic BGS stress grid."""
import argparse
import gzip
import json
from pathlib import Path
import numpy as np
from common import digest, write_json


def merge(intervals, length):
    result=[]
    for lo,hi in sorted((max(0,int(a)),min(length,int(b))) for a,b in intervals):
        if hi<=lo:continue
        if result and lo<=result[-1][1]:result[-1][1]=max(result[-1][1],hi)
        else:result.append([lo,hi])
    return np.asarray(result,dtype=np.int64).reshape(-1,2)


def dilate_to_fraction(intervals,length,fraction):
    # Synthetic stress target: deterministic dilation of observed annotations,
    # not a claim that these added bases are functional in humans.
    target=int(length*fraction)
    def expanded(pad):return merge([(a-pad,b+pad) for a,b in intervals],length)
    if np.diff(intervals,axis=1).sum()>=target:return intervals,0
    lo,hi=0,length
    while lo<hi:
        mid=(lo+hi)//2
        if np.diff(expanded(mid),axis=1).sum()>=target:hi=mid
        else:lo=mid+1
    return expanded(lo),lo


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--phastcons',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    region=json.loads((a.base/'region.json').read_text());base=np.load(a.base/'maps.npz')
    assert digest(a.base/'maps.npz')==region['maps_sha256']
    length=region['length'];intervals=[]
    with gzip.open(a.phastcons,'rt') as f:
        for line in f:
            row=line.split()
            if row[1]!='chr'+region['chrom']:continue
            lo,hi=int(row[2]),int(row[3])
            if lo<region['right'] and hi>region['left']:intervals.append([lo-region['left'],hi-region['left']])
    phast=merge(intervals,length);assert len(phast)
    conserved=merge(np.vstack([base['exons'],phast]),length)
    targets={'annotated':(conserved,0)}
    for fraction in [.15,.30]:targets[f'density{int(fraction*100)}']=dilate_to_fraction(conserved,length,fraction)
    models=[('annotated_gamma','annotated','Gamma_K17',None,.3,'annotation-based extension, DFE assumption uncalibrated'),
            ('annotated_s1e3','annotated','fixed',-.001,0.,'annotation-based extension, all target mutations deleterious'),
            ('density15_s1e3','density15','fixed',-.001,0.,'synthetic selected-target-density stress test'),
            ('density30_s1e3','density30','fixed',-.001,0.,'synthetic selected-target-density stress test'),
            ('density30_s1e4','density30','fixed',-.0001,0.,'synthetic selected-target-density and DFE stress test'),
            ('density30_s1e2','density30','fixed',-.01,0.,'synthetic selected-target-density and DFE stress test')]
    manifest=[]
    for name,target,dfe,s,neutral,note in models:
        selected,pad=targets[target];out=a.out/name;out.mkdir()
        model=dict(id=name,dfe=dfe,s=s,h=.5,neutral_fraction=neutral,interpretation=note,
            selected_bp=int(np.diff(selected,axis=1).sum()),target_fraction=float(np.diff(selected,axis=1).sum()/length),
            annotation_dilation_bp=pad,synthetic=target!='annotated')
        np.savez_compressed(out/'maps.npz',map_position=base['map_position'],map_rate=base['map_rate'],
            exons=base['exons'],phastcons=phast,selected_intervals=selected)
        r=dict(region,base_maps_sha256=region['maps_sha256'],maps_sha256=digest(out/'maps.npz'),
            selection_model=model,annotation_id='Havana exons union UCSC hg38 phastConsElements100way',
            phastcons_url='https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/phastConsElements100way.txt.gz',
            phastcons_sha256=digest(a.phastcons),phastcons_bp=int(np.diff(phast,axis=1).sum()),
            selected_target_bp=model['selected_bp'],source_sha256=digest(__file__))
        write_json(out/'region.json',r)
        z=np.load(out/'maps.npz')
        assert np.array_equal(z['map_position'],base['map_position']) and np.array_equal(z['map_rate'],base['map_rate'])
        manifest.append(dict(name=name,region=str(out.resolve()),model=model))
    write_json(a.out/'models.json',dict(models=manifest,recombination_map_identical=True,
        selection_grid_chosen_without_sweep_outcomes=True,source_sha256=digest(__file__)))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
