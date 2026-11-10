"""Freeze exon/conserved annotation for the existing immutable 3-Mb targets."""
import argparse,gzip,json
from pathlib import Path
import numpy as np
import stdpopsim
from metrics import digest,write_json


def merge(intervals,length):
    result=[]
    for lo,hi in sorted((max(0,int(a)),min(length,int(b))) for a,b in intervals):
        if hi<=lo:continue
        if result and lo<=result[-1][1]:result[-1][1]=max(result[-1][1],hi)
        else:result.append([lo,hi])
    return np.asarray(result,dtype=np.int64).reshape(-1,2)


def main():
    p=argparse.ArgumentParser();p.add_argument('--targets',type=Path,required=True)
    p.add_argument('--phastcons',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    annot=stdpopsim.get_species('HomSap').get_annotations('ensembl_havana_104_exons')
    for key in ['GRK2_GIH','TREM2_IBS']:
        meta=json.loads((a.targets/key/'manifest.json').read_text());left,right=meta['left0'],meta['right0'];length=right-left
        exons=annot.get_chromosome_annotations(meta['chromosome'])
        exons=merge(exons[(exons[:,1]>left)&(exons[:,0]<right)]-left,length)
        conserved=[]
        with gzip.open(a.phastcons,'rt') as f:
            for line in f:
                r=line.split()
                if r[1]=='chr'+meta['chromosome'] and int(r[2])<right and int(r[3])>left:
                    conserved.append([int(r[2])-left,int(r[3])-left])
        union=merge(list(exons)+conserved,length)
        out=a.out/key;out.mkdir();np.savez_compressed(out/'annotation.npz',exons=exons,selected_intervals=union)
        write_json(out/'manifest.json',dict(target_manifest_sha256=digest(a.targets/key/'manifest.json'),
            annotation='Ensembl Havana 104 exons union UCSC hg38 phastConsElements100way',
            selected_bp=int(np.diff(union,axis=1).sum()),length=length,
            coordinate_convention='zero-based half-open, relative to 3-Mb target left edge',
            phastcons_sha256=digest(a.phastcons),annotation_sha256=digest(out/'annotation.npz'),
            source_sha256=digest(__file__)))
        print(key,len(union),int(np.diff(union,axis=1).sum()),flush=True)

if __name__=='__main__':main()
