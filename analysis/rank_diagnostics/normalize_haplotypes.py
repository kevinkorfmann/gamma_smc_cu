"""Genome-wide XP-EHH normalization and separate gene/flank summaries."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from common import POPS,sha,write_json,summaries

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--haplotypes',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pop',choices=POPS,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for c in range(1,23):
        folder=a.haplotypes/f'chr{c}_{a.pop}'
        r=json.loads((folder/'COMPLETE.json').read_text())
        assert r['pop']==a.pop and r['chrom']==c
        for name,h in r['outputs'].items():
            if sha(folder/name)!=h:raise ValueError(f'Changed input {folder/name}')
        records.append(r)
    n=sum(r['xp_finite'] for r in records)
    mu=sum(r['xp_sum'] for r in records)/n
    sd=np.sqrt(sum(r['xp_sumsq'] for r in records)/n-mu*mu)
    assert np.isfinite(sd) and sd>0
    atlas=pd.read_csv(a.input/'atlas.csv')
    result=[];observed_n=0;zsum=0.;zsumsq=0.
    for c in range(1,23):
        folder=a.haplotypes/f'chr{c}_{a.pop}'
        genes=atlas.loc[atlas.chr==c,['gene_id','gene_name','chr','start','end']]
        with np.load(folder/'xp.npz') as z:
            pos=z['pos'];score=(z['raw']-mu)/sd
        f=np.isfinite(score);observed_n+=int(f.sum());zsum+=score[f].sum();zsumsq+=np.square(score[f]).sum()
        d=summaries(genes,pos,score,'xp','positive')
        ihs=pd.read_csv(folder/'ihs.csv.gz').drop(columns=['gene_name','chr','start','end'])
        d=d.merge(ihs,on='gene_id',validate='one_to_one').merge(pd.read_csv(folder/'site_counts.csv.gz'),on='gene_id',validate='one_to_one')
        d['pop']=a.pop;d['reference']=records[0]['reference'];result.append(d)
        for fp in folder.glob('*_tracks.npz'):
            with np.load(fp) as z:arrays={k:z[k] for k in z.files}
            arrays['xp_z']=(arrays.pop('xp_raw')-mu)/sd
            np.savez_compressed(a.output/f'{fp.stem}_{a.pop}.npz',**arrays)
    assert observed_n==n and abs(zsum/n)<1e-10 and abs(zsumsq/n-1)<1e-9
    d=pd.concat(result,ignore_index=True)
    assert len(d)==19119 and not d.gene_id.duplicated().any()
    d.to_csv(a.output/f'{a.pop}.csv.gz',index=False)
    write_json(a.output/f'{a.pop}.json',dict(pop=a.pop,reference=records[0]['reference'],n_sites=n,
        raw_mean=mu,raw_sd=sd,z_mean=float(zsum/n),z_second_moment=float(zsumsq/n),
        original_ihs_rows=sum(r['ihs_original_rows'] for r in records),
        aligned_ihs_rows=sum(r['ihs_aligned_rows'] for r in records),
        chromosome_records=records,output_sha256=sha(a.output/f'{a.pop}.csv.gz'),source_sha256=sha(__file__)))
    print(f'{a.pop}/{records[0]["reference"]}: {n} finite XP-EHH sites; {int((d.ihs_body_n>0).sum())} genes with iHS, {int((d.xp_body_n>0).sum())} with XP-EHH',flush=True)

if __name__=='__main__':main()
