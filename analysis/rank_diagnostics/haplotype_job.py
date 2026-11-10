"""One full chromosome/population XP-EHH and archived-iHS summary task."""
import argparse
from pathlib import Path
import os
import time
import numpy as np
import pandas as pd
import allel
from common import POPS, sha, write_json, normalize_ihs, summaries

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--ihs-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--chr',type=int,required=True)
    p.add_argument('--pop',choices=POPS,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    out=a.output/f'chr{a.chr}_{a.pop}'
    if (out/'COMPLETE.json').exists() and a.resume:return
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    started=time.monotonic()
    genes=pd.read_csv(a.input_root/'atlas.csv')
    genes=genes.loc[genes.chr==a.chr,['gene_id','gene_name','chr','start','end']]
    raw=a.ihs_root/f'chr{a.chr}_{a.pop}'/'ihs.ihs.out'
    ihs=pd.read_csv(raw,sep='\t')
    required={'pos','freq','ihs'}
    if not required.issubset(ihs):raise ValueError(f'Unexpected iHS schema: {ihs.columns}')
    ihs=ihs.sort_values('pos')
    z=normalize_ihs(ihs)
    cache=a.cache/f'chr{a.chr}'
    corrected_pos=np.load(cache/'positions.npy',mmap_mode='r')
    # Historical inputs include duplicated positions/other filtered variants.
    # Retain the historical normalization, then require unique corrected cores.
    raw_ihs_rows=len(ihs)
    unique=~ihs.pos.duplicated(keep=False)
    aligned=np.isin(ihs.pos.to_numpy(),corrected_pos)&np.asarray(unique)
    ihs=ihs.loc[aligned].copy();z=z[aligned]
    summaries(genes,ihs.pos.to_numpy(),z,'ihs','abs').to_csv(out/'ihs.csv.gz',index=False)
    sample_ids=np.load(cache/'sample_ids.npy',allow_pickle=True).astype(str)
    members=pd.read_csv(a.samples,sep=r'\s+').set_index('SampleID')
    labels=members.loc[sample_ids,'Population'].to_numpy()
    reference='CEU' if a.pop=='YRI' else 'YRI'
    def indices(pop):
        s=np.flatnonzero(labels==pop)
        return (s[:,None]*2+np.array([0,1])).reshape(-1)
    focal,ref=indices(a.pop),indices(reference)
    assert len(focal)>0 and len(ref)>0 and not np.intersect1d(focal,ref).size
    G=np.load(cache/'G.npy',mmap_mode='r')
    pos=np.load(cache/'positions.npy',mmap_mode='r')
    # Read only focal/reference rows; union-polymorphic includes fixed differences.
    f=np.asarray(G[focal,:]);r=np.asarray(G[ref,:])
    ac=f.sum(axis=0,dtype=np.int32)+r.sum(axis=0,dtype=np.int32)
    keep=(ac>0)&(ac<len(focal)+len(ref))
    pos=np.array(pos[keep],dtype=np.int64)
    f=np.ascontiguousarray(f[:,keep].T);r=np.ascontiguousarray(r[:,keep].T)
    print(f'XP-EHH chr{a.chr} {a.pop}/{reference}: {len(pos)} sites, {len(focal)}/{len(ref)} haplotypes',flush=True)
    xp=allel.xpehh(f,r,pos,min_ehh=.05,include_edges=False,
                  gap_scale=20000,max_gap=200000,use_threads=True)
    finite=np.isfinite(xp)
    np.savez_compressed(out/'xp.npz',pos=pos,raw=xp)
    # Preserve eligible and union SNP counts to distinguish missing-score causes.
    focal_count=f.sum(axis=1,dtype=np.int32)
    focal_seg=(focal_count>0)&(focal_count<len(focal))
    focal_common=(focal_count/len(focal)>=.05)&(focal_count/len(focal)<=.95)
    count=genes[['gene_id']].copy()
    for name,mask in [('corrected_polymorphic',focal_seg),('corrected_maf05',focal_common)]:
        sp=pos[mask]
        count[name]=np.searchsorted(sp,genes.end,'right')-np.searchsorted(sp,genes.start,'left')
    count.to_csv(out/'site_counts.csv.gz',index=False)
    # Retain named regional tracks for the same final genome-wide normalization.
    for gene in ['GRK2','TREM2','TREML1']:
        g=genes.loc[genes.gene_name==gene]
        if len(g)==1:
            g=g.iloc[0];lo,hi=int(g.start)-500000,int(g.end)+500000
            si=ihs.pos.between(lo,hi);sx=(pos>=lo)&(pos<=hi)
            np.savez_compressed(out/f'{gene}_tracks.npz',ihs_pos=ihs.loc[si,'pos'].to_numpy(),
                ihs_z=z[np.asarray(si)],xp_pos=pos[sx],xp_raw=xp[sx])
    record=dict(pop=a.pop,reference=reference,chrom=a.chr,elapsed_seconds=time.monotonic()-started,
        focal_haplotypes=len(focal),reference_haplotypes=len(ref),union_sites=len(pos),
        xp_finite=int(finite.sum()),xp_sum=float(xp[finite].sum()),xp_sumsq=float(np.square(xp[finite]).sum()),
        ihs_original_rows=raw_ihs_rows,ihs_aligned_rows=len(ihs),
        ihs_raw_sha256=sha(raw),cache_manifest_sha256=sha(cache/'READY.json'),
        samples_sha256=sha(a.samples),source_sha256=sha(__file__),common_sha256=sha(Path(__file__).with_name('common.py')),
        scikit_allel=allel.__version__,numpy=np.__version__,slurm_job=os.environ.get('SLURM_JOB_ID'))
    record['outputs']={q.name:sha(q) for q in out.iterdir() if q.is_file()}
    write_json(out/'COMPLETE.json',record)
    print(f'Completed in {record["elapsed_seconds"]:.1f} s',flush=True)

if __name__=='__main__':main()
