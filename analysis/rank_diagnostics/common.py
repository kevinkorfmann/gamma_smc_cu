"""Small, explicit numerical/coordinate helpers for rank diagnostics."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

POPS = ['ACB','ASW','LWK','GWD','MSL','ESN','YRI','CEU','TSI','FIN','GBR','IBS',
        'CHB','JPT','CHS','CDX','KHV','GIH','PJL','BEB','STU','ITU','MXL','PUR','CLM','PEL']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''): h.update(block)
    return h.hexdigest()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False,
        default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')

def normalize_ihs(df):
    """Match the manuscript's chromosome/population, 20-bin convention."""
    bins=pd.cut(df.freq, np.linspace(0,1,21), include_lowest=True, labels=False)
    out=np.full(len(df),np.nan)
    for b in range(20):
        keep=(bins==b)&np.isfinite(df.ihs)
        v=df.loc[keep,'ihs'].to_numpy()
        if len(v)>=50 and v.std(ddof=0)>0:
            out[np.asarray(keep)]=(v-v.mean())/v.std(ddof=0)
    return out

def interval_summary(pos, values, starts, ends, mode='abs', threshold=2.):
    """One-based closed query intervals, preserving finite denominator."""
    pos=np.asarray(pos);values=np.asarray(values)
    if np.any(np.diff(pos)<0): raise ValueError('Unsorted positions')
    rows=[]
    for lo,hi in zip(starts,ends):
        i,j=np.searchsorted(pos,[lo,hi],side='left')
        j=np.searchsorted(pos,hi,side='right')
        v=values[i:j];v=v[np.isfinite(v)]
        x=np.abs(v) if mode=='abs' else v
        rows.append((j-i,len(x),float(np.max(x)) if len(x) else np.nan,
                     float(np.sum(x>threshold)),float(np.sum(x)),float(np.sum(x*x))))
    return np.asarray(rows,float)

def combine_summaries(a,b):
    out=a+b
    out[:,2]=np.fmax(a[:,2],b[:,2])
    return out

def summaries(genes, pos, values, prefix, mode, threshold=2.):
    result=genes[['gene_id','gene_name','chr','start','end']].copy()
    for region,flank in [('body',0),('flank100',100000),('flank500',500000)]:
        if not flank:
            z=interval_summary(pos,values,genes.start,genes.end,mode,threshold)
        else:
            z=combine_summaries(
                interval_summary(pos,values,np.maximum(1,genes.start-flank),genes.start-1,mode,threshold),
                interval_summary(pos,values,genes.end+1,genes.end+flank,mode,threshold))
        for k,col in enumerate(['returned','n','max','extreme_count','sum','sumsq']):
            result[f'{prefix}_{region}_{col}']=z[:,k]
        result[f'{prefix}_{region}_fraction']=np.divide(z[:,3],z[:,1],out=np.full(len(z),np.nan),where=z[:,1]>0)
    return result

def overlap_mean(starts,ends,values,lo,hi):
    """Map and query coordinates zero-based half-open; never fill gaps."""
    i=np.searchsorted(ends,lo,side='right');j=np.searchsorted(starts,hi,side='left')
    w=np.maximum(0,np.minimum(ends[i:j],hi)-np.maximum(starts[i:j],lo))
    keep=np.isfinite(values[i:j]);w=w[keep];v=values[i:j][keep]
    covered=w.sum()
    return (float(np.dot(w,v)/covered) if covered else np.nan,float(covered/(hi-lo)))
