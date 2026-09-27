#!/usr/bin/env python3
"""Extract published Buffalo–Kern B' predictions without importing bgspy.

The trusted Dryad SHA-256 is checked before a restricted pickle reader loads
NumPy arrays and inert bgspy records. No bgspy model code is executed.
"""
import argparse
import builtins
import collections
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_SHA='f0f4ec46b170780f17b2d85873463e7e0afb97d9eb0f53c530ed748e7e66bf21'
SUMMARY_SHA='8faef5cce203c1d652b9cc182b1ae44fd8784c6e45717451df2d2d6dd6e6fd85'
AUTHOR_COMMIT='348f00bf017287be2cd6d9be135d8cacfcd0e26a'

def sha(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

class Record:
    def __new__(cls,*args,**kwargs):return object.__new__(cls)
    def __setstate__(self,state):
        self.__dict__.update(state if isinstance(state,dict) else {'state':state})

def placeholder(*args,**kwargs):
    obj=Record();obj.args=args;return obj

class PublishedReader(pickle.Unpickler):
    def find_class(self,module,name):
        if (module,name) in {('bgspy.pipeline','ModelDir'),('bgspy.likelihood','SimplexModel'),
                            ('bgspy.data','GenomicBinnedData'),('bgspy.optim','OptimResult')}:
            return Record
        if module in ['numpy.core.multiarray','numpy._core.multiarray'] and name in ['_reconstruct','scalar']:
            core=getattr(np,'_core',None) or np.core
            return getattr(core.multiarray,name)
        if module=='numpy' and name in ['ndarray','dtype']:return getattr(np,name)
        if module=='collections' and name=='defaultdict':return collections.defaultdict
        if module=='builtins' and name in ['set','frozenset','slice','dict']:return getattr(builtins,name)
        # DataFrames in unused substitution-rate outputs are kept as inert records.
        if (module,name) in {('pandas.core.frame','DataFrame'),('pandas.core.internals.managers','BlockManager'),
                            ('pandas.core.indexes.base','Index'),('pandas.core.indexes.range','RangeIndex'),
                            ('pandas.core.indexes.numeric','Int64Index')}:
            return Record
        if (module,name) in {('pandas._libs.internals','_unpickle_block'),('pandas.core.indexes.base','_new_Index')}:
            return placeholder
        raise pickle.UnpicklingError(f'Unapproved pickle global: {module}.{name}')

def predictions(model):
    """Author predict_simplex, with pi0=1: exp(-mu * sum(W * logB_fit))."""
    theta=model.theta_;coeff=model.logB_fit
    fixed=model._fixed_mu
    mu=theta[1] if fixed is None else fixed
    weights=theta[2:] if fixed is None else theta[1:]
    assert coeff.shape[1]==1
    weights=weights.reshape(coeff.shape[2:])
    values=np.exp(-mu*np.einsum('ijk,jk->i',coeff[:,0,:,:],weights))
    # Independent loop as in bgspy.likelihood.predict_simplex at AUTHOR_COMMIT.
    loop=np.zeros(coeff.shape[0])
    for j in range(coeff.shape[2]):
        for k in range(coeff.shape[3]):loop += -mu*weights[j,k]*coeff[:,0,j,k]
    assert np.allclose(values,np.exp(loop),rtol=1e-14,atol=1e-15)
    rows=[]
    for chrom,edges in model.bins.bins_.items():
        mask=model.bins.masks_[chrom]
        assert len(mask)==len(edges)-1
        rows.extend((chrom,int(s),int(e)) for s,e,keep in zip(edges[:-1],edges[1:],mask) if keep)
    assert len(rows)==len(values),(len(rows),len(values))
    assert np.all(np.isfinite(values)) and np.all((values>0)&(values<=1))
    frame=pd.DataFrame(rows,columns=['chrom','start','end']);frame['Bprime']=values
    return frame

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--summary',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--populations',nargs='+',choices=['YRI','CEU','CHB'],default=['YRI','CEU'])
    a=ap.parse_args();assert sha(a.model)==MODEL_SHA;assert sha(a.summary)==SUMMARY_SHA
    with a.model.open('rb') as f:obj=PublishedReader(f).load()
    print('Available fits:',list(obj.fits),flush=True)
    reference=pd.read_csv(a.summary,sep='\t')
    checks={}
    for pop in ['yri','ceu','chb']:
        large=predictions(obj.fits[(pop,'1000000','sparse')]['mbp'])
        merged=large.merge(reference,on=['chrom','start','end'],validate='one_to_one')
        assert len(merged)>2000
        error=float(np.max(np.abs(merged.Bprime-merged['B_'+pop])))
        checks[pop]={'n_model_bins':len(large),'n_shared_bins':len(merged),'max_absolute_error':error}
    a.output.mkdir(parents=True,exist_ok=True)
    population_maps={}
    output_hashes={}
    for population in dict.fromkeys(a.populations):
        fit_key=(population.lower(),'100000','sparse')
        fine=predictions(obj.fits[fit_key]['mbp'])
        dest=a.output/f'buffalo_kern2024_{population}_CADD6_100kb.tsv.gz'
        fine.to_csv(dest,sep='\t',index=False,compression={'method':'gzip','mtime':0})
        population_maps[population]={'fit_key':list(fit_key),'n_bins':len(fine),'file':dest.name}
        output_hashes[dest.name]=sha(dest)
    meta={'paper_doi':'10.1371/journal.pgen.1011144','dataset_doi':'10.5061/dryad.qnk98sfnv',
          'dryad_version':275846,'source_model':a.model.name,'source_model_sha256':MODEL_SHA,
          'released_summary_sha256':SUMMARY_SHA,'author_code_commit':AUTHOR_COMMIT,
          'assembly':'GRCh38','coordinates':'0-based half-open','population_maps':population_maps,
          'annotation':'CADD top 6%','recombination_map':'deCODE','bin_width_bp':100000,
          'model':'Bprime, initial fit (not rescaled)','populations':list(population_maps),
          'interpretation':'Retained neutral diversity fraction; lower Bprime means stronger predicted BGS.',
          'missing_bins':'Author masks retained; no interpolation or gap filling.',
          'validation':'Vectorized predictions agree with author predict_simplex summation at rtol=1e-14.',
          'comparison_to_released_1Mb_summary':checks,
          'summary_difference':'The authors revisions.ipynb exports cadd6__decode; the deposited fitted object is cadd6__decode__altgrid. Their values differ; the TSV is not used as an equality oracle or as the plotted map.',
          'output_sha256':output_hashes}
    (a.output/'buffalo_kern2024_provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))

if __name__=='__main__':main()
