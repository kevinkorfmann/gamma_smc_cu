"""Annotate the corrected atlas and quantify genomic covariate dependence."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.spatial import cKDTree
from common import POPS,sha,write_json,overlap_mean

COVS=['length_bp','n_sites','recombination','bgs']
FOCAL=[('GRK2','GIH'),('TREM2','IBS'),('TREML1','IBS'),('IFIH1','IBS'),
       ('CCDC92','CDX'),('CLEC6A','CDX'),('SLC6A15','CHS'),('BPIFA2','GIH'),
       ('LCT','CEU'),('SLC24A5','TSI'),('EDAR','CHB'),('ABCC11','CHB')]

def correlation_matrix(values):
    z=np.column_stack([rankdata(v,method='average') for v in np.asarray(values).T])
    cor=np.corrcoef(z.T)
    precision=np.linalg.inv(cor)
    partial=-precision[0,1:]/np.sqrt(precision[0,0]*np.diag(precision)[1:])
    return cor[0,1:],partial

def annotate(atlas,root):
    result=atlas[['gene_id','gene_name','chr','start','end','is_sd']].copy()
    result['length_bp']=result.end-result.start+1
    for label,prefix in [('recombination','DeCodeSexAveraged_GRCh38'),('pyrho','PyrhoYRI_GRCh38')]:
        result[label]=np.nan;result[label+'_coverage']=0.
        for c,g in result.groupby('chr'):
            d=pd.read_csv(root/prefix/f'chr{c}.txt',sep=r'\s+')
            x=d.iloc[:,1].to_numpy(dtype=np.int64);v=d.iloc[:-1,2].to_numpy(dtype=float)
            if np.any(np.diff(x)<=0):raise ValueError(f'Non-increasing {prefix} chr{c}')
            v[~np.isfinite(v)|(v<0)]=np.nan
            vals=[overlap_mean(x[:-1],x[1:],v,int(r.start)-1,int(r.end)) for r in g.itertuples()]
            result.loc[g.index,[label,label+'_coverage']]=vals
    for pop in ['YRI','CEU']:
        d=pd.read_csv(root/f'buffalo_{pop}.tsv.gz',sep='\t')
        col='bgs' if pop=='YRI' else 'bgs_ceu'
        result[col]=np.nan;result[col+'_coverage']=0.
        for c,g in result.groupby('chr'):
            dd=d.loc[d.chrom==f'chr{c}'].sort_values('start')
            starts,ends=dd.start.to_numpy(),dd.end.to_numpy();v=dd.Bprime.to_numpy()
            assert np.all(ends>starts) and np.all(ends[:-1]<=starts[1:])
            vals=[overlap_mean(starts,ends,v,int(r.start)-1,int(r.end)) for r in g.itertuples()]
            result.loc[g.index,[col,col+'_coverage']]=vals
    b=pd.read_csv(root/'barroso.csv.gz')
    check=result.merge(b[['gene_id','chr','start','end']],on='gene_id',validate='one_to_one',suffixes=('','_b'))
    assert all(np.array_equal(check[c],check[c+'_b']) for c in ['chr','start','end'])
    result=result.merge(b[['gene_id','b_regulatory_mean','b_phastcons_mean']],on='gene_id',validate='one_to_one')
    return result

def percentile_features(df):
    return np.column_stack([rankdata(df[c],method='average')/len(df) for c in COVS])

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((a.input/'SOURCES.json').read_text())
    for name,record in manifest.items():
        if 'sha256' in record and sha(a.input/name)!=record['sha256']:raise ValueError(f'Input changed: {name}')
    atlas=pd.read_csv(a.input/'atlas.csv');counts=pd.read_csv(a.input/'site_counts.csv.gz')
    assert len(atlas)==19119 and len(counts)==19119*26 and not counts.duplicated(['gene_id','pop']).any()
    ann=annotate(atlas,a.input);ann.to_csv(a.output/'gene_covariates.csv.gz',index=False)
    assembled=[];corrs=[];bins=[];focal=[];coverage=[];sensitivity=[]
    maxerr=0.
    for pop in POPS:
        d=ann.merge(counts.loc[counts['pop']==pop],on=['gene_id','chr'],validate='one_to_one')
        d=d.merge(atlas[['gene_id',f'{pop}_rank',f'{pop}_tmrca']],on='gene_id',validate='one_to_one').rename(columns={f'{pop}_rank':'rank',f'{pop}_tmrca':'tmrca'})
        reconstructed=np.exp(np.divide(d.log_sum,d.pairs,out=np.full(len(d),np.nan),where=d.pairs>0))
        finite=np.isfinite(d.tmrca)
        assert np.array_equal(finite,(d.n_sites>=2)&(d.pairs>0))
        assert np.allclose(reconstructed[finite],d.loc[finite,'tmrca'],rtol=1e-10,atol=1e-8)
        maxerr=max(maxerr,float(np.max(np.abs(reconstructed[finite]/d.loc[finite,'tmrca']-1))))
        valid=~d.is_sd&np.isfinite(d['rank'])
        d['eligible']=valid&(d.recombination_coverage>=.95)&(d.bgs_coverage>=.95)&np.isfinite(d[COVS]).all(axis=1)
        coverage.append(dict(pop=pop,atlas_genes=len(d),non_sd_finite=int(valid.sum()),
            complete_covariates=int(d.eligible.sum()),recombination_covered=int((valid&(d.recombination_coverage>=.95)).sum()),
            bgs_covered=int((valid&(d.bgs_coverage>=.95)).sum())))
        g=d.loc[d.eligible].copy();z=g[['rank']+COVS].to_numpy();chrom=g.chr.to_numpy()
        raw,partial=correlation_matrix(z)
        replicates=np.array([correlation_matrix(z[chrom!=c]) for c in range(1,23)])
        se=np.sqrt(21/22*np.sum((replicates-replicates.mean(axis=0))**2,axis=0))
        for j,cov in enumerate(COVS):
            for k,(kind,vals) in enumerate([('marginal',raw),('partial',partial)]):
                corrs.append(dict(pop=pop,covariate=cov,kind=kind,n=len(g),rho=vals[j],
                    ci_low=max(-1,vals[j]-1.96*se[k,j]),ci_high=min(1,vals[j]+1.96*se[k,j])))
            q=pd.qcut(g[cov].rank(method='average'),5,labels=False,duplicates='drop')
            for b,gg in g.groupby(q):
                bins.append(dict(pop=pop,covariate=cov,quintile=int(b)+1,n=len(gg),median_rank=gg['rank'].median(),
                    low1_count=int((gg['rank']<.01).sum()),low1_fraction=float((gg['rank']<.01).mean())))
        # Independent reference-map/model sensitivity, with its actual coverage.
        for rec,bgs,bcover in [('pyrho','bgs','bgs_coverage'),('recombination','bgs_ceu','bgs_ceu_coverage'),
                               ('recombination','b_regulatory_mean',None),('recombination','b_phastcons_mean',None)]:
            keep=valid&(d[rec+'_coverage']>=.95)&np.isfinite(d[bgs])
            if bcover:keep &= d[bcover]>=.95
            gg=d.loc[keep]
            r,pa=correlation_matrix(gg[['rank','length_bp','n_sites',rec,bgs]].to_numpy())
            for j,name in enumerate(['length_bp','n_sites',rec,bgs]):
                sensitivity.append(dict(pop=pop,recombination_map=rec,bgs_map=bgs,covariate=name,n=len(gg),rho=r[j],partial_rho=pa[j]))
        x=percentile_features(g)
        for name,fpop in FOCAL:
            if fpop!=pop:continue
            ix=np.flatnonzero(g.gene_name.to_numpy()==name)
            record=dict(gene=name,pop=pop,original_rank=float(d.loc[d.gene_name==name,'rank'].iloc[0]),
                        matched_rank=None,n_controls=0,maximum_covariate_distance=None)
            if len(ix)==1:
                ix=int(ix[0]);distance=np.abs(x-x[ix]);allowed=(chrom!=chrom[ix])&(distance.max(axis=1)<=.20)
                candidates=np.flatnonzero(allowed)
                near=candidates[np.argsort(np.sum(distance[candidates]**2,axis=1),kind='stable')[:500]]
                record['n_controls']=len(near)
                if len(near):record['maximum_covariate_distance']=float(distance[near].max())
                if len(near)>=100:
                    v=g['rank'].to_numpy();r=v[ix]
                    record['matched_rank']=float((1+np.sum(v[near]<r)+.5*np.sum(v[near]==r))/(1+len(near)))
                    pd.DataFrame({'gene':name,'pop':pop,'control_gene_id':g.iloc[near].gene_id.to_numpy(),
                        'control_rank':v[near],'distance':np.sqrt(np.sum(distance[near]**2,axis=1))}).to_csv(a.output/f'matches_{name}_{pop}.csv',index=False)
            focal.append(record)
        assembled.append(d)
    full=pd.concat(assembled,ignore_index=True)
    full.to_csv(a.output/'gene_population_covariates.csv.gz',index=False)
    for name,data in [('covariate_correlations',corrs),('covariate_quintiles',bins),('focal_matched_ranks',focal),
                      ('coverage',coverage),('map_sensitivity',sensitivity)]:
        pd.DataFrame(data).to_csv(a.output/f'{name}.csv',index=False)
    write_json(a.output/'covariate_verification.json',dict(genes=19119,populations=26,rows=len(full),
        exact_accumulator_files=572,max_relative_tmrca_reconstruction_error=maxerr,
        input_manifest_sha256=sha(a.input/'SOURCES.json'),source_sha256=sha(__file__),
        uncertainty='22-autosome delete-one jackknife; 1.96 standard-error intervals; no independent-gene p-values',
        output_sha256={q.name:sha(q) for q in a.output.glob('*.csv*')}))
    print(pd.DataFrame(coverage).to_string(index=False),flush=True)

if __name__=='__main__':main()
