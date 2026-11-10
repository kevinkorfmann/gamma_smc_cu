"""Quantitative genome-wide haplotype agreement, availability and matching."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.spatial import cKDTree
from common import POPS,sha,write_json
from covariates import COVS,FOCAL,percentile_features

METRICS=['ihs_body','ihs_flank100','xp_body','xp_flank100']
EDGES=np.array([0,.01,.10,.20,.30,.40,.50,.60,.70,.80,.90,1.000001])

def correlation(y,s,x=None):
    values=np.column_stack([y,s] if x is None else [y,s,x])
    if any(np.ptp(v)==0 for v in values.T):return np.nan
    z=np.column_stack([rankdata(v) for v in values.T])
    c=np.corrcoef(z.T)
    if x is None:return c[0,1]
    inv=np.linalg.inv(c)
    return -inv[0,1]/np.sqrt(inv[0,0]*inv[1,1])

def jackknife_ci(estimate,replicates):
    v=np.asarray(replicates,float);v=v[np.isfinite(v)]
    if len(v)!=22:return np.nan,np.nan
    se=np.sqrt(21/22*np.sum((v-v.mean())**2))
    return estimate-1.96*se,estimate+1.96*se

def matching(g,drop_chrom=None,keep_pairs=False):
    """Exact missingness, different chromosome and fixed covariate calipers."""
    x=g[[f'match_{c}' for c in COVS]].to_numpy()
    chrom=g.chr.to_numpy();young=g['rank'].to_numpy()<.01;avail=g.ihs_body_n.to_numpy()>0
    usable=np.ones(len(g),bool) if drop_chrom is None else chrom!=drop_chrom
    rows=[];pairs=[]
    for available in [False,True]:
        cases=np.flatnonzero(usable&young&(avail==available))
        controls=np.flatnonzero(usable&~young&(avail==available))
        if not len(controls):continue
        tree=cKDTree(x[controls])
        for i in cases:
            candidate=controls[tree.query_ball_point(x[i],.15,p=np.inf)]
            candidate=candidate[chrom[candidate]!=chrom[i]]
            distance=np.sum((x[candidate]-x[i])**2,axis=1)
            chosen=candidate[np.argsort(distance,kind='stable')[:20]]
            for metric in METRICS:
                score=g[metric+'_fraction'].to_numpy();tail=g[metric+'_top5'].to_numpy()
                finite_controls=chosen[np.isfinite(score[chosen])]
                computable=np.isfinite(score[i]) and len(finite_controls)>=5 and len(chosen)>=5
                rows.append(dict(pop=g['pop'].iloc[0],gene_id=g.gene_id.iloc[i],chr=int(chrom[i]),
                    ihs_available=bool(available),metric=metric,matched=len(chosen)>=5,n_controls=len(chosen),
                    computable=bool(computable),n_finite_controls=len(finite_controls),
                    case_top5=float(tail[i]) if computable else np.nan,
                    control_top5=float(np.mean(tail[finite_controls])) if computable else np.nan))
            if keep_pairs and len(chosen)>=5:
                for j in chosen:
                    pairs.append(dict(pop=g['pop'].iloc[0],case=g.gene_id.iloc[i],control=g.gene_id.iloc[j],
                        case_chr=int(chrom[i]),control_chr=int(chrom[j]),
                        max_covariate_distance=float(np.max(np.abs(x[i]-x[j]))),ihs_available=bool(available)))
    return rows,pairs

def matched_summary(rows):
    df=pd.DataFrame(rows)
    result={}
    for (available,metric),g in df.groupby(['ihs_available','metric']):
        good=g.loc[g.computable]
        result[(bool(available),metric)]=dict(n_candidates=len(g),n_matched=int(g.matched.sum()),
            n_computable=len(good),case_rate=float(good.case_top5.mean()),control_rate=float(good.control_top5.mean()),
            difference=float((good.case_top5-good.control_top5).mean()))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--normalized',type=Path,required=True);a=p.parse_args()
    cov=pd.read_csv(a.summary/'gene_population_covariates.csv.gz')
    full=[];corrs=[];focal=[];status=[];match_rows=[];match_pairs=[];groups=[]
    # Shared chromosome blocks preserve dependence across populations and genes.
    totals=np.zeros((22,26,len(EDGES)-1));available=np.zeros(totals.shape+(4,))
    for pi,pop in enumerate(POPS):
        file=a.normalized/f'{pop}.csv.gz';receipt=json.loads((a.normalized/f'{pop}.json').read_text())
        assert sha(file)==receipt['output_sha256']
        h=pd.read_csv(file)
        h=h.drop(columns=['gene_name','chr','start','end'])
        d=cov.loc[cov['pop']==pop].merge(h,on=['gene_id','pop'],validate='one_to_one')
        if not np.array_equal(d.n_sites,d.corrected_polymorphic):
            bad=d.loc[d.n_sites!=d.corrected_polymorphic,['gene_id','n_sites','corrected_polymorphic']]
            bad.to_csv(a.summary/f'FAILED_site_count_{pop}.csv',index=False)
            raise ValueError(f'XP-EHH and corrected atlas use different SNP counts: {pop}')
        primary=~d.is_sd&np.isfinite(d['rank'])
        d['ihs_state']=np.where(d.ihs_body_n>0,'finite score',np.where(d.corrected_maf05==0,'no MAF-eligible core','eligible core; no finite score'))
        for metric in METRICS+['ihs_flank500','xp_flank500']:
            s=d[metric+'_fraction']
            ok=primary&np.isfinite(s)
            d[metric+'_rank']=np.nan
            d.loc[ok,metric+'_rank']=rankdata(-s.loc[ok],method='average')/ok.sum()
            d[metric+'_top5']=np.where(np.isfinite(s),d[metric+'_rank']<.05,np.nan)
            for summary_name in ['fraction','max']:
                score=d[metric+'_'+summary_name]
                for scope,mask,partial in [('all',primary&np.isfinite(score),False),
                       ('common',d.eligible&np.isfinite(score),False),('partial',d.eligible&np.isfinite(score),True)]:
                    g=d.loc[mask];y=-np.log10(g['rank'].to_numpy());sv=g[metric+'_'+summary_name].to_numpy()
                    x=g[COVS].to_numpy() if partial else None
                    estimate=correlation(y,sv,x);chrom=g.chr.to_numpy()
                    jk=[correlation(y[chrom!=c],sv[chrom!=c],None if x is None else x[chrom!=c]) for c in range(1,23)]
                    lo,hi=jackknife_ci(estimate,jk)
                    corrs.append(dict(pop=pop,metric=metric,summary=summary_name,scope=scope,n=len(g),rho=estimate,ci_low=lo,ci_high=hi))
        for state,g in d.loc[primary].groupby('ihs_state'):
            status.append(dict(pop=pop,state=state,n=len(g),low1=int((g['rank']<.01).sum()),
                xp_body_available=int((g.xp_body_n>0).sum()),xp_flank_available=int((g.xp_flank100_n>0).sum()),
                ihs_flank_available=int((g.ihs_flank100_n>0).sum())))
        for c,g in d.loc[primary].groupby('chr'):
            binid=np.searchsorted(EDGES,g['rank'].to_numpy(),side='right')-1
            totals[c-1,pi]=np.bincount(binid,minlength=len(EDGES)-1)
            for mi,m in enumerate(METRICS):
                available[c-1,pi,:,mi]=np.bincount(binid,weights=(g[m+'_n']>0).astype(float),minlength=len(EDGES)-1)
        for gene,fpop in FOCAL:
            if fpop==pop:
                g=d.loc[d.gene_name==gene].iloc[0]
                for m in METRICS:
                    focal.append(dict(gene=gene,pop=pop,metric=m,n_finite=int(g[m+'_n']),rank=g[m+'_rank'],
                        fraction=g[m+'_fraction'],max_score=g[m+'_max']))
        g=d.loc[d.eligible].copy().reset_index(drop=True)
        features=percentile_features(g)
        for j,c in enumerate(COVS):g['match_'+c]=features[:,j]
        rows,pairs=matching(g,keep_pairs=True);match_rows.extend(rows);match_pairs.extend(pairs);groups.append(g)
        full.append(d)
        print('Compared',pop,flush=True)
    out=pd.concat(full,ignore_index=True)
    out.to_csv(a.summary/'all_gene_population_comparisons.csv.gz',index=False)
    for name,data in [('haplotype_correlations',corrs),('haplotype_missingness',status),('focal_haplotype_ranks',focal),
                      ('matched_candidates',match_rows),('matched_control_pairs',match_pairs)]:
        pd.DataFrame(data).to_csv(a.summary/f'{name}.csv.gz' if name=='matched_control_pairs' else a.summary/f'{name}.csv',index=False)
    estimate=available.sum(axis=0)/totals.sum(axis=0)[...,None]
    mean=estimate.mean(axis=0)
    rng=np.random.default_rng(20260927);w=rng.multinomial(22,np.repeat(1/22,22),size=1000)
    count=np.einsum('bc,cpd->bpd',w,totals)
    nums=np.einsum('bc,cpdm->bpdm',w,available)
    boot=np.nanmean(nums/count[...,None],axis=1)
    lower,upper=np.nanquantile(boot,[.025,.975],axis=0)
    rows=[]
    for b in range(len(EDGES)-1):
        for j,m in enumerate(METRICS):
            rows.append(dict(bin=b,lower_rank=EDGES[b],upper_rank=min(1,EDGES[b+1]),metric=m,
                fraction=mean[b,j],ci_low=lower[b,j],ci_high=upper[b,j],gene_population_n=int(totals[:,:,b].sum())))
    pd.DataFrame(rows).to_csv(a.summary/'availability_by_rank.csv',index=False)
    estimates=matched_summary(match_rows);replicates={key:[] for key in estimates}
    for c in range(1,23):
        rr=[]
        for g in groups:rr.extend(matching(g,drop_chrom=c)[0])
        stats=matched_summary(rr)
        for key in estimates:replicates[key].append(stats.get(key,{}).get('difference',np.nan))
        print('Matched chromosome jackknife',c,flush=True)
    rows=[]
    for key,s in estimates.items():
        lo,hi=jackknife_ci(s['difference'],replicates[key])
        rows.append(dict(ihs_available=key[0],metric=key[1],**s,ci_low=lo,ci_high=hi))
    pd.DataFrame(rows).to_csv(a.summary/'matched_support.csv',index=False)
    write_json(a.summary/'comparison_verification.json',dict(rows=len(out),populations=26,
        corrected_site_counts_equal=True,normalized_population_receipts={p:sha(a.normalized/f'{p}.json') for p in POPS},
        source_sha256=sha(__file__),matched_cases=len({(r['pop'],r['gene_id']) for r in match_rows if r['matched']}),
        outputs={q.name:sha(q) for q in a.summary.glob('*.csv*')}))

if __name__=='__main__':main()
