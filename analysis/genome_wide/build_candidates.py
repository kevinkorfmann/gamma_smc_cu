#!/usr/bin/env python3
"""Rebuild ranks, unique-bp SD mask, candidate cascade and descriptive scores.

All outputs go to a new directory. Inputs are never modified. The default
requires all 22 x 26 inference outputs. Scores have no calibrated p/FDR meaning.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sd_mask import load_sd_intervals, coverage_fraction
from rerun_support import fingerprint

SUPERPOPS={'AFR':['ACB','ASW','ESN','GWD','LWK','MSL','YRI'],
 'EUR':['CEU','FIN','GBR','IBS','TSI'],'EAS':['CDX','CHB','CHS','JPT','KHV'],
 'SAS':['BEB','GIH','ITU','PJL','STU'],'AMR':['CLM','MXL','PEL','PUR']}
POPS=sorted(sum(SUPERPOPS.values(),[]))
REFERENCE_GENES=['SLC24A5','ABCC11','LCT','EDAR','FADS1','KITLG','TRPV6','ALDH2','HERC2','ADH1B','IRGM','SH2B3','ADH4','SLC45A2','ACKR1','OAS1','HMGA2','CASP12','TYRP1','OCA2','MC1R','EPAS1','HBB']


def aggregate(results, chromosomes, populations, allow_partial=False):
    from campaign_identity import scientific_identity
    frames=[];missing=[];metadata=[];campaign=None;chromosome_inputs={}
    for chrom in chromosomes:
        combined=None
        for pop in populations:
            path=Path(results)/f'chr{chrom}'/(pop+'.npz')
            marker=path.with_suffix('.metadata.json')
            if not path.exists() or not marker.exists():
                missing.append(str(path));continue
            meta=json.loads(marker.read_text())
            if not meta.get('complete') or meta.get('calibration',{}).get('time_units')!='generations':
                raise ValueError(f'Incomplete or non-generation-calibrated input {path}')
            common, chromosome_input=scientific_identity(meta)
            if campaign is None:campaign=common
            elif campaign!=common:raise ValueError(f'Mixed scientific campaigns/settings in {path}')
            if chrom in chromosome_inputs and chromosome_inputs[chrom]!=chromosome_input:
                raise ValueError(f'Mixed input caches/annotations for chromosome {chrom}')
            chromosome_inputs[chrom]=chromosome_input
            if meta['identity'].get('chromosome')!=chrom or meta['identity'].get('population')!=pop:
                raise ValueError(f'Chromosome/population identity mismatch in {path}')
            if meta['output_sha256']['.npz'] != fingerprint(path)['sha256']:
                raise ValueError(f'Output checksum mismatch {path}')
            metadata.append({'path':str(path),'metadata_sha256':fingerprint(marker)['sha256']})
            with np.load(path,allow_pickle=False) as z:
                count=z['count']; expected=int(z['n_pairs_total'])
                eligible=z['n_sites_per_gene']>=2
                if not np.array_equal(count>0,eligible) or np.any(count[eligible]!=expected):
                    raise ValueError(f'Incomplete pair aggregation in {path}')
                with np.errstate(divide='ignore',invalid='ignore'):
                    values=np.where(count>0,np.exp(z['log_sum']/count),np.nan)
                part=pd.DataFrame({k:z[k] for k in ['gene_id','gene_name','start','end']})
                part['chr']=chrom;part[pop+'_tmrca']=values
            if combined is None:combined=part
            else:
                cols=['gene_id','gene_name','start','end','chr']
                if not combined[cols].equals(part[cols]):raise ValueError(f'Gene annotation/order mismatch {path}')
                combined[pop+'_tmrca']=part[pop+'_tmrca']
        if combined is not None:frames.append(combined)
    if missing and not allow_partial:raise FileNotFoundError(f'Missing {len(missing)} completed chromosome/population inputs; first: {missing[0]}')
    if not frames:raise ValueError('No completed inputs')
    df=pd.concat(frames,ignore_index=True)
    for p in populations:
        c=p+'_tmrca'
        if c not in df:df[c]=np.nan
        df[p+'_rank']=df[c].rank(pct=True)
    rc=[p+'_rank' for p in populations]
    df['min_rank']=df[rc].min(axis=1);df['max_rank']=df[rc].max(axis=1);df['rank_range']=df.max_rank-df.min_rank
    valid=df[rc].notna().any(axis=1)
    df['min_pop']=None;df.loc[valid,'min_pop']=df.loc[valid,rc].idxmin(axis=1).str.removesuffix('_rank')
    return df,missing,metadata


def galwey(matrix):
    eigenvalues=np.linalg.eigvalsh(np.asarray(matrix,dtype=float))
    eigenvalues=np.clip(eigenvalues,0,None)
    if eigenvalues.sum()<=0:raise ValueError('Rank correlation matrix has no positive eigenvalues')
    return float(np.sqrt(eigenvalues).sum()**2/eigenvalues.sum())


def replication_scores(df):
    cols=[p+'_rank' for p in POPS]
    missing=set(cols)-set(df)
    if missing:raise ValueError('Descriptive population replication requires all 26 rank columns')
    complete=df[cols].dropna()
    if len(complete)<3:raise ValueError('At least three genes complete in all26 populations required')
    corr=complete.corr(method='spearman')
    if corr.isna().any().any():raise ValueError('Undefined population rank correlation')
    neff={};out=df.copy()
    for sp,pops in SUPERPOPS.items():
        c=[p+'_rank' for p in pops];neff[sp]=galwey(corr.loc[c,c])
        # A missing population cannot count as within-continent replication.
        worst=out[c].max(axis=1).where(out[c].notna().all(axis=1))
        out[sp+'_max_rank']=worst
        out[sp+'_descriptive_score']=worst**neff[sp]
    scores=[sp+'_descriptive_score' for sp in SUPERPOPS]
    best=out[scores].min(axis=1)
    out['descriptive_replication_score']=np.minimum(1,5*best)
    out['score_rank_all_genes']=out.descriptive_replication_score.rank(method='min')
    sas_eur=[p+'_rank' for p in SUPERPOPS['SAS']+SUPERPOPS['EUR']]
    neff['SAS_EUR']=galwey(corr.loc[sas_eur,sas_eur])
    return out,corr,{'n_complete_genes':len(complete),'effective_counts':neff,
        'method':'Galwey (sum sqrt nonnegative eigenvalues)^2 / sum eigenvalues',
        'score':'min(1, 5 * min_continent(max_population_rank ** effective_count))',
        'interpretation':'Descriptive only; empirical ranks do not establish calibrated p-values or false-discovery rates.'}


def cascade(df, rank_threshold=.01, replication_threshold=.05, exclusion_bp=500000, cluster_bp=1000000, reference_genes=REFERENCE_GENES):
    stage1=df[~df.is_sd].copy();stage2=stage1[stage1.min_rank<rank_threshold].copy()
    eligible={sp:(stage2[[p+'_rank' for p in pp]]<replication_threshold).all(axis=1) for sp,pp in SUPERPOPS.items()}
    stage3=stage2[np.column_stack(list(eligible.values())).any(axis=1)].copy()
    reference=df[df.gene_name.isin(reference_genes)]
    missing=set(reference_genes)-set(reference.gene_name)
    if missing:raise ValueError(f'Reference genes absent: {sorted(missing)}')
    keep=[]
    for g in stage3.itertuples():
        same=reference[reference.chr==g.chr]
        keep.append(not ((g.start-exclusion_bp<=same.end)&(g.end+exclusion_bp>=same.start)).any())
    stage4=stage3[keep].copy();loci=[];members=[]
    for chrom, group in stage4.groupby('chr'):
        clusters=[];current=[];right=None
        for idx,g in group.sort_values(['start','end','gene_id']).iterrows():
            if right is not None and g.start-right>cluster_bp:clusters.append(current);current=[];right=None
            current.append(idx);right=max(right or g.end,g.end)
        if current:clusters.append(current)
        for cl in clusters:
            rows=stage4.loc[cl];representative=rows.sort_values(['min_rank','gene_id']).iloc[0]
            qualifying=[sp for sp,pops in SUPERPOPS.items() if (representative[[p+'_rank' for p in pops]]<replication_threshold).all()]
            focal_sp=min(qualifying,key=lambda sp:representative.get(sp+'_descriptive_score',representative[[p+'_rank' for p in SUPERPOPS[sp]]].max()))
            focal_pop=min(SUPERPOPS[focal_sp],key=lambda p:representative[p+'_rank'])
            locus_id=f'chr{chrom}:{int(rows.start.min())}-{int(rows.end.max())}'
            record=representative.to_dict();record.update(locus_id=locus_id,representative_gene=representative.gene_name,cluster_size=len(rows),cluster_start=int(rows.start.min()),cluster_end=int(rows.end.max()),cluster_span_bp=int(rows.end.max()-rows.start.min()+1),cluster_members=';'.join(rows.gene_name.astype(str)),focal_superpopulation=focal_sp,focal_population=focal_pop)
            loci.append(record)
            for idx,g in rows.iterrows():members.append({'locus_id':locus_id,'gene_id':g.gene_id,'gene_name':g.gene_name,'representative_gene':representative.gene_name})
    stage5=pd.DataFrame(loci)
    if len(stage5):stage5=stage5.sort_values(['min_rank','gene_id'])
    return stage4,stage5,pd.DataFrame(members),{'stage0':len(df),'stage1':len(stage1),'stage2':len(stage2),'stage3':len(stage3),'stage4':len(stage4),'stage5':len(stage5)}


def main():
    p=argparse.ArgumentParser(description=__doc__);source=p.add_mutually_exclusive_group(required=True)
    source.add_argument('--results-dir');source.add_argument('--ranks-csv',help='Existing full precision rank matrix, for postprocessing checks')
    p.add_argument('--sd-track',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--allow-partial',action='store_true',help='Diagnostics only; candidate/replication output still requires all26 populations')
    p.add_argument('--chromosomes',nargs='+',type=int,default=list(range(1,23)))
    args=p.parse_args();out=Path(args.output_dir)
    if out.exists():raise FileExistsError('Choose a new output directory; existing outputs are never replaced')
    if args.ranks_csv:
        df=pd.read_csv(args.ranks_csv);missing=[];provenance=[fingerprint(args.ranks_csv)]
    else:df,missing,provenance=aggregate(args.results_dir,args.chromosomes,POPS,args.allow_partial)
    if df.gene_id.duplicated().any():raise ValueError('Duplicate gene IDs')
    intervals=load_sd_intervals(args.sd_track)
    df['sd_unique_bp_fraction']=[coverage_fraction(g.chr,g.start,g.end,intervals) for g in df.itertuples()]
    df['is_sd']=df.sd_unique_bp_fraction>=.5
    df,corr,scores=replication_scores(df)
    s4,s5,members,counts=cascade(df)
    sensitivity=[]
    for label,rt,rp,cd in [('baseline',.01,.05,1000000),('strict_rank',.005,.05,1000000),('loose_rank',.02,.05,1000000),('strict_replication',.01,.01,1000000),('loose_replication',.01,.10,1000000),('tight_cluster',.01,.05,500000),('loose_cluster',.01,.05,2000000)]:
        _,_,_,c=cascade(df,rt,rp,cluster_bp=cd);sensitivity.append(dict(configuration=label,rank_threshold=rt,replication_threshold=rp,cluster_bp=cd,**c))
    out.mkdir(parents=True)
    df.to_csv(out/'genome_wide_ranks.csv',index=False)
    df[['gene_id','gene_name','chr','start','end','min_rank','min_pop','max_rank','rank_range','is_sd','sd_unique_bp_fraction']].sort_values('min_rank').to_csv(out/'genome_wide_stats.csv',index=False)
    df[~df.is_sd].sort_values('min_rank').head(50).to_csv(out/'top50_sd_masked.csv',index=False)
    df[df.gene_name.isin(REFERENCE_GENES)].sort_values('min_rank').to_csv(out/'reference_loci.csv',index=False)
    df[['gene_id','gene_name']+[c for c in df if 'descriptive' in c or c=='score_rank_all_genes']].to_csv(out/'descriptive_replication_scores.csv',index=False)
    s4.to_csv(out/'stage4_genes.csv',index=False);s5.to_csv(out/'stage5_loci.csv',index=False);members.to_csv(out/'cluster_members.csv',index=False)
    corr.to_csv(out/'population_rank_correlations.csv');pd.DataFrame(sensitivity).to_csv(out/'threshold_sensitivity.csv',index=False)
    manifest={'inputs':provenance,'sd_track':fingerprint(args.sd_track),'missing_inputs':missing,'counts':counts,'replication':scores,'sd_mask':'union of UCSC 0-based half-open intervals; GENCODE1-based closed genes','rank_precision':'raw NPZ floating-point accumulators, not rounded display CSVs'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(counts))

if __name__=='__main__':main()
