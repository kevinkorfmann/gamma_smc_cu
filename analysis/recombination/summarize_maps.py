#!/usr/bin/env python3
"""Validate map outputs and summarize exact gene-body/flank interval overlaps."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from infer_maps import weighted_intervals, sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--ranks',type=Path,required=True);ap.add_argument('--case-data',type=Path)
    a=ap.parse_args();out=a.root/'results';out.mkdir(exist_ok=True)
    targets=pd.read_csv(a.root/'inputs/targets.tsv',sep='\t')
    ranks=pd.read_csv(a.ranks).set_index('gene_id');rows=[];missing=[];splits=[];spatial=[]
    bgs=pd.read_csv(a.root/'bgs/buffalo_kern2024_YRI_CADD6_100kb.tsv.gz',sep='\t')
    bgmeta=json.loads((a.root/'bgs/buffalo_kern2024_provenance.json').read_text())
    for name,digest in bgmeta['output_sha256'].items():assert sha(a.root/'bgs'/name)==digest
    bgrows=[]
    for t in targets.itertuples():
        bg=bgs[bgs.chrom==t.chrom]
        bv,bc=weighted_intervals(bg.start.to_numpy(),bg.end.to_numpy(),bg.Bprime.to_numpy(),
                                [t.gene_start0,t.display_start0,t.gene_end0],
                                [t.gene_end0,t.gene_start0,t.display_end0])
        bgrow=dict(gene=t.gene,gene_Bprime_YRI=bv[0],gene_Bprime_coverage=bc[0]/(t.gene_end0-t.gene_start0),
                   flank_Bprime_YRI=np.nansum(bv[1:]*bc[1:])/bc[1:].sum() if bc[1:].sum()>0 else np.nan,
                   flank_Bprime_coverage=bc[1:].sum()/(t.display_end0-t.display_start0-t.gene_end0+t.gene_start0))
        bgrows.append(bgrow)
        for pop in t.populations.split(','):
            prefix=a.root/'maps'/f'{t.gene}.{pop}'
            if not Path(str(prefix)+'.json').exists():missing.append(f'{t.gene}/{pop}');continue
            meta=json.loads(Path(str(prefix)+'.json').read_text())
            for name,digest in meta['output_sha256'].items():assert sha(a.root/'maps'/name)==digest,name
            z=np.load(str(prefix)+'.intervals.npz');left,right=z['pos_left'],z['pos_right']
            rho=z['rho_per_bp'];assert np.isfinite(rho).all() and np.all(rho>0)
            assert np.allclose(z['r_per_bp'],rho/(4*meta['Ne_used']))
            avg,cov=weighted_intervals(left,right,rho,[t.gene_start0,t.display_start0,t.gene_end0],[t.gene_end0,t.gene_start0,t.display_end0])
            flank=np.nansum(avg[1:]*cov[1:])/cov[1:].sum()
            r=ranks.loc[t.gene_id]
            rows.append(dict(gene=t.gene,population=pop,role='focal' if pop==t.focal_population else 'YRI comparison',
                chrom=t.chrom,gene_start0=t.gene_start0,gene_end0=t.gene_end0,n_diploid=meta['n_diploid'],
                retained_snps=meta['intervals']+1,gene_coverage_fraction=cov[0]/(t.gene_end0-t.gene_start0),
                gene_rho_per_bp=avg[0],flank_rho_per_bp=flank,gene_cM_Mb_Ne10000=avg[0]/40000*1e8,
                flank_cM_Mb_Ne10000=flank/40000*1e8,gene_to_flank_rate_ratio=avg[0]/flank,
                tmrca_generations=r[f'{pop}_tmrca'],tmrca_rank=r[f'{pop}_rank'],Ne_auxiliary=meta['Ne_auxiliary_estimate'],
                sources=t.sources))
            if t.gene not in ['GRK2','TREM2']:continue
            windows=pd.read_csv(str(prefix)+'.windows.tsv',sep='\t')
            case=np.load(a.case_data/f'{t.gene}_plot_data.npz') if a.case_data else None
            tag='control' if pop=='YRI' else 'focal'
            for size in [10000,50000]:
                d=windows[windows.window_bp==size];centers=(d.start+d.end)/2+1
                if case is not None:
                    valid=(centers>=case[tag+'_positions'][0])&(centers<=case[tag+'_positions'][-1])&(d.coverage_fraction>=.99)
                    tmrca=np.interp(centers,case[tag+'_positions'],case[tag+'_median'])
                    finite=valid&np.isfinite(d.rho_per_bp)&(d.rho_per_bp>0)&(tmrca>0)
                    spatial.append(dict(gene=t.gene,population=pop,window_bp=size,n_windows=int(finite.sum()),
                        spearman_tmrca_rho=float(spearmanr(tmrca[finite],d.rho_per_bp[finite]).statistic),
                        note='Descriptive spatial association; adjacent windows are correlated, no independence-based P value.'))
                paths=[a.root/'maps'/f'{t.gene}.{pop}_half{k}.windows.tsv' for k in [1,2]]
                if all(p.exists() for p in paths):
                    h=[pd.read_csv(p,sep='\t').query('window_bp==@size') for p in paths]
                    assert np.array_equal(h[0].start,h[1].start)
                    ok=(h[0].coverage_fraction>=.99)&(h[1].coverage_fraction>=.99)
                    splits.append(dict(gene=t.gene,population=pop,window_bp=size,n_windows=int(ok.sum()),
                        spearman_halves=float(spearmanr(h[0].rho_per_bp[ok],h[1].rho_per_bp[ok]).statistic),
                        median_half1_over_half2=float(np.median(h[0].rho_per_bp[ok]/h[1].rho_per_bp[ok]))))
    bgtab=pd.DataFrame(bgrows);bgtab.to_csv(out/'gene_BGS_summary.tsv',sep='\t',index=False)
    pd.DataFrame(rows).merge(bgtab,on='gene',validate='many_to_one').to_csv(out/'gene_recombination_summary.tsv',sep='\t',index=False)
    pd.DataFrame(splits).to_csv(out/'focal_split_repeatability.tsv',sep='\t',index=False)
    pd.DataFrame(spatial).to_csv(out/'focal_tmrca_association.tsv',sep='\t',index=False)
    report=dict(expected_genes=len(targets),expected_maps=sum(len(x.split(',')) for x in targets.populations),
                completed_maps=len(rows),completed_genes=len(set(r['gene'] for r in rows)),missing=missing,
                validated_hashes_and_rate_scaling=True,BGS_source_sha256_validated=True,
                spatial_associations_recomputed=a.case_data is not None,
                genes_with_complete_BGS_body_coverage=int((bgtab.gene_Bprime_coverage>=.999999).sum()))
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
