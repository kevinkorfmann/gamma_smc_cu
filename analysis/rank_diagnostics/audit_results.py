"""Fail-closed consistency and matching-balance audit of completed results."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from common import sha,write_json
from covariates import COVS
from compare import METRICS

def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True);a=p.parse_args()
    r=json.loads((a.summary/'comparison_verification.json').read_text())
    for file,digest in r['outputs'].items():assert sha(a.summary/file)==digest,file
    d=pd.read_csv(a.summary/'all_gene_population_comparisons.csv.gz')
    assert len(d)==19119*26 and not d.duplicated(['pop','gene_id']).any()
    assert np.array_equal(d.n_sites,d.corrected_polymorphic)
    for m in METRICS+['ihs_flank500','xp_flank500']:
        assert np.array_equal(d[m+'_n']==0,~np.isfinite(d[m+'_fraction']))
        assert np.all(d[m+'_n']<=d[m+'_returned'])
        assert np.all(d[m+'_extreme_count']<=d[m+'_n'])
        finite=d[m+'_n']>0
        np.testing.assert_allclose(d.loc[finite,m+'_fraction'],d.loc[finite,m+'_extreme_count']/d.loc[finite,m+'_n'])
    pairs=pd.read_csv(a.summary/'matched_control_pairs.csv.gz')
    assert not pairs.duplicated(['pop','case','control']).any()
    assert np.all(pairs.case_chr!=pairs.control_chr)
    assert pairs.max_covariate_distance.max()<=.1500000001
    sizes=pairs.groupby(['pop','case']).size();assert sizes.min()>=5 and sizes.max()<=20
    idx=d.set_index(['pop','gene_id'])
    cases=idx.loc[list(zip(pairs['pop'],pairs['case']))].reset_index(drop=True)
    controls=idx.loc[list(zip(pairs['pop'],pairs.control))].reset_index(drop=True)
    assert cases.eligible.all() and controls.eligible.all()
    assert (cases['rank']<.01).all() and (controls['rank']>=.01).all()
    assert np.array_equal(cases.ihs_body_n>0,controls.ihs_body_n>0)
    balance=[]
    for pop,g in d.loc[d.eligible].groupby('pop'):
        pg=pairs.loc[pairs['pop']==pop]
        for cov in COVS:
            percentile=pd.Series(rankdata(g[cov],method='average')/len(g),index=g.gene_id)
            delta=percentile.loc[pg['case']].to_numpy()-percentile.loc[pg.control].to_numpy()
            balance.append(dict(pop=pop,covariate=cov,n_pairs=len(pg),
                mean_case_minus_control_percentile=float(np.mean(delta)),
                mean_absolute_percentile_distance=float(np.mean(np.abs(delta))),
                p95_absolute_percentile_distance=float(np.quantile(np.abs(delta),.95)),
                maximum_absolute_percentile_distance=float(np.max(np.abs(delta)))))
    pd.DataFrame(balance).to_csv(a.summary/'matching_balance.csv',index=False)
    mr=pd.read_csv(a.summary/'matched_candidates.csv')
    assert mr.loc[~mr.computable,['case_top5','control_top5']].isna().all().all()
    assert not mr.loc[(~mr.ihs_available)&(mr.metric=='ihs_body'),'computable'].any()
    av=pd.read_csv(a.summary/'availability_by_rank.csv')
    assert av[['fraction','ci_low','ci_high']].ge(0).all().all()
    assert av[['fraction','ci_low','ci_high']].le(1.0000000001).all().all()
    focal=pd.read_csv(a.summary/'focal_matched_ranks.csv')
    for row in focal.itertuples():
        if not np.isfinite(row.matched_rank):continue
        f=pd.read_csv(a.summary/f'matches_{row.gene}_{row.pop}.csv')
        expected=(1+(f.control_rank<row.original_rank).sum()+.5*(f.control_rank==row.original_rank).sum())/(1+len(f))
        assert abs(expected-row.matched_rank)<1e-12
    report=dict(passed=True,rows=len(d),matched_pairs=len(pairs),matched_gene_populations=len(sizes),
        maximum_covariate_percentile_distance=float(pairs.max_covariate_distance.max()),
        checks=['input hashes','unique gene-population observations','exact retained-site counts',
                'finite-score denominators and missingness','matched controls on other chromosomes',
                'caliper and control counts','exact body-iHS-availability matching','low/non-low case definitions',
                'missing scores never zero','focal conditional ranks'],
        source_sha256=sha(__file__),comparison_receipt_sha256=sha(a.summary/'comparison_verification.json'),
        matching_balance_sha256=sha(a.summary/'matching_balance.csv'))
    write_json(a.summary/'audit_verification.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
