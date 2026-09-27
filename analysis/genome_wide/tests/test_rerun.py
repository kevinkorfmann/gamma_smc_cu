"""CPU regressions for the scientific rerun pipeline (no CUDA required)."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import types
import numpy as np
import pandas as pd
import pytest

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from sd_mask import merge_intervals, coverage_fraction
from rerun_support import calibrate, completed_or_reserve, finish
from build_candidates import galwey, replication_scores, cascade, POPS


def test_sd_overlap_is_counted_once_and_uses_gencode_coordinates():
    intervals={1:merge_intervals([(0,10),(0,10),(5,15),(12,20)])}
    assert intervals[1]==[(0,20)]
    assert coverage_fraction(1,1,100,intervals)==.2
    assert coverage_fraction(1,20,21,intervals)==.5
    assert coverage_fraction(1,21,21,intervals)==0


def test_calibration_physical_time_depends_on_mutation_rate_not_arbitrary_Ne():
    g=np.array([[0,0,1],[1,0,1],[0,1,0],[1,1,1]],dtype=np.uint8)
    a=calibrate(g,np.array([10,20,99]),1e-8,2e-8)
    b=calibrate(g,np.array([10,20,99]),2e-8,4e-8)
    assert a['theta']==pytest.approx(.015)
    assert a['generations_per_coalescent_unit']==pytest.approx(.015/(2e-8))
    assert b['generations_per_coalescent_unit']==pytest.approx(a['generations_per_coalescent_unit']/2)
    assert a['scaled_recombination_rate']==b['scaled_recombination_rate']
    with pytest.raises(ValueError):calibrate(np.zeros((4,3),dtype=np.uint8),np.array([1,2,3]),1e-8,1e-8)


def test_galwey_limits():
    assert galwey(np.eye(5))==pytest.approx(5)
    assert galwey(np.ones((5,5)))==pytest.approx(1,abs=1e-7)


def test_scores_use_complete_gene_set_and_do_not_admit_missing_population():
    d=pd.DataFrame({p+'_rank':[.1,.2,.3,.4] for p in POPS})
    d.loc[0,'ACB_rank']=np.nan
    scored,corr,meta=replication_scores(d)
    assert meta['n_complete_genes']==3
    assert np.isnan(scored.loc[0,'AFR_max_rank'])
    assert scored.loc[0,'EUR_max_rank']==.1
    assert 'p_value' not in scored and 'q_value' not in scored


def test_single_linkage_handles_nested_genes():
    d=pd.DataFrame({'gene_id':['a','b','c','ref'],'gene_name':['A','B','C','REF'],
                    'chr':[1,1,1,2],'start':[1,10,150,1],'end':[200,20,160,10],
                    'min_rank':[.001,.002,.003,.1],'is_sd':[False]*4})
    for p in POPS:d[p+'_rank']=[.001,.002,.003,.1]
    _,loci,members,counts=cascade(d,cluster_bp=20,reference_genes=['REF'])
    assert counts['stage5']==1
    assert loci.iloc[0].cluster_size==3
    assert loci.iloc[0].representative_gene=='A'


def test_resume_checks_manifest_and_output_hashes(tmp_path):
    (tmp_path/'CEU.csv').write_text('a\n1\n')
    np.savez(tmp_path/'CEU.npz',x=np.arange(3))
    identity={'run':'abc'}
    finish(tmp_path,'CEU',identity,{'time_units':'generations'},{})
    assert completed_or_reserve(tmp_path,'CEU',identity,True)
    with pytest.raises(FileExistsError):completed_or_reserve(tmp_path,'CEU',{'run':'other'},True)
    (tmp_path/'CEU.csv').write_text('a\n2\n')
    with pytest.raises(FileExistsError):completed_or_reserve(tmp_path,'CEU',identity,True)


def test_genome_runner_aggregates_all_pairs_and_records_physical_units(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('scan_runner',HERE/'infer_chromosome.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cache=tmp_path/'cache';(cache/'parsed').mkdir(parents=True);(cache/'genes').mkdir()
    positions=np.array([10,20,30,40]);g=np.array([[0,0,0,0],[1,1,0,0],[0,1,0,1],[1,0,1,0]],dtype=np.uint8)
    np.savez(cache/'parsed/chr1.npz',G=g,positions=positions,sample_ids=np.array(['a','b']))
    (cache/'genes/chr1_genes.tsv').write_text('gene_id\tgene_name\tstart\tend\nid\tGENE\t10\t40\n')
    leads=tmp_path/'leads.tsv'
    leads.write_text('CHROM\tPOS\tRSID\tX\tS\tPOSTERIOR\n1\t30\trsLead\t2\t.1\t.99\n')
    samples=tmp_path/'samples.txt';samples.write_text('header\n0 a 0 0 0 CEU EUR\n0 b 0 0 0 CEU EUR\n')
    output=tmp_path/'output';calls=[]
    def fake_infer(G,pos,**kw):
        calls.append(kw)
        return {'positions':pos,'mean':np.broadcast_to(pos[:,None]*10.0,(len(pos),len(kw['pairs']))).copy(),
                'pairs':kw['pairs'],
                'metadata':{'time_units':'generations','generations_per_coalescent_unit':2*kw['Ne']}}
    monkeypatch.setitem(sys.modules,'gamma_smc_cu',types.SimpleNamespace(infer_blockwise=fake_infer))
    monkeypatch.setattr(sys,'argv',[str(HERE/'infer_chromosome.py')])
    args=argparse.Namespace(cache_dir=str(cache),samples=str(samples),output_dir=str(output),mu=1.25e-8,rho=1e-8,core_block_sites=4096,flank_sites=2048,pair_chunk=2,sequence_length=None,resume=False,chr=1,lead_variants=str(leads),lead_half_bp=10)
    module.CACHE_DIR=str(cache);module.RESULTS_DIR=str(output);module.PAIR_CHUNK=2
    module.run_chromosome(1,['CEU'],args)
    with np.load(output/'chr1/CEU.npz') as z:
        assert z['count'].tolist()==[6]
        assert np.exp(z['log_sum']/z['count'])[0]==pytest.approx((100*200*300*400)**.25)
        assert z['n_geom_lt_1000'].tolist()==[6] and z['n_arith_lt_1000'].tolist()==[6]
        assert z['histogram'].sum()==6 and z['histogram_underflow'].sum()==0
        # A separate lead bank uses positions20,30,40 from the exact SAME
        # chromosome posterior and does not alter the one-gene array shape.
        assert z['lead_count'].tolist()==[6] and z['lead_n_sites'].tolist()==[3]
        assert z['lead_center_pos'].tolist()==[30]
        assert np.exp(z['lead_log_sum']/z['lead_count'])[0]==pytest.approx((200*300*400)**(1/3))
    meta=json.loads((output/'chr1/CEU.metadata.json').read_text())
    assert meta['complete'] and meta['calibration']['time_units']=='generations'
    assert len(calls)==3 and all(not c['auto_estimate_theta'] for c in calls)
    assert all(c['physical_mu']==args.mu for c in calls)
    from lead_summaries import extract
    table=extract(output,tmp_path/'extracted',chromosomes=[1],populations=['CEU'])
    assert len(table)==1 and table.iloc[0].geom_mean_tmrca==pytest.approx((200*300*400)**(1/3))
    assert table.iloc[0].n_pairs==6 and len(calls)==3
    manifest=json.loads((tmp_path/'extracted/manifest.json').read_text())
    assert manifest['status']=='complete' and manifest['n_leads']==1


def test_invalid_posteriors_fail_before_any_floor_or_summary():
    from rerun_support import validate_posterior_means, per_pair_moments
    for bad in [0., -1., np.nan, np.inf, -np.inf]:
        values=np.array([[1.,2.],[3.,bad]],dtype=np.float32)
        with pytest.raises(ValueError, match='finite and strictly positive'):
            validate_posterior_means(values,2,2)
        with pytest.raises(ValueError, match='finite and strictly positive'):
            per_pair_moments(values,np.array([0,1]))
    with pytest.raises(ValueError, match='shape'):
        validate_posterior_means(np.ones((2,3)),2,4)


def test_float64_moments_preserve_subgeneration_times_and_small_contributions():
    from rerun_support import per_pair_moments
    values=np.array([[.25,2**24],[.5,1.],[.75,1.]],dtype=np.float32)
    linear,logarithmic=per_pair_moments(values,np.arange(3))
    assert linear.dtype==np.float64 and logarithmic.dtype==np.float64
    assert linear[0]==.5
    assert linear[1]==(2**24+2)/3  # float32 reduction loses both small terms
    assert np.exp(logarithmic[0])==pytest.approx((.25*.5*.75)**(1/3))
    np.testing.assert_array_equal(values[:,0],[.25,.5,.75])


def test_exact_threshold_counts_and_separate_histogram_tails():
    from rerun_support import per_pair_moments,pair_distribution_counts
    # The mixed pair has geometric mean100 <1000 but arithmetic mean5000.5.
    values=np.array([[.25,5,999,1001,2e6,1],[.25,5,999,1001,2e6,10000]],dtype=np.float32)
    linear,logs=per_pair_moments(values,np.array([0,1]))
    result=pair_distribution_counts(linear,logs,np.linspace(np.log(10),np.log(1e6),51))
    assert result['n_geom_lt_1000']==4
    assert result['n_arith_lt_1000']==3
    assert result['histogram_underflow']==2
    assert result['histogram_overflow']==1
    assert result['histogram'].sum()==3
    assert result['histogram'].sum()+result['histogram_underflow']+result['histogram_overflow']==6
