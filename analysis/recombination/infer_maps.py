#!/usr/bin/env python3
"""Infer population maps on Sesame; resume from hashed input and output records.

No population pooling, imputation, MAF thinning, or extrapolation outside SNPs.
All stored genomic intervals are GRCh38, 0-based, half-open. Absolute rates are
conditional on Ne=10,000, matching the Gamma-SMC study; raw rho is also retained.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path

import certifi
os.environ['CURL_CA_BUNDLE']=certifi.where()
os.environ['SSL_CERT_FILE']=certifi.where()
os.environ['HTS_SSL_CAINFO']=certifi.where()
import numpy as np
import pandas as pd

VCF_URL=('https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/'
         '1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/'
         '1kGP_high_coverage_Illumina.{chrom}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz')
EXPECTED={'model.ckpt':'d1ef4caae0c16b8c0cb1e5cfc79d1982950bff2e3509ea08a9f244fc9bb3f79f',
          'feat_stats.npz':'6198b021f1df6bba48cb672a71386d188714f3e87c5cb166dbdee25c497e55ba'}
MU=1.25e-8
NE=10000


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()


def weighted_intervals(left,right,values,starts,ends):
    """Return overlap-weighted means and covered bp, with no extrapolation."""
    left,right=np.asarray(left),np.asarray(right)
    result=np.full(len(starts),np.nan);covered=np.zeros(len(starts))
    for i,(start,end) in enumerate(zip(starts,ends)):
        j,k=np.searchsorted(right,start,side='right'),np.searchsorted(left,end,side='left')
        widths=np.maximum(0,np.minimum(right[j:k],end)-np.maximum(left[j:k],start))
        covered[i]=widths.sum()
        if covered[i]>0:result[i]=np.dot(widths,np.asarray(values)[j:k])/covered[i]
    return result,covered


def extract(row,panel,root):
    from cyvcf2 import VCF
    path=root/'cache'/f'{row.gene}.npz'
    pops=row.populations.split(',')
    wanted=sorted(panel.loc[panel['pop'].isin(pops),'sample'])
    spec={'url':VCF_URL.format(chrom=row.chrom),'start0':int(row.extract_start0),'end0':int(row.extract_end0),'samples':wanted,
          'filters':'PASS/dot, biallelic ACGT SNP, diploid phased, complete within each population, segregating within population; duplicate positions removed'}
    fingerprint=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    if path.exists():
        z=np.load(path)
        if str(z['fingerprint'])!=fingerprint:raise ValueError(f'Input cache specification changed: {path}')
        return {k:z[k] for k in z.files}
    print(f'EXTRACT {row.gene} {row.chrom}:{row.extract_start0+1}-{row.extract_end0}',flush=True)
    bcf=root/'cache'/f'{row.gene}.bcf'
    if not bcf.exists():
        sample_file=root/'cache'/f'{row.gene}.samples.txt'
        sample_file.write_text('\n'.join(wanted)+'\n')
        temporary=bcf.with_suffix('.partial.bcf')
        command=[os.environ.get('BCFTOOLS','bcftools'),'view','--no-version','-r',
                 f'{row.chrom}:{row.extract_start0+1}-{row.extract_end0}',
                 '-S',str(sample_file),'-Ob','-o',str(temporary),spec['url']]
        for attempt in range(3):
            try:
                subprocess.run(command,check=True,timeout=900)
                temporary.replace(bcf);break
            except (subprocess.SubprocessError,OSError):
                if attempt==2:raise
                time.sleep(3)
    v=VCF(str(bcf),strict_gt=True)
    if set(v.samples)!=set(wanted):raise ValueError(f'Missing expected samples: {set(wanted)-set(v.samples)}')
    sample_pop=panel.set_index('sample')['pop'].to_dict()
    selections={p:np.array([i for i,s in enumerate(v.samples) if sample_pop[s]==p]) for p in pops}
    samples={p:np.array([v.samples[i] for i in idx]) for p,idx in selections.items()}
    positions={p:[] for p in pops};matrices={p:[] for p in pops}
    counters={p:dict(missing=0,unphased=0,monomorphic=0,retained_before_duplicate_filter=0) for p in pops}
    total=0;eligible=0
    for rec in v:
        total+=1
        if rec.FILTER not in (None,'PASS','.') or len(rec.REF)!=1 or rec.REF not in 'ACGT' or len(rec.ALT)!=1 or len(rec.ALT[0])!=1 or rec.ALT[0] not in 'ACGT':continue
        eligible+=1
        gt=np.asarray(rec.genotypes,dtype=np.int8)
        if gt.shape[1]!=3:raise ValueError('Non-diploid input')
        for pop,idx in selections.items():
            g=gt[idx]
            if np.any(g[:,:2]<0) or np.any(g[:,:2]>1):counters[pop]['missing']+=1;continue
            if np.any((g[:,0]!=g[:,1]) & (g[:,2]==0)):counters[pop]['unphased']+=1;continue
            hap=g[:,:2].reshape(-1)
            if hap.min()==hap.max():counters[pop]['monomorphic']+=1;continue
            positions[pop].append(rec.POS-1);matrices[pop].append(hap)
            counters[pop]['retained_before_duplicate_filter']+=1
    v.close()
    out={'fingerprint':np.array(fingerprint)}
    for p in pops:
        pos=np.array(positions[p],dtype=np.int64)
        if len(pos)<100:raise ValueError(f'{row.gene}/{p}: fewer than 100 SNPs')
        unique,counts=np.unique(pos,return_counts=True);ok=~np.isin(pos,unique[counts>1])
        out[f'{p}_pos']=pos[ok];out[f'{p}_gm']=np.stack(matrices[p],axis=1)[:,ok]
        out[f'{p}_samples']=samples[p];counters[p]['duplicate_records_removed']=int((~ok).sum())
        counters[p]['retained_snps']=int(ok.sum());counters[p]['diploid_samples']=len(samples[p])
    out['metadata']=np.array(json.dumps(dict(spec=spec,qc=counters,records=total,eligible_biallelic_snps=eligible)))
    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**out);temp.replace(path)
    return out


def save_prediction(pred,row,pop,samples,root,metadata):
    prefix=root/'maps'/f'{row.gene}.{pop}'
    left,right=pred['pos_left'],pred['pos_right']
    assert np.all(right>left) and np.all(left[1:]==right[:-1])
    assert np.all(np.isfinite(pred['rho_per_bp'])) and np.all(pred['rho_per_bp']>0)
    assert np.allclose(pred['r_per_bp'],pred['rho_per_bp']/(4*NE))
    np.savez_compressed(str(prefix)+'.intervals.npz',**pred)
    rows=[]
    for window in [10000,50000]:
        starts=np.arange(row.display_start0,row.display_end0,window,dtype=np.int64)
        ends=np.minimum(starts+window,row.display_end0)
        d={'chrom':row.chrom,'start':starts,'end':ends,'window_bp':window}
        for key in ['rho_per_bp','r_per_bp','rho_ci_lo','rho_ci_hi']:
            d[key],covered=weighted_intervals(left,right,pred[key],starts,ends)
        d['covered_bp']=covered;d['coverage_fraction']=covered/(ends-starts)
        d['cM_Mb_Ne10000']=d['r_per_bp']*1e8
        d['snps']=np.searchsorted(left,ends)-np.searchsorted(left,starts)
        rows.append(pd.DataFrame(d))
    pd.concat(rows).to_csv(str(prefix)+'.windows.tsv',sep='\t',index=False)
    intervals=pd.DataFrame({'chrom':row.chrom,'start':left.astype(int),'end':right.astype(int),'rho_per_bp':pred['rho_per_bp'],'r_per_bp_Ne10000':pred['r_per_bp']})
    intervals.to_csv(str(prefix)+'.bed.gz',sep='\t',index=False)
    report={**metadata,'gene':row.gene,'population':pop,'assembly':'GRCh38','coordinate_system':'0-based half-open',
            'samples':samples.tolist(),'n_diploid':len(samples),'intervals':len(left),
            'snp_span0':[float(left[0]),float(right[-1])],'Ne_used':float(pred['Ne_used']),
            'Ne_auxiliary_estimate':float(pred['Ne_estimated']),'mutation_rate':MU,
            'display_region0':[int(row.display_start0),int(row.display_end0)],
            'extraction_region0':[int(row.extract_start0),int(row.extract_end0)],
            'maximum_adjacent_snp_gap_bp':float((right-left).max()),
            'interval_bound_note':'Per-interval model predictive bounds conditional on Ne; windowed averages of bounds are not calibrated window confidence intervals.',
            'output_sha256':{p.name:sha(p) for p in [Path(str(prefix)+'.intervals.npz'),Path(str(prefix)+'.windows.tsv'),Path(str(prefix)+'.bed.gz')]}}
    Path(str(prefix)+'.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1)
    ap.add_argument('--genes',nargs='*');ap.add_argument('--split-check',action='store_true');a=ap.parse_args()
    for d in ['maps','cache','logs']:(a.root/d).mkdir(parents=True,exist_ok=True)
    hashes={n:sha(a.bundle/n) for n in EXPECTED};assert hashes==EXPECTED,'Model bundle checksum mismatch'
    import torch,fastrho
    torch.set_num_threads(6)
    model,cfg,stats=fastrho.load_model(a.bundle/'model.ckpt',a.bundle/'feat_stats.npz',device='cuda:0')
    targets=pd.read_csv(a.root/'inputs/targets.tsv',sep='\t')
    if a.genes:targets=targets[targets.gene.isin(a.genes)]
    targets=targets.iloc[a.shard::a.shards]
    panel=pd.read_csv(a.root/'inputs/phase3.panel',sep=r'\s+',usecols=[0,1,2,3])
    provenance=dict(fastrho_version=fastrho.__version__,torch_version=torch.__version__,python=platform.python_version(),
                    host=platform.node(),gpu=torch.cuda.get_device_name(0),model_id='domain-randomized-v1',
                    model_sha256=hashes,script_sha256=sha(__file__),manifest_sha256=sha(a.root/'inputs/targets.tsv'),
                    panel_sha256=sha(a.root/'inputs/phase3.panel'),input_mode='phased')
    (a.root/'logs'/f'environment.{a.shard}.json').write_text(json.dumps(provenance,indent=2)+'\n')
    failures=[]
    for row in targets.itertuples(index=False):
        try:
            populations=row.populations.split(',')
            if all((a.root/'maps'/f'{row.gene}.{p}.json').exists() for p in populations) and not a.split_check:continue
            z=extract(row,panel,a.root)
            for pop in populations:
                jobs=[(pop,z[f'{pop}_gm'],z[f'{pop}_samples'])]
                if a.split_check and row.gene in ['GRK2','TREM2']:
                    # Interleave diploid IDs in VCF sample order, keeping both haplotypes together.
                    for half in [0,1]:
                        idx=np.arange(len(z[f'{pop}_samples']))[half::2]
                        hidx=np.column_stack([2*idx,2*idx+1]).ravel()
                        jobs.append((f'{pop}_half{half+1}',z[f'{pop}_gm'][hidx],z[f'{pop}_samples'][idx]))
                for label,gm,samples in jobs:
                    if (a.root/'maps'/f'{row.gene}.{label}.json').exists():continue
                    t=time.time();pos=z[f'{pop}_pos']-row.extract_start0
                    print(f'INFER {row.gene}/{label} {gm.shape}',flush=True)
                    pred=fastrho.predict_map_from_genotype_matrix(gm,pos,model,cfg,stats,mutation_rate=MU,Ne=NE,device='cuda:0',input_mode='phased')
                    pred['pos_left']+=row.extract_start0;pred['pos_right']+=row.extract_start0
                    metadata={**provenance,'seconds':time.time()-t,'input_cache_sha256':sha(a.root/'cache'/f'{row.gene}.npz'),
                              'input_qc':json.loads(str(z['metadata'])),'split_check':label!=pop}
                    save_prediction(pred,row,label,samples,a.root,metadata)
                    print(f'DONE {row.gene}/{label} {time.time()-t:.1f}s',flush=True)
        except Exception:
            err={'gene':row.gene,'error':traceback.format_exc()};failures.append(err);print(err['error'],flush=True)
            (a.root/'logs'/f'failures.{a.shard}.json').write_text(json.dumps(failures,indent=2)+'\n')
    (a.root/'logs'/f'complete.{a.shard}.json').write_text(json.dumps({'target_genes':len(targets),'failures':failures},indent=2)+'\n')
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
