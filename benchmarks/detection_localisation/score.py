"""Fresh matched Gamma-SMC, iHS and XP-EHH scores on simulated genotypes."""
import argparse, gzip, hashlib, importlib.metadata, json, os, subprocess, time
from pathlib import Path
import numpy as np
import allel


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def run(a):
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    start=time.monotonic();record=json.loads((a.input/'COMPLETE.json').read_text())
    assert sha(a.input/'input.npz')==record['input_sha256']
    a.out.mkdir(parents=True,exist_ok=False)
    z=np.load(a.input/'input.npz');G=z['G'];pos=z['positions'];labels=z['population']
    f=np.ascontiguousarray(G[labels==0]);r=np.ascontiguousarray(G[labels==1]);n=f.shape[0]
    assert n==100 and r.shape[0]==100 and not np.any(pos==z['selected_position0'])
    ac=f.sum(axis=0);seg=(ac>0)&(ac<n);common=(ac>=.05*n)&(ac<=.95*n)
    ipos=pos[common];h=np.ascontiguousarray(f[:,common].T)
    ihs=allel.ihs(h,ipos,min_ehh=.05,min_maf=.05,include_edges=False,gap_scale=20000,max_gap=200000,use_threads=False)
    hf=np.ascontiguousarray(f.T);hr=np.ascontiguousarray(r.T)
    xp=allel.xpehh(hf,hr,pos,min_ehh=.05,include_edges=False,gap_scale=20000,max_gap=200000,use_threads=False)
    if a.audit:
        reverse=allel.xpehh(hr,hf,pos,min_ehh=.05,include_edges=False,gap_scale=20000,max_gap=200000,use_threads=False)
        assert np.array_equal(np.isfinite(xp),np.isfinite(reverse))
        assert np.allclose(xp[np.isfinite(xp)],-reverse[np.isfinite(xp)],atol=1e-12)
    pairs=np.column_stack(np.triu_indices(n,1));gpos=pos[seg]
    if a.backend=='gpu':
        import gamma_smc_cu
        pred=gamma_smc_cu.infer(np.ascontiguousarray(f[:,seg]),gpos,pairs=pairs.tolist(),Ne=10000,
              mu=1.25e-8,rho=1e-8,physical_mu=1.25e-8,auto_estimate_theta=False,mean_only=True)
        assert pred['metadata']['time_units']=='generations'
        assert np.array_equal(pred['positions'],gpos) and pred['mean'].shape==(len(gpos),4950)
        assert np.all(np.isfinite(pred['mean'])) and np.all(pred['mean']>0)
        logtmrca=np.log(pred['mean'].astype(np.float64)).mean(axis=1)
        gamma_metadata=pred['metadata'];api_sha=sha(Path(gamma_smc_cu.__file__).with_name('infer.py'))
    else:
        import zstandard
        binary=Path(os.environ['GAMMA_SMC_BIN']);flow=Path(os.environ['GAMMA_FLOW'])
        vcf=a.out/'input.vcf.gz';fg=f[:,seg]
        with gzip.open(vcf,'wt') as out:
            out.write('##fileformat=VCFv4.2\n##contig=<ID=1,length=1000000>\n##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
            out.write('#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t'+'\t'.join(f'S{i}' for i in range(50))+'\n')
            for j,position in enumerate(gpos):
                gt='\t'.join(f'{fg[2*i,j]}|{fg[2*i+1,j]}' for i in range(50))
                out.write(f'1\t{position+1}\tS{j}\tA\tC\t.\tPASS\t.\tGT\t{gt}\n')
        posterior=a.out/'posterior.zst'
        command=[str(binary),'-i',str(vcf),'-o',str(posterior),'-m','0.0005','-r','0.0004','-f',str(flow),'-h']
        process=subprocess.run(command,capture_output=True,text=True,timeout=600)
        (a.out/'gamma_stdout.txt').write_text(process.stdout);(a.out/'gamma_stderr.txt').write_text(process.stderr)
        if process.returncode:raise RuntimeError('CPU reference failed; see gamma_stderr.txt')
        meta=json.loads(Path(str(posterior)+'.meta').read_text())
        assert meta['num_pairs']==4950 and np.array_equal(meta['output_positions'],gpos)
        assert np.isclose(meta['scaled_mutation_rate'],.0005) and np.isclose(meta['scaled_recombination_rate'],.0004)
        chunks=int(np.ceil(meta['num_pairs']/meta['chunk_size']))
        with posterior.open('rb') as stream:
            raw=np.frombuffer(zstandard.ZstdDecompressor().stream_reader(stream).read(),dtype=np.float32)
        raw=raw.reshape(chunks,2,meta['sequence_length'],meta['chunk_size'])
        params=np.transpose(raw,[1,2,0,3]).reshape(2,meta['sequence_length'],-1)[:,:,:4950]
        assert np.all(np.isfinite(params)) and np.all(params>0)
        logtmrca=(np.log(params[0].astype(float))-np.log(params[1].astype(float))+np.log(20000.)).mean(axis=1)
        gamma_metadata=dict(backend='CPU Gamma-SMC reference',time_units='generations',Ne=10000,mu=1.25e-8,r=1e-8,
             native_theta=meta['scaled_mutation_rate'],native_rho=meta['scaled_recombination_rate'],
             generations_per_coalescent_unit=20000,flow_sha256=sha(flow),binary_sha256=sha(binary),posterior_sha256=sha(posterior))
        api_sha=sha(binary)
    np.savez_compressed(a.out/'raw_scores.npz',gpos=gpos,logtmrca=logtmrca,
         ipos=ipos,ihs=ihs,derived_frequency=ac[common]/n,xpos=pos,xp=xp,
         selected_position0=z['selected_position0'])
    result=dict(simulation=record['config'],outcome=record['outcome'],seconds=time.monotonic()-start,
        input_sha256=record['input_sha256'],source_sha256=sha(__file__),scores_sha256=sha(a.out/'raw_scores.npz'),
        gamma_metadata=gamma_metadata,gamma_api_sha256=api_sha,backend=a.backend,
        versions={m:importlib.metadata.version(m) for m in ['numpy','scikit-allel']},
        counts=dict(pooled=len(pos),focal_segregating=len(gpos),focal_common=len(ipos),
                    ihs_finite=int(np.isfinite(ihs).sum()),xp_finite=int(np.isfinite(xp).sum())),
        xp_sign_audit=a.audit,slurm_job=os.environ['SLURM_JOB_ID'])
    (a.out/'COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--audit',action='store_true');p.add_argument('--backend',choices=['gpu','cpu'],default='gpu');run(p.parse_args())
