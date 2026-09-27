"""Gamma-SMC-CU on actual simulated polymorphisms; score identical sample pairs."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import tskit
from common import digest, extract_inputs, interval_means, snp_cells, write_json


def infer_stage(path, cfg, out, crops):
    import gamma_smc_cu
    from gamma_smc_cu.infer import _estimate_scaled_params
    ts=tskit.load(path)
    G,pos,dropped=extract_inputs(ts)
    pairs=np.array([(i,j) for i in range(len(G)) for j in range(i)],dtype=np.int32)
    region=cfg['region'];L=region['length']
    edges=np.arange(0,L+1,10000,dtype=float)
    if edges[-1]!=L:edges=np.r_[edges,L]
    truth=ts.divergence(sample_sets=[[int(s)] for s in ts.samples()],indexes=pairs,
        windows=edges,mode='branch')/2
    core=region.get('scoring_interval0',[100000,L-100000])
    core_mask=(edges[:-1]>=core[0])&(edges[1:]<=core[1])
    out.mkdir(exist_ok=False,parents=True)
    info=dict(tree_sha256=digest(path),source_sha256=digest(__file__),
        genotype_shape=list(G.shape),dropped=dropped,pairs=len(pairs),
        map_input='same scalar mean of generating deCODE map; local map not accepted by this API',
        mu=cfg['mu'],rho=region['mean_r'],Ne=cfg['Ne'],
        core_interval0=core,selection_aware_prior=False,
        input_type='real simulated variants: selected mutations plus neutral overlays')
    pkg=Path(gamma_smc_cu.__file__).parent
    info['software_files_sha256']={f.name:digest(f) for f in [pkg/'infer.py',pkg/'default_flow_field.txt',*pkg.glob('_core*.so')]}
    info['api_sha256']=digest(Path(gamma_smc_cu.__file__).with_name('infer.py'))
    start=time.monotonic();arrays=dict(edges=edges,truth=truth,pairs=pairs,positions=pos)
    for mode,auto in [('auto',True),('fixed',False)]:
        result=gamma_smc_cu.infer(G,pos,pairs=pairs.tolist(),mu=cfg['mu'],
            rho=region['mean_r'],Ne=cfg['Ne'],mean_only=True,auto_estimate_theta=auto)
        if not np.array_equal(result['positions'],pos):raise ValueError('Changed SNP set')
        raw=result['mean']
        if raw.shape!=(len(pos),len(pairs)) or np.any(~np.isfinite(raw)) or np.any(raw<=0):
            raise ValueError('Invalid Gamma-SMC-CU estimates')
        pred,cov=interval_means(*snp_cells(pos),raw,edges)
        valid=core_mask&(cov>.999)
        error=np.log(pred[valid])-np.log(truth[valid])
        arrays.update({mode:pred,mode+'_coverage':cov})
        info[mode]=dict(core_coverage=float(valid.sum()/core_mask.sum()),
            mean_log_error=float(error.mean()),rmse_log=float(np.sqrt((error**2).mean())))
        if auto:
            km,kr=_estimate_scaled_params(G,pos,cfg['mu'],region['mean_r'],cfg['Ne'])
            info[mode].update(effective_kernel_mu=km,effective_kernel_rho=kr)
    # Common generating genealogy, cropped inputs: quantify inference context
    # sensitivity separately from biological region-length sensitivity.
    if crops:
        info['context_sensitivity']={}
        focal=cfg['focal_position0']
        for width in [3016000,6000000]:
            lo=focal-width/2;hi=focal+width/2
            use=(pos>=lo)&(pos<hi)
            for mode,auto in [('auto',True),('fixed',False)]:
                # The API estimates theta over [0,last_SNP+1). A crop is a
                # new contig and MUST start at coordinate zero for calibration.
                result=gamma_smc_cu.infer(G[:,use],pos[use]-lo,pairs=pairs.tolist(),
                    mu=cfg['mu'],rho=region['mean_r'],Ne=cfg['Ne'],
                    mean_only=True,auto_estimate_theta=auto)
                pred,cov=interval_means(*snp_cells(pos[use]),result['mean'],edges)
                interior=core_mask&(cov>.999)&(edges[:-1]>=lo+250000)&(edges[1:]<=hi-250000)
                delta=np.abs(pred[interior]-arrays[mode][interior])/arrays[mode][interior]
                name=f'{mode}_{width}'
                info['context_sensitivity'][name]=dict(sites=int(use.sum()),
                    median_relative_change=float(np.median(delta)),p95_relative_change=float(np.quantile(delta,.95)),
                    assessed_interval0=[float(edges[:-1][interior].min()),float(edges[1:][interior].max())])
                arrays[name]=pred
    counts=np.histogram(pos,bins=edges)[0]
    arrays['snp_counts']=counts
    info.update(seconds=time.monotonic()-start,total_snps=len(pos),
        core_snps=int(((pos>=core[0])&(pos<core[1])).sum()),
        focal100kb_snps=int((np.abs(pos-cfg['focal_position0'])<50000).sum()),
        no_extrapolation_beyond_first_last_snp=True)
    np.savez_compressed(out/'predictions.npz',**arrays)
    write_json(out/'metrics.json',info)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--lock',type=Path,required=True);p.add_argument('--crops',action='store_true')
    a=p.parse_args();cfg=json.loads((a.run/'config.json').read_text())
    with a.lock.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for path in sorted(a.run.glob('*.variants.trees')):
            out=a.run/'gamma'/path.name.removesuffix('.variants.trees')
            if (out/'metrics.json').exists():continue
            infer_stage(path,cfg,out,a.crops)


if __name__=='__main__':main()
