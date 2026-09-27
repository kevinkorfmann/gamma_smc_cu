"""Generate fresh ASMC quantities from corrected, pinned haploid demography.

CPU-only setup. The built-in population histories were calibrated to mu=1.65e-8.
Both time and haploid population size are rescaled to the requested physical mu.
No old externally named CEU_50 decoding file is trusted or overwritten.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import traceback
import numpy as np

PREPARE_VERSION = '2.2.5'
SOURCE_MU = 1.65e-8
POPS = ['ACB','ASW','BEB','CDX','CEU','CHB','CHS','CLM','ESN','FIN','GBR','GIH',
        'GWD','IBS','ITU','JPT','KHV','LWK','MSL','MXL','PEL','PJL','PUR','STU','TSI','YRI']


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for part in iter(lambda:stream.read(1<<20),b''):h.update(part)
    return h.hexdigest()


def rescale_demography(data, physical_mu):
    data=np.asarray(data,dtype=np.float64)
    if (data.ndim!=2 or data.shape[1]!=2 or len(data)<2 or not np.isfinite(data).all()
            or data[0,0]!=0 or np.any(np.diff(data[:,0])<=0) or np.any(data[:,1]<=0)):
        raise ValueError('Demography must contain increasing times starting at0 and positive haploid population sizes')
    if not np.isfinite(physical_mu) or physical_mu<=0:
        raise ValueError('Physical mutation rate must be finite and positive')
    # Preserve mu*t and mu*N, hence the model in mutation-scaled units.
    return data*(SOURCE_MU/physical_mu)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True,help='new immutable directory')
    parser.add_argument('--demography',choices=POPS,default='CEU',help='corrected built-in haploid history; CEU preserves the original reference model')
    parser.add_argument('--samples',type=int,choices=[50,100,200,300],default=300,help='CSFS haploid sample count, not number of diploid input samples')
    parser.add_argument('--quantiles',type=int,default=50)
    parser.add_argument('--mu',type=float,default=1.25e-8)
    args=parser.parse_args()
    if args.quantiles<2 or not np.isfinite(args.mu) or args.mu<=0:
        parser.error('positive physical mu and at least2 quantiles are required')
    version=importlib.metadata.version('asmc-preparedecoding')
    if version!=PREPARE_VERSION:
        raise RuntimeError(f'Requires asmc-preparedecoding=={PREPARE_VERSION}; found {version}')
    from asmc.preparedecoding import prepare_decoding, save_demography
    output=Path(args.output_dir).resolve()
    output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    try:
        original=output/'builtin_demography';original.mkdir()
        save_demography(str(original)+'/',args.demography)
        original_file=original/f'{args.demography}.demo'
        data=np.loadtxt(original_file)
        rescaled=rescale_demography(data,args.mu)
        model=output/f'{args.demography}.physical_mu.demo'
        np.savetxt(model,rescaled,fmt='%.17g',delimiter='\t')
        dq=prepare_decoding(demography=str(model),discretization=[args.quantiles],
                            frequencies='UKBB',samples=args.samples,mutation_rate=args.mu)
        prefix=str(output/'decoding')
        dq.save_decoding_quantities(prefix)
        dq.save_csfs(prefix)
        dq.save_intervals(prefix)
        dq.save_discretization(prefix)
        if int(dq.samples)!=args.samples or not np.isclose(float(dq.mu),args.mu,rtol=1e-12,atol=0):
            raise ValueError('Generated quantities report unexpected sample count or mutation rate')
        metadata={'schema_version':1,'status':'complete','arguments':vars(args),
            'preparedecoding_version':version,'python':sys.version,'platform':platform.platform(),
            'script_sha256':sha256(__file__),'runtime_seconds':time.perf_counter()-start,
            'demography':args.demography,'population_size_units':'haploid',
            'builtin_demography_mutation_rate':SOURCE_MU,'physical_mutation_rate':args.mu,
            'time_and_population_size_multiplier':SOURCE_MU/args.mu,
            'csfs_samples_haploid':int(dq.samples),'actual_states':int(dq.states),
            'discretization_specification':[args.quantiles],
            'frequency_model':'built-in UKBB; ASMC runs must select sequence mode for unascertained sequence emissions',
            'recommended_asmc_version':'1.4.0',
            'source_documentation':['https://github.com/PalamaraLab/ASMC_data#demographies',
              'https://github.com/PalamaraLab/PrepareDecoding/blob/main/docs/api.md',
              'https://github.com/PalamaraLab/ASMC/blob/main/docs/asmc.md#decoding-quantities-decodingquantitiesgz'],
            'files':{str(p.relative_to(output)):{'sha256':sha256(p),'bytes':p.stat().st_size}
                     for p in sorted(output.rglob('*')) if p.is_file()}}
        (output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps(metadata),flush=True)
    except BaseException:
        (output/'FAILED.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__':
    main()
