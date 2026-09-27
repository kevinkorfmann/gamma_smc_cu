"""Extract the exact trusted deposited fit used in the specificity experiment."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from common import digest, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    source=Path(__file__).resolve().parents[2]/'analysis/recombination/extract_buffalo_kern.py'
    spec=importlib.util.spec_from_file_location('trusted_bprime_reader',source)
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    if digest(a.model)!=reader.MODEL_SHA:raise ValueError('Untrusted source pickle')
    with a.model.open('rb') as f:obj=reader.PublishedReader(f).load()
    m=obj.fits[('yri','100000','sparse')]['mbp']
    if list(m.features)!=['cadd6'] or m._fixed_mu is not None or m._indices_fit is not None:
        raise ValueError('Unsupported model')
    frame=reader.predictions(m)
    a.out.mkdir(exist_ok=False,parents=True)
    np.savez_compressed(a.out/'published_fit.npz',coeff=m.logB_fit[:,0,:,0],
        Y=m.Y,theta=m.theta_,t=m.t,chrom=frame.chrom.to_numpy(dtype=str),
        start=frame.start.to_numpy(),end=frame.end.to_numpy(),Bprime=frame.Bprime.to_numpy(),
        log10_pi0_bounds=m.log10_pi0_bounds,log10_mu_bounds=m.log10_mu_bounds)
    write_json(a.out/'provenance.json',dict(source_model=str(a.model),source_model_sha256=reader.MODEL_SHA,
        author_code_commit=reader.AUTHOR_COMMIT,model='YRI CADD6 deCODE altgrid 100kb sparse mbp initial',
        compact_sha256=digest(a.out/'published_fit.npz'),published_nll=float(m.nll_),rows=len(frame),
        features=list(m.features),extractor_sha256=digest(__file__),reader_sha256=digest(source)))


if __name__=='__main__':main()
