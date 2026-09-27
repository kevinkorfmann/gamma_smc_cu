#!/usr/bin/env python3
"""Audit every final map, its generating manifest, samples and split membership."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from infer_maps import sha, EXPECTED

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args()
    targets=pd.read_csv(a.root/'inputs/targets.tsv',sep='\t')
    panel=pd.read_csv(a.root/'inputs/phase3.panel',sep=r'\s+',usecols=[0,1,2,3])
    manifests={sha(p):pd.read_csv(p,sep='\t').set_index('gene') for p in (a.root/'inputs').glob('targets*.tsv')}
    script=Path(__file__).with_name('infer_maps.py');n=0;halves=0;sample_sizes={}
    for t in targets.itertuples():
        for pop in t.populations.split(','):
            labels=[pop]+([f'{pop}_half{k}' for k in [1,2]] if t.gene in ['GRK2','TREM2'] else [])
            memberships=[]
            for label in labels:
                prefix=a.root/'maps'/f'{t.gene}.{label}'
                m=json.loads(Path(str(prefix)+'.json').read_text())
                assert m['script_sha256']==sha(script)
                assert m['model_sha256']==EXPECTED
                assert m['panel_sha256']==sha(a.root/'inputs/phase3.panel')
                origin=manifests[m['manifest_sha256']].loc[t.gene]
                assert pop in origin.populations.split(',')
                assert m['display_region0']==[t.display_start0,t.display_end0]==[int(origin.display_start0),int(origin.display_end0)]
                assert m['extraction_region0']==[t.extract_start0,t.extract_end0]
                assert m['gene']==t.gene and m['population']==label
                samples=set(m['samples']);assert len(samples)==m['n_diploid']==len(m['samples'])
                memberships.append(samples)
                if label==pop:
                    assert samples==set(panel[panel['pop']==pop]['sample'])
                    sample_sizes[pop]=len(samples);n+=1
                else:halves+=1
                for name,digest in m['output_sha256'].items():assert sha(a.root/'maps'/name)==digest
                z=np.load(str(prefix)+'.intervals.npz')
                lo,hi=z['pos_left'],z['pos_right'];rho=z['rho_per_bp']
                assert len(lo)==m['intervals'] and np.all(hi>lo) and np.array_equal(lo[1:],hi[:-1])
                assert lo[0]>=t.extract_start0 and hi[-1]<=t.extract_end0
                assert np.all(np.isfinite(rho)) and np.all(rho>0)
                assert np.allclose(z['r_per_bp'],rho/40000)
                assert float(z['Ne_used'])==10000 and m['mutation_rate']==1.25e-8
            if len(memberships)==3:
                assert not (memberships[1]&memberships[2])
                assert memberships[1]|memberships[2]==memberships[0]
    report={'primary_maps':n,'split_maps':halves,'genes':len(targets),'all_checks_passed':True,
            'checks':['output checksums','model/script/panel provenance','generating manifests',
                      'sample membership','coordinate spans','contiguous intervals','finite positive rho',
                      'fixed-Ne rate scaling','disjoint diploid halves'],
            'diploid_sample_sizes':sample_sizes}
    (a.root/'results/run_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
