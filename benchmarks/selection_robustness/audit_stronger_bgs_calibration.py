"""Independent tree-based checks of all BGS-only calibration families."""
import argparse
import json
from pathlib import Path
import numpy as np
import tskit
from common import digest, write_json


def main():
    p=argparse.ArgumentParser()
    for k in ['bank','summary','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();m=json.loads((a.bank/'manifest.json').read_text())
    s=json.loads(a.summary.read_text());base=Path(m['baseline'])
    assert s['bank_manifest_sha256']==digest(a.bank/'manifest.json')
    assert digest(base/'manifest.json')==m['baseline_manifest_sha256']
    assert len(s['results'])==6
    reference_region=json.loads((base/'neutral_family_000/control/config.json').read_text())['region']
    reference_path=next(p.parent/'maps.npz' for p in (base.parents[2]/'regions').rglob('region.json')
        if json.loads(p.read_text()).get('maps_sha256')==reference_region['maps_sha256'])
    assert digest(reference_path)==reference_region['maps_sha256']
    refmap=np.load(reference_path)
    records=[];seeds=set()
    for model in m['models']:
        region=Path(model['region']);r=json.loads((region/'region.json').read_text());z=np.load(region/'maps.npz')
        assert digest(region/'maps.npz')==r['maps_sha256']
        assert r['selection_model']==model['model']
        np.testing.assert_array_equal(z['map_position'],refmap['map_position'])
        np.testing.assert_array_equal(z['map_rate'],refmap['map_rate'])
        target=z['selected_intervals'];assert np.diff(target,axis=1).sum()==model['model']['selected_bp']
        family_means=[]
        for i in range(6):
            fam=a.bank/model['name']/f'family_{i:03d}'
            assert json.loads((fam/'family_status.json').read_text())['status']=='completed'
            for kind,stages,end in [('founder',{'burn5Ne','final'},100001),('control',{'control_t100','control_t300','control_t1000','final'},102501)]:
                d=fam/kind;cfg=json.loads((d/'config.json').read_text());pr=json.loads((d/'processed.json').read_text())
                assert cfg['bgs'] and not cfg['sweep'] and cfg['Ne']==10000 and cfg['mu']==1.29e-8 and cfg['scaling_factor']==1
                assert cfg['source_sha256']==pr['processor_sha256']==m['sources']['stronger_bgs.py']
                assert cfg['region']==r and cfg['end_tick']==end
                assert cfg['seed'] not in seeds;seeds.add(cfg['seed'])
                assert {v['stage'] for v in pr['results']}==stages
                assert all(v['sample_size']==50 for v in pr['results'])
                if kind=='control':assert cfg['parent_sha256']==json.loads((fam/'founder/simulation_complete.json').read_text())['final_sha256']
            d=fam/'control';reporter=tskit.load(d/'final.sample.trees');real=tskit.load(d/'final.variants.trees')
            assert reporter.num_samples==real.num_samples==50
            assert np.all(real.nodes_time[real.samples()]==0)
            data=np.load(d/'final.npz');edges=data['edges'];mask=abs((edges[:-1]+edges[1:])/2-5000000)<50000
            tmrca=real.diversity(windows=edges,mode='branch')/2
            np.testing.assert_allclose(tmrca,data['tmrca'],rtol=1e-12)
            np.testing.assert_allclose(reporter.diversity(windows=edges),data['pi'],rtol=1e-12)
            np.testing.assert_allclose(real.diversity(mode='branch'),reporter.diversity(mode='branch'),rtol=1e-12)
            family_means.append(float(tmrca[mask].mean()))
            if i==0:
                raw=tskit.load(d/'final.raw.trees')
                coeff=[q['selection_coeff'] for u in raw.mutations() for q in u.metadata['mutation_list']]
                assert coeff and any(v<0 for v in coeff)
                if model['model']['dfe']=='fixed':
                    assert all(np.isclose(v,model['model']['s']) for v in coeff)
                    for u in real.mutations():
                        pos=real.site(u.site).position;j=np.searchsorted(target[:,0],pos,side='right')-1
                        if j>=0 and pos<target[j,1] and u.time<102500:
                            assert all(q['mutation_type']!=99 for q in u.metadata['mutation_list'])
            records.append(dict(model=model['name'],family=i,truth_reporter_genealogy_identical=True,
                direct_tree_tmrca=family_means[-1],samples=50))
        row=next(v for v in s['results'] if v['model']==model['name'])
        np.testing.assert_allclose(np.mean(family_means),row['effects']['focal100kb_tmrca']['mean'],rtol=1e-12)
    for f in sorted(base.glob('neutral_family_*')):
        cfg=json.loads((f/'control/config.json').read_text())
        assert not cfg['bgs'] and not cfg['sweep'] and cfg['Ne']==10000 and cfg['mu']==1.29e-8
        assert cfg['region']['map_archive_sha256']==r['map_archive_sha256']
        assert cfg['region']['left']==r['left'] and cfg['region']['right']==r['right']
    write_json(a.out,dict(passed=True,families=36,independent_reference_families=30,
        distinct_simulation_seeds=len(seeds),checks=records,source_sha256=digest(__file__),summary_sha256=digest(a.summary)))
    print('Independent calibration audit passed: 36 families, 72 distinct simulation seeds, tree truth and neutral reporters verified.')


if __name__=='__main__':main()
