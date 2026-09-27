"""Validate new annotation merging and selected/neutral mutation allocation."""
import argparse
import json
from pathlib import Path
import numpy as np
import tskit
from common import digest,write_json
from prepare_stronger_bgs import merge,dilate_to_fraction


def main():
    p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--regions',type=Path,required=True);p.add_argument('--base',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    np.testing.assert_array_equal(merge([[-2,5],[4,9],[12,15],[15,18]],20),[[0,9],[12,18]])
    x,pad=dilate_to_fraction(np.array([[10,15],[18,20]]),100,.30)
    assert int(np.diff(x,axis=1).sum())>=30
    assert np.diff(merge([[10-pad+1,15+pad-1],[18-pad+1,20+pad-1]],100),axis=1).sum()<30
    base=np.load(a.base/'maps.npz');models=json.loads((a.regions/'models.json').read_text())['models']
    for model in models:
        region=Path(model['region']);z=np.load(region/'maps.npz')
        np.testing.assert_array_equal(z['map_position'],base['map_position'])
        np.testing.assert_array_equal(z['map_rate'],base['map_rate'])
        x=z['selected_intervals'];assert np.all(x[:,1]>x[:,0]) and np.all(x[1:,0]>x[:-1,1])
        assert np.diff(x,axis=1).sum()==model['model']['selected_bp']
    checked=[]
    for kind in ['founder','control']:
        run=a.fixture/kind;cfg=json.loads((run/'config.json').read_text())
        raw=tskit.load(run/'final.raw.trees');real=tskit.load(run/'final.variants.trees')
        reporter=tskit.load(run/'final.sample.trees')
        assert raw.metadata['SLiM']['tick']==cfg['end_tick']
        assert real.num_samples==reporter.num_samples==50
        assert np.all(real.nodes_time[real.samples()]==0)
        np.testing.assert_allclose(real.diversity(mode='branch'),reporter.diversity(mode='branch'),rtol=1e-12)
        coeff=[q['selection_coeff'] for m in raw.mutations() for q in m.metadata['mutation_list']]
        assert len(coeff)>0 and all(np.isclose(s,-.001) for s in coeff)
        intervals=np.load(a.regions/cfg['region']['selection_model']['id']/'maps.npz')['selected_intervals']
        count=0
        for m in real.mutations():
            pos=real.site(m.site).position;i=np.searchsorted(intervals[:,0],pos,side='right')-1
            in_target=i>=0 and pos<intervals[i,1]
            if in_target and m.time<raw.metadata['SLiM']['tick']-1:
                assert all(q['mutation_type']!=99 for q in m.metadata['mutation_list']), 'Neutral overlay leaked into fully selected forward target'
                count+=1
        assert count>0
        checked.append(dict(kind=kind,persisted_deleterious_entries=len(coeff),checked_forward_target_mutations=count,
            final_tick=raw.metadata['SLiM']['tick'],genealogies_identical=True,neutral_overlay_allocation_correct=True))
    write_json(a.out,dict(passed=True,technical_fixture_only=True,Ne=100,simulations_excluded_from_production=True,
        interval_merge_and_dilation=True,all_six_recombination_maps_identical=True,checks=checked,
        source_sha256=digest(__file__),simulator_sha256=digest(Path(__file__).with_name('stronger_bgs.py'))))
    print('Stronger-BGS technical validation passed.')


if __name__=='__main__':main()
